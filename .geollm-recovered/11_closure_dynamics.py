#!/usr/bin/env python3
"""
GeoLLM v6 — closure dynamics
============================
Faltz's full ingest loop, ported to LM. The genome only stores observations
that SURPRISE the model. Predictable transitions silently update cell_c
without writing.

Algorithm:
  cell_c = IDENTITY
  for each train token t:
      carrier = embed(t)
      total   = ZREAD(genome, query=cell_c)              # what the field says
      known   = cell_c                                    # what model predicts
      sigma_pred = sigma(compose(total, inverse(known)))  # prediction tension
      actual_residual = compose(carrier, inverse(known))
      sigma_actual    = sigma(actual_residual)            # observation gap
      if sigma_actual > pi/4:
          # CLOSE: write to genome, update cell_c toward observation
          genome.append((cell_c.copy(), carrier))
          cell_c = slerp(cell_c, carrier, alpha=0.3)
          cell_c = normalize(cell_c)
      else:
          # No closure: cell_c drifts gently
          cell_c = slerp(cell_c, carrier, alpha=0.05)
          cell_c = normalize(cell_c)

Inference is the same v3 ZREAD with full+phase scoring; we just query
against the (much smaller) closure-genome.
"""
import math, hashlib
from pathlib import Path
import numpy as np

I_q = np.array([1.0, 0.0, 0.0, 0.0])

# ── Quaternion ops (subset reused across files) ───────────────────────
def hamilton(a, b):
    w1,x1,y1,z1 = a; w2,x2,y2,z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2])

def normalize(v, eps=1e-12):
    n = np.linalg.norm(v); return v if n < eps else v / n

def inverse(q):
    """Conjugate / inverse of a unit quaternion."""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def sigma(q):
    """Geodesic distance from q to identity, in [0, π]."""
    return math.acos(max(-1.0, min(1.0, q[0])))

def slerp(a, b, t):
    """Spherical linear interpolation on S^3."""
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot < 0:  # take shorter arc
        b = -b; dot = -dot
    if dot > 0.9995:
        return normalize(a + t*(b - a))
    omega = math.acos(dot)
    so = math.sin(omega)
    return (math.sin((1-t)*omega)/so) * a + (math.sin(t*omega)/so) * b

def axis_angle(axis, theta):
    a = np.asarray(axis, dtype=np.float64); a = a / (np.linalg.norm(a) + 1e-12)
    c, s = math.cos(theta/2), math.sin(theta/2)
    return np.array([c, s*a[0], s*a[1], s*a[2]])

# ── Hopf / domain embed ───────────────────────────────────────────────
def hopf_decompose(q):
    w,x,y,z = q
    base = np.array([2*(x*z+w*y), 2*(y*z-w*x), w*w+z*z-x*x-y*y])
    n = np.linalg.norm(base); base = base/n if n > 1e-15 else np.array([0.,0.,1.])
    phase = (math.atan2(z, w) + math.atan2(x, y)) % (2*math.pi)
    return base, phase

def carrier_from_hopf(base, phase):
    base = base / (np.linalg.norm(base) + 1e-12); phase = phase % (2*math.pi)
    bz = max(-1.0, min(1.0, base[2]))
    sin_t = math.sqrt(max(0.0, 1.0 - bz*bz))
    if sin_t < 1e-12:
        if bz >= 0: return np.array([math.cos(phase),0,0,math.sin(phase)])
        return np.array([0, math.sin(phase), math.cos(phase), 0])
    theta = math.acos(bz); delta = math.atan2(base[1], base[0])
    alpha = 0.5*(phase+delta); beta = 0.5*(phase-delta)
    ct, st = math.cos(0.5*theta), math.sin(0.5*theta)
    q = np.array([ct*math.cos(alpha), st*math.sin(beta),
                  st*math.cos(beta),  ct*math.sin(alpha)])
    return q / np.linalg.norm(q)

def semantic_base_from_bytes(data):
    h = hashlib.sha256(data).digest()
    u1 = (int.from_bytes(h[:8],'little')+1)/(2**64+1)
    u2 = (int.from_bytes(h[8:16],'little')+1)/(2**64+1)
    z = 1.0 - 2.0*u1; r = math.sqrt(max(0., 1-z*z)); phi = 2*math.pi*u2
    return np.array([r*math.cos(phi), r*math.sin(phi), z])

def domain_embed(token):
    return carrier_from_hopf(semantic_base_from_bytes(token.encode('utf-8')), 0.0)

# ── Operators (carried over from v0..v5; tested null already) ─────────
R_i_pi  = axis_angle([1,0,0], math.pi)
R_j_pi  = axis_angle([0,1,0], math.pi)
R_j_pi2 = axis_angle([0,1,0], math.pi/2)
R_k_pi2 = axis_angle([0,0,1], math.pi/2)
R_diag  = axis_angle([1,1,1],  2*math.pi/3)
R_diag2 = axis_angle([1,1,1], -2*math.pi/3)

OPERATORS = {
    'is':I_q,'are':I_q,'was':I_q,'were':I_q,'be':I_q,'am':I_q,'been':I_q,'being':I_q,
    'the':I_q,'a':I_q,'an':I_q,
    'not':R_i_pi,"n't":R_i_pi,'no':R_i_pi,'never':R_i_pi,
    'and':R_j_pi2,'or':R_k_pi2,'but':R_j_pi,
    'of':I_q,'to':I_q,'in':I_q,'on':I_q,'at':I_q,'for':I_q,'with':I_q,
    'by':I_q,'from':I_q,'as':I_q,'into':I_q,'about':I_q,
    'i':I_q,'you':I_q,'he':I_q,'she':I_q,'it':I_q,'we':I_q,'they':I_q,
    'this':I_q,'that':I_q,'these':I_q,'those':I_q,
    'will':R_diag,'would':R_diag2,'shall':R_diag,
    'can':R_diag,'could':R_diag2,'may':R_diag,'might':R_diag2,
    'should':R_diag2,'must':R_diag,
    'have':I_q,'has':I_q,'had':I_q,
    'do':I_q,'does':I_q,'did':I_q,
}

def _zread_predict(state, genome_states, genome_targets,
                   pending_states=None, pending_targets=None,
                   cutoff=math.pi/3):
    """ZREAD soft attention. Includes pending (unrestacked) entries so a
    closure that just fired is visible to the very next step's
    prediction; without this, the genome would silently grow until
    restack and bypass the σ-threshold gate."""
    parts_states = [genome_states]
    parts_targets = [genome_targets]
    if pending_states:
        parts_states.append(np.stack(pending_states, axis=0))
        parts_targets.append(np.stack(pending_targets, axis=0))
    states  = np.concatenate(parts_states,  axis=0) if parts_states  else np.zeros((0,4))
    targets = np.concatenate(parts_targets, axis=0) if parts_targets else np.zeros((0,4))
    if len(states) == 0:
        return I_q.copy()
    dots = np.clip(np.abs(states @ state), -1.0, 1.0)
    gaps = np.arccos(dots)
    w = np.cos(gaps); w[gaps > cutoff] = 0.0
    w = np.clip(w, 0.0, 1.0)
    if w.sum() < 1e-12:
        return I_q.copy()
    avg = (w[:, None] * targets).sum(axis=0) / w.sum()
    return normalize(avg)

# ── Closure dynamics: σ-thresholded ingest, ARCHITECTURE-CORRECT ──────
def train_closure(tokens, carriers, ops, wid, threshold=math.pi/4,
                  context_len=3, restack_every=64):
    """Walk tokens. At each step:
       - state = compose(left context of length `context_len`)
       - predicted = ZREAD(genome, query=state)
       - actual    = embed(token)
       - residual  = compose(actual, inverse(predicted))
       - if σ(residual) > threshold: close (write (state, actual) to genome)

    Returns list of (state, actual_carrier) entries.

    `restack_every` controls how often we re-pack the genome into a numpy
    array for fast ZREAD; small values are slow, large values let the
    genome grow stale before its predictions update. 512 is a reasonable
    default for hundreds of thousands of observations."""
    genome = []
    g_states  = np.zeros((0, 4), dtype=np.float64)
    g_targets = np.zeros((0, 4), dtype=np.float64)
    pending_states  = []
    pending_targets = []

    n = len(tokens)
    for i in range(n):
        target = tokens[i]
        if target not in wid:
            continue
        # Allow partial context at the start so closures can fire from step 0
        # (matches the architecture: cell_c starts at identity, first
        # observation is by definition surprising).
        ctx = tokens[max(0, i-context_len):i]
        state = compose_state(ctx, carriers, ops)
        actual = carriers[target]
        predicted = _zread_predict(state, g_states, g_targets,
                                    pending_states, pending_targets)
        residual = hamilton(actual, inverse(predicted))
        sigma_actual = sigma(residual)
        if sigma_actual > threshold:
            genome.append((state.copy(), actual.copy()))
            pending_states.append(state)
            pending_targets.append(actual)
            if len(pending_states) >= restack_every:
                # Re-pack into numpy for fast ZREAD on subsequent steps.
                g_states  = np.concatenate([g_states,  np.stack(pending_states,  0)], axis=0)
                g_targets = np.concatenate([g_targets, np.stack(pending_targets, 0)], axis=0)
                pending_states.clear(); pending_targets.clear()

    # Flush any remaining pendings (so test introspection of the genome
    # length matches what train_closure returned).
    if pending_states:
        g_states  = np.concatenate([g_states,  np.stack(pending_states,  0)], axis=0)
        g_targets = np.concatenate([g_targets, np.stack(pending_targets, 0)], axis=0)
    return genome

# ── Inference: ZREAD-style population read ────────────────────────────
def compose_state(ctx_tokens, carriers, ops):
    q = I_q.copy()
    for t in ctx_tokens:
        if t in ops:    q = hamilton(ops[t], q)
        elif t in carriers: q = hamilton(q, carriers[t])
        q = normalize(q)
    return q

def predict_topk(ctx, genome, carriers, ops, wid, V, k=10, cutoff=math.pi/3):
    """Score next tokens via cos(σ) over the closure-genome (full channel),
    return ids of the top-k by score. Returns [] if genome empty."""
    if not genome:
        return []
    q = compose_state(ctx, carriers, ops)
    # Build a vocab map from carrier -> wid via nearest-base lookup.
    # Each genome entry's stored carrier is one of the vocab carriers.
    # We bin by "which vocab token's carrier this is" by searching for an
    # exact match (carrier comes straight from carriers[w]); fall back to
    # nearest-base.
    # Since genome stores carriers[w].copy(), we can identify the token.
    # Build an index from carrier-bytes to wid for fast lookup.
    cid_to_wid = {}
    for w, c in carriers.items():
        cid_to_wid[c.tobytes()] = wid[w]
    states = np.stack([g[0] for g in genome], axis=0).astype(np.float64)
    nxt_ids = []
    for _, c in genome:
        wid_match = cid_to_wid.get(c.tobytes())
        if wid_match is None:
            # Fallback: nearest base
            b_c, _ = hopf_decompose(c)
            best_w = -1; best_d = math.pi
            for w, cc in carriers.items():
                b_cc, _ = hopf_decompose(cc)
                d = math.acos(max(-1.0, min(1.0, abs(np.dot(b_c, b_cc)))))
                if d < best_d:
                    best_d = d; best_w = wid[w]
            wid_match = best_w
        nxt_ids.append(wid_match)
    N = np.array(nxt_ids, dtype=np.int64)
    # Full-channel σ
    dots = np.clip(np.abs(states @ q), -1.0, 1.0)
    gaps = np.arccos(dots)
    w_full = np.cos(gaps); w_full[gaps > cutoff] = 0.0
    s = np.bincount(N, weights=np.clip(w_full, 0, 1), minlength=V)
    if s.sum() == 0:
        return []
    return list(np.argsort(-s)[:k])

# ── If run as script, do a tiny sanity demo ───────────────────────────
def _demo():
    import re, random
    from collections import Counter
    here = Path(__file__).parent / 'corpora'
    text = ' '.join((here/'shakespeare.txt').read_text(errors='ignore').lower()
                    for _ in [0])
    text += ' ' + (here/'pride_prejudice.txt').read_text(errors='ignore').lower()
    toks = re.findall(r"[a-z']+", text)
    n_train = int(len(toks)*0.9); train, test = toks[:n_train], toks[n_train:]
    counts = Counter(train)
    vocab = [w for w,c in counts.most_common(2500) if c >= 10]
    wid = {w:i for i,w in enumerate(vocab)}; V = len(vocab)
    carriers = {w: domain_embed(w) for w in vocab}
    print(f"Vocab {V} · train {len(train):,} · test {len(test):,}")
    print(f"Threshold sweep:")
    for thr_div in [3, 4, 6, 8]:
        thr = math.pi/thr_div
        genome = train_closure(train, carriers, OPERATORS, wid, threshold=thr)
        # Quick eval
        random.seed(42)
        valid = [t for t in test if t in wid]
        pos = random.sample(range(5, len(valid)), 500)
        correct = 0; correct5 = 0; correct10 = 0
        for p in pos:
            ctx = valid[p-5:p]; target_id = wid[valid[p]]
            ranked = predict_topk(ctx, genome, carriers, OPERATORS, wid, V, k=10)
            if ranked and ranked[0] == target_id: correct += 1
            if target_id in ranked[:5]: correct5 += 1
            if target_id in ranked[:10]: correct10 += 1
        print(f"  thr=pi/{thr_div}  genome={len(genome):>6,d}  "
              f"top1={correct/len(pos):.4f}  top5={correct5/len(pos):.4f}  "
              f"top10={correct10/len(pos):.4f}")

if __name__ == '__main__':
    _demo()

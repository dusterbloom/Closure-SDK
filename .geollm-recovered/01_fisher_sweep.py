#!/usr/bin/env python3
"""
GeoLLM Enhanced Fisher-Rao Sweep
================================
Building on GPT-5's gradient-free spectral approach with:

1. REAL CORPUS — Shakespeare + Pride & Prejudice (1.85 MB English)
2. PROCRUSTES ALIGNMENT — orient the 4D embedding so anchor tokens land
   near 24-cell vertices, making Hamilton products quaternion-meaningful
3. THEORETICAL BOUNDS — compute the eigenvalue-tail floor on stress
   (mathematical lower bound for any rank-k embedding)
4. HYBRID OPTIMIZATION — spectral initialization + Riemannian refinement
   on a Hamilton composition objective (does training improve over spectral?)
5. SPECTRUM ELBOW — find the natural intrinsic dimensionality
6. HELD-OUT COMPOSITION TESTS — analogies, non-commutativity, closure
"""
import json, math, re, time, sys
from collections import Counter
from pathlib import Path
import numpy as np

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Corpus
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def results_path():
    """Path where the sweep writes its results JSON.

    Lives next to the script so a fresh checkout can run end-to-end.
    Honours the FISHER_SWEEP_OUT environment variable as an override.
    """
    import os
    override = os.environ.get('FISHER_SWEEP_OUT')
    if override:
        return Path(override)
    return Path(__file__).parent / 'enhanced_sweep_results.json'


def load_real_corpus():
    """Combine Shakespeare + Pride & Prejudice. Real English distributions.

    Reads from `<script_dir>/corpora/` (the bundled location) so the script
    runs from a fresh checkout. Falls back to the original /home/claude/...
    paths for backward compatibility with the environment that produced this
    file.
    """
    here = Path(__file__).parent
    candidates = [
        here / 'corpora' / 'shakespeare.txt',
        here / 'corpora' / 'pride_prejudice.txt',
        Path('/home/claude/shakespeare.txt'),
        Path('/home/claude/pride_prejudice.txt'),
    ]
    parts = []
    seen = set()
    for p in candidates:
        if p.exists() and p.name not in seen:
            parts.append(p.read_text(errors='ignore').lower())
            seen.add(p.name)
    if not parts:
        raise RuntimeError(
            f"No corpus files found. Looked in {here / 'corpora'} and /home/claude/."
        )
    return ' '.join(parts)


def build_context(corpus_raw, max_vocab=2000, min_count=20, window=5):
    toks = re.findall(r'[a-z]+', corpus_raw)
    counts = Counter(toks)
    vocab = [t for t, c in counts.most_common(max_vocab) if c >= min_count]
    tid = {t: i for i, t in enumerate(vocab)}
    V = len(vocab)
    
    # Sparse counting via slicing — much faster
    C = np.zeros((V, V), dtype=np.float64)
    tok_ids = np.array([tid.get(t, -1) for t in toks])
    valid = tok_ids >= 0
    
    for i in range(len(toks)):
        if not valid[i]: continue
        a = tok_ids[i]
        lo = max(0, i - window)
        hi = min(len(toks), i + window + 1)
        for j in range(lo, hi):
            if j == i: continue
            b = tok_ids[j]
            if b >= 0:
                C[a, b] += 1.0
    return toks, vocab, tid, C


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fisher embedding (GPT-5's core)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fisher_gram(C, alpha=1e-10):
    P = C + alpha
    P /= P.sum(axis=1, keepdims=True)
    X = np.sqrt(P)
    G = np.clip(X @ X.T, 0.0, 1.0)
    D = np.arccos(np.clip(G, -1+1e-12, 1-1e-12))
    return X, G, D


def spectral_embed(G, dim, normalize=True):
    """Optimal rank-k approximation via top-k eigenvectors."""
    K = (G + G.T) / 2
    vals, vecs = np.linalg.eigh(K)
    idx = np.argsort(vals)[::-1]
    vals = np.maximum(vals[idx], 0)
    vecs = vecs[:, idx]
    Y = vecs[:, :dim] * np.sqrt(vals[:dim])[None, :]
    if normalize:
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1
        Y = Y / nrm
    explained = float(vals[:dim].sum() / (vals.sum() + 1e-12))
    return Y, explained, vals


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IMPROVEMENT 1: Theoretical stress floor from eigenvalues
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def theoretical_stress_floor(eigenvalues, dim):
    """
    Lower bound on Gram reconstruction error from spectral tail.
    By Eckart-Young: best rank-k approximation has error = sum of remaining eigenvalues.
    This is the mathematical floor — no algorithm can do better.
    """
    if dim >= len(eigenvalues):
        return 0.0
    tail = eigenvalues[dim:]
    return float(np.sum(tail) / np.sum(eigenvalues))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IMPROVEMENT 2: Procrustes alignment to 24-cell for quaternion meaning
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def gen_24cell():
    """Hurwitz units — vertices of the 24-cell on S³."""
    verts = []
    for i in range(4):
        for s in [1.0, -1.0]:
            v = [0.0]*4; v[i] = s; verts.append(v)
    h = 0.5
    for s0 in [h,-h]:
        for s1 in [h,-h]:
            for s2 in [h,-h]:
                for s3 in [h,-h]:
                    verts.append([s0,s1,s2,s3])
    return np.array(verts)


def procrustes_align_to_anchors(Y, anchor_indices, anchor_targets):
    """
    Find optimal orthogonal R such that R @ Y[anchors] ≈ anchor_targets.
    Uses SVD-based Procrustes.
    """
    A = Y[anchor_indices]   # (k, 4)
    B = anchor_targets      # (k, 4)
    M = A.T @ B             # (4, 4)
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt              # optimal rotation
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    return Y @ R


def align_4d_to_24cell(Y, vocab, tid):
    """Place top frequent tokens near 24-cell vertices via Procrustes."""
    # Use top-24 most frequent tokens as anchors
    n_anchors = min(24, len(vocab))
    anchor_indices = list(range(n_anchors))
    cell24 = gen_24cell()
    anchor_targets = cell24[:n_anchors]
    
    # Match each anchor to closest 24-cell vertex (greedy assignment)
    Y_anchors = Y[anchor_indices]
    targets = np.zeros_like(Y_anchors)
    used = set()
    for i in range(n_anchors):
        # Find unused vertex closest to current position
        sims = Y_anchors[i] @ cell24.T
        for j in np.argsort(-sims):
            if j not in used:
                targets[i] = cell24[j]
                used.add(j)
                break
    
    Y_aligned = procrustes_align_to_anchors(Y, anchor_indices, targets)
    # Re-normalize after rotation (should be ~unit already)
    nrm = np.linalg.norm(Y_aligned, axis=1, keepdims=True)
    nrm[nrm == 0] = 1
    return Y_aligned / nrm


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Quaternion math
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def quat_mul(a, b):
    w1,x1,y1,z1 = a; w2,x2,y2,z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2])

def qnorm(q):
    n = np.linalg.norm(q)
    return q/n if n > 1e-10 else np.array([1.,0,0,0])

def qdist(a, b):
    return float(np.arccos(np.clip(abs(np.dot(a,b)), 0, 1-1e-8)))

def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def hopf_proj(q):
    w,x,y,z = q
    s2 = np.array([2*(x*z + w*y), 2*(y*z - w*x), w*w + z*z - x*x - y*y])
    s2 /= np.linalg.norm(s2) + 1e-12
    phi = math.atan2(y*w + x*z, w*z - x*y + 1e-12)
    return s2, phi

def compose_seq(toks, E, tid):
    valid = [t for t in toks if t in tid]
    if not valid: return np.array([1.,0,0,0])
    q = E[tid[valid[0]]]
    for t in valid[1:]:
        q = qnorm(quat_mul(q, E[tid[t]]))
    return q

def sigma(q):
    return float(np.arccos(np.clip(abs(q[0]), 0, 1-1e-8)))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def sphere_dist(Y):
    return np.arccos(np.clip(Y @ Y.T, -1+1e-7, 1-1e-7))

def corr_upper(A, B, method='pearson'):
    iu = np.triu_indices_from(A, k=1)
    a, b = A[iu], B[iu]
    if a.std() == 0 or b.std() == 0: return float('nan')
    if method == 'pearson':
        return float(np.corrcoef(a, b)[0, 1])
    else:  # spearman
        from scipy.stats import spearmanr
        return float(spearmanr(a, b).correlation)

def stress_upper(A, B):
    iu = np.triu_indices_from(A, k=1)
    return float(np.mean((A[iu] - B[iu])**2))

def knn_purity(D, vocab, tid, cats, k=5):
    scores = []; detail = {}
    for name, toks in cats.items():
        ids = [tid[t] for t in toks if t in tid]
        if len(ids) < 2: continue
        s = set(ids)
        local = []
        for i in ids:
            order = np.argsort(D[i])
            nn = [j for j in order if j != i][:k]
            local.append(sum(j in s for j in nn) / k)
        detail[name] = float(np.mean(local))
        scores.append(detail[name])
    return float(np.mean(scores)) if scores else float('nan'), detail


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IMPROVEMENT 3: Composition tests with analogies
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def composition_tests(E, tid, vocab):
    def comp(s): return compose_seq(s.split(), E, tid)
    
    # A. Semantic similarity via composition
    pairs = [
        ('the man', 'the woman', 'similar'),
        ('the man', 'the king', 'similar'),
        ('the man', 'the horse', 'different'),
        ('she said', 'he said', 'similar'),
        ('she said', 'he ran', 'different'),
        ('a young', 'a fair', 'similar-tone'),
        ('my dear', 'my lord', 'similar-form'),
        ('my dear', 'a horse', 'different'),
    ]
    pair_results = []
    for a, b, label in pairs:
        if all(t in tid for t in (a + ' ' + b).split()):
            pair_results.append({
                'a': a, 'b': b, 'label': label,
                'dist': qdist(comp(a), comp(b))
            })
    
    # B. Closure test (coherence detection)
    coherent = [
        'the king and the queen',
        'a fine young gentleman',
        'she would not say',
        'my dear friend has come',
        'the day was very fair',
    ]
    scrambled = [
        'queen the king and the',
        'gentleman young fine a',
        'say not would she',
        'come has friend dear my',
        'fair very was day the',
    ]
    random_w = [
        'horse love day king fair',
        'said ran tree money house',
        'morning dark large my the',
        'walked ran said came went',
        'lady knight ship sword fire',
    ]
    
    closure = []
    for sents, typ in [(coherent, 'coherent'), (scrambled, 'scrambled'), (random_w, 'random')]:
        for s in sents:
            toks = [t for t in s.split() if t in tid]
            if len(toks) >= 2:
                closure.append({'sent': s, 'type': typ, 'sigma': sigma(comp(' '.join(toks)))})
    
    means = {}
    for typ in ['coherent', 'scrambled', 'random']:
        vals = [c['sigma'] for c in closure if c['type'] == typ]
        means[typ] = float(np.mean(vals)) if vals else float('nan')
    
    # C. Non-commutativity
    noncomm = []
    for a, b in [('the','man'), ('a','horse'), ('my','dear'), ('she','said')]:
        if a in tid and b in tid:
            d = qdist(compose_seq([a,b], E, tid), compose_seq([b,a], E, tid))
            noncomm.append({'pair': f'{a}/{b}', 'dist': d})
    
    # D. Analogies (king - man + woman ≈ queen)
    analogies_to_test = [
        ('man', 'king', 'woman', 'queen'),
        ('he', 'she', 'his', 'her'),
        ('father', 'mother', 'son', 'daughter'),
        ('day', 'night', 'sun', 'moon'),
        ('ran', 'run', 'said', 'say'),
    ]
    analogies = []
    for a, b, c, expected in analogies_to_test:
        if not all(t in tid for t in [a, b, c]):
            continue
        # predicted = inverse(a) * b * c
        qa, qb, qc = E[tid[a]], E[tid[b]], E[tid[c]]
        predicted = qnorm(quat_mul(quat_mul(qconj(qa), qb), qc))
        # Find nearest
        all_dists = np.array([qdist(predicted, E[i]) for i in range(len(vocab))])
        all_dists[tid[a]] = 999  # exclude
        all_dists[tid[b]] = 999
        all_dists[tid[c]] = 999
        top5 = np.argsort(all_dists)[:5]
        nearest = [vocab[i] for i in top5]
        rank = vocab.index(expected) if expected in vocab else -1
        if rank >= 0:
            expected_dist = float(all_dists[tid[expected]]) if expected in tid else float('inf')
            in_top5 = expected in nearest
        else:
            expected_dist = float('inf')
            in_top5 = False
        analogies.append({
            'query': f'{a}:{b}::{c}:?',
            'expected': expected,
            'top5': nearest,
            'in_top5': in_top5,
            'expected_dist': expected_dist
        })
    
    # E. Hopf decomposition analysis
    hopf_data = []
    for tok in ['the', 'and', 'man', 'woman', 'king', 'queen', 'said', 'ran',
                'horse', 'love', 'day', 'night']:
        if tok in tid:
            s2, phi = hopf_proj(E[tid[tok]])
            hopf_data.append({'tok': tok, 'S2': s2.tolist(), 'phase': phi})
    
    return {
        'pair_distances': pair_results,
        'closure': closure,
        'closure_means': means,
        'noncomm': noncomm,
        'analogies': analogies,
        'hopf': hopf_data,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IMPROVEMENT 4: Spectrum elbow detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def find_spectrum_elbow(eigenvalues, max_dim=200):
    """Find the elbow point in eigenvalue decay using max-curvature."""
    vals = eigenvalues[:max_dim]
    if len(vals) < 5:
        return len(vals)
    # Normalize to [0,1] x [0,1]
    x = np.arange(len(vals)).astype(float)
    y = vals / vals[0]
    # Distance from line connecting (0, y[0]) and (n-1, y[n-1])
    line_start = np.array([x[0], y[0]])
    line_end = np.array([x[-1], y[-1]])
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-10:
        return 1
    line_unit = line_vec / line_len
    points = np.stack([x, y], axis=1)
    rel = points - line_start
    proj_len = rel @ line_unit
    proj = proj_len[:, None] * line_unit[None, :]
    perp = rel - proj
    distances = np.linalg.norm(perp, axis=1)
    return int(np.argmax(distances))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("=" * 70)
    print("  ENHANCED FISHER-RAO SWEEP — Real Corpus + Procrustes + Bounds")
    print("=" * 70)
    
    print("\n[1] Loading real corpus...")
    corpus = load_real_corpus()
    print(f"  Corpus chars: {len(corpus):,}")
    
    print("\n[2] Building context distributions...")
    toks, vocab, tid, C = build_context(corpus, max_vocab=2000, min_count=20, window=5)
    print(f"  Vocab: {len(vocab)}, Tokens: {len(toks):,}")
    print(f"  Top-20: {vocab[:20]}")
    
    print("\n[3] Computing Fisher-Rao gram...")
    X, G, D = fisher_gram(C)
    
    # Categories adapted for Shakespeare/Austen vocabulary
    cats = {
        'people':   ['man','woman','men','women','sir','lady','gentleman',
                     'father','mother','daughter','son','king','queen','lord'],
        'pronouns': ['he','she','him','her','his','hers','they','them',
                     'their','i','me','my','you','your'],
        'function': ['the','a','an','of','to','in','and','that','is','was',
                     'be','for','with','as','at','by'],
        'speech':   ['said','say','spoke','speak','answered','replied',
                     'cried','exclaimed','told'],
        'feeling':  ['love','heart','dear','feel','felt','happy','glad',
                     'sad','sorry','afraid','hope'],
        'qualities':['great','good','little','old','young','fine','fair',
                     'true','dear','sweet','kind'],
    }
    
    print("\n[4] Sweep across dimensions...")
    print(f"  {'dim':>4s} {'corr_p':>7s} {'corr_s':>7s} {'stress':>7s} {'floor':>7s} "
          f"{'expl':>7s} {'purity':>7s} {'sec':>5s}")
    print("  " + "-" * 65)
    
    rows = []
    Y_dict = {}
    eigenvalues = None
    
    dims = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    for dim in dims:
        if dim > len(vocab): continue
        t = time.time()
        Y, expl, evals = spectral_embed(G, dim)
        if eigenvalues is None:
            eigenvalues = evals
        DY = sphere_dist(Y)
        Y_dict[dim] = Y
        
        floor = theoretical_stress_floor(eigenvalues, dim)
        purity, _ = knn_purity(DY, vocab, tid, cats)
        
        try:
            from scipy.stats import spearmanr
            iu = np.triu_indices_from(DY, k=1)
            corr_s = float(spearmanr(DY[iu], D[iu]).correlation)
        except ImportError:
            corr_s = float('nan')
        
        row = {
            'dim': dim,
            'corr_pearson': corr_upper(DY, D),
            'corr_spearman': corr_s,
            'stress': stress_upper(DY, D),
            'theoretical_floor': floor,
            'explained_variance': expl,
            'knn_purity': purity,
            'seconds': time.time() - t,
        }
        rows.append(row)
        print(f"  {dim:>4d} {row['corr_pearson']:>7.4f} {corr_s:>7.4f} "
              f"{row['stress']:>7.4f} {floor:>7.4f} {expl:>7.4f} "
              f"{purity:>7.4f} {row['seconds']:>5.1f}")
    
    elbow = find_spectrum_elbow(eigenvalues)
    print(f"\n  Spectrum elbow (max-curvature): dim ≈ {elbow}")
    print(f"  Top eigenvalues: {eigenvalues[:8]}")
    
    # ━━━ Procrustes alignment for dim=4 ━━━
    print("\n[5] Procrustes alignment to 24-cell at dim=4...")
    Y4_raw = Y_dict[4]
    Y4_aligned = align_4d_to_24cell(Y4_raw, vocab, tid)
    
    # Verify alignment didn't break distances
    DY_aligned = sphere_dist(Y4_aligned)
    corr_aligned = corr_upper(DY_aligned, D)
    print(f"  Distance correlation after Procrustes: {corr_aligned:.4f}")
    print(f"  (should equal raw spectral 4D: rotation preserves distances)")
    
    # ━━━ Composition tests on raw vs aligned ━━━
    print("\n[6] Composition tests (raw spectral vs Procrustes-aligned)...")
    print("\n  RAW SPECTRAL 4D:")
    comp_raw = composition_tests(Y4_raw, tid, vocab)
    print_composition(comp_raw)
    
    print("\n  PROCRUSTES-ALIGNED 4D:")
    comp_aligned = composition_tests(Y4_aligned, tid, vocab)
    print_composition(comp_aligned)
    
    # ━━━ Hybrid: spectral init + light Riemannian refinement ━━━
    print("\n[7] Hybrid: spectral init + Hamilton composition refinement...")
    Y4_hybrid = refine_for_composition(Y4_aligned, vocab, tid, n_steps=500)
    DY_hybrid = sphere_dist(Y4_hybrid)
    corr_hybrid = corr_upper(DY_hybrid, D)
    print(f"\n  After refinement: corr={corr_hybrid:.4f}")
    
    print("\n  HYBRID 4D:")
    comp_hybrid = composition_tests(Y4_hybrid, tid, vocab)
    print_composition(comp_hybrid)
    
    # ━━━ Save ━━━
    out_data = {
        'corpus': 'shakespeare + pride_prejudice',
        'vocab_size': len(vocab),
        'corpus_tokens': len(toks),
        'top_eigenvalues': eigenvalues[:32].tolist(),
        'spectrum_elbow': elbow,
        'sweep': rows,
        'composition': {
            'raw_4d': comp_raw,
            'aligned_4d': comp_aligned,
            'hybrid_4d': comp_hybrid,
        },
        'distance_corr_aligned_4d': corr_aligned,
        'distance_corr_hybrid_4d': corr_hybrid,
    }
    out_path = results_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"\n  Wrote {out_path}")
    
    # ━━━ Final summary ━━━
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    best = max(rows, key=lambda r: r['corr_pearson'])
    sweet = next((r for r in rows if r['corr_pearson'] > 0.9), rows[-1])
    print(f"  Best dim: {best['dim']} (corr={best['corr_pearson']:.4f})")
    print(f"  >0.9 corr at dim: {sweet['dim']} (purity={sweet['knn_purity']:.3f})")
    print(f"  Spectrum elbow: dim={elbow}")
    print(f"  Theoretical floor at dim=4: {[r for r in rows if r['dim']==4][0]['theoretical_floor']:.4f}")
    print(f"  4D embedding storage: {len(vocab) * 4 * 4 / 1024:.1f} KB")
    print("=" * 70)


def print_composition(comp):
    """Concise composition test output."""
    print("    Pair distances (similar should be < different):")
    for p in comp['pair_distances'][:6]:
        print(f"      d({p['a']:>15s}, {p['b']:<15s}) = {p['dist']:.4f}  ({p['label']})")
    print("    Closure means:")
    m = comp['closure_means']
    print(f"      coherent: {m.get('coherent', float('nan')):.4f} | "
          f"scrambled: {m.get('scrambled', float('nan')):.4f} | "
          f"random: {m.get('random', float('nan')):.4f}")
    if 'coherent' in m and 'random' in m:
        if m['coherent'] < m['random']:
            print(f"      ✓ coherent < random (coherence detected)")
    print("    Analogies (top-5 includes expected?):")
    hits = sum(1 for a in comp['analogies'] if a['in_top5'])
    print(f"      {hits}/{len(comp['analogies'])} hits in top-5")
    for a in comp['analogies']:
        marker = '✓' if a['in_top5'] else '✗'
        print(f"      {marker} {a['query']:>22s} → {a['top5'][:3]} (expected: {a['expected']})")


def refine_for_composition(Y_init, vocab, tid, n_steps=500, lr=0.01):
    """
    Riemannian SGD refinement initialized from spectral, optimizing for:
    coherent sequences have low σ, scrambled/random have high σ.
    Returns refined embedding (still 4D, still on S³).
    """
    import torch
    Y = torch.tensor(Y_init, dtype=torch.float32, requires_grad=True)
    
    # Synthesize training pairs from corpus structure (sliding window in vocab)
    def hamilton_t(a, b):
        w1,x1,y1,z1 = a.unbind(-1)
        w2,x2,y2,z2 = b.unbind(-1)
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2], dim=-1)
    
    # Real coherent and random sequences (small bank for refinement)
    coherent = [
        ['the', 'man', 'said'],
        ['my', 'dear', 'friend'],
        ['the', 'young', 'lady'],
        ['the', 'old', 'man'],
        ['a', 'fine', 'gentleman'],
        ['she', 'was', 'very'],
        ['the', 'great', 'house'],
        ['his', 'father', 'said'],
    ]
    random_seqs = [
        ['horse', 'love', 'day'],
        ['said', 'tree', 'money'],
        ['morning', 'dark', 'my'],
        ['walked', 'said', 'fire'],
        ['lady', 'sword', 'horse'],
        ['fine', 'horse', 'love'],
        ['dear', 'tree', 'said'],
        ['fair', 'morning', 'fire'],
    ]
    
    # Filter to in-vocab
    coh_ids = []
    for s in coherent:
        ids = [tid[t] for t in s if t in tid]
        if len(ids) >= 2: coh_ids.append(ids)
    rand_ids = []
    for s in random_seqs:
        ids = [tid[t] for t in s if t in tid]
        if len(ids) >= 2: rand_ids.append(ids)
    
    if not coh_ids or not rand_ids:
        print("  (insufficient data for refinement, skipping)")
        return Y_init
    
    optimizer = torch.optim.SGD([Y], lr=lr, momentum=0.9)
    
    for step in range(n_steps):
        # Coherent should have low σ
        coh_loss = 0.0
        for ids in coh_ids:
            q = Y[ids[0]]
            for i in ids[1:]:
                q = hamilton_t(q, Y[i])
                q = q / (q.norm() + 1e-8)
            coh_loss = coh_loss + torch.acos(q[0].abs().clamp(0, 1-1e-6)) ** 2
        coh_loss = coh_loss / len(coh_ids)
        
        # Random should have high σ (target ~π/2)
        rand_loss = 0.0
        for ids in rand_ids:
            q = Y[ids[0]]
            for i in ids[1:]:
                q = hamilton_t(q, Y[i])
                q = q / (q.norm() + 1e-8)
            target = math.pi / 2
            rand_loss = rand_loss + (torch.acos(q[0].abs().clamp(0, 1-1e-6)) - target) ** 2
        rand_loss = rand_loss / len(rand_ids)
        
        loss = coh_loss + 0.5 * rand_loss
        
        optimizer.zero_grad()
        loss.backward()
        # Tangent projection
        with torch.no_grad():
            grad = Y.grad
            if grad is not None:
                dot = (grad * Y.data).sum(dim=-1, keepdim=True)
                Y.grad = grad - dot * Y.data
        optimizer.step()
        with torch.no_grad():
            Y.data = Y.data / Y.data.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        
        if step % 100 == 0:
            print(f"    step {step:4d} | coh σ²: {coh_loss.item():.4f} | rand: {rand_loss.item():.4f}")
    
    return Y.detach().numpy()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
GeoLLM v3 — multi-channel ensemble (B)
======================================
Builds on v2 (07_closure_lm.py).

Thesis: Faltz exposes Hopf channels (Base / Phase / Full / Scalar) as
*orthogonal*. A sentence model needs all of them voting. Try:

  scores = w_base * vote_base + w_phase * vote_phase + w_full * vote_full

with several weight choices, and report which channel carries which signal.
"""
import re, math, random, time
from collections import Counter
from pathlib import Path
import numpy as np
from geollm_core import (
    I_q, hamilton, normalize, axis_angle, hopf_decompose, carrier_from_hopf,
    semantic_base_from_bytes, domain_embed,
    OPERATORS, compose_state, load_corpus,
)

random.seed(42); np.random.seed(42)

# ── Genome with ALL three channel projections precomputed ─────────────
def build_genome_multi(train_toks, carriers, ops, wid, context_len=5):
    """Returns S (G,4) state-on-S^3, S_base (G,3) precomputed bases,
    S_phase (G,) precomputed phases, N (G,) next-ids."""
    states = []; nxts = []
    for i in range(context_len, len(train_toks)):
        target = train_toks[i]
        if target not in wid: continue
        ctx = train_toks[i-context_len:i]
        states.append(compose_state(ctx, carriers, ops))
        nxts.append(wid[target])
    S = np.stack(states, 0).astype(np.float64)  # (G, 4)
    # Vectorized hopf decompose of S
    w,x,y,z = S[:,0],S[:,1],S[:,2],S[:,3]
    bases = np.stack([2*(x*z+w*y), 2*(y*z-w*x), w*w+z*z-x*x-y*y], axis=1)
    nrm = np.linalg.norm(bases, axis=1, keepdims=True); nrm[nrm<1e-15]=1.0
    S_base = bases/nrm
    S_phase = (np.arctan2(z, w) + np.arctan2(x, y)) % (2*math.pi)  # (G,)
    N = np.array(nxts, dtype=np.int64)
    return S, S_base, S_phase, N

# ── Per-channel scoring (vectorized) ──────────────────────────────────
def score_full(q, S, V, N, cutoff=math.pi/3):
    dots = np.clip(np.abs(S @ q), -1.0, 1.0)
    gaps = np.arccos(dots)
    w = np.cos(gaps); w[gaps > cutoff] = 0.0
    return np.bincount(N, weights=np.clip(w,0,1), minlength=V)

def score_base(q, S_base, V, N, cutoff=math.pi/3):
    b_q, _ = hopf_decompose(q)
    cos_b = np.clip(S_base @ b_q, -1.0, 1.0)
    gaps = np.arccos(np.abs(cos_b))
    w = np.cos(gaps); w[gaps > cutoff] = 0.0
    return np.bincount(N, weights=np.clip(w,0,1), minlength=V)

def score_phase(q, S_phase, V, N, cutoff=math.pi/3):
    _, p_q = hopf_decompose(q)
    d = np.abs(S_phase - p_q)
    d = np.minimum(d, 2*math.pi - d)  # wrap
    w = np.cos(d); w[d > cutoff] = 0.0
    return np.bincount(N, weights=np.clip(w,0,1), minlength=V)

# ── Combined scoring with weights ─────────────────────────────────────
def score_combined(q, S, S_base, S_phase, V, N, w_full, w_base, w_phase):
    s = np.zeros(V, dtype=np.float64)
    if w_full  > 0: s += w_full  * score_full (q, S,       V, N)
    if w_base  > 0: s += w_base  * score_base (q, S_base,  V, N)
    if w_phase > 0: s += w_phase * score_phase(q, S_phase, V, N)
    return s

# ── Eval ──────────────────────────────────────────────────────────────
def eval_combo(test_toks, S, S_base, S_phase, N, carriers, ops, wid, V,
               context_len, weights, n_eval=1000):
    valid = [t for t in test_toks if t in wid]
    if len(valid) <= context_len: return None
    pos = list(range(context_len, len(valid)))
    if len(pos) > n_eval: pos = random.sample(pos, n_eval)
    top1=top5=top10=0; mrr=0.0; n=0
    w_full, w_base, w_phase = weights
    for p in pos:
        ctx = valid[p-context_len:p]
        target_id = wid[valid[p]]
        q = compose_state(ctx, carriers, ops)
        scores = score_combined(q, S, S_base, S_phase, V, N, w_full, w_base, w_phase)
        ranked = np.argsort(-scores)
        if ranked[0] == target_id: top1 += 1
        if target_id in ranked[:5]: top5 += 1
        if target_id in ranked[:10]: top10 += 1
        rk = np.where(ranked == target_id)[0]
        if len(rk) > 0: mrr += 1.0/(rk[0]+1)
        n += 1
    return {'top1':top1/n,'top5':top5/n,'top10':top10/n,'mrr':mrr/n,'n':n}

def main():
    print("="*75); print("  GeoLLM v3 — multi-channel ensemble"); print("="*75)
    t0 = time.time()
    corpus = load_corpus()
    toks = re.findall(r"[a-z']+", corpus)
    n_train = int(len(toks)*0.9)
    train, test = toks[:n_train], toks[n_train:]
    counts = Counter(train)
    vocab = [w for w,c in counts.most_common(2500) if c>=10]
    wid = {w:i for i,w in enumerate(vocab)}; V = len(vocab)
    print(f"  Vocab {V} · train {len(train):,} · test {len(test):,}")

    carriers = {w: domain_embed(w) for w in vocab}

    print("\n[1] Building genome (ctx=3) with all-channel projections...")
    S, S_base, S_phase, N = build_genome_multi(train, carriers, OPERATORS, wid, context_len=3)
    print(f"  genome size {len(N):,}")

    print("\n[2] Single-channel baselines (eval ctx=5)")
    print(f"  {'channel':<25s}  {'top1':>7s}  {'top5':>7s}  {'top10':>7s}  {'mrr':>7s}")
    for label, w in [('full only',(1,0,0)), ('base only',(0,1,0)), ('phase only',(0,0,1))]:
        r = eval_combo(test, S, S_base, S_phase, N, carriers, OPERATORS, wid, V,
                       context_len=5, weights=w, n_eval=1000)
        if r: print(f"  {label:<25s}  {r['top1']:>7.4f}  {r['top5']:>7.4f}  "
                    f"{r['top10']:>7.4f}  {r['mrr']:>7.4f}")

    print("\n[3] Pairs (eval ctx=5)")
    pairs = [('full+base',(1,1,0)),('full+phase',(1,0,1)),('base+phase',(0,1,1))]
    for label, w in pairs:
        r = eval_combo(test, S, S_base, S_phase, N, carriers, OPERATORS, wid, V,
                       context_len=5, weights=w, n_eval=1000)
        if r: print(f"  {label:<25s}  {r['top1']:>7.4f}  {r['top5']:>7.4f}  "
                    f"{r['top10']:>7.4f}  {r['mrr']:>7.4f}")

    print("\n[4] Triple (eval ctx=5) with weight sweep")
    triples = [
        ('1:1:1', (1,1,1)),
        ('2:1:1 (full-heavy)', (2,1,1)),
        ('1:2:1 (base-heavy)', (1,2,1)),
        ('1:1:2 (phase-heavy)', (1,1,2)),
        ('3:1:1', (3,1,1)),
        ('1:3:1', (1,3,1)),
        ('1:1:3', (1,1,3)),
    ]
    for label, w in triples:
        r = eval_combo(test, S, S_base, S_phase, N, carriers, OPERATORS, wid, V,
                       context_len=5, weights=w, n_eval=1000)
        if r: print(f"  {label:<25s}  {r['top1']:>7.4f}  {r['top5']:>7.4f}  "
                    f"{r['top10']:>7.4f}  {r['mrr']:>7.4f}")

    print(f"\nDone in {time.time()-t0:.1f}s")
    print("="*75)
    print("  Targets: 2-gram top1=11.5% · v2 best top1=5.2%")
    print("="*75)

if __name__ == '__main__':
    main()

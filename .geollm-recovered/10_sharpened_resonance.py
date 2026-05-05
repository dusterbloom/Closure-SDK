#!/usr/bin/env python3
"""
GeoLLM v5 — sharpened resonance (A)
====================================
Builds on v3 (full+phase). Sharpening choices:

  1. TOP-K only — cap how many genome entries vote (closer to closure_ea's
     resonate_spectrum, which ranks all but uses [0] for hard attention).
  2. Gaussian falloff — weight = exp(-gap^2 / (2*sigma^2)) instead of
     cos(gap) clipped at pi/3.
  3. Combined: Gaussian within top-K.

Sweep K in {20, 50, 100, 500, 5000, all} and sigma in {pi/12, pi/8, pi/6, pi/4, pi/3}.
Best target: lift past the v3 5.2% top-1 on full+phase.
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

def build_genome(train_toks, carriers, ops, wid, context_len=3):
    states = []; nxts = []
    for i in range(context_len, len(train_toks)):
        target = train_toks[i]
        if target not in wid: continue
        ctx = train_toks[i-context_len:i]
        states.append(compose_state(ctx, carriers, ops))
        nxts.append(wid[target])
    S = np.stack(states, 0).astype(np.float64)
    w,x,y,z = S[:,0],S[:,1],S[:,2],S[:,3]
    bases = np.stack([2*(x*z+w*y), 2*(y*z-w*x), w*w+z*z-x*x-y*y], axis=1)
    nrm = np.linalg.norm(bases,axis=1,keepdims=True); nrm[nrm<1e-15]=1.0
    S_base = bases/nrm
    S_phase = (np.arctan2(z, w) + np.arctan2(x, y)) % (2*math.pi)
    N = np.array(nxts, dtype=np.int64)
    return S, S_base, S_phase, N

# ── Sharpened scoring ─────────────────────────────────────────────────
def gaps_full(q, S):
    dots = np.clip(np.abs(S @ q), -1.0, 1.0)
    return np.arccos(dots)

def gaps_phase(q, S_phase):
    _, p_q = hopf_decompose(q)
    d = np.abs(S_phase - p_q)
    return np.minimum(d, 2*math.pi - d)

def score_sharpened(q, S, S_phase, V, N, *, top_k=None, sigma=None,
                     channels=('full','phase')):
    """
    top_k: keep only the K closest entries by combined gap (None = all).
    sigma: if not None, weight = exp(-(gap/sigma)^2 / 2). Else cos+cutoff(pi/3).
    channels: tuple of ('full','phase','base'). Sum scores.
    """
    g_full  = gaps_full (q, S) if 'full'  in channels else None
    g_phase = gaps_phase(q, S_phase) if 'phase' in channels else None
    # Use the MIN gap across active channels for top-K candidate selection
    # (so each channel can pull in its closest neighbors).
    candidates_mask = np.ones(len(N), dtype=bool)
    if top_k is not None:
        # Sort by min gap across active channels, take top_k
        gaps_min = np.full(len(N), np.inf)
        if g_full  is not None: gaps_min = np.minimum(gaps_min, g_full)
        if g_phase is not None: gaps_min = np.minimum(gaps_min, g_phase)
        idx = np.argpartition(gaps_min, min(top_k, len(N)-1))[:top_k]
        candidates_mask[:] = False
        candidates_mask[idx] = True

    def to_weight(g):
        if sigma is not None:
            return np.exp(-(g*g) / (2*sigma*sigma))
        else:
            w = np.cos(g)
            w[g > math.pi/3] = 0.0
            return np.clip(w, 0, 1)

    s = np.zeros(V, dtype=np.float64)
    if g_full is not None:
        w = to_weight(g_full)
        w = w * candidates_mask
        s += np.bincount(N, weights=w, minlength=V)
    if g_phase is not None:
        w = to_weight(g_phase)
        w = w * candidates_mask
        s += np.bincount(N, weights=w, minlength=V)
    return s

def eval_config(test_toks, S, S_phase, N, carriers, ops, wid, V,
                context_len, n_eval, top_k, sigma):
    valid = [t for t in test_toks if t in wid]
    if len(valid) <= context_len: return None
    pos = list(range(context_len, len(valid)))
    if len(pos) > n_eval: pos = random.sample(pos, n_eval)
    top1 = top5 = top10 = 0; mrr = 0.0; n = 0
    for p in pos:
        ctx = valid[p-context_len:p]
        target_id = wid[valid[p]]
        q = compose_state(ctx, carriers, ops)
        scores = score_sharpened(q, S, S_phase, V, N, top_k=top_k, sigma=sigma)
        ranked = np.argsort(-scores)
        if ranked[0] == target_id: top1 += 1
        if target_id in ranked[:5]: top5 += 1
        if target_id in ranked[:10]: top10 += 1
        rk = np.where(ranked == target_id)[0]
        if len(rk) > 0: mrr += 1.0/(rk[0]+1)
        n += 1
    return {'top1':top1/n,'top5':top5/n,'top10':top10/n,'mrr':mrr/n,'n':n}

def main():
    print("="*75); print("  GeoLLM v5 — sharpened resonance (top-K + Gaussian falloff)"); print("="*75)
    t0 = time.time()
    corpus = load_corpus()
    toks = re.findall(r"[a-z']+", corpus)
    n_train = int(len(toks)*0.9); train, test = toks[:n_train], toks[n_train:]
    counts = Counter(train)
    vocab = [w for w,c in counts.most_common(2500) if c>=10]
    wid = {w:i for i,w in enumerate(vocab)}; V = len(vocab)
    print(f"  Vocab {V} · train {len(train):,} · test {len(test):,}")
    carriers = {w: domain_embed(w) for w in vocab}

    print("\n[1] Building genome (ctx=3)...")
    S, _, S_phase, N = build_genome(train, carriers, OPERATORS, wid, context_len=3)
    print(f"  size {len(N):,}")

    print("\n[2] cos+cutoff baseline (v3 best replication)...")
    r = eval_config(test, S, S_phase, N, carriers, OPERATORS, wid, V,
                    context_len=5, n_eval=1500, top_k=None, sigma=None)
    print(f"  cos+cutoff, all entries  top1={r['top1']:.4f}  top5={r['top5']:.4f}  "
          f"top10={r['top10']:.4f}  mrr={r['mrr']:.4f}")

    print("\n[3] Top-K with cos+cutoff (no Gaussian)")
    print(f"  {'top_k':>7s}  {'top1':>7s}  {'top5':>7s}  {'top10':>7s}  {'mrr':>7s}")
    for K in [10, 50, 200, 1000, 5000]:
        r = eval_config(test, S, S_phase, N, carriers, OPERATORS, wid, V,
                        context_len=5, n_eval=1500, top_k=K, sigma=None)
        if r: print(f"  {K:>7d}  {r['top1']:>7.4f}  {r['top5']:>7.4f}  "
                    f"{r['top10']:>7.4f}  {r['mrr']:>7.4f}")

    print("\n[4] Gaussian falloff (no top-K)")
    print(f"  {'sigma':>7s}  {'top1':>7s}  {'top5':>7s}  {'top10':>7s}  {'mrr':>7s}")
    for s in [math.pi/12, math.pi/8, math.pi/6, math.pi/4, math.pi/3]:
        r = eval_config(test, S, S_phase, N, carriers, OPERATORS, wid, V,
                        context_len=5, n_eval=1500, top_k=None, sigma=s)
        if r: print(f"  pi/{math.pi/s:>4.1f}  {r['top1']:>7.4f}  {r['top5']:>7.4f}  "
                    f"{r['top10']:>7.4f}  {r['mrr']:>7.4f}")

    print("\n[5] Top-K + Gaussian (sigma=pi/8)")
    print(f"  {'top_k':>7s}  {'top1':>7s}  {'top5':>7s}  {'top10':>7s}  {'mrr':>7s}")
    for K in [10, 50, 200, 1000]:
        r = eval_config(test, S, S_phase, N, carriers, OPERATORS, wid, V,
                        context_len=5, n_eval=1500, top_k=K, sigma=math.pi/8)
        if r: print(f"  {K:>7d}  {r['top1']:>7.4f}  {r['top5']:>7.4f}  "
                    f"{r['top10']:>7.4f}  {r['mrr']:>7.4f}")

    print("\n[6] Hard top-1 (closure_ea-style hard attention)")
    valid = [t for t in test if t in wid]
    pos = random.sample(range(5, len(valid)), min(1500, len(valid)-5))
    top1 = top5 = top10 = 0; mrr = 0.0; n = 0
    for p in pos:
        ctx = valid[p-5:p]; target_id = wid[valid[p]]
        q = compose_state(ctx, carriers, OPERATORS)
        gf = gaps_full(q, S); gp = gaps_phase(q, S_phase)
        gmin = np.minimum(gf, gp)
        # Single nearest entry's next token
        i = int(np.argmin(gmin))
        # Use it as a one-hot prediction
        prediction = N[i]
        if prediction == target_id: top1 += 1
        # MRR: rank of target by gap (the only sensible rank for hard attention)
        # Use 'top5/10' loosely as "target is among the K nearest entries' next tokens"
        order = np.argsort(gmin)
        unique_predictions = []
        seen = set()
        for j in order[:50]:
            tok = int(N[j])
            if tok not in seen:
                unique_predictions.append(tok); seen.add(tok)
            if len(unique_predictions) >= 10: break
        if target_id in unique_predictions[:5]:  top5 += 1
        if target_id in unique_predictions[:10]: top10 += 1
        if target_id in unique_predictions:
            mrr += 1.0 / (unique_predictions.index(target_id) + 1)
        n += 1
    print(f"  hard-attention (1-NN)    top1={top1/n:.4f}  top5={top5/n:.4f}  "
          f"top10={top10/n:.4f}  mrr={mrr/n:.4f}  n={n}")

    print(f"\nDone in {time.time()-t0:.1f}s")
    print("="*75)
    print("  Targets: 2-gram top1=11.5% · v3 best=5.2%")
    print("="*75)

if __name__ == '__main__':
    main()

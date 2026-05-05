#!/usr/bin/env python3
"""
Diagnostic: what's the ceiling for a rank-r bilinear next-token model?
Truncated SVD of the bigram transition matrix gives the BEST POSSIBLE
r-floats-per-token bilinear LM. This sets the ceiling that any geometric
construction at the same param budget can reach.

If rank-4 hits ~10% top-1 -> our geometric v0's 1% is a predictor/composition
problem, not a dimensional one.
If rank-4 hits ~1% -> 4 floats really is too compressed; need >= 16 dim.
"""
import re, random, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

random.seed(42); np.random.seed(42)

def load_corpus():
    here = Path(__file__).parent / 'corpora'
    parts = []
    for p in [here/'shakespeare.txt', here/'pride_prejudice.txt']:
        if p.exists(): parts.append(p.read_text(errors='ignore').lower())
    return ' '.join(parts)

def main():
    print("="*75)
    print("  DIAGNOSTIC — rank-r SVD of bigram transition matrix")
    print("="*75)
    t0 = time.time()

    corpus = load_corpus()
    toks = re.findall(r"[a-z']+", corpus)
    n_train = int(len(toks) * 0.9)
    train, test = toks[:n_train], toks[n_train:]
    counts = Counter(train)
    vocab = [w for w, c in counts.most_common(2500) if c >= 10]
    wid = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    print(f"  Vocab {V} · train {len(train):,} · test {len(test):,}")

    # Build asymmetric bigram count matrix B[i, j] = count(i -> j)
    print("\n[1] Building bigram count matrix...")
    B = np.zeros((V, V), dtype=np.float64)
    train_ids = [wid[t] for t in train if t in wid]
    for i in range(len(train_ids) - 1):
        B[train_ids[i], train_ids[i+1]] += 1.0
    print(f"  Nonzeros: {(B>0).sum():,}")

    # Full bigram baseline (the ceiling for ANY model)
    print("\n[2] Full bigram baseline...")
    test_ids = [wid[t] for t in test if t in wid]
    n_eval = 2000
    pos = random.sample(range(1, len(test_ids)), min(n_eval, len(test_ids)-1))
    unigram = Counter(train_ids)
    rows_argmax_full = [w for w, _ in unigram.most_common()]  # backoff
    top1 = top5 = top10 = 0; mrr = 0.0; n = 0
    for p in pos:
        prev, tgt = test_ids[p-1], test_ids[p]
        row = B[prev]
        if row.sum() == 0:
            ranked = rows_argmax_full
        else:
            ranked = np.argsort(-row).tolist()
        if ranked[0] == tgt: top1 += 1
        if tgt in ranked[:5]: top5 += 1
        if tgt in ranked[:10]: top10 += 1
        try:
            mrr += 1.0 / (ranked.index(tgt) + 1)
        except ValueError:
            pass
        n += 1
    print(f"  bigram (full)   top1={top1/n:.4f}  top5={top5/n:.4f}  top10={top10/n:.4f}  mrr={mrr/n:.4f}")

    # Rank-r truncated-SVD ceiling
    print("\n[3] Computing SVD of bigram counts (one-shot)...")
    P = B / (B.sum(axis=1, keepdims=True) + 1e-12)  # row-stochastic transition matrix
    U, S, Vt = np.linalg.svd(P, full_matrices=False)
    print(f"  Top 10 singular values: {S[:10].round(3)}")

    print("\n[4] Top-k accuracy at increasing rank r:")
    print(f"  {'r':>4s}  {'params':>8s}  {'top1':>7s}  {'top5':>7s}  {'top10':>7s}  {'mrr':>7s}")
    for r in [2, 4, 8, 16, 32, 64, 128]:
        if r > min(P.shape): continue
        # Best rank-r approx
        Pr = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]
        # Score: for each prev token, rank by Pr[prev, :]
        top1 = top5 = top10 = 0; mrr = 0.0; n = 0
        for p in pos:
            prev, tgt = test_ids[p-1], test_ids[p]
            scores = Pr[prev]
            ranked = np.argsort(-scores)
            if ranked[0] == tgt: top1 += 1
            if tgt in ranked[:5]: top5 += 1
            if tgt in ranked[:10]: top10 += 1
            rank_arr = np.where(ranked == tgt)[0]
            if len(rank_arr) > 0:
                mrr += 1.0 / (rank_arr[0] + 1)
            n += 1
        params = r * V * 2  # left + right embedding
        print(f"  {r:>4d}  {params:>8d}  {top1/n:>7.4f}  {top5/n:>7.4f}  "
              f"{top10/n:>7.4f}  {mrr/n:>7.4f}")

    print(f"\nDone in {time.time()-t0:.1f}s")
    print("="*75)
    print("  Reading: rank-r is the BEST POSSIBLE r-floats-per-token bilinear LM.")
    print("  Geometric v0 at d=4 got top1=0.01. If rank-4 here gets top1>>0.01,")
    print("  the geometric loss is in the composer/predictor, not the dimension.")
    print("="*75)

if __name__ == '__main__':
    main()

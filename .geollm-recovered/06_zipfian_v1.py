#!/usr/bin/env python3
"""
GeoLLM v1 — separate input/output embeddings, hand-assigned operators
=====================================================================
Diagnostic (script #5) showed rank-4 SVD of the bigram transition gets
8.1% top-1 with 20K params. v0 got 1% with ~10K params (one-embedding).
v1 fixes the deficit:

  - TWO embeddings on S^3:
      E_in[w]   = used during composition (Hamilton on running state)
      E_out[w]  = used at prediction (cos(q, E_out[w]))
    Both come from the SVD of the asymmetric bigram count matrix:
      P[i,j] = count(i->j); P = U S V^T;
      E_in[w]  = normalize(U[w] sqrt(S))
      E_out[w] = normalize(V^T[:,w] sqrt(S))

  - Hand-assigned operators (function words) act ONLY on the running state q
    via left-Hamilton. Operator output embedding is op(I)/||op(I)||.

  - Win condition: beat rank-4 SVD baseline (8.1% top-1) AT MULTI-STEP CONTEXT.
    A win means composition of multi-token history adds value over prev-token
    bilinear model at matched param budget.
"""
import re, math, random, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from geollm_core import I_q, hamilton, normalize, axis_angle, OPERATORS, load_corpus

random.seed(42); np.random.seed(42)

# ── Build dual embedding from SVD of asymmetric bigram counts ─────────
def build_dual_embedding(train_ids, V, rank=4, normalize_rows=False):
    """SVD of bigram transition matrix.
    With normalize_rows=False, E_out @ E_in[prev] reproduces rank-r SVD of P
    (recovers the 8.1% top-1 ceiling at rank=4).
    With normalize_rows=True, both rows are unit-normalized to S^{r-1};
    information in row magnitudes is destroyed (sanity check)."""
    B = np.zeros((V, V), dtype=np.float64)
    for i in range(len(train_ids)-1):
        B[train_ids[i], train_ids[i+1]] += 1.0
    P = B / (B.sum(axis=1, keepdims=True) + 1e-12)
    U, S, Vt = np.linalg.svd(P, full_matrices=False)
    s = np.sqrt(S[:rank])
    E_in  = U[:, :rank] * s[None, :]            # (V, r)
    E_out = (Vt[:rank, :] * s[:, None]).T       # (V, r)
    if normalize_rows:
        E_in  = E_in  / (np.linalg.norm(E_in,  axis=1, keepdims=True) + 1e-12)
        E_out = E_out / (np.linalg.norm(E_out, axis=1, keepdims=True) + 1e-12)
    return E_in, E_out, B

def install_operators(E_in, E_out, wid, ops):
    """Override E_in[w]=op(I), E_out[w]=op(I) for operator words.
    The operator's input embedding is what gets right-multiplied as a content
    proxy if it slipped through, but composer dispatches on word so it's not
    used for operators — purely defensive.  E_out is used to predict the
    operator as a next token: keep it as op(I) so 'is/the/and...' have
    distinct, structured output vectors."""
    for w, op in ops.items():
        if w in wid:
            v = normalize(op)
            E_in[wid[w]]  = v
            E_out[wid[w]] = v
    return E_in, E_out

# ── Composition: walk left context as program ────────────────────────
def compose(ctx, wid, E_in, ops, renorm=True):
    """Walk left-context as a program. If renorm=True, normalize state
    after each step (pure S^3 dynamics — destroys magnitudes carrying SVD
    signal). If renorm=False, leave magnitudes alone (Hamilton-on-B^4)."""
    q = I_q.copy()
    for t in ctx:
        if t in ops:
            q = hamilton(ops[t], q)
        elif t in wid:
            q = hamilton(q, E_in[wid[t]])
        if renorm:
            q = normalize(q)
        else:
            # Guard against numerical blowup over long contexts
            n = np.linalg.norm(q)
            if n > 1e6 or n < 1e-6:
                q = normalize(q)
    return q

# ── Evaluation ────────────────────────────────────────────────────────
def eval_program(test_ids, wid_inv, wid, E_in, E_out, ops, context_len, n_eval=2000, renorm=True):
    test_toks = [wid_inv[i] for i in test_ids]
    if len(test_toks) <= context_len:
        return None
    pos = list(range(context_len, len(test_toks)))
    if len(pos) > n_eval: pos = random.sample(pos, n_eval)

    top1 = top5 = top10 = 0; mrr = 0.0; n = 0
    for p in pos:
        ctx = test_toks[p-context_len:p]
        target_id = test_ids[p]
        q = compose(ctx, wid, E_in, ops, renorm=renorm)
        sims = E_out @ q
        ranked = np.argsort(-sims)
        if ranked[0] == target_id: top1 += 1
        if target_id in ranked[:5]: top5 += 1
        if target_id in ranked[:10]: top10 += 1
        rk = np.where(ranked == target_id)[0]
        if len(rk) > 0: mrr += 1.0 / (rk[0] + 1)
        n += 1
    return {'top1':top1/n,'top5':top5/n,'top10':top10/n,'mrr':mrr/n,'n':n}

def eval_bilinear_prev_only(test_ids, E_in, E_out, n_eval=2000):
    """Sanity check: cos(E_in[prev], E_out[w]) for every w. Should ~match
    rank-r SVD baseline."""
    pos = list(range(1, len(test_ids)))
    if len(pos) > n_eval: pos = random.sample(pos, n_eval)
    top1 = top5 = top10 = 0; mrr = 0.0; n = 0
    for p in pos:
        prev, tgt = test_ids[p-1], test_ids[p]
        sims = E_out @ E_in[prev]
        ranked = np.argsort(-sims)
        if ranked[0] == tgt: top1 += 1
        if tgt in ranked[:5]: top5 += 1
        if tgt in ranked[:10]: top10 += 1
        rk = np.where(ranked == tgt)[0]
        if len(rk) > 0: mrr += 1.0 / (rk[0] + 1)
        n += 1
    return {'top1':top1/n,'top5':top5/n,'top10':top10/n,'mrr':mrr/n,'n':n}

# ── Main ──────────────────────────────────────────────────────────────
def main():
    print("="*75)
    print("  GeoLLM v1 — dual S^3 embedding (input/output) + Zipfian operators")
    print("="*75)
    t0 = time.time()

    corpus = load_corpus()
    toks = re.findall(r"[a-z']+", corpus)
    n_train = int(len(toks) * 0.9)
    train, test = toks[:n_train], toks[n_train:]
    counts = Counter(train)
    vocab = [w for w, c in counts.most_common(2500) if c >= 10]
    wid = {w: i for i, w in enumerate(vocab)}
    wid_inv = {i: w for w, i in wid.items()}
    V = len(vocab)
    train_ids = [wid[t] for t in train if t in wid]
    test_ids  = [wid[t] for t in test  if t in wid]
    print(f"  Vocab {V} · train {len(train_ids):,} · test {len(test_ids):,}")

    print("\n[1] Dual embedding from rank-4 SVD of bigram counts (UN-normalized)...")
    E_in, E_out, B = build_dual_embedding(train_ids, V, rank=4, normalize_rows=False)
    norms_in  = np.linalg.norm(E_in,  axis=1)
    norms_out = np.linalg.norm(E_out, axis=1)
    print(f"  E_in  norms: min={norms_in.min():.3f}  max={norms_in.max():.3f}  median={np.median(norms_in):.3f}")
    print(f"  E_out norms: min={norms_out.min():.3f}  max={norms_out.max():.3f}  median={np.median(norms_out):.3f}")

    print("\n[2] Sanity: prev-only bilinear — should match rank-4 SVD (~8.1%)")
    r = eval_bilinear_prev_only(test_ids, E_in, E_out)
    print(f"  bilinear (prev-only)  top1={r['top1']:.4f}  top5={r['top5']:.4f}  "
          f"top10={r['top10']:.4f}  mrr={r['mrr']:.4f}")

    # Build op-installed copies. For un-normalized embeddings, install operators
    # using op(I) at typical-token magnitude to keep regimes comparable.
    typical_in  = np.median(norms_in)
    typical_out = np.median(norms_out)
    def install_at_scale(E, wid, ops, scale):
        E2 = E.copy()
        for w, op in ops.items():
            if w in wid:
                v = normalize(op) * scale
                E2[wid[w]] = v
        return E2

    print(f"\n[3] Install Zipfian operators (scale to typical row norm)...")
    print(f"  typical in={typical_in:.3f}  out={typical_out:.3f}")
    E_in_ops  = install_at_scale(E_in,  wid, OPERATORS, typical_in)
    E_out_ops = install_at_scale(E_out, wid, OPERATORS, typical_out)

    print("\n[4] Geometric program — Hamilton on B^4 (no renorm), varying context")
    print(f"  {'context':>7s}  {'top1':>7s}  {'top5':>7s}  {'top10':>7s}  {'mrr':>7s}")
    for cl in [1, 2, 4, 6, 8, 12, 20]:
        r = eval_program(test_ids, wid_inv, wid, E_in_ops, E_out_ops, OPERATORS,
                         context_len=cl, n_eval=1500, renorm=False)
        if r is None: continue
        print(f"  {cl:>7d}  {r['top1']:>7.4f}  {r['top5']:>7.4f}  "
              f"{r['top10']:>7.4f}  {r['mrr']:>7.4f}")

    print("\n[5] Same as [4] but WITH renormalization (S^3 dynamics)")
    print(f"  {'context':>7s}  {'top1':>7s}  {'top5':>7s}  {'top10':>7s}  {'mrr':>7s}")
    for cl in [1, 2, 4, 8]:
        r = eval_program(test_ids, wid_inv, wid, E_in_ops, E_out_ops, OPERATORS,
                         context_len=cl, n_eval=1500, renorm=True)
        if r is None: continue
        print(f"  {cl:>7d}  {r['top1']:>7.4f}  {r['top5']:>7.4f}  "
              f"{r['top10']:>7.4f}  {r['mrr']:>7.4f}")

    print("\n[6] Ablation — all operators replaced by identity (B^4 mode)")
    OPS_id = {w: I_q for w in OPERATORS}
    E_in_id  = install_at_scale(E_in,  wid, OPS_id, typical_in)
    E_out_id = install_at_scale(E_out, wid, OPS_id, typical_out)
    print(f"  {'context':>7s}  {'top1':>7s}  {'top5':>7s}  {'top10':>7s}  {'mrr':>7s}")
    for cl in [1, 4, 8]:
        r = eval_program(test_ids, wid_inv, wid, E_in_id, E_out_id, OPS_id,
                         context_len=cl, n_eval=1500, renorm=False)
        if r is None: continue
        print(f"  {cl:>7d}  {r['top1']:>7.4f}  {r['top5']:>7.4f}  "
              f"{r['top10']:>7.4f}  {r['mrr']:>7.4f}")

    print(f"\nDone in {time.time()-t0:.1f}s")
    print("="*75)
    print("  Reading:")
    print("  [2] = rank-4 SVD ceiling (prev-only baseline). Should be ~0.08")
    print("  [4] cl=1 should ~match [2]: Hamilton(I, E_in[w]) = E_in[w]")
    print("  [4] cl>1 beats [2] -> multi-step composition adds value")
    print("  [4] beats [6] -> Zipfian operator assignments carry signal")
    print("  [5] vs [4]    -> S^3-renormalized vs B^4 magnitudes-kept")
    print("="*75)

if __name__ == '__main__':
    main()

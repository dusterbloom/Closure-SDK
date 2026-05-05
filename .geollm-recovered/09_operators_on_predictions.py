#!/usr/bin/env python3
"""
GeoLLM v4 — operators act on PREDICTIONS, not on state (C)
==========================================================
Hypothesis: in v0/v1/v2, hand-assigned function-word operators left-multiplied
the running state q. Three runs in a row show this does nothing. Maybe the
operators belong at the OTHER boundary: not on the cognitive state during
composition, but on the SCORE VECTOR after resonance. That's closer to
Faltz's "verify" verb than to Hamilton ingest.

Algorithm:
  1. Compose state q the v2 way (operators rotate state, content via Hamilton)
  2. Resonate -> score vector S in R^V
  3. Walk operators in the last `K` context tokens, RIGHT TO LEFT:
       - NOT/no/never/n't  -> S' = max(S) - S          (rank inversion)
       - and/or/but        -> no-op for now (binary, needs second operand)
       - everything else   -> identity (no-op)
  4. Predict argmax of S'.

Test with two cuts of the data:
  - All test positions
  - Only positions where 'not' or 'no' appears in the last 3 context tokens
    (subset where score-side NOT is supposed to matter)
Compare with-inversion vs without-inversion on the negation subset.

Falsification target:
  - On all positions: score-side NOT does NOT hurt.
  - On negation-subset: score-side NOT helps.
If both pass, "operators on predictions" carries real signal.
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

NEGATION_WORDS = {'not', "n't", 'no', 'never'}

# ── Score-side operator: NOT inversion ────────────────────────────────
def apply_score_not(scores):
    """Rank-inversion: new_scores = max(scores) - scores. Tokens with low
    pre-NOT scores become high; tokens with high pre-NOT scores become low.
    Sum and ordering are reversed; ties preserved."""
    return scores.max() - scores

def apply_score_operators(scores, ctx_tokens, last_k=3):
    """Walk last_k tokens of context right-to-left. Apply score-side
    operators to the scores in encounter order."""
    out = scores.copy()
    tail = ctx_tokens[-last_k:] if len(ctx_tokens) >= last_k else ctx_tokens
    # Apply from right to left so "not the X" applies NOT after determining
    # the X-class; iterate backwards through tail.
    for tok in reversed(tail):
        if tok in NEGATION_WORDS:
            out = apply_score_not(out)
    return out

# ── Composition + genome (v2 style) ───────────────────────────────────
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

# ── Best v3 score function: full+phase ────────────────────────────────
def score_fullphase(q, S, S_phase, V, N, cutoff=math.pi/3):
    # full
    dots = np.clip(np.abs(S @ q), -1.0, 1.0)
    gaps = np.arccos(dots)
    w_full = np.cos(gaps); w_full[gaps > cutoff] = 0.0
    s_full = np.bincount(N, weights=np.clip(w_full,0,1), minlength=V)
    # phase
    _, p_q = hopf_decompose(q)
    d = np.abs(S_phase - p_q); d = np.minimum(d, 2*math.pi - d)
    w_phase = np.cos(d); w_phase[d > cutoff] = 0.0
    s_phase = np.bincount(N, weights=np.clip(w_phase,0,1), minlength=V)
    return s_full + s_phase

# ── Eval: compare with-vs-without score-side NOT ─────────────────────
def eval_compare(test_toks, S, S_phase, N, carriers, ops, wid, V,
                 context_len=5, n_eval=1500):
    valid = [t for t in test_toks if t in wid]
    if len(valid) <= context_len: return None, None, None
    pos = list(range(context_len, len(valid)))
    if len(pos) > n_eval: pos = random.sample(pos, n_eval)

    # Buckets:
    #   all_no_op: full test set, no score-side ops (v3 baseline)
    #   all_with_op: full test set, score-side NOT applied
    #   neg_subset_with_op vs without: positions where NOT is in last 3
    all_no, all_yes = stats_init(), stats_init()
    neg_no, neg_yes = stats_init(), stats_init()

    for p in pos:
        ctx = valid[p-context_len:p]
        target_id = wid[valid[p]]
        q = compose_state(ctx, carriers, ops)
        scores = score_fullphase(q, S, S_phase, V, N)
        scores_op = apply_score_operators(scores, ctx, last_k=3)

        has_neg = any(t in NEGATION_WORDS for t in ctx[-3:])

        update_stats(all_no, scores, target_id)
        update_stats(all_yes, scores_op, target_id)
        if has_neg:
            update_stats(neg_no, scores, target_id)
            update_stats(neg_yes, scores_op, target_id)

    return finalize(all_no), finalize(all_yes), finalize(neg_no), finalize(neg_yes)

def stats_init():
    return {'top1':0,'top5':0,'top10':0,'mrr':0.0,'n':0}

def update_stats(d, scores, target_id):
    ranked = np.argsort(-scores)
    if ranked[0] == target_id: d['top1'] += 1
    if target_id in ranked[:5]: d['top5'] += 1
    if target_id in ranked[:10]: d['top10'] += 1
    rk = np.where(ranked == target_id)[0]
    if len(rk) > 0: d['mrr'] += 1.0 / (rk[0] + 1)
    d['n'] += 1

def finalize(d):
    if d['n'] == 0: return d
    return {'top1':d['top1']/d['n'],'top5':d['top5']/d['n'],
            'top10':d['top10']/d['n'],'mrr':d['mrr']/d['n'],'n':d['n']}

def main():
    print("="*75); print("  GeoLLM v4 — operators on PREDICTIONS (score-side NOT)"); print("="*75)
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

    print("\n[1] Building genome (ctx=3)...")
    S, S_base, S_phase, N = build_genome(train, carriers, OPERATORS, wid, context_len=3)
    print(f"  size {len(N):,}")

    print("\n[2] Counting test positions with negation in last 3 ctx tokens...")
    valid = [t for t in test if t in wid]
    pos = list(range(5, len(valid)))
    n_with_neg = sum(1 for p in pos if any(t in NEGATION_WORDS for t in valid[p-3:p]))
    print(f"  total test positions {len(pos):,} · with negation {n_with_neg} ({n_with_neg/len(pos)*100:.1f}%)")

    print("\n[3] Eval: score-side NOT vs not, on full test + on negation subset")
    all_no, all_yes, neg_no, neg_yes = eval_compare(
        test, S, S_phase, N, carriers, OPERATORS, wid, V,
        context_len=5, n_eval=2000)

    print(f"\n  ── ALL POSITIONS ──")
    print(f"  {'condition':<25s}  {'top1':>7s}  {'top5':>7s}  {'top10':>7s}  {'mrr':>7s}  {'n':>5s}")
    print(f"  {'no score-op':<25s}  {all_no['top1']:>7.4f}  {all_no['top5']:>7.4f}  "
          f"{all_no['top10']:>7.4f}  {all_no['mrr']:>7.4f}  {all_no['n']:>5d}")
    print(f"  {'WITH score-side NOT':<25s}  {all_yes['top1']:>7.4f}  {all_yes['top5']:>7.4f}  "
          f"{all_yes['top10']:>7.4f}  {all_yes['mrr']:>7.4f}  {all_yes['n']:>5d}")

    print(f"\n  ── NEGATION SUBSET (NOT/no/never/n't in last 3 ctx tokens) ──")
    print(f"  {'condition':<25s}  {'top1':>7s}  {'top5':>7s}  {'top10':>7s}  {'mrr':>7s}  {'n':>5s}")
    if neg_no['n'] > 0:
        print(f"  {'no score-op':<25s}  {neg_no['top1']:>7.4f}  {neg_no['top5']:>7.4f}  "
              f"{neg_no['top10']:>7.4f}  {neg_no['mrr']:>7.4f}  {neg_no['n']:>5d}")
        print(f"  {'WITH score-side NOT':<25s}  {neg_yes['top1']:>7.4f}  {neg_yes['top5']:>7.4f}  "
              f"{neg_yes['top10']:>7.4f}  {neg_yes['mrr']:>7.4f}  {neg_yes['n']:>5d}")
    else:
        print("  (no negation positions sampled)")

    print(f"\nDone in {time.time()-t0:.1f}s")
    print("="*75)
    print("  Falsification:")
    print("   - On ALL: score-NOT must NOT hurt. (Hurts -> hypothesis dead.)")
    print("   - On negation subset: score-NOT must HELP.")
    print("="*75)

if __name__ == '__main__':
    main()

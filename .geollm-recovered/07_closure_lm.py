#!/usr/bin/env python3
"""
GeoLLM v2 — ClosureLM, Faltz-aligned non-parametric kNN on S² (with S¹ phase)
============================================================================

What changed vs v0/v1:
  - Token carriers come from domain_embed(token_bytes, phase=0): SHA-256 hash
    of bytes -> S² base direction. NO learned embedding from co-occurrence.
  - "Learning" = storing (state_after_left_context, true_next_carrier) pairs
    in a genome. Non-parametric. Same as kNN.
  - Prediction = resonate over the genome:
      gap_w = sigma_base(q_state, entry_state)   [S² distance, axis only]
      vote[next_token] += cos(gap_w) clamped >= 0    (cutoff at pi/3)
    Top-k by accumulated vote. Sub-linear with vectorized numpy.
  - Operators (function words) still hand-assigned (Faltz's "bootstrap, don't
    learn"). They left-multiply the running state on S^3.

This is what the closure_ea Rust crate already does for its ingest loop, ported
into the smallest standalone numpy that reproduces the LM-prediction question.

Win condition: beat full-bigram (top1=11.5%) on Shakespeare+P&P with token
carriers fixed by hashing — no embedding parameters trained from data, only
the genome as memory.
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

# ── Genome (vectorized): a stack of (state, next_token_id) pairs ──────
def build_genome(train_toks, carriers, ops, vocab_set, context_len=5,
                max_entries=None):
    """Walk training tokens, compose the left-context state at each position
    where the next token is in vocab, store (state, next_token_id) pair.
    """
    # Filter to known tokens (skip OOV) — keeps composition stable.
    known_idx = [i for i, t in enumerate(train_toks) if t in vocab_set]
    states = []
    nxt_ids = []
    for j, i in enumerate(known_idx):
        if i < context_len: continue
        ctx = train_toks[max(0, i-context_len):i]
        q = compose_state(ctx, carriers, ops)
        states.append(q)
        nxt_ids.append(known_idx_to_vocab_id(t=train_toks[i], wid=None))
    return states, nxt_ids

# Helper: vocab id encoder
def make_token_to_id(vocab):
    return {t: i for i, t in enumerate(vocab)}

def known_idx_to_vocab_id(t, wid):
    # Placeholder for build_genome refactor below.
    return None

def build_genome_v2(train_toks, carriers, ops, wid, context_len=5):
    """Vectorized state collection: store states as (G, 4) array and next-ids
    as (G,) int array."""
    states = []
    nxts = []
    n = len(train_toks)
    for i in range(context_len, n):
        target = train_toks[i]
        if target not in wid:
            continue
        ctx = train_toks[i-context_len:i]
        q = compose_state(ctx, carriers, ops)
        states.append(q)
        nxts.append(wid[target])
    S = np.stack(states, axis=0).astype(np.float64)  # (G, 4)
    N = np.array(nxts, dtype=np.int64)               # (G,)
    return S, N

# ── Vectorized resonance prediction ───────────────────────────────────
def predict_resonate(q_state, S, N, V, top_k=10, mode='base', cutoff=math.pi/3):
    """Resonate q against genome states S, weight by cos(gap), aggregate
    per-next-token, return ranked list.

    mode='base': S² distance — only axis matches (token identity).
    mode='full': full geodesic σ on S³.
    mode='phase': S¹ distance (cyclic position only — for diagnostics).
    """
    if mode == 'full':
        # σ(q, s) = arccos(|q·s|)  -- on S^3, unit quaternions
        dots = np.clip(np.abs(S @ q_state), -1.0, 1.0)
        gaps = np.arccos(dots)
    elif mode == 'base':
        # Decompose q and S to bases, S² distance
        # q: (4,) -> base (3,)
        b_q, _ = hopf_decompose(q_state)
        # vectorize hopf decompose for S
        w = S[:, 0]; x = S[:, 1]; y = S[:, 2]; z = S[:, 3]
        bases = np.stack([
            2.0*(x*z + w*y),
            2.0*(y*z - w*x),
            w*w + z*z - x*x - y*y
        ], axis=1)
        nrm = np.linalg.norm(bases, axis=1, keepdims=True)
        nrm[nrm < 1e-15] = 1.0
        bases = bases / nrm
        cosines = np.clip(bases @ b_q, -1.0, 1.0)
        gaps = np.arccos(np.abs(cosines))  # use |cos| because base is in RP^2
    else:
        raise ValueError(mode)

    # Coupling: cos(gap), 0 below cutoff
    weights = np.cos(gaps)
    weights[gaps > cutoff] = 0.0
    weights = np.clip(weights, 0.0, 1.0)

    # Aggregate per token via np.bincount (fast)
    scores = np.bincount(N, weights=weights, minlength=V)
    return scores  # (V,)

def evaluate(test_toks, S, N, carriers, ops, wid, V, context_len=5,
             n_eval=1500, mode='base', top_k=10):
    valid_test = [t for t in test_toks if t in wid]
    if len(valid_test) <= context_len:
        return None
    pos = list(range(context_len, len(valid_test)))
    if len(pos) > n_eval:
        pos = random.sample(pos, n_eval)

    top1 = top5 = top10 = 0; mrr = 0.0; n = 0
    for p in pos:
        ctx = valid_test[p-context_len:p]
        target_id = wid[valid_test[p]]
        q = compose_state(ctx, carriers, ops)
        scores = predict_resonate(q, S, N, V, top_k=top_k, mode=mode)
        ranked = np.argsort(-scores)
        if ranked[0] == target_id: top1 += 1
        if target_id in ranked[:5]: top5 += 1
        if target_id in ranked[:10]: top10 += 1
        rk = np.where(ranked == target_id)[0]
        if len(rk) > 0: mrr += 1.0 / (rk[0] + 1)
        n += 1
    return {'top1':top1/n,'top5':top5/n,'top10':top10/n,'mrr':mrr/n,'n':n}

# ── Main ──────────────────────────────────────────────────────────────
def main():
    print("="*75)
    print("  GeoLLM v2 — ClosureLM (Faltz-aligned: hash-embed + genome resonance)")
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

    print("\n[1] Hash-embedding tokens (no learning) -> S^2 base via SHA-256...")
    carriers = {w: domain_embed(w) for w in vocab}
    print(f"  Built {len(carriers)} carriers, all unit on S^3 ({carriers[vocab[0]].round(3)})")

    print("\n[2] Building genome from train: (state | left_ctx, next_token)")
    for ctx_len in [3, 5]:
        S, N = build_genome_v2(train, carriers, OPERATORS, wid, context_len=ctx_len)
        print(f"  ctx_len={ctx_len}: genome size {len(N):,}")

        print(f"\n[3] Resonance prediction (mode=base, ctx={ctx_len}):")
        for cl_eval in [3, 5, 8]:
            r = evaluate(test, S, N, carriers, OPERATORS, wid, V,
                         context_len=cl_eval, n_eval=1000, mode='base')
            if r: print(f"  eval ctx={cl_eval}  top1={r['top1']:.4f}  top5={r['top5']:.4f}  "
                        f"top10={r['top10']:.4f}  mrr={r['mrr']:.4f}")

        print(f"\n[4] Resonance prediction (mode=full, ctx={ctx_len}):")
        for cl_eval in [3, 5, 8]:
            r = evaluate(test, S, N, carriers, OPERATORS, wid, V,
                         context_len=cl_eval, n_eval=1000, mode='full')
            if r: print(f"  eval ctx={cl_eval}  top1={r['top1']:.4f}  top5={r['top5']:.4f}  "
                        f"top10={r['top10']:.4f}  mrr={r['mrr']:.4f}")

    print("\n[5] Ablation — operators all replaced by identity")
    OPS_id = {w: I_q for w in OPERATORS}
    S, N = build_genome_v2(train, carriers, OPS_id, wid, context_len=5)
    r = evaluate(test, S, N, carriers, OPS_id, wid, V,
                 context_len=5, n_eval=1000, mode='base')
    if r: print(f"  identity-ops ctx=5 mode=base  top1={r['top1']:.4f}  "
                f"top5={r['top5']:.4f}  top10={r['top10']:.4f}  mrr={r['mrr']:.4f}")

    print(f"\nDone in {time.time()-t0:.1f}s")
    print("="*75)
    print("  Reading: target to beat = 2-gram (top1≈0.115).")
    print("  This is non-parametric: NO trainable embedding parameters.")
    print("  Token carriers come from SHA-256 of bytes; learning = genome storage.")
    print("="*75)

if __name__ == '__main__':
    main()

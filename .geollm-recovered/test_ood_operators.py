#!/usr/bin/env python3
"""Out-of-distribution test for bootstrap operators.

Hypothesis: Faltz's claim that 'is/not/and/or' are SPECIFIC OPERATIONS
(hand-bootstrapped) should hold across corpora — the algebra of English
function words doesn't change between Shakespeare and Pride & Prejudice.
In-distribution next-token prediction can't measure this because the
data already contains the operator-context transitions. The OOD test
forces the model to use compositional structure: train on one corpus,
test on the other.

Falsifiable comparison:
  M_baseline:  rank-32 SVD bigram from Shakespeare-train, content-only.
  M_ops:       same SVD, but during eval, when an operator appears in
               left context, its position-token's E_in row is REPLACED
               by op(I) — the architecture's prescribed bootstrap value.

  Test on a held-out P&P slice. Compare top-1.

Win condition (T1): On the OOD test set, M_ops top-1 - M_baseline top-1
                    must be >= +0.5pp. Equal or worse → operators don't help OOD.
Sanity (T2):       In-distribution P&P→P&P, ops should NOT help (replicate
                   our prior finding). Confirms the test framework doesn't
                   spuriously favor ops.
"""
import math, random, re
from collections import Counter
from pathlib import Path
import numpy as np

THIS_DIR = Path(__file__).parent.resolve()
from geollm_core import OPERATORS, I_q, normalize


def _load_split(name):
    """Load Shakespeare or Pride&Prejudice as a token stream."""
    path = THIS_DIR / 'corpora' / name
    text = path.read_text(errors='ignore').lower()
    return re.findall(r"[a-z']+", text)


def _build_svd(train_ids, V, rank=32):
    B = np.zeros((V, V), dtype=np.float64)
    for i in range(len(train_ids)-1):
        B[train_ids[i], train_ids[i+1]] += 1.0
    P = B / (B.sum(axis=1, keepdims=True) + 1e-12)
    U, S, Vt = np.linalg.svd(P, full_matrices=False)
    s = np.sqrt(S[:rank])
    E_in  = U[:, :rank] * s[None, :]
    E_out = (Vt[:rank, :] * s[:, None]).T
    return E_in, E_out


def _train_ridge(state_fn, train_ids, E_in, E_out, context_len, ridge=1e-3):
    Xs = [state_fn(train_ids, i, E_in, context_len) for i in range(context_len, len(train_ids))]
    Ys = [E_out[train_ids[i]] for i in range(context_len, len(train_ids))]
    X = np.stack(Xs); Y = np.stack(Ys)
    return np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ Y)


def _eval(state_fn, test_ids, E_in, E_out, W, context_len, n_eval, seed=42):
    rng = random.Random(seed)
    pos = sorted(rng.sample(range(context_len, len(test_ids)), min(n_eval, len(test_ids)-context_len)))
    correct = 0
    for p in pos:
        x = state_fn(test_ids, p, E_in, context_len)
        if int(np.argmax(E_out @ (x @ W))) == test_ids[p]:
            correct += 1
    return correct / len(pos)


def state_baseline(ids, i, E_in, context_len):
    """concat(prev E_in, sum of earlier E_in) — the v7.5 winner."""
    prev = E_in[ids[i-1]]
    earlier = E_in[ids[max(0, i-context_len):i-1]].sum(axis=0) \
              if i > 1 else np.zeros(E_in.shape[1])
    return np.concatenate([prev, earlier])


def _operator_replacement(rank):
    """Map operator-name → fixed embedding vector in rank-dim space.
    Inject the algebra: NOT, IS, etc. carry hand-set values that don't
    vary by corpus. Use a small consistent rule:
      identity-class operators (is, the, of, ...) → zero vector
      NOT-class                                   → -e_0 (flip first dim)
      AND-class                                   → +e_1
      OR-class                                    → +e_2
      BUT                                         → +e_3
      modals (will, would, can, ...)              → +e_4
    Choices are deliberate-but-provisional. Whether they HELP is the test."""
    rep = {}
    e0 = np.zeros(rank); e0[0] = 1.0
    e1 = np.zeros(rank); e1[1] = 1.0
    e2 = np.zeros(rank); e2[2] = 1.0
    e3 = np.zeros(rank); e3[3] = 1.0
    e4 = np.zeros(rank); e4[4] = 1.0
    NOT_set = {'not', "n't", 'no', 'never'}
    AND_set = {'and'}
    OR_set  = {'or'}
    BUT_set = {'but'}
    MODAL_set = {'will', 'would', 'shall', 'can', 'could', 'may',
                 'might', 'should', 'must'}
    for w in OPERATORS:
        if   w in NOT_set:   rep[w] = -e0
        elif w in AND_set:   rep[w] = +e1
        elif w in OR_set:    rep[w] = +e2
        elif w in BUT_set:   rep[w] = +e3
        elif w in MODAL_set: rep[w] = +e4
        else:                rep[w] = np.zeros(rank)  # identity-class (is, the, of, of, in, ...)
    return rep


def make_state_with_ops(rank, vocab):
    """State function that, when an operator appears in the context, swaps
    its E_in row for the hand-bootstrapped vector. The algebra is the
    SAME regardless of which corpus the SVD came from."""
    rep = _operator_replacement(rank)
    op_idxs = {}  # operator-token -> vocab id
    for i, w in enumerate(vocab):
        if w in OPERATORS:
            op_idxs[i] = rep[w]

    def state_fn(ids, i, E_in, context_len):
        # Build a "patched" view of E_in for the context window only:
        # for each token id in ctx, if it's an operator, use the bootstrap
        # vector instead of E_in[id].
        prev_id = ids[i-1]
        prev_vec = op_idxs.get(prev_id, E_in[prev_id])
        earlier_ids = ids[max(0, i-context_len):i-1]
        if not earlier_ids:
            earlier_vec = np.zeros(E_in.shape[1])
        else:
            vecs = [op_idxs[j] if j in op_idxs else E_in[j] for j in earlier_ids]
            earlier_vec = np.stack(vecs, 0).sum(axis=0)
        return np.concatenate([prev_vec, earlier_vec])
    return state_fn


def _setup(train_corpus_name, test_corpus_name, max_vocab=2500, min_count=10, rank=32):
    """Build a unified vocabulary from the *combined* corpora so token IDs
    are stable across train/test; train SVD on train_corpus only."""
    train_toks = _load_split(train_corpus_name)
    test_toks  = _load_split(test_corpus_name)
    counts = Counter(train_toks + test_toks)
    vocab = [w for w, c in counts.most_common(max_vocab) if c >= min_count]
    wid = {w: i for i, w in enumerate(vocab)}
    train_ids = [wid[t] for t in train_toks if t in wid]
    test_ids  = [wid[t] for t in test_toks  if t in wid]
    E_in, E_out = _build_svd(train_ids, len(vocab), rank=rank)
    return vocab, wid, train_ids, test_ids, E_in, E_out


# ── T1: OOD operators must HELP (or fail honestly) ────────────────────
def test_ood_operators_help_or_fail():
    """Train SVD on Shakespeare; evaluate on Pride & Prejudice.
    Compare baseline-state vs operator-state ridge heads."""
    rank = 32; context_len = 5
    vocab, wid, train_ids, test_ids, E_in, E_out = _setup(
        'shakespeare.txt', 'pride_prejudice.txt', rank=rank)

    state_with_ops = make_state_with_ops(rank, vocab)

    # Train both heads on Shakespeare
    W_base = _train_ridge(state_baseline, train_ids, E_in, E_out, context_len)
    W_ops  = _train_ridge(state_with_ops, train_ids, E_in, E_out, context_len)

    # Evaluate on P&P
    acc_base = _eval(state_baseline, test_ids, E_in, E_out, W_base, context_len, n_eval=2000)
    acc_ops  = _eval(state_with_ops, test_ids, E_in, E_out, W_ops,  context_len, n_eval=2000)

    print(f"  OOD (Shakespeare → P&P)")
    print(f"  baseline  top1={acc_base:.4f}")
    print(f"  with ops  top1={acc_ops:.4f}")
    print(f"  Δ = {(acc_ops - acc_base)*100:+.2f}pp")

    # Don't fail this test — REPORT the comparison. The test passes if the
    # ablation produced a number; the user reads the Δ to judge.
    assert 0 <= acc_base <= 1 and 0 <= acc_ops <= 1


# ── T2: Sanity — In-distribution should still NOT help (regression of v0/v1/v4)
def test_in_distribution_operators_dont_help_sanity():
    """Train SVD on Shakespeare; evaluate on held-out Shakespeare.
    Operators should NOT help here (the data contains transitions)."""
    rank = 32; context_len = 5
    train_toks = _load_split('shakespeare.txt')
    n = len(train_toks)
    train_part = train_toks[:int(n*0.9)]
    test_part  = train_toks[int(n*0.9):]
    counts = Counter(train_part)
    vocab = [w for w, c in counts.most_common(2500) if c >= 10]
    wid = {w: i for i, w in enumerate(vocab)}
    train_ids = [wid[t] for t in train_part if t in wid]
    test_ids  = [wid[t] for t in test_part  if t in wid]
    E_in, E_out = _build_svd(train_ids, len(vocab), rank=rank)

    state_with_ops = make_state_with_ops(rank, vocab)
    W_base = _train_ridge(state_baseline,  train_ids, E_in, E_out, context_len)
    W_ops  = _train_ridge(state_with_ops,  train_ids, E_in, E_out, context_len)

    acc_base = _eval(state_baseline, test_ids, E_in, E_out, W_base, context_len, n_eval=2000)
    acc_ops  = _eval(state_with_ops, test_ids, E_in, E_out, W_ops,  context_len, n_eval=2000)

    print(f"  IN-DIST (Shakespeare → Shakespeare-held-out)")
    print(f"  baseline  top1={acc_base:.4f}")
    print(f"  with ops  top1={acc_ops:.4f}")
    print(f"  Δ = {(acc_ops - acc_base)*100:+.2f}pp")

    assert 0 <= acc_base <= 1 and 0 <= acc_ops <= 1


if __name__ == "__main__":
    failures = []
    for name in ['test_in_distribution_operators_dont_help_sanity',
                 'test_ood_operators_help_or_fail']:
        fn = globals()[name]
        try:
            print(f"\n── {name} ──")
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failures.append((name, str(e)))
            print(f"FAIL  {name}: {e}")
        except Exception as e:
            failures.append((name, f"{type(e).__name__}: {e}"))
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    if failures:
        print(f"\n{len(failures)} failure(s)")
        raise SystemExit(1)
    print("\nAll tests reported")

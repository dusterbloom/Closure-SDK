#!/usr/bin/env python3
"""
Re-test next-token prediction with Hopf-style channel split:
  - Function words (operators) live in a SEPARATE W register
  - Content tokens live in the RGB content register (rank-32 SVD as v7.5)
  - The two registers NEVER mix
  - Predictor reads concat(W_state, RGB_prev, RGB_earlier_sum)

Hypothesis after this morning's compositional-generalization win:
  V7.5 (mixed channels) hit 9.07% top-1.
  V8 (Hopf-split channels) should beat it because the operator signal,
  preserved orthogonally to content, can now help where it was previously
  destroyed by mixing.

Win condition: top-1 > 9.5% on the same fixed test positions.
Falsification: top-1 ≤ 9.07% — the Hopf split hypothesis fails on real text
                even though it succeeded on synthetic minimal pairs.
"""
import re, math, random, hashlib
from collections import Counter
from pathlib import Path
import numpy as np

from geollm_core import OPERATORS

random.seed(42); np.random.seed(42)

THIS_DIR = Path(__file__).parent.resolve()
W_DIM = 8      # operator register size
RANK  = 32     # content SVD rank
CTX   = 5
N_EVAL = 1500


def load_corpus():
    here = THIS_DIR / 'corpora'
    parts = []
    for p in [here/'shakespeare.txt', here/'pride_prejudice.txt']:
        if p.exists(): parts.append(p.read_text(errors='ignore').lower())
    return ' '.join(parts)


def build_svd_embeddings(train_ids, V, rank=RANK):
    B = np.zeros((V, V), dtype=np.float64)
    for i in range(len(train_ids) - 1):
        B[train_ids[i], train_ids[i+1]] += 1.0
    P = B / (B.sum(axis=1, keepdims=True) + 1e-12)
    U, S, Vt = np.linalg.svd(P, full_matrices=False)
    s = np.sqrt(S[:rank])
    return U[:, :rank] * s[None, :], (Vt[:rank, :] * s[:, None]).T


def build_op_w_vectors(w_dim=W_DIM, seed=42, narrow=False):
    """If narrow=True, ONLY treat algebraically-stable logical operators
    as Zipfian operators (NOT, AND, OR, BUT). All other 'function words'
    fall through to content treatment.
    If narrow=False (broad), categorize all 50+ entries in OPERATORS."""
    rng = np.random.RandomState(seed)
    A = rng.randn(8, w_dim)
    Q, _ = np.linalg.qr(A.T)
    Q = Q.T
    NEG = {'not', "n't", 'no', 'never'}
    AND = {'and'}
    OR  = {'or'}
    BUT = {'but'}
    if narrow:
        # Only logical operators get a W-channel marker. Everything else
        # — determiners, prepositions, copulas, modals, pronouns — stays
        # in the content (RGB) channel like ordinary words.
        out = {}
        for w in NEG: out[w] = Q[0]
        for w in AND: out[w] = Q[1]
        for w in OR:  out[w] = Q[2]
        for w in BUT: out[w] = Q[3]
        return out
    # Broad: all OPERATORS entries get classified
    MODAL = {'will','would','shall','can','could','may','might','should','must'}
    AUX = {'have','has','had','do','does','did','be','been','being','am','are','is','was','were'}
    DET = {'the','a','an'}
    PREP = {'of','to','in','on','at','for','with','by','from','as','into','about'}
    PRON = {'i','you','he','she','it','we','they','this','that','these','those'}
    classes = [NEG, AND, OR, BUT, MODAL, AUX, DET, PREP, PRON]
    out = {}
    for w in OPERATORS:
        for ci, cls in enumerate(classes):
            if w in cls:
                out[w] = Q[ci % Q.shape[0]]
                break
        else:
            out[w] = np.zeros(w_dim)
    return out


def make_state_v75(ids, i, E_in):
    """v7.5 baseline: concat(prev_E_in, sum-earlier-E_in). No channel split."""
    prev = E_in[ids[i-1]]
    earlier = E_in[ids[max(0, i-CTX):i-1]].sum(axis=0) if i > 1 else np.zeros(RANK)
    return np.concatenate([prev, earlier])


def make_state_hopf(ids, i, E_in, op_vecs, vocab):
    """Hopf split: W register holds operator markers, RGB holds content."""
    # W register: sum of operator markers in left CTX
    w_state = np.zeros(W_DIM)
    rgb_prev = np.zeros(RANK)
    rgb_earlier = np.zeros(RANK)
    lo = max(0, i - CTX)
    for j in range(lo, i):
        tok = vocab[ids[j]]
        if tok in op_vecs:
            w_state = w_state + op_vecs[tok]
        else:  # content
            if j == i - 1:
                rgb_prev = E_in[ids[j]]
            else:
                rgb_earlier = rgb_earlier + E_in[ids[j]]
    n = np.linalg.norm(w_state); w_state = w_state / n if n > 1e-12 else w_state
    return np.concatenate([w_state, rgb_prev, rgb_earlier])


def train_ridge(state_fn, train_ids, E_in, E_out, ridge=1e-3, **kwargs):
    Xs, Ys = [], []
    for i in range(CTX, len(train_ids)):
        Xs.append(state_fn(train_ids, i, E_in, **kwargs))
        Ys.append(E_out[train_ids[i]])
    X = np.stack(Xs); Y = np.stack(Ys)
    return np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ Y)


def eval_ridge(state_fn, test_ids, E_in, E_out, W, n_eval=N_EVAL, **kwargs):
    rng = random.Random(42)
    pos = sorted(rng.sample(range(CTX, len(test_ids)), min(n_eval, len(test_ids)-CTX)))
    correct = top5 = top10 = 0
    for p in pos:
        x = state_fn(test_ids, p, E_in, **kwargs)
        pred = x @ W
        ranked = np.argsort(-(E_out @ pred))
        if int(ranked[0]) == test_ids[p]: correct += 1
        if test_ids[p] in ranked[:5]: top5 += 1
        if test_ids[p] in ranked[:10]: top10 += 1
    n = len(pos)
    return {'top1': correct/n, 'top5': top5/n, 'top10': top10/n, 'n': n}


def main():
    print("="*75)
    print("  Hopf channel-split next-token LM (v8)")
    print("="*75)

    corpus = load_corpus()
    toks = re.findall(r"[a-z']+", corpus)
    n_train = int(len(toks) * 0.9)
    train, test = toks[:n_train], toks[n_train:]
    counts = Counter(train)
    vocab = [w for w, c in counts.most_common(2500) if c >= 10]
    wid = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    train_ids = [wid[t] for t in train if t in wid]
    test_ids  = [wid[t] for t in test  if t in wid]

    print(f"  V={V} · train {len(train_ids):,} · test {len(test_ids):,}")
    print(f"  W register dim={W_DIM}, content rank={RANK}, ctx={CTX}")

    print("\n  Building SVD embeddings (rank=32)...")
    E_in, E_out = build_svd_embeddings(train_ids, V, rank=RANK)

    op_vecs_broad  = build_op_w_vectors(w_dim=W_DIM, narrow=False)
    op_vecs_narrow = build_op_w_vectors(w_dim=W_DIM, narrow=True)
    print(f"  Broad operators: {len(op_vecs_broad)} tokens (all of OPERATORS)")
    print(f"  Narrow operators: {len(op_vecs_narrow)} tokens (logic only: NOT/AND/OR/BUT/etc)")

    print("\n  Training v7.5 baseline (mixed channels)...")
    W_base = train_ridge(make_state_v75, train_ids, E_in, E_out)
    r_base = eval_ridge(make_state_v75, test_ids, E_in, E_out, W_base)

    print("  Training v8-broad (Hopf split, all function words → W)...")
    W_broad = train_ridge(make_state_hopf, train_ids, E_in, E_out,
                          op_vecs=op_vecs_broad, vocab=vocab)
    r_broad = eval_ridge(make_state_hopf, test_ids, E_in, E_out, W_broad,
                         op_vecs=op_vecs_broad, vocab=vocab)

    print("  Training v8-narrow (Hopf split, ONLY logical ops → W)...")
    W_narrow = train_ridge(make_state_hopf, train_ids, E_in, E_out,
                           op_vecs=op_vecs_narrow, vocab=vocab)
    r_narrow = eval_ridge(make_state_hopf, test_ids, E_in, E_out, W_narrow,
                          op_vecs=op_vecs_narrow, vocab=vocab)

    print(f"\n  {'model':<48s}  {'top1':>7s}  {'top5':>7s}  {'top10':>7s}")
    print(f"  {'v7.5 (mixed channels, baseline)':<48s}  {r_base['top1']:>7.4f}  {r_base['top5']:>7.4f}  {r_base['top10']:>7.4f}")
    print(f"  {'v8-broad (all 50+ Zipfian → W)':<48s}  {r_broad['top1']:>7.4f}  {r_broad['top5']:>7.4f}  {r_broad['top10']:>7.4f}")
    print(f"  {'v8-narrow (only NOT/AND/OR/BUT → W)':<48s}  {r_narrow['top1']:>7.4f}  {r_narrow['top5']:>7.4f}  {r_narrow['top10']:>7.4f}")
    print(f"\n  Δ top1:  v8-broad - v7.5 = {(r_broad['top1']-r_base['top1'])*100:+.2f} pp")
    print(f"           v8-narrow - v7.5 = {(r_narrow['top1']-r_base['top1'])*100:+.2f} pp")
    print(f"           v8-narrow - v8-broad = {(r_narrow['top1']-r_broad['top1'])*100:+.2f} pp")
    print("="*75)
    if r_narrow['top1'] > r_base['top1'] + 0.005:
        print("  ✓ ZIPF SAVES IT — but only for the algebraically stable subset.")
        print("    Logical operators DO transfer; broad function-word categories don't.")
    elif r_narrow['top1'] > r_broad['top1'] + 0.005:
        print("  ~ Narrow > Broad: limiting to logical operators recovers SOME ground.")
    else:
        print("  ✗ Even narrow logical operators don't help on real text.")

    # ── ABLATION: channel-split architecture WITHOUT operators ──
    # If v8-narrow > v8-empty (no operators in W register), then operators
    # specifically contribute. If v8-narrow ≈ v8-empty, the lift is from the
    # architecture (the extra W register dims) not the operator semantics.
    print("\n  ── ABLATION: isolate operator contribution ──")
    op_vecs_empty = {}  # no operators get W markers
    W_empty = train_ridge(make_state_hopf, train_ids, E_in, E_out,
                          op_vecs=op_vecs_empty, vocab=vocab)
    r_empty = eval_ridge(make_state_hopf, test_ids, E_in, E_out, W_empty,
                         op_vecs=op_vecs_empty, vocab=vocab)
    print(f"  {'v8-empty (channel-split, no ops in W)':<48s}  {r_empty['top1']:>7.4f}  {r_empty['top5']:>7.4f}  {r_empty['top10']:>7.4f}")
    delta_op = (r_narrow['top1'] - r_empty['top1']) * 100
    print(f"\n  v8-narrow - v8-empty = {delta_op:+.2f} pp  (operator-specific contribution)")
    if delta_op > 0.5:
        print(f"  ✓ Operators carry their own signal beyond the architecture.")
    elif delta_op > 0.0:
        print(f"  ~ Operators contribute mildly; most lift is architectural.")
    else:
        print(f"  ~ Lift is purely architectural; operators don't add signal.")


if __name__ == "__main__":
    main()

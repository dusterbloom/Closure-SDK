#!/usr/bin/env python3
"""
The test Faltz's hypothesis was actually designed for, that we'd been
avoiding: compositional generalization on synthetic minimal pairs,
with held-out content tokens.

Setup:
  Vocabulary: 50 'subject' tokens (sub_0..sub_49), 50 'predicate' tokens
              (pred_0..pred_49), 4 operator tokens (is, not, and, or),
              4 reply tokens (CONFIRM, DENY, CONJ, DISJ).
  Sentences (3 tokens each, predict the 4th):
       "<s> is <p>"        -> next = CONFIRM
       "<s> is not <p>"    -> next = DENY     (4 tokens, predict 5th)
       "<s> and <p>"       -> next = CONJ
       "<s> or <p>"        -> next = DISJ
       (Use 3-token contexts; we encode IS-NOT as a separate context type.)

Train/test split is on the SUBJECT and PREDICATE token sets:
  Train pairs = first 40 subjects × first 40 predicates  (1600 (s,p) combos)
  Test pairs  = last 10 subjects × last 10 predicates    (100 held-out combos
                                                         — content tokens
                                                         NEVER seen with these
                                                         specific operators
                                                         during training)

Models compared:
  (A) WITHOUT operators: each token gets a learned 16-dim embedding via
      least-squares (closed form on bigram-style state). No operator algebra.
  (B) WITH operators: same content embeddings, but is/not/and/or are
      replaced by FIXED rotations on the running state (Faltz-style).
      The WIN: held-out (s,p) pairs predict the correct reply because the
      operator rotation acts identically regardless of content seen.

Falsification:
  T1. Both models converge to ~100% on TRAIN (sanity).
  T2. Model (B) WITH operators beats Model (A) WITHOUT operators on
      HELD-OUT pairs by >= 10pp.

If T2 passes: Faltz's bootstrap-operator hypothesis is empirically
              validated for compositional generalization. Day's headline.
If T2 fails:  even on the task designed for the hypothesis, hand-assigned
              operators don't transfer better than a learned head.
              Hypothesis is dead for empirical regimes.
"""
import math, random, hashlib
import numpy as np

random.seed(42); np.random.seed(42)


def build_corpus(n_subjects=50, n_predicates=50, train_subj=40, train_pred=40):
    subjects   = [f"sub_{i}"  for i in range(n_subjects)]
    predicates = [f"pred_{i}" for i in range(n_predicates)]
    operators  = ["is", "not", "and", "or"]
    replies    = ["CONFIRM", "DENY", "CONJ", "DISJ"]
    vocab = subjects + predicates + operators + replies
    wid = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)

    train_subjects = subjects[:train_subj]
    test_subjects  = subjects[train_subj:]
    train_preds    = predicates[:train_pred]
    test_preds     = predicates[train_pred:]

    def make_sentences(subjects_set, preds_set):
        sents = []
        for s in subjects_set:
            for p in preds_set:
                sents.append(([s, "is", p],          "CONFIRM"))
                sents.append(([s, "is", "not", p],   "DENY"))
                sents.append(([s, "and", p],         "CONJ"))
                sents.append(([s, "or", p],          "DISJ"))
        return sents

    train_sents = make_sentences(train_subjects, train_preds)
    test_sents  = make_sentences(test_subjects,  test_preds)
    return vocab, wid, train_sents, test_sents, replies


# ── Embeddings: learned via least-squares from training data ─────────
def hash_unit(token, dim, seed=0):
    h = hashlib.sha256(f"{seed}:{token}".encode()).digest()
    rng = np.random.RandomState(int.from_bytes(h[:4], 'little'))
    v = rng.randn(dim); return v / np.linalg.norm(v)


def make_state_no_ops(ctx, E_in, wid):
    """State = sum of context embeddings. No operator special-casing."""
    vec = np.zeros(E_in.shape[1])
    for t in ctx:
        if t in wid:
            vec = vec + E_in[wid[t]]
    n = np.linalg.norm(vec)
    return vec / n if n > 1e-12 else vec


# ── Operator table (model B): fixed rotations in embed space ─────────
def make_operator_rotations(dim, seed=42):
    """Each operator gets a fixed orthogonal rotation matrix.
    The rotations are RANDOM but FIXED — same matrix used at train and
    test time. The hypothesis: applying the same rotation regardless of
    content lets generalization transfer to unseen content tokens."""
    rng = np.random.RandomState(seed)
    mats = {}
    for name in ["is", "not", "and", "or"]:
        # Each operator gets a distinct random orthogonal rotation.
        # Even IS gets a non-identity rotation so the running state
        # encodes that "is" was the operator (otherwise IS=identity is
        # indistinguishable from no-operator and the model cannot learn
        # to predict CONFIRM specifically for "s is p" patterns).
        A = rng.randn(dim, dim)
        Q, _ = np.linalg.qr(A)
        mats[name] = Q
    return mats


def make_state_with_ops(ctx, E_in, wid, op_mats):
    """Walk context: content tokens add to state; operators rotate state.
    Then unit-normalize."""
    dim = E_in.shape[1]
    state = np.zeros(dim)
    for t in ctx:
        if t in op_mats:
            state = op_mats[t] @ state
        elif t in wid:
            state = state + E_in[wid[t]]
    n = np.linalg.norm(state)
    return state / n if n > 1e-12 else state


def make_state_hopf_split(ctx, E_in, wid, op_vecs, w_dim=4):
    """Hopf-style channel split: operators act on the W register only,
    content sums into the RGB register only. State = concat(W, RGB).
    Operators DON'T touch content; content DOESN'T touch operator state.
    This is what S^3 → S^2 × S^1 was designed for."""
    dim = E_in.shape[1]
    rgb_dim = dim - w_dim
    w_state = np.zeros(w_dim)
    rgb_state = np.zeros(rgb_dim)
    for t in ctx:
        if t in op_vecs:
            # Operator adds its W-channel embedding (orthogonal markers)
            w_state = w_state + op_vecs[t]
        elif t in wid:
            # Content adds to RGB channel (last (dim-w_dim) coords of E_in)
            rgb_state = rgb_state + E_in[wid[t], w_dim:]
    # Unit-normalize each channel independently
    nw = np.linalg.norm(w_state); w_state = w_state / nw if nw > 1e-12 else w_state
    nr = np.linalg.norm(rgb_state); rgb_state = rgb_state / nr if nr > 1e-12 else rgb_state
    return np.concatenate([w_state, rgb_state])


def make_operator_w_vectors(w_dim=4, seed=42):
    """Each operator gets a fixed vector in the W channel — designed
    so that the 4 operators are mutually orthogonal markers."""
    rng = np.random.RandomState(seed)
    # Use 4 mutually-orthogonal vectors from a random orthonormal basis
    A = rng.randn(w_dim, w_dim)
    Q, _ = np.linalg.qr(A)
    return {
        "is":  Q[0],
        "not": Q[1],
        "and": Q[2],
        "or":  Q[3],
    }


# ── Train ridge head ────────────────────────────────────────────────
def train_ridge(sentences, E_in, wid, state_fn, target_to_id, dim, ridge=1e-3,
                **state_kwargs):
    Xs, Ys = [], []
    for ctx, target in sentences:
        x = state_fn(ctx, E_in, wid, **state_kwargs)
        y_onehot = np.zeros(len(target_to_id))
        y_onehot[target_to_id[target]] = 1.0
        Xs.append(x); Ys.append(y_onehot)
    X = np.stack(Xs); Y = np.stack(Ys)
    W = np.linalg.solve(X.T @ X + ridge * np.eye(dim), X.T @ Y)
    return W


def evaluate(sentences, E_in, wid, W, state_fn, target_to_id, **state_kwargs):
    correct = 0; n = 0
    id_to_target = {v: k for k, v in target_to_id.items()}
    for ctx, target in sentences:
        x = state_fn(ctx, E_in, wid, **state_kwargs)
        scores = x @ W
        pred_id = int(np.argmax(scores))
        if id_to_target[pred_id] == target:
            correct += 1
        n += 1
    return correct / n


def hopfield_retrieve(query, keys, values, beta=32.0):
    """Modern Hopfield retrieval: softmax(β · K·q) · V.
    keys: (N, dim), values: (N, n_classes) one-hot, query: (dim,)
    Returns (n_classes,) probability over classes."""
    scores = beta * (keys @ query)
    scores = scores - scores.max()
    w = np.exp(scores); w /= w.sum()
    return w @ values


def evaluate_hopfield(test_sents, E_in, wid, train_keys, train_values,
                      state_fn, target_to_id, beta=32.0, **state_kwargs):
    correct = 0; n = 0
    id_to_target = {v: k for k, v in target_to_id.items()}
    for ctx, target in test_sents:
        q = state_fn(ctx, E_in, wid, **state_kwargs)
        probs = hopfield_retrieve(q, train_keys, train_values, beta=beta)
        if id_to_target[int(np.argmax(probs))] == target:
            correct += 1
        n += 1
    return correct / n


def build_hopfield_memory(sentences, E_in, wid, state_fn, target_to_id,
                           **state_kwargs):
    Ks, Vs = [], []
    for ctx, target in sentences:
        x = state_fn(ctx, E_in, wid, **state_kwargs)
        y = np.zeros(len(target_to_id))
        y[target_to_id[target]] = 1.0
        Ks.append(x); Vs.append(y)
    return np.stack(Ks), np.stack(Vs)


def main():
    print("="*75)
    print("  Compositional generalization on minimal pairs")
    print("="*75)

    vocab, wid, train_sents, test_sents, replies = build_corpus()
    V = len(vocab)
    print(f"  V={V} · train sents={len(train_sents)} · test sents (held-out (s,p))={len(test_sents)}")

    target_to_id = {r: i for i, r in enumerate(replies)}

    # Frozen embeddings: random unit vectors per token, dim=16
    DIM = 16
    E_in = np.stack([hash_unit(w, DIM) for w in vocab], axis=0)

    # ── Model A: NO OPERATORS (operators are just summed like content) ──
    W_a = train_ridge(train_sents, E_in, wid, make_state_no_ops, target_to_id, DIM)
    train_acc_a = evaluate(train_sents, E_in, wid, W_a, make_state_no_ops, target_to_id)
    test_acc_a  = evaluate(test_sents,  E_in, wid, W_a, make_state_no_ops, target_to_id)

    # ── Model B: WITH OPERATORS (fixed rotations on running state) ──
    op_mats = make_operator_rotations(DIM)
    W_b = train_ridge(train_sents, E_in, wid, make_state_with_ops, target_to_id, DIM,
                      op_mats=op_mats)
    train_acc_b = evaluate(train_sents, E_in, wid, W_b, make_state_with_ops, target_to_id,
                           op_mats=op_mats)
    test_acc_b  = evaluate(test_sents,  E_in, wid, W_b, make_state_with_ops, target_to_id,
                           op_mats=op_mats)

    # ── Model C: WITH OPERATORS + Hopfield (bilinear) head ──
    K_c, V_c = build_hopfield_memory(train_sents, E_in, wid, make_state_with_ops,
                                      target_to_id, op_mats=op_mats)
    train_acc_c = evaluate_hopfield(train_sents, E_in, wid, K_c, V_c,
                                     make_state_with_ops, target_to_id, beta=32.0,
                                     op_mats=op_mats)
    test_acc_c = evaluate_hopfield(test_sents, E_in, wid, K_c, V_c,
                                    make_state_with_ops, target_to_id, beta=32.0,
                                    op_mats=op_mats)

    # ── Model D: NO OPERATORS + Hopfield head (control for C) ──
    K_d, V_d = build_hopfield_memory(train_sents, E_in, wid, make_state_no_ops,
                                      target_to_id)
    train_acc_d = evaluate_hopfield(train_sents, E_in, wid, K_d, V_d,
                                     make_state_no_ops, target_to_id, beta=32.0)
    test_acc_d = evaluate_hopfield(test_sents, E_in, wid, K_d, V_d,
                                    make_state_no_ops, target_to_id, beta=32.0)

    # ── Model E: HOPF-CHANNELIZED operators (W) + content (RGB), ridge ──
    op_vecs = make_operator_w_vectors(w_dim=4)
    W_e = train_ridge(train_sents, E_in, wid, make_state_hopf_split, target_to_id,
                       DIM, op_vecs=op_vecs, w_dim=4)
    train_acc_e = evaluate(train_sents, E_in, wid, W_e, make_state_hopf_split,
                            target_to_id, op_vecs=op_vecs, w_dim=4)
    test_acc_e = evaluate(test_sents, E_in, wid, W_e, make_state_hopf_split,
                           target_to_id, op_vecs=op_vecs, w_dim=4)

    # ── Model F: HOPF-CHANNELIZED + Hopfield head ──
    K_f, V_f = build_hopfield_memory(train_sents, E_in, wid, make_state_hopf_split,
                                      target_to_id, op_vecs=op_vecs, w_dim=4)
    train_acc_f = evaluate_hopfield(train_sents, E_in, wid, K_f, V_f,
                                     make_state_hopf_split, target_to_id,
                                     beta=32.0, op_vecs=op_vecs, w_dim=4)
    test_acc_f = evaluate_hopfield(test_sents, E_in, wid, K_f, V_f,
                                    make_state_hopf_split, target_to_id,
                                    beta=32.0, op_vecs=op_vecs, w_dim=4)

    print(f"\n  {'model':<48s}  {'train':>8s}  {'test':>18s}")
    print(f"  {'A: NO ops + ridge':<48s}  {train_acc_a:>8.4f}  {test_acc_a:>18.4f}")
    print(f"  {'B: ops-rotate-content + ridge':<48s}  {train_acc_b:>8.4f}  {test_acc_b:>18.4f}")
    print(f"  {'C: ops-rotate-content + Hopfield':<48s}  {train_acc_c:>8.4f}  {test_acc_c:>18.4f}")
    print(f"  {'D: NO ops + Hopfield':<48s}  {train_acc_d:>8.4f}  {test_acc_d:>18.4f}")
    print(f"  {'E: Hopf-split (W=op, RGB=content) + ridge':<48s}  {train_acc_e:>8.4f}  {test_acc_e:>18.4f}")
    print(f"  {'F: Hopf-split + Hopfield':<48s}  {train_acc_f:>8.4f}  {test_acc_f:>18.4f}")
    print(f"\n  Δ test accuracy on held-out (s,p):")
    print(f"    B vs A:  {(test_acc_b - test_acc_a)*100:+.2f} pp  (ops mixed with content, ridge)")
    print(f"    C vs D:  {(test_acc_c - test_acc_d)*100:+.2f} pp  (ops mixed with content, Hopfield)")
    print(f"    E vs A:  {(test_acc_e - test_acc_a)*100:+.2f} pp  (Hopf channel split, ridge)")
    print(f"    F vs D:  {(test_acc_f - test_acc_d)*100:+.2f} pp  (Hopf channel split, Hopfield)")
    print("="*75)

    # ── Falsifiability check on the operator+bilinear hypothesis ──
    if test_acc_c > test_acc_d + 0.10:
        print("  ✓ BILINEAR HYPOTHESIS VALIDATED: with Hopfield retrieval,")
        print("    bootstrap operators (rotations) DO transfer compositionally.")
    elif test_acc_c > test_acc_d + 0.02:
        print("  Mild bilinear win (2-10pp). Suggestive.")
    elif abs(test_acc_c - test_acc_d) <= 0.02:
        print("  Bilinear retrieval gives no advantage to operators either.")
    else:
        print("  Operators+Hopfield WORSE than no-ops+Hopfield. Decisively dead.")


if __name__ == "__main__":
    main()

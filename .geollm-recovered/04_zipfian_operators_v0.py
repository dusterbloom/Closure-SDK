#!/usr/bin/env python3
"""
GeoLLM v0 — Zipfian operators + spectral content states
========================================================
Hypothesis (Faltz): the high-freq Zipfian head is the ALGEBRA of language;
the long tail is what the algebra acts on. Operators are built-in, not
learned. Run a sentence as a program on S^3.

Departures from script #2 (which lost to bigram by ~7x):
  1. ASYMMETRIC context (left only) — preserves word order in the gram
  2. RANKED spectrum kept (one S^3 embedding from top-4 eigenvectors)
  3. HAND-ASSIGNED operators for ~30 function words (negation, copula,
     determiners, conjunctions, prepositions, pronouns)
  4. Composition runs LEFT->RIGHT as a program: ops apply to running state,
     content is multiplied in via Hamilton
  5. Predictor: cos(q, E_w) over a UNIFIED embedding where each operator's
     "embedding" is op(identity) — so the predictor is the same for content
     and function words

Falsifiability: must beat 2-gram top-1 (12.5%) on Shakespeare+P&P held-out.
"""
import re, math, random, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from geollm_core import I_q, hamilton, normalize, axis_angle, OPERATORS, load_corpus

random.seed(42)
np.random.seed(42)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Quaternion algebra (S^3)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

I = I_q  # keep local alias for backward compat with this script's code

# 24-cell vertex set (unit quaternions of binary tetrahedral group):
# 8 from {+/-1, +/-i, +/-j, +/-k}, 16 from (+/-1 +/- i +/- j +/- k)/2
def cell24():
    V = []
    for a in [-1, 1]:
        for sign_pos in range(4):
            v = np.zeros(4); v[sign_pos] = a; V.append(v)
    for sa in [-1, 1]:
        for sb in [-1, 1]:
            for sc in [-1, 1]:
                for sd in [-1, 1]:
                    V.append(np.array([sa, sb, sc, sd]) / 2.0)
    return np.array(V)  # (24, 4)

CELL24 = cell24()

def tokenize_split(corpus, train_frac=0.9):
    toks = re.findall(r"[a-z']+", corpus)
    n_train = int(len(toks) * train_frac)
    return toks[:n_train], toks[n_train:]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Asymmetric (left-only) co-occurrence on CONTENT words
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def split_vocab(train_toks, max_vocab=2500, min_count=10):
    """Build vocab; tag each as operator or content."""
    counts = Counter(train_toks)
    vocab = [w for w, c in counts.most_common(max_vocab) if c >= min_count]
    op_set = set(OPERATORS.keys())
    content = [w for w in vocab if w not in op_set]
    operators_in_vocab = [w for w in vocab if w in op_set]
    return vocab, content, operators_in_vocab, counts

def asymmetric_cooc(train_toks, content_set, win=4):
    """C[a,b] += 1 when content word b appears AFTER content word a within
    `win` content-word positions, skipping operator words."""
    cw = [w for w in train_toks if w in content_set]
    n_c = len(cw)
    cid = {w: i for i, w in enumerate(sorted(content_set))}
    V = len(cid)
    C = np.zeros((V, V), dtype=np.float64)
    for i in range(n_c):
        a = cid[cw[i]]
        for j in range(i+1, min(i+1+win, n_c)):
            b = cid[cw[j]]
            C[a, b] += 1.0  # asymmetric: a -> b only
    return cid, C

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Spectral S^3 embedding from asymmetric counts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fisher_embed_s3(C, alpha=1e-10):
    """Asymmetric C -> Fisher rep -> SVD -> top-4 left singular vectors,
    normalize to S^3."""
    P = C + alpha
    P /= P.sum(axis=1, keepdims=True)
    X = np.sqrt(P)              # Fisher chart, shape (V, V)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Y = U[:, :4] * S[:4][None, :]
    nrm = np.linalg.norm(Y, axis=1, keepdims=True)
    nrm[nrm == 0] = 1
    return Y / nrm              # (V, 4) on S^3

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Compose-as-program: walk left context, apply ops, multiply content
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compose_program(context_toks, wid, E, ops):
    """
    Run the left-context as a program:
      q = identity
      for tok in context:
        if tok in ops:        q = ops[tok] (X) q   (operator left-multiplies)
        elif tok in wid:      q = q (X) E[wid[tok]] (content right-multiplies)
        else:                 skip (OOV)
        q = normalize(q)
    """
    q = I.copy()
    for t in context_toks:
        if t in ops:
            q = hamilton(ops[t], q)
        elif t in wid:
            q = hamilton(q, E[wid[t]])
        q = normalize(q)
    return q

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Build a UNIFIED embedding table (content + operators) for prediction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_unified_embedding(content_words, content_emb, ops_in_vocab, ops):
    """Each operator gets embedding op(I) so the predictor is uniform."""
    words = list(content_words) + list(ops_in_vocab)
    wid = {w: i for i, w in enumerate(words)}
    E = np.zeros((len(words), 4), dtype=np.float64)
    for w in content_words:
        E[wid[w]] = content_emb[w]
    for w in ops_in_vocab:
        # operator embedding = op acting on identity
        E[wid[w]] = normalize(ops[w])  # op (X) I = op
    return wid, E, words

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Next-token evaluation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def eval_program(test_toks, wid, E, ops, context_len=8, n_eval=2000):
    valid = [t for t in test_toks if t in wid]
    if len(valid) <= context_len:
        return None
    positions = list(range(context_len, len(valid)))
    if len(positions) > n_eval:
        positions = random.sample(positions, n_eval)

    top1 = top5 = top10 = 0
    mrr = 0.0
    n = 0
    for pos in positions:
        ctx = valid[pos-context_len:pos]
        target = valid[pos]
        q = compose_program(ctx, wid, E, ops)
        sims = E @ q
        ranked = np.argsort(-sims)
        target_id = wid[target]
        if ranked[0] == target_id: top1 += 1
        if target_id in ranked[:5]: top5 += 1
        if target_id in ranked[:10]: top10 += 1
        rank_arr = np.where(ranked == target_id)[0]
        if len(rank_arr) > 0:
            mrr += 1.0 / (rank_arr[0] + 1)
        n += 1
    return {'top1': top1/n, 'top5': top5/n, 'top10': top10/n, 'mrr': mrr/n, 'n': n}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Bigram baseline for comparison
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def bigram_baseline(train_toks, test_toks, wid, n_eval=2000):
    train_ids = [wid[t] for t in train_toks if t in wid]
    test_ids = [wid[t] for t in test_toks if t in wid]
    bigram = defaultdict(Counter)
    for i in range(len(train_ids)-1):
        bigram[train_ids[i]][train_ids[i+1]] += 1
    unigram = Counter(train_ids)

    positions = list(range(1, len(test_ids)))
    if len(positions) > n_eval:
        positions = random.sample(positions, n_eval)

    top1 = top5 = top10 = 0; mrr = 0.0; n = 0
    for pos in positions:
        prev = test_ids[pos-1]; tgt = test_ids[pos]
        dist = bigram.get(prev, unigram)
        ranked = [w for w, _ in dist.most_common()]
        if not ranked: continue
        if ranked[0] == tgt: top1 += 1
        if tgt in ranked[:5]: top5 += 1
        if tgt in ranked[:10]: top10 += 1
        if tgt in ranked[:10]:
            mrr += 1.0 / (ranked.index(tgt) + 1)
        n += 1
    return {'top1': top1/n, 'top5': top5/n, 'top10': top10/n, 'mrr': mrr/n, 'n': n}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("="*75)
    print("  GeoLLM v0 — Zipfian operators + spectral content (4 floats/token)")
    print("="*75)

    t0 = time.time()
    print("\n[1] Loading corpus...")
    corpus = load_corpus()
    train_toks, test_toks = tokenize_split(corpus, train_frac=0.9)
    print(f"  Train {len(train_toks):,} tokens · Test {len(test_toks):,} tokens")

    print("\n[2] Building vocab + Zipf inspection...")
    vocab, content, ops_in_vocab, counts = split_vocab(train_toks)
    print(f"  Vocab {len(vocab)} · operators {len(ops_in_vocab)} · content {len(content)}")
    print(f"  Top 10 by freq: {[w for w,_ in counts.most_common(10)]}")
    op_coverage = sum(counts[w] for w in ops_in_vocab) / sum(counts.values())
    print(f"  Operator-word coverage of corpus: {op_coverage:.2%}")

    print("\n[3] Asymmetric co-occurrence on content words (left->right, win=4)...")
    cid, C = asymmetric_cooc(train_toks, set(content), win=4)
    print(f"  Content gram shape {C.shape} · nonzeros {(C>0).sum():,}")

    print("\n[4] Spectral S^3 embedding from SVD of Fisher rep...")
    Y = fisher_embed_s3(C)
    content_emb = {w: Y[cid[w]] for w in cid}
    print(f"  Content embedding shape {Y.shape}")

    print("\n[5] Unified embedding (content + operators)...")
    wid, E, words = build_unified_embedding(content, content_emb, ops_in_vocab, OPERATORS)
    print(f"  Total embedded words {len(words)}")

    print("\n[6] Evaluation — bigram baseline first...")
    r_bg = bigram_baseline(train_toks, test_toks, wid)
    print(f"  bigram                top1={r_bg['top1']:.4f}  top5={r_bg['top5']:.4f}  "
          f"top10={r_bg['top10']:.4f}  mrr={r_bg['mrr']:.4f}  n={r_bg['n']}")

    print("\n[7] Geometric program (Zipfian ops, varying context)...")
    print(f"  {'context':>7s}  {'top1':>7s}  {'top5':>7s}  {'top10':>7s}  {'mrr':>7s}")
    for cl in [2, 4, 6, 8, 12, 20]:
        r = eval_program(test_toks, wid, E, OPERATORS, context_len=cl, n_eval=1500)
        if r is None: continue
        print(f"  {cl:>7d}  {r['top1']:>7.4f}  {r['top5']:>7.4f}  "
              f"{r['top10']:>7.4f}  {r['mrr']:>7.4f}")

    print("\n[8] Ablation — same code with ALL operators replaced by identity...")
    OPS_id = {w: I for w in OPERATORS}
    r_id = eval_program(test_toks, wid, E, OPS_id, context_len=8, n_eval=1500)
    print(f"  identity ops (ctx=8)  top1={r_id['top1']:.4f}  top5={r_id['top5']:.4f}  "
          f"top10={r_id['top10']:.4f}  mrr={r_id['mrr']:.4f}")

    print(f"\nDone in {time.time()-t0:.1f}s")
    print("="*75)
    print(f"  Bigram top1: {r_bg['top1']:.4f}  · target to beat")
    print(f"  Identity-ops top1 (ctx=8): {r_id['top1']:.4f}")
    print("="*75)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
GeoLLM World Model Validation
=============================
The real test: does the Geometric Computer's composition do LANGUAGE MODELING?

Tests:
  1. (S³)^N vs k-dim sphere — same parameter count comparison
  2. Hamilton product vs naive averaging vs random — algebra earning its keep?
  3. Next-token prediction — actual language modeling on held-out data
  4. Polysemy via context rotation — does q|context = R(context) ⊗ q?
"""
import json, math, re, time, random
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, "/mnt/user-data/outputs/geollm")

random.seed(42)
np.random.seed(42)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Corpus + train/test split
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_corpus():
    parts = []
    here = Path(__file__).parent / 'corpora'
    for p in [here / 'shakespeare.txt', here / 'pride_prejudice.txt']:
        if Path(p).exists():
            parts.append(Path(p).read_text(errors='ignore').lower())
    return ' '.join(parts)


def tokenize_split(corpus, train_frac=0.9):
    toks = re.findall(r'[a-z]+', corpus)
    n_train = int(len(toks) * train_frac)
    return toks[:n_train], toks[n_train:]


def build_vocab_and_counts(train_toks, max_vocab=2000, min_count=20, window=5):
    counts = Counter(train_toks)
    vocab = [t for t, c in counts.most_common(max_vocab) if c >= min_count]
    tid = {t: i for i, t in enumerate(vocab)}
    V = len(vocab)
    
    C = np.zeros((V, V), dtype=np.float64)
    valid_ids = np.array([tid.get(t, -1) for t in train_toks])
    
    for i in range(len(train_toks)):
        a = valid_ids[i]
        if a < 0: continue
        lo, hi = max(0, i - window), min(len(train_toks), i + window + 1)
        for j in range(lo, hi):
            if j == i: continue
            b = valid_ids[j]
            if b >= 0:
                C[a, b] += 1.0
    return vocab, tid, C


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Spectral embedding
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fisher_gram(C, alpha=1e-10):
    P = C + alpha
    P /= P.sum(axis=1, keepdims=True)
    X = np.sqrt(P)
    G = np.clip(X @ X.T, 0.0, 1.0)
    return X, G


def spectral_embed(G, dim, normalize=True):
    K = (G + G.T) / 2
    vals, vecs = np.linalg.eigh(K)
    idx = np.argsort(vals)[::-1]
    vals = np.maximum(vals[idx], 0)
    vecs = vecs[:, idx]
    Y = vecs[:, :dim] * np.sqrt(vals[:dim])[None, :]
    if normalize:
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1
        Y = Y / nrm
    return Y, vals


def spectral_quat_stack(G, n_quats):
    """
    (S³)^n_quats: stack n_quats independent S³ embeddings.
    Each uses a different 4-dim slice of the spectrum, then normalized.
    Total params: n_quats * 4 floats per token.
    """
    K = (G + G.T) / 2
    vals, vecs = np.linalg.eigh(K)
    idx = np.argsort(vals)[::-1]
    vals = np.maximum(vals[idx], 0)
    vecs = vecs[:, idx]
    
    embeddings = []
    for i in range(n_quats):
        slice_start = i * 4
        Y = vecs[:, slice_start:slice_start+4] * np.sqrt(vals[slice_start:slice_start+4])[None, :]
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1
        Y = Y / nrm
        embeddings.append(Y)
    return embeddings  # list of (V, 4)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Composition operators
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def hamilton(a, b):
    w1,x1,y1,z1 = a; w2,x2,y2,z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2])


def hamilton_batch(A, B):
    """Hamilton product applied per quaternion in (V, 4n) tensors."""
    n = A.shape[-1] // 4
    out = np.zeros_like(A)
    for k in range(n):
        a = A[..., k*4:(k+1)*4]
        b = B[..., k*4:(k+1)*4]
        w1,x1,y1,z1 = a[...,0], a[...,1], a[...,2], a[...,3]
        w2,x2,y2,z2 = b[...,0], b[...,1], b[...,2], b[...,3]
        out[..., k*4]   = w1*w2 - x1*x2 - y1*y2 - z1*z2
        out[..., k*4+1] = w1*x2 + x1*w2 + y1*z2 - z1*y2
        out[..., k*4+2] = w1*y2 - x1*z2 + y1*w2 + z1*x2
        out[..., k*4+3] = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return out


def normalize(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    n = np.where(n < 1e-10, 1.0, n)
    return v / n


def normalize_quat_stack(v):
    """Normalize each 4-block separately to keep each quaternion on S³."""
    n = v.shape[-1] // 4
    out = np.zeros_like(v)
    for k in range(n):
        block = v[..., k*4:(k+1)*4]
        out[..., k*4:(k+1)*4] = normalize(block)
    return out


def compose_hamilton(toks, E_stack, tid):
    """E_stack is list of (V, 4) arrays. Composition is per-quaternion Hamilton."""
    valid = [t for t in toks if t in tid]
    if not valid: return None
    
    # Concatenate quaternions for each token: (V, 4*n_quats)
    n_quats = len(E_stack)
    embs_for_seq = np.zeros((len(valid), 4 * n_quats))
    for i, t in enumerate(valid):
        for k, E in enumerate(E_stack):
            embs_for_seq[i, k*4:(k+1)*4] = E[tid[t]]
    
    result = embs_for_seq[0:1]  # (1, 4n)
    for i in range(1, len(valid)):
        result = hamilton_batch(result, embs_for_seq[i:i+1])
        result = normalize_quat_stack(result)
    return result.squeeze(0)


def compose_avg(toks, E_full, tid):
    """Naive: average and normalize."""
    valid_ids = [tid[t] for t in toks if t in tid]
    if not valid_ids: return None
    avg = E_full[valid_ids].mean(axis=0)
    return normalize(avg)


def compose_mult(toks, E_full, tid):
    """Element-wise product + normalize."""
    valid_ids = [tid[t] for t in toks if t in tid]
    if not valid_ids: return None
    prod = E_full[valid_ids[0]].copy()
    for vid in valid_ids[1:]:
        prod = prod * E_full[vid]
    return normalize(prod)


def compose_random(toks, E_full, tid, rng=None):
    """Random rotation per step (control)."""
    if rng is None: rng = np.random.RandomState(123)
    d = E_full.shape[1]
    valid_ids = [tid[t] for t in toks if t in tid]
    if not valid_ids: return None
    state = E_full[valid_ids[0]].copy()
    for vid in valid_ids[1:]:
        # Random orthogonal rotation
        A = rng.randn(d, d)
        Q, _ = np.linalg.qr(A)
        state = Q @ state
        state = normalize(state)
    return state


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Next-token prediction (THE REAL TEST)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def evaluate_next_token_prediction(test_toks, vocab, tid, E_target, compose_fn,
                                    E_full, n_quats, context_len=5, n_eval=2000,
                                    top_k=10):
    """
    For each position in test set:
      1. Take previous `context_len` tokens
      2. Compose them via compose_fn
      3. Find nearest neighbors in E_target (target embedding for prediction)
      4. Check if true next token is in top-k
    
    E_target: the embedding to compare against (for nearest-neighbor lookup)
    compose_fn: function (toks, E_full_or_stack, tid) → composed state
    E_full: the embedding stack to use during composition
    n_quats: how many quaternions stacked (1 = simple S³, 3 = (S³)^3)
    """
    valid_test = [t for t in test_toks if t in tid]
    if len(valid_test) <= context_len:
        return {'top1': 0.0, 'top5': 0.0, 'top10': 0.0, 'mrr': 0.0, 'n': 0}
    
    positions = list(range(context_len, len(valid_test)))
    if len(positions) > n_eval:
        positions = random.sample(positions, n_eval)
    
    top1_hits = 0
    top5_hits = 0
    top10_hits = 0
    mrr_sum = 0.0
    n = 0
    
    for pos in positions:
        context = valid_test[pos - context_len:pos]
        target = valid_test[pos]
        
        composed = compose_fn(context, E_full, tid)
        if composed is None: continue
        
        # Cosine similarity (for sphere embeddings, this == -geodesic dist monotonically)
        # E_target is (V, dim) — composed is (dim,)
        if composed.ndim == 1 and len(composed) == E_target.shape[1]:
            sims = E_target @ composed
        elif composed.ndim == 1 and len(composed) == 4 * n_quats:
            # composed is the quaternion stack — average similarities
            sims = np.zeros(E_target.shape[0])
            for k in range(n_quats):
                comp_k = composed[k*4:(k+1)*4]
                # E_target should also be the stacked rep
                if E_target.shape[1] == 4 * n_quats:
                    targ_k = E_target[:, k*4:(k+1)*4]
                    sims += targ_k @ comp_k
                else:
                    sims += E_target @ comp_k  # mismatched, skip
                    break
            sims /= n_quats
        else:
            continue
        
        # Top predictions
        top_indices = np.argsort(-sims)[:top_k]
        target_id = tid[target]
        
        if top_indices[0] == target_id:
            top1_hits += 1
        if target_id in top_indices[:5]:
            top5_hits += 1
        if target_id in top_indices[:10]:
            top10_hits += 1
        
        # MRR
        rank_arr = np.where(top_indices == target_id)[0]
        if len(rank_arr) > 0:
            mrr_sum += 1.0 / (rank_arr[0] + 1)
        
        n += 1
    
    if n == 0:
        return {'top1': 0.0, 'top5': 0.0, 'top10': 0.0, 'mrr': 0.0, 'n': 0}
    return {
        'top1': top1_hits / n,
        'top5': top5_hits / n,
        'top10': top10_hits / n,
        'mrr': mrr_sum / n,
        'n': n,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Baseline: ngram model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def ngram_baseline(train_toks, test_toks, vocab, tid, n=3, n_eval=2000, top_k=10):
    """N-gram model with backoff for comparison."""
    train_ids = [tid[t] for t in train_toks if t in tid]
    test_ids = [tid[t] for t in test_toks if t in tid]
    
    if len(test_ids) <= n:
        return {'top1': 0.0, 'top5': 0.0, 'top10': 0.0, 'mrr': 0.0, 'n': 0}
    
    # Build n-gram → next token distribution
    ngram_counts = defaultdict(Counter)
    for i in range(len(train_ids) - n):
        context = tuple(train_ids[i:i+n-1])
        nxt = train_ids[i+n-1]
        ngram_counts[context][nxt] += 1
    
    # Unigram fallback
    unigram = Counter(train_ids)
    
    # Eval
    positions = list(range(n-1, len(test_ids)))
    if len(positions) > n_eval:
        positions = random.sample(positions, n_eval)
    
    top1 = top5 = top10 = 0
    mrr_sum = 0.0
    n_eval_done = 0
    
    for pos in positions:
        context = tuple(test_ids[pos-(n-1):pos])
        target = test_ids[pos]
        
        if context in ngram_counts:
            dist = ngram_counts[context]
        else:
            dist = unigram
        
        ranked = [w for w, _ in dist.most_common()]
        
        if not ranked: continue
        if ranked[0] == target: top1 += 1
        if target in ranked[:5]: top5 += 1
        if target in ranked[:10]: top10 += 1
        
        if target in ranked[:top_k]:
            r = ranked.index(target)
            mrr_sum += 1.0 / (r + 1)
        
        n_eval_done += 1
    
    if n_eval_done == 0:
        return {'top1': 0.0, 'top5': 0.0, 'top10': 0.0, 'mrr': 0.0, 'n': 0}
    return {
        'top1': top1 / n_eval_done,
        'top5': top5 / n_eval_done,
        'top10': top10 / n_eval_done,
        'mrr': mrr_sum / n_eval_done,
        'n': n_eval_done,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run all experiments
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("=" * 75)
    print("  WORLD MODEL VALIDATION — Does Hamilton composition predict text?")
    print("=" * 75)
    
    print("\n[1] Loading corpus + train/test split...")
    corpus = load_corpus()
    train_toks, test_toks = tokenize_split(corpus, train_frac=0.9)
    print(f"  Train: {len(train_toks):,} tokens")
    print(f"  Test:  {len(test_toks):,} tokens")
    
    print("\n[2] Building vocabulary from TRAIN ONLY...")
    vocab, tid, C = build_vocab_and_counts(train_toks, max_vocab=2000, min_count=20, window=5)
    print(f"  Vocab: {len(vocab)}")
    
    print("\n[3] Computing Fisher gram from TRAIN counts...")
    X, G = fisher_gram(C)
    
    # ━━━ Build embeddings: spheres of various dims, and (S³)^N stacks ━━━
    print("\n[4] Building embeddings...")
    
    spheres = {}
    for d in [4, 8, 12, 16, 32, 64]:
        Y, _ = spectral_embed(G, d)
        spheres[d] = Y
        print(f"  S^{d-1}: shape={Y.shape}")
    
    # (S³)^N stacks
    stacks = {}
    for n in [1, 2, 3, 4, 8, 16]:
        if n * 4 <= len(vocab):
            stacks[n] = spectral_quat_stack(G, n)
            # Concatenated form for cosine similarity (still per-quat normalized)
            print(f"  (S³)^{n}: {n} quaternions, total {n*4} floats")
    
    # ━━━ Test 1: Distance preservation comparison ━━━
    print("\n" + "=" * 75)
    print("  TEST 1: Distance preservation — same param count")
    print("=" * 75)
    print(f"  {'config':<15s} {'params':>7s} {'corr':>8s}")
    
    iu = np.triu_indices(len(vocab), k=1)
    D_true = np.arccos(np.clip(G[iu], -1+1e-7, 1-1e-7))
    
    for d in [4, 8, 12, 16, 32, 64]:
        Y = spheres[d]
        DY = np.arccos(np.clip(Y @ Y.T, -1+1e-7, 1-1e-7))[iu]
        c = np.corrcoef(DY, D_true)[0, 1]
        print(f"  Sphere d={d:>3d}    {d:>7d}  {c:>8.4f}")
    
    print()
    for n, stack in stacks.items():
        # Concatenate stacks for distance measurement
        Y_concat = np.concatenate(stack, axis=1)
        Y_concat = normalize(Y_concat)  # Normalize the concatenation
        DY = np.arccos(np.clip(Y_concat @ Y_concat.T, -1+1e-7, 1-1e-7))[iu]
        c = np.corrcoef(DY, D_true)[0, 1]
        print(f"  (S³)^{n}        {n*4:>7d}  {c:>8.4f}")
    
    # ━━━ Test 2: Next-token prediction ━━━
    print("\n" + "=" * 75)
    print("  TEST 2: Next-token prediction on HELD-OUT data (the real test)")
    print("=" * 75)
    
    # Baseline: bigram and trigram
    print("\n  N-gram baselines:")
    print(f"  {'model':<25s} {'top-1':>7s} {'top-5':>7s} {'top-10':>7s} {'mrr':>7s} {'n':>5s}")
    for ng in [2, 3, 4]:
        r = ngram_baseline(train_toks, test_toks, vocab, tid, n=ng, n_eval=2000)
        print(f"  {ng}-gram                     {r['top1']:>7.4f} {r['top5']:>7.4f} "
              f"{r['top10']:>7.4f} {r['mrr']:>7.4f} {r['n']:>5d}")
    
    # Geometric Computer compositions
    print("\n  Geometric Computer compositions:")
    print(f"  {'config':<25s} {'top-1':>7s} {'top-5':>7s} {'top-10':>7s} {'mrr':>7s}")
    
    # (S³)^N with Hamilton product
    for n in [1, 2, 3, 4, 8, 16]:
        if n not in stacks: continue
        stack = stacks[n]
        E_concat = np.concatenate(stack, axis=1)  # (V, 4n)
        # E_target should also be the per-quaternion-normalized stack
        # For prediction lookup, normalize each quaternion separately for the target too
        E_target = E_concat.copy()
        for k in range(n):
            E_target[:, k*4:(k+1)*4] = normalize(E_target[:, k*4:(k+1)*4])
        
        r = evaluate_next_token_prediction(
            test_toks, vocab, tid, E_target,
            compose_hamilton, stack, n_quats=n, context_len=4, n_eval=2000)
        print(f"  (S³)^{n} Hamilton          {r['top1']:>7.4f} {r['top5']:>7.4f} "
              f"{r['top10']:>7.4f} {r['mrr']:>7.4f}")
    
    # k-dim sphere with Hamilton (only meaningful for d=4 since Hamilton needs quaternions)
    # vs simple averaging on k-dim spheres
    print("\n  Naive averaging on spheres (control):")
    for d in [4, 8, 12, 16, 32, 64]:
        Y = spheres[d]
        r = evaluate_next_token_prediction(
            test_toks, vocab, tid, Y,
            compose_avg, Y, n_quats=1, context_len=4, n_eval=2000)
        print(f"  Sphere d={d:>3d} avg          {r['top1']:>7.4f} {r['top5']:>7.4f} "
              f"{r['top10']:>7.4f} {r['mrr']:>7.4f}")
    
    print("\n  Element-wise multiply (control):")
    for d in [4, 8, 12, 16, 32, 64]:
        Y = spheres[d]
        r = evaluate_next_token_prediction(
            test_toks, vocab, tid, Y,
            compose_mult, Y, n_quats=1, context_len=4, n_eval=2000)
        print(f"  Sphere d={d:>3d} mult         {r['top1']:>7.4f} {r['top5']:>7.4f} "
              f"{r['top10']:>7.4f} {r['mrr']:>7.4f}")
    
    # Hamilton on dim-4 sphere only
    print("\n  Hamilton product (dim must be multiple of 4):")
    for d, label in [(4, '(S³)^1'), (8, '(S³)^2'), (12, '(S³)^3'),
                      (16, '(S³)^4'), (32, '(S³)^8'), (64, '(S³)^16')]:
        n = d // 4
        if n not in stacks: continue
        stack = stacks[n]
        E_concat = np.concatenate(stack, axis=1)
        E_target = E_concat.copy()
        for k in range(n):
            E_target[:, k*4:(k+1)*4] = normalize(E_target[:, k*4:(k+1)*4])
        r = evaluate_next_token_prediction(
            test_toks, vocab, tid, E_target,
            compose_hamilton, stack, n_quats=n, context_len=4, n_eval=2000)
        print(f"  d={d:>3d} {label:<10s}        {r['top1']:>7.4f} {r['top5']:>7.4f} "
              f"{r['top10']:>7.4f} {r['mrr']:>7.4f}")
    
    # ━━━ Test 3: Context length sensitivity ━━━
    print("\n" + "=" * 75)
    print("  TEST 3: Context length sensitivity (does longer context help?)")
    print("=" * 75)
    
    n = 4  # (S³)^4 = 16 floats
    stack = stacks[n]
    E_concat = np.concatenate(stack, axis=1)
    E_target = E_concat.copy()
    for k in range(n):
        E_target[:, k*4:(k+1)*4] = normalize(E_target[:, k*4:(k+1)*4])
    
    print(f"  Using (S³)^4 (16 floats):")
    print(f"  {'context':>7s} {'top-1':>7s} {'top-5':>7s} {'top-10':>7s} {'mrr':>7s}")
    for cl in [1, 2, 3, 5, 8, 12, 20]:
        r = evaluate_next_token_prediction(
            test_toks, vocab, tid, E_target,
            compose_hamilton, stack, n_quats=n, context_len=cl, n_eval=1000)
        print(f"  {cl:>7d} {r['top1']:>7.4f} {r['top5']:>7.4f} "
              f"{r['top10']:>7.4f} {r['mrr']:>7.4f}")
    
    # ━━━ Test 4: Composition operator comparison at SAME dim ━━━
    print("\n" + "=" * 75)
    print("  TEST 4: Composition operator at fixed dim=16 (Hamilton vs naïve)")
    print("=" * 75)
    
    # Use 16-dim sphere (single embedding) and (S³)^4 — same param count
    Y16 = spheres[16]
    print(f"  {'method':<25s} {'top-1':>7s} {'top-5':>7s} {'top-10':>7s} {'mrr':>7s}")
    
    r = evaluate_next_token_prediction(
        test_toks, vocab, tid, Y16, compose_avg, Y16, n_quats=1,
        context_len=4, n_eval=2000)
    print(f"  Average on S^15          {r['top1']:>7.4f} {r['top5']:>7.4f} "
          f"{r['top10']:>7.4f} {r['mrr']:>7.4f}")
    
    r = evaluate_next_token_prediction(
        test_toks, vocab, tid, Y16, compose_mult, Y16, n_quats=1,
        context_len=4, n_eval=2000)
    print(f"  Mult on S^15             {r['top1']:>7.4f} {r['top5']:>7.4f} "
          f"{r['top10']:>7.4f} {r['mrr']:>7.4f}")
    
    # Hamilton on (S³)^4 — same 16 floats
    n = 4
    stack = stacks[n]
    E_concat = np.concatenate(stack, axis=1)
    E_target = E_concat.copy()
    for k in range(n):
        E_target[:, k*4:(k+1)*4] = normalize(E_target[:, k*4:(k+1)*4])
    r = evaluate_next_token_prediction(
        test_toks, vocab, tid, E_target,
        compose_hamilton, stack, n_quats=n, context_len=4, n_eval=2000)
    print(f"  Hamilton on (S³)^4       {r['top1']:>7.4f} {r['top5']:>7.4f} "
          f"{r['top10']:>7.4f} {r['mrr']:>7.4f}")
    
    # Random rotation control
    r = evaluate_next_token_prediction(
        test_toks, vocab, tid, Y16,
        lambda toks, E, t: compose_random(toks, E, t),
        Y16, n_quats=1, context_len=4, n_eval=2000)
    print(f"  Random rotations         {r['top1']:>7.4f} {r['top5']:>7.4f} "
          f"{r['top10']:>7.4f} {r['mrr']:>7.4f}")
    
    print("\n" + "=" * 75)
    print("  CONCLUSION")
    print("=" * 75)
    print("  If Hamilton on (S³)^4 ≈ Average on S^15:")
    print("    → Algebraic structure doesn't help; spectrum is what matters")
    print("  If Hamilton beats Average:")
    print("    → Non-commutativity captures word-order signal")
    print("  If both beat n-gram:")
    print("    → Geometric composition CAN do language modeling")
    print("  If neither beats n-gram:")
    print("    → 4-16 floats per token is too compressed for real LM")
    print("=" * 75)


if __name__ == '__main__':
    main()

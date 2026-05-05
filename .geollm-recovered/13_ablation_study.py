#!/usr/bin/env python3
"""Carrier × Multi-rate × Threshold ablation study.

Tests three levers on the recursive predictive state model:
1. Carrier: random vs SVD embeddings
2. Multi-rate: 1/2/3/4 timescale state vectors
3. Closure threshold: binary vs surprise-weighted
"""
import numpy as np, time, sys, os

VOCAB = 64
K = 5

# --- Load corpus ---
for path in ['/tmp/shakespeare_input.txt',
             os.path.expanduser('~/Dev/Closure-SDK/.geollm-recovered/shakespeare_input.txt')]:
    if os.path.exists(path):
        raw = open(path, 'rb').read()
        break
else:
    sys.exit("Corpus not found")

tokens = np.frombuffer(raw, dtype=np.uint8) % VOCAB
N = len(tokens)
TRAIN = 200000
TEST = 3000
train_tok = tokens[:TRAIN]
test_tok = tokens[TRAIN:TRAIN+TEST]
print(f"Corpus: {N} tokens | Train: {TRAIN} | Test: {TEST}", flush=True)

# --- Bigram matrix + SVD ---
P = np.zeros((VOCAB, VOCAB), dtype=np.float64)
for i in range(TRAIN - 1):
    P[train_tok[i], train_tok[i+1]] += 1
rs = P.sum(axis=1, keepdims=True)
rs[rs == 0] = 1
P /= rs
U, S, Vt = np.linalg.svd(P, full_matrices=False)

# --- Carriers ---
def svd_carrier(rank):
    emb = (U[:, :rank] * S[:rank]).astype(np.float32)
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-10)
    return emb

def random_carrier(rank, seed=42):
    rng = np.random.RandomState(seed)
    emb = rng.randn(VOCAB, rank).astype(np.float32)
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-10)
    return emb

# --- Engine ---
def run(emb, alphas, train_n, gcap, k=K, thresh='binary', thresh_val=0.3):
    d = emb.shape[1]
    nr = len(alphas)
    kdim = d * nr
    tsub = train_tok[:train_n]

    t0 = time.time()
    # Pre-build all training keys
    keys = np.empty((train_n, kdim), dtype=np.float32)
    states = [emb[tsub[0]].copy() for _ in alphas]
    keys[0] = np.concatenate(states)
    n0 = np.linalg.norm(keys[0])
    if n0 > 1e-10:
        keys[0] /= n0

    for i in range(1, train_n):
        e = emb[tsub[i]]
        parts = []
        for j, a in enumerate(alphas):
            states[j] = a * e + (1 - a) * states[j]
            nm = np.linalg.norm(states[j])
            if nm > 1e-10:
                states[j] /= nm
            parts.append(states[j])
        keys[i] = np.concatenate(parts)
        nm = np.linalg.norm(keys[i])
        if nm > 1e-10:
            keys[i] /= nm

    # Build genome
    gkeys = np.empty((gcap, kdim), dtype=np.float32)
    gobs = np.empty(gcap, dtype=np.int32)
    gs = 0

    for i in range(train_n - 1):
        if gs == 0:
            gkeys[0] = keys[i]
            gobs[0] = tsub[i+1]
            gs = 1
            continue

        sims = gkeys[:gs] @ keys[i]
        kk = min(k, gs)
        if gs <= kk:
            idx = np.arange(gs)
        else:
            idx = np.argpartition(sims, -kk)[-kk:]

        votes = np.zeros(VOCAB, dtype=np.float32)
        for j in idx:
            if sims[j] > 0:
                votes[gobs[j]] += sims[j]
        pred = np.argmax(votes)
        obs = tsub[i+1]

        store = False
        if thresh == 'binary':
            store = (pred != obs)
        elif thresh == 'surprise':
            total = votes.sum()
            p_obs = votes[obs] / total if total > 0 else 0.0
            store = (p_obs < thresh_val)

        if store and gs < gcap:
            gkeys[gs] = keys[i]
            gobs[gs] = obs
            gs += 1

    train_time = time.time() - t0

    # Evaluate on test set
    t0 = time.time()
    test_keys = np.empty((TEST, kdim), dtype=np.float32)
    states = [emb[test_tok[0]].copy() for _ in alphas]
    test_keys[0] = np.concatenate(states)
    nm = np.linalg.norm(test_keys[0])
    if nm > 1e-10:
        test_keys[0] /= nm

    for i in range(1, TEST):
        e = emb[test_tok[i]]
        parts = []
        for j, a in enumerate(alphas):
            states[j] = a * e + (1 - a) * states[j]
            nm = np.linalg.norm(states[j])
            if nm > 1e-10:
                states[j] /= nm
            parts.append(states[j])
        test_keys[i] = np.concatenate(parts)
        nm = np.linalg.norm(test_keys[i])
        if nm > 1e-10:
            test_keys[i] /= nm

    correct = 0
    CHUNK = 500
    for start in range(0, TEST - 1, CHUNK):
        end = min(start + CHUNK, TEST - 1)
        batch = test_keys[start:end]
        all_sims = batch @ gkeys[:gs].T
        for ri in range(end - start):
            row = all_sims[ri]
            kk = min(k, gs)
            if gs <= kk:
                idx = np.arange(gs)
            else:
                idx = np.argpartition(row, -kk)[-kk:]
            votes = np.zeros(VOCAB, dtype=np.float32)
            for j in idx:
                if row[j] > 0:
                    votes[gobs[j]] += row[j]
            if np.argmax(votes) == test_tok[start + ri + 1]:
                correct += 1

    acc = correct / (TEST - 1) * 100
    eval_time = time.time() - t0
    return acc, gs, train_time, eval_time


# ============================================================
print("\n" + "="*60, flush=True)
print("EXPERIMENT 1: CARRIER ABLATION", flush=True)
print("(single rate α=0.9, train=200K, genome=50K, k=5)", flush=True)
print("="*60, flush=True)

for name, emb in [
    ('Random R^24',  random_carrier(24)),
    ('SVD rank-8',   svd_carrier(8)),
    ('SVD rank-24',  svd_carrier(24)),
]:
    acc, gs, tt, et = run(emb, [0.9], TRAIN, 50000)
    print(f"  {name:20s}: {acc:6.2f}%  genome={gs:6d}  train={tt:.1f}s  eval={et:.1f}s", flush=True)

# ============================================================
print("\n" + "="*60, flush=True)
print("EXPERIMENT 2: MULTI-RATE STATE", flush=True)
print("(SVD rank-24 carrier, train=200K, genome=50K, k=5)", flush=True)
print("="*60, flush=True)

emb24 = svd_carrier(24)
for name, alphas in [
    ('1-rate [0.9]',             [0.9]),
    ('2-rate [0.95, 0.5]',      [0.95, 0.5]),
    ('2-rate [0.9, 0.3]',       [0.9, 0.3]),
    ('3-rate [0.95, 0.7, 0.3]', [0.95, 0.7, 0.3]),
    ('4-rate [.97,.8,.5,.2]',   [0.97, 0.8, 0.5, 0.2]),
]:
    acc, gs, tt, et = run(emb24, alphas, TRAIN, 50000)
    print(f"  {name:30s}: {acc:6.2f}%  genome={gs:6d}  train={tt:.1f}s  eval={et:.1f}s", flush=True)

# ============================================================
print("\n" + "="*60, flush=True)
print("EXPERIMENT 3: CLOSURE THRESHOLD", flush=True)
print("(SVD rank-24, train=200K, genome=50K, k=5)", flush=True)
print("="*60, flush=True)

for rname, alphas in [('[0.9]', [0.9]), ('[.95,.7,.3]', [0.95, 0.7, 0.3])]:
    for tname, thresh, tv in [('binary', 'binary', 0), ('surprise<0.3', 'surprise', 0.3), ('surprise<0.5', 'surprise', 0.5)]:
        acc, gs, tt, et = run(emb24, alphas, TRAIN, 50000, thresh=thresh, thresh_val=tv)
        print(f"  rates={rname:15s} gate={tname:15s}: {acc:6.2f}%  genome={gs:6d}  train={tt:.1f}s  eval={et:.1f}s", flush=True)

# ============================================================
print("\n" + "="*60, flush=True)
print("EXPERIMENT 4: INTERACTIONS & SCALING", flush=True)
print("="*60, flush=True)

# Carrier × multirate interaction
acc, gs, tt, et = run(random_carrier(24), [0.95, 0.7, 0.3], TRAIN, 50000)
print(f"  Random×3-rate:           {acc:6.2f}%  genome={gs:6d}  train={tt:.1f}s  eval={et:.1f}s", flush=True)

# Best combo scaled up
acc, gs, tt, et = run(emb24, [0.95, 0.7, 0.3], TRAIN, 80000)
print(f"  SVD×3-rate, g=80K:       {acc:6.2f}%  genome={gs:6d}  train={tt:.1f}s  eval={et:.1f}s", flush=True)

acc, gs, tt, et = run(emb24, [0.95, 0.7, 0.3], TRAIN, 80000, thresh='surprise', thresh_val=0.4)
print(f"  SVD×3-rate×surp, g=80K:  {acc:6.2f}%  genome={gs:6d}  train={tt:.1f}s  eval={et:.1f}s", flush=True)

# ============================================================
print("\n" + "="*60, flush=True)
print("N-GRAM BASELINES (same test set)", flush=True)
print("="*60, flush=True)

# Bigram
bi_correct = sum(1 for i in range(TEST-1) if np.argmax(P[test_tok[i]]) == test_tok[i+1])
print(f"  Bigram:  {bi_correct/(TEST-1)*100:.2f}%", flush=True)

# Trigram
from collections import Counter
tri = Counter()
for i in range(TRAIN - 2):
    tri[(train_tok[i], train_tok[i+1], train_tok[i+2])] += 1
tri_best = {}
for (a, b, c), cnt in tri.items():
    key = (a, b)
    if key not in tri_best or cnt > tri_best[key][1]:
        tri_best[key] = (c, cnt)

tri_correct = 0
for i in range(TEST - 2):
    key = (test_tok[i], test_tok[i+1])
    if key in tri_best:
        pred = tri_best[key][0]
    else:
        pred = np.argmax(P[test_tok[i+1]])
    if pred == test_tok[i+2]:
        tri_correct += 1
print(f"  Trigram: {tri_correct/(TEST-2)*100:.2f}%", flush=True)

print("\nDone.", flush=True)

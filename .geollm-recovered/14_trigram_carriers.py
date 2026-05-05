#!/usr/bin/env python3
"""Higher-order carrier experiment.

Tests whether trigram/skip-gram structure in embeddings
breaks through the trigram ceiling.

Carriers tested:
  random    — R^24 random unit vectors (control)
  bigram    — SVD of bigram transition matrix (current baseline)
  trigram   — SVD of trigram predictive matrix M[b, a*64+c] = P(c|a,b)
  skipgram  — SVD of PPMI co-occurrence matrix (window=5)
  bi_lr     — bigram left+right singular vectors concatenated
  bi_tri    — bigram SVD (rank 12) + trigram SVD (rank 12) concatenated
"""
import numpy as np, time, os, sys
from collections import Counter

VOCAB = 64; K = 5; TRAIN = 200000; TEST = 3000

raw = open('/tmp/shakespeare_input.txt', 'rb').read()
tokens = np.frombuffer(raw, dtype=np.uint8) % VOCAB
train_tok = tokens[:TRAIN]
test_tok = tokens[TRAIN:TRAIN+TEST]
print(f"Corpus: {len(tokens)} | Train: {TRAIN} | Test: {TEST}\n", flush=True)

# === Statistical matrices ===
print("Building matrices...", flush=True)
t0 = time.time()

# Bigram
P = np.zeros((VOCAB, VOCAB), dtype=np.float64)
np.add.at(P, (train_tok[:-1], train_tok[1:]), 1)
rs = P.sum(axis=1, keepdims=True); rs[rs==0] = 1; P /= rs
U_bi, S_bi, Vt_bi = np.linalg.svd(P, full_matrices=False)

# Trigram tensor → predictive matrix
T = np.zeros((VOCAB, VOCAB, VOCAB), dtype=np.float64)
np.add.at(T, (train_tok[:-2], train_tok[1:-1], train_tok[2:]), 1)
M_tri = np.zeros((VOCAB, VOCAB*VOCAB), dtype=np.float64)
for b in range(VOCAB):
    for a in range(VOCAB):
        s = T[a, b, :].sum()
        if s > 0:
            M_tri[b, a*VOCAB:(a+1)*VOCAB] = T[a, b, :] / s
U_tri, S_tri, _ = np.linalg.svd(M_tri, full_matrices=False)

# Skip-gram PPMI (window=5)
C = np.zeros((VOCAB, VOCAB), dtype=np.float64)
for w in range(1, 6):
    np.add.at(C, (train_tok[:-w], train_tok[w:]), 1.0/w)
    np.add.at(C, (train_tok[w:], train_tok[:-w]), 1.0/w)
tc = C.sum(); rc = C.sum(1, keepdims=True); cc = C.sum(0, keepdims=True)
rc[rc==0] = 1; cc[cc==0] = 1
pmi = np.log2(np.maximum(C * tc / (rc * cc), 1e-10))
ppmi = np.maximum(pmi, 0)
U_sg, S_sg, _ = np.linalg.svd(ppmi, full_matrices=False)

print(f"Matrices built in {time.time()-t0:.1f}s\n", flush=True)

# === Carriers ===
def carrier(name, rank=24):
    if name == 'random':
        emb = np.random.RandomState(42).randn(VOCAB, rank).astype(np.float32)
    elif name == 'bigram':
        emb = (U_bi[:, :rank] * S_bi[:rank]).astype(np.float32)
    elif name == 'trigram':
        emb = (U_tri[:, :rank] * S_tri[:rank]).astype(np.float32)
    elif name == 'skipgram':
        emb = (U_sg[:, :rank] * np.sqrt(S_sg[:rank])).astype(np.float32)
    elif name == 'bi_lr':
        r = rank // 2
        emb = np.concatenate([
            (U_bi[:, :r] * S_bi[:r]).astype(np.float32),
            (Vt_bi.T[:, :r] * S_bi[:r]).astype(np.float32)
        ], axis=1)
    elif name == 'bi_tri':
        r = rank // 2
        emb = np.concatenate([
            (U_bi[:, :r] * S_bi[:r]).astype(np.float32),
            (U_tri[:, :r] * S_tri[:r]).astype(np.float32)
        ], axis=1)
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-10)
    return emb

# === Engine ===
def run(emb, alphas, gcap, k=K, thresh='binary', tv=0.5):
    d = emb.shape[1]; nr = len(alphas); kdim = d * nr
    t0 = time.time()

    keys = np.empty((TRAIN, kdim), dtype=np.float32)
    states = [emb[train_tok[0]].copy() for _ in alphas]
    keys[0] = np.concatenate(states)
    nm = np.linalg.norm(keys[0])
    if nm > 1e-10: keys[0] /= nm

    for i in range(1, TRAIN):
        e = emb[train_tok[i]]
        for j, a in enumerate(alphas):
            states[j] = a * e + (1-a) * states[j]
            nm = np.linalg.norm(states[j])
            if nm > 1e-10: states[j] /= nm
        keys[i] = np.concatenate(states)
        nm = np.linalg.norm(keys[i])
        if nm > 1e-10: keys[i] /= nm

    gk = np.empty((gcap, kdim), dtype=np.float32)
    go = np.empty(gcap, dtype=np.int32)
    gs = 0

    for i in range(TRAIN - 1):
        if gs == 0:
            gk[0] = keys[i]; go[0] = train_tok[i+1]; gs = 1; continue
        sims = gk[:gs] @ keys[i]
        kk = min(k, gs)
        idx = np.arange(gs) if gs <= kk else np.argpartition(sims, -kk)[-kk:]
        votes = np.zeros(VOCAB, dtype=np.float32)
        for j in idx:
            if sims[j] > 0: votes[go[j]] += sims[j]
        pred = np.argmax(votes); obs = train_tok[i+1]
        if thresh == 'binary':
            store = (pred != obs)
        else:
            total = votes.sum()
            store = ((votes[obs]/total if total > 0 else 0) < tv)
        if store and gs < gcap:
            gk[gs] = keys[i]; go[gs] = obs; gs += 1

    tt = time.time() - t0; t0 = time.time()

    tk = np.empty((TEST, kdim), dtype=np.float32)
    states = [emb[test_tok[0]].copy() for _ in alphas]
    tk[0] = np.concatenate(states)
    nm = np.linalg.norm(tk[0])
    if nm > 1e-10: tk[0] /= nm
    for i in range(1, TEST):
        e = emb[test_tok[i]]
        for j, a in enumerate(alphas):
            states[j] = a * e + (1-a) * states[j]
            nm = np.linalg.norm(states[j])
            if nm > 1e-10: states[j] /= nm
        tk[i] = np.concatenate(states)
        nm = np.linalg.norm(tk[i])
        if nm > 1e-10: tk[i] /= nm

    correct = 0
    for s in range(0, TEST-1, 500):
        e = min(s+500, TEST-1)
        sm = tk[s:e] @ gk[:gs].T
        for r in range(e-s):
            row = sm[r]; kk = min(k, gs)
            idx = np.arange(gs) if gs <= kk else np.argpartition(row, -kk)[-kk:]
            v = np.zeros(VOCAB, dtype=np.float32)
            for j in idx:
                if row[j] > 0: v[go[j]] += row[j]
            if np.argmax(v) == test_tok[s+r+1]: correct += 1

    return correct/(TEST-1)*100, gs, tt, time.time()-t0


# ============================================================
print("="*60, flush=True)
print("EXP 1: CARRIER SWEEP — single-rate [0.9], g=50K", flush=True)
print("="*60, flush=True)
for name in ['random', 'bigram', 'trigram', 'skipgram', 'bi_lr', 'bi_tri']:
    acc, gs, tt, et = run(carrier(name), [0.9], 50000)
    print(f"  {name:15s}: {acc:6.2f}%  g={gs:6d}  t={tt:.0f}s", flush=True)

print("\n" + "="*60, flush=True)
print("EXP 2: CARRIER SWEEP — 2-rate [0.95, 0.5], g=50K", flush=True)
print("="*60, flush=True)
for name in ['random', 'bigram', 'trigram', 'skipgram', 'bi_lr', 'bi_tri']:
    acc, gs, tt, et = run(carrier(name), [0.95, 0.5], 50000)
    print(f"  {name:15s}: {acc:6.2f}%  g={gs:6d}  t={tt:.0f}s", flush=True)

print("\n" + "="*60, flush=True)
print("EXP 3: BEST COMBOS — 2-rate + surprise<0.5 + g=80K", flush=True)
print("="*60, flush=True)
for name in ['bigram', 'trigram', 'skipgram', 'bi_tri']:
    acc, gs, tt, et = run(carrier(name), [0.95, 0.5], 80000, thresh='surprise')
    print(f"  {name:15s}: {acc:6.2f}%  g={gs:6d}  t={tt:.0f}s", flush=True)

print("\n" + "="*60, flush=True)
print("EXP 4: RATE SWEEP — trigram carrier, g=80K", flush=True)
print("="*60, flush=True)
emb_t = carrier('trigram')
for rn, als in [('[0.9]', [0.9]), ('[.95,.5]', [0.95, 0.5]),
                ('[.95,.7,.3]', [0.95, 0.7, 0.3]),
                ('[.98,.8,.5,.2]', [0.98, 0.8, 0.5, 0.2])]:
    acc, gs, tt, et = run(emb_t, als, 80000)
    print(f"  {rn:25s}: {acc:6.2f}%  g={gs:6d}  t={tt:.0f}s", flush=True)

# N-gram baselines
print("\n" + "="*60, flush=True)
print("N-GRAM BASELINES (same test set)", flush=True)
print("="*60, flush=True)

bi_c = sum(1 for i in range(TEST-1) if np.argmax(P[test_tok[i]]) == test_tok[i+1])
print(f"  Bigram:  {bi_c/(TEST-1)*100:.2f}%", flush=True)

tri_counts = Counter()
for i in range(TRAIN-2):
    tri_counts[(train_tok[i], train_tok[i+1], train_tok[i+2])] += 1
tri_best = {}
for (a,b,c), cnt in tri_counts.items():
    kk = (a,b)
    if kk not in tri_best or cnt > tri_best[kk][1]: tri_best[kk] = (c, cnt)
tri_c = sum(1 for i in range(TEST-2)
            if ((test_tok[i], test_tok[i+1]) in tri_best and
                tri_best[(test_tok[i], test_tok[i+1])][0] == test_tok[i+2])
            or ((test_tok[i], test_tok[i+1]) not in tri_best and
                np.argmax(P[test_tok[i+1]]) == test_tok[i+2]))
print(f"  Trigram: {tri_c/(TEST-2)*100:.2f}%", flush=True)

fg = Counter()
for i in range(TRAIN-4):
    fg[tuple(train_tok[i:i+5])] += 1
fb = {}
for g, cnt in fg.items():
    k4 = g[:4]
    if k4 not in fb or cnt > fb[k4][1]: fb[k4] = (g[4], cnt)
fc = 0
for i in range(TEST-4):
    k4 = tuple(test_tok[i:i+4])
    if k4 in fb:
        if fb[k4][0] == test_tok[i+4]: fc += 1
    elif (test_tok[i+2], test_tok[i+3]) in tri_best:
        if tri_best[(test_tok[i+2], test_tok[i+3])][0] == test_tok[i+4]: fc += 1
    elif np.argmax(P[test_tok[i+3]]) == test_tok[i+4]:
        fc += 1
print(f"  5-gram:  {fc/(TEST-4)*100:.2f}%", flush=True)

print("\nDone.", flush=True)

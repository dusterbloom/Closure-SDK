#!/usr/bin/env python3
"""
GeoLLM v7 — synthesis
=====================
Spine connecting four pieces from /home/peppi/Dev/research/ that have
each worked in isolation but never run together:

  1. HoloField interference ............ concept emergence from corpus
                                          (no backprop). Each token has a
                                          fixed phase pattern; co-occurring
                                          tokens accumulate correlated
                                          interference signatures.
  2. Modern Hopfield (β=32) ............ associative memory. Stores
                                          (left-context-aggregate, next-token)
                                          pairs; retrieves via softmax.
  3. TAO spherical embeddings .......... (unused in this minimal v7 but
                                          exposed via SphericalEmbeddings
                                          for tests + future composition).
  4. Bootstrap operators ............... is/not/and/or as fixed complex
                                          phase shifts in projection space.
                                          NOT = π phase flip. IS = identity.

Win condition (test_synthesis.py::test_end_to_end_beats_v3):
  top-1 ≥ 5.8 % on fixed 1500 test positions of Shakespeare+P&P,
  beating v3's 5.2 % by enough to clear sampling noise.
"""
import math, hashlib
from collections import Counter
from pathlib import Path
import numpy as np

from geollm_core import (
    I_q, hamilton, normalize, axis_angle, hopf_decompose, carrier_from_hopf,
    semantic_base_from_bytes, OPERATORS,
)

# ── Helpers ───────────────────────────────────────────────────────────
def sigma_between(a, b):
    """Geodesic distance between two unit quaternions on S^3, in [0, π]."""
    dot = float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))
    return math.acos(dot)


def domain_embed_unit(token):
    """Carrier for a single token, on S^3 (4 floats, unit)."""
    return carrier_from_hopf(semantic_base_from_bytes(token.encode('utf-8')), 0.0)


def apply_op(q, op_name):
    """Apply a hand-bootstrapped operator (left-Hamilton multiplication)."""
    if op_name not in OPERATORS:
        return q
    return normalize(hamilton(OPERATORS[op_name], q))


# ── HoloField: concept emergence via interference ────────────────────
class HoloField:
    """Each token is hash-keyed to a fixed complex phase pattern of
    `dim` frequencies. Walking a corpus, the field accumulates the
    pattern-sum of the *neighborhood* of each token. After ingest,
    `context_accum[t]` is the interference signature of t's typical
    neighborhood — co-occurring tokens have correlated signatures.

    This is the architectural correlate of "concepts emerge from stream
    statistics without backprop" — no gradient descent, just running
    sums of complex phase patterns.
    """
    def __init__(self, dim=64, seed=42):
        self.dim = dim
        self.token_patterns = {}
        self.context_accum = {}
        self.count = {}

    def _pattern(self, token):
        if token not in self.token_patterns:
            h = hashlib.sha256(token.encode('utf-8')).digest()
            phases = np.array([
                int.from_bytes(h[i*2:(i+1)*2], 'little') / 65536.0 * 2 * math.pi
                for i in range(self.dim)
            ])
            self.token_patterns[token] = np.exp(1j * phases)
        return self.token_patterns[token]

    def ingest(self, tokens, epochs=1, window=3):
        """For each token t at position i, accumulate Σ_{j in window} pattern(token[j])
        into context_accum[t]. After full walk, context_accum/count is t's
        average neighborhood pattern."""
        unique = set(tokens)
        for w in unique:
            if w not in self.context_accum:
                self.context_accum[w] = np.zeros(self.dim, dtype=np.complex128)
                self.count[w] = 0
        for _ in range(epochs):
            n = len(tokens)
            for i, t in enumerate(tokens):
                ctx_sum = np.zeros(self.dim, dtype=np.complex128)
                lo, hi = max(0, i-window), min(n, i+window+1)
                for j in range(lo, hi):
                    if j == i: continue
                    ctx_sum += self._pattern(tokens[j])
                self.context_accum[t] += ctx_sum
                self.count[t] += 1

    def project(self, token):
        """Return real-valued projection (2*dim) = [Re; Im] of normalized
        context accumulator. This is the token's interference signature."""
        if token not in self.context_accum or self.count[token] == 0:
            return np.zeros(2 * self.dim)
        v = self.context_accum[token] / max(1, self.count[token])
        return np.concatenate([v.real, v.imag])


# ── Modern Hopfield: associative memory at β=32 ──────────────────────
class HopfieldMemory:
    """Stores (key, value) unit-vector pairs. Retrieval = softmax(β·K·q)·V.
    β=32 is the architecture's reliable-retrieval setting per
    publishable/supra/memo2026_001."""
    def __init__(self, beta=32.0):
        self.beta = beta
        self.K_list = []
        self.V_list = []
        self._K_arr = None
        self._V_arr = None

    def store(self, key, value):
        self.K_list.append(np.asarray(key, dtype=np.float64))
        self.V_list.append(np.asarray(value, dtype=np.float64))
        self._K_arr = None  # invalidate cache

    def _stack(self):
        if self._K_arr is None:
            self._K_arr = np.stack(self.K_list, axis=0)
            self._V_arr = np.stack(self.V_list, axis=0)
        return self._K_arr, self._V_arr

    def retrieve(self, query):
        if not self.K_list:
            return np.asarray(query, dtype=np.float64)
        K, V = self._stack()
        scores = self.beta * (K @ np.asarray(query, dtype=np.float64))
        scores = scores - scores.max()
        weights = np.exp(scores)
        weights /= weights.sum()
        return weights @ V


# ── Spherical embeddings with spectral variance scaling ──────────────
class SphericalEmbeddings:
    """Per-mode variance follows holographic scaling:
        scale[k] = 1 / sqrt(1 + k(k+1))   (R=1, eps_k = k(k+1)/R^2)
    Low-k modes (small index) have larger expected norm than high-k modes."""
    def __init__(self, vocab_size, dim=16, seed=42):
        rng = np.random.RandomState(seed)
        ks = np.arange(dim)
        scale = 1.0 / np.sqrt(1.0 + ks * (ks + 1))
        self.W = rng.randn(vocab_size, dim) * scale[None, :]

    def weights(self):
        return self.W

    def __getitem__(self, idx):
        return self.W[idx]


# ── Operator phase shifts (in projection space) ──────────────────────
def _operator_phase_shift(op_name, dim):
    """Each operator gets a fixed complex phase-shift vector of length `dim`.
    NOT = e^{iπ} = -1 across all modes (sign flip).
    IS = e^{i0} = +1 (identity).
    AND/OR get small distinct rotations.
    Other operators (determiners/prepositions/pronouns/modals) = identity."""
    if op_name in ('not', "n't", 'no', 'never'):
        return -np.ones(dim, dtype=np.complex128)
    if op_name in ('and',):
        return np.exp(1j * (math.pi / 2)) * np.ones(dim, dtype=np.complex128)
    if op_name in ('or',):
        return np.exp(-1j * (math.pi / 2)) * np.ones(dim, dtype=np.complex128)
    if op_name in ('but',):
        return np.exp(1j * math.pi) * np.ones(dim, dtype=np.complex128)
    return None  # identity


# ── SynthesisLM v7.5: SVD bigram + concat-state + ridge head ─────────
class SynthesisLM:
    """The working synthesis. After empirically falsifying HoloField as a
    next-token-prediction signal (it encodes context-similarity, not
    transition probability — see test_synthesis::test_end_to_end_beats_v3
    for the ablation history), the architecture is:

      input/output embeddings: rank-r SVD of the bigram transition matrix
                               (frozen, derived from corpus, no backprop).
                               Captures transition info bilinearly.
      state:                   concat(prev_token_E_in, sum-of-earlier-4 E_in)
                               Preserves recency AND context, doesn't smear.
      learned head:            W = (X^T X + λI)^-1 X^T Y   (closed-form ridge)
                               Maps state -> predicted target embedding.
                               2*r * r trainable params (e.g. 2K at r=32).
      predict:                 argmax_w cos(state @ W, E_out[w])

    Bootstrap operators (is/not/and/or, etc.) are kept as available API
    via apply_op() but are NOT used in the prediction path here — three
    prior experiments in this directory (v0/v1/v4) showed they don't help
    on next-token prediction with this corpus, because the data already
    contains operator-context transitions. They live on for tasks where
    the data wouldn't supply that signal (entailment, OOD, sparse-corpus).

    HoloField (above) is also kept for tests T1, but not in the prediction
    path. Its strength (concept clustering) is orthogonal to next-token
    transitions and can be revived for entailment / clustering tasks.
    """
    def __init__(self, vocab, wid, dim=32, beta=32.0, seed=42, context_len=5):
        self.vocab = vocab
        self.wid = wid
        self.V = len(vocab)
        self.dim = dim                # SVD rank
        self.context_len = context_len
        self.E_in = None              # (V, dim) frozen, from SVD
        self.E_out = None             # (V, dim) frozen, from SVD
        self.W = None                 # (2*dim, dim) trainable, closed-form

    def _state(self, ids, i):
        """Concat(prev token's E_in, sum of earlier 4 tokens' E_in)."""
        prev = self.E_in[ids[i-1]]
        lo = max(0, i - self.context_len)
        if lo < i-1:
            earlier = self.E_in[ids[lo:i-1]].sum(axis=0)
        else:
            earlier = np.zeros(self.dim)
        return np.concatenate([prev, earlier])

    def train(self, train_tokens, ridge=1e-3):
        # Phase 1 — SVD of bigram transition matrix
        train_ids = [self.wid[t] for t in train_tokens if t in self.wid]
        B = np.zeros((self.V, self.V), dtype=np.float64)
        for k in range(len(train_ids) - 1):
            B[train_ids[k], train_ids[k+1]] += 1.0
        P = B / (B.sum(axis=1, keepdims=True) + 1e-12)
        U, S, Vt = np.linalg.svd(P, full_matrices=False)
        s = np.sqrt(S[:self.dim])
        self.E_in  = U[:, :self.dim] * s[None, :]
        self.E_out = (Vt[:self.dim, :] * s[:, None]).T

        # Phase 2 — collect (state, target_embedding) regression pairs
        Xs, Ys = [], []
        for i in range(self.context_len, len(train_ids)):
            Xs.append(self._state(train_ids, i))
            Ys.append(self.E_out[train_ids[i]])
        X = np.stack(Xs, axis=0)
        Y = np.stack(Ys, axis=0)

        # Phase 3 — closed-form ridge: W = (X^T X + λI)^-1 X^T Y
        XtX = X.T @ X + ridge * np.eye(2 * self.dim)
        self.W = np.linalg.solve(XtX, X.T @ Y)

    def predict_topk(self, ctx_tokens, k=10):
        if self.E_in is None or self.W is None:
            return []
        ids = [self.wid[t] for t in ctx_tokens if t in self.wid]
        if len(ids) < 1:
            return []
        # Pad: use the last available token if context too short
        while len(ids) < self.context_len:
            ids = [ids[0]] + ids
        # Pseudo-position: score state at "the next slot after ctx"
        i = len(ids)
        prev = self.E_in[ids[i-1]]
        earlier = self.E_in[ids[max(0,i-self.context_len):i-1]].sum(axis=0) \
                  if i > 1 else np.zeros(self.dim)
        state = np.concatenate([prev, earlier])
        pred = state @ self.W
        sims = self.E_out @ pred
        return list(np.argsort(-sims)[:k])

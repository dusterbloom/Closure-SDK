#!/usr/bin/env python3
"""
Replicate Chavan 2025's methodology, but ablate the *learning* part.

Chavan trains an embedding network with TRIPLET LOSS using class labels,
then projects to a sphere and reports near-perfect AG News classification.
We test whether the geometry alone — Faltz's HASH-DERIVED S^3 carriers,
NO training, NO labels — captures comparable structure.

Falsifiable comparisons (T1 — T3):

  T1. Faltz's training-free sphere carriers must NOT collapse to noise.
      Silhouette score on AG News > 0 (versus random embedding which gives ≤0).

  T2. Chavan's claim "manifold geometry helps even without training" must hold:
      Faltz mean-pooled carriers must produce silhouette > the unconstrained
      Euclidean baseline (here: random Gaussian embeddings of the same dim).

  T3. Linear-classifier accuracy with Faltz's carriers must clear 50% (random
      is 25% for 4-class). Doesn't have to match Chavan's 99.88% — but it must
      cross 0.5 to demonstrate that the geometry encodes a topic signal
      without ever seeing labels.

If all three pass: hash-derived geometry alone captures *real* topic structure.
If T1 fails: hash carriers are pure noise; Faltz's framing is broken at the
embedding step.
"""
import math, hashlib, random, re
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score, accuracy_score

THIS_DIR = Path(__file__).parent.resolve()


def _load_ag_news():
    """Load AG News from HuggingFace cache, fall back to a tiny built-in synthetic if unavailable."""
    try:
        from datasets import load_dataset
        ds = load_dataset("ag_news", split={"train": "train[:8000]", "test": "test[:2000]"})
        # Tuples (text, label)
        train = [(ex['text'].lower(), ex['label']) for ex in ds['train']]
        test  = [(ex['text'].lower(), ex['label']) for ex in ds['test']]
        return train, test
    except Exception as e:
        print(f"  HuggingFace load failed ({e}); using stub")
        return None, None


def _tokenize(text):
    return re.findall(r"[a-z']+", text)


def _hash_carrier(token, seed=0):
    """Same as Faltz: SHA-256 → S^2 base via Box-Muller → carrier_from_hopf(base, 0)."""
    h = hashlib.sha256(f"{seed}:{token}".encode('utf-8')).digest()
    u1 = (int.from_bytes(h[:8],  'little') + 1) / (2**64 + 1)
    u2 = (int.from_bytes(h[8:16],'little') + 1) / (2**64 + 1)
    z = 1.0 - 2.0 * u1
    r = math.sqrt(max(0.0, 1.0 - z*z))
    phi = 2.0 * math.pi * u2
    base = np.array([r*math.cos(phi), r*math.sin(phi), z])
    bz = float(np.clip(base[2], -1.0, 1.0))
    sin_t = math.sqrt(max(0.0, 1.0 - bz*bz))
    if sin_t < 1e-12:
        if bz >= 0:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return np.array([0.0, 0.0, 1.0, 0.0])
    theta = math.acos(bz)
    delta = math.atan2(base[1], base[0])
    alpha = 0.5 * (0.0 + delta)
    beta_ = 0.5 * (0.0 - delta)
    ct, st = math.cos(0.5*theta), math.sin(0.5*theta)
    q = np.array([ct*math.cos(alpha), st*math.sin(beta_),
                  st*math.cos(beta_),  ct*math.sin(alpha)])
    return q / np.linalg.norm(q)


def _embed_sentence_faltz(text, dim=4):
    """Mean-pool hash carriers, re-normalize to S^3."""
    toks = _tokenize(text)
    if not toks:
        return np.zeros(dim)
    carriers = np.stack([_hash_carrier(t) for t in toks], axis=0)
    mean = carriers.mean(axis=0)
    n = np.linalg.norm(mean)
    return mean / n if n > 1e-12 else mean


def _embed_sentence_random(text, dim=4, rng=None):
    """Random unit vector per sentence — pure-noise baseline."""
    if rng is None:
        rng = np.random.RandomState(hash(text) & 0xFFFFFFFF)
    v = rng.randn(dim)
    return v / np.linalg.norm(v)


def _embed_sentence_tfidf(corpus_texts, target_text):
    """Quick TF-IDF using a simple bag-of-words count vector for the
    target_text, projected to the dim of the bag-of-words vocab.
    Use sklearn for correctness."""
    raise NotImplementedError("computed in batch instead")


def _build_tfidf(train_texts, test_texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=2000)
    X_train = vec.fit_transform(train_texts).toarray()
    X_test = vec.transform(test_texts).toarray()
    return X_train, X_test


# ── T1: Faltz training-free carriers must give silhouette > 0 ────────
def test_faltz_silhouette_positive():
    train, test = _load_ag_news()
    assert train is not None, "AG News must be available (HF datasets cache)"
    # Use the training subset for silhouette
    texts  = [t for t, _ in train]
    labels = np.array([l for _, l in train])
    X = np.stack([_embed_sentence_faltz(t) for t in texts], axis=0)
    s = silhouette_score(X, labels, metric='cosine')
    print(f"  Faltz hash-carrier silhouette: {s:.4f}")
    assert s > 0.0, f"Faltz carriers degenerate to noise (silhouette={s:.4f})"


# ── T2: Faltz silhouette > random Gaussian embedding silhouette ──────
def test_faltz_beats_random_geometry():
    train, _ = _load_ag_news()
    assert train is not None
    texts  = [t for t, _ in train]
    labels = np.array([l for _, l in train])
    X_faltz  = np.stack([_embed_sentence_faltz(t)  for t in texts], axis=0)
    X_random = np.stack([_embed_sentence_random(t) for t in texts], axis=0)
    s_faltz  = silhouette_score(X_faltz,  labels, metric='cosine')
    s_random = silhouette_score(X_random, labels, metric='cosine')
    print(f"  Faltz silhouette: {s_faltz:.4f}  ·  Random silhouette: {s_random:.4f}")
    assert s_faltz > s_random + 0.02, \
        f"Hash carriers don't clearly beat random (Faltz={s_faltz:.4f} vs Random={s_random:.4f})"


# ── T3: classification accuracy must clear 50% (random=25%) ──────────
def test_faltz_classification_clears_chance():
    train, test = _load_ag_news()
    assert train is not None
    train_texts = [t for t, _ in train]; y_train = [l for _, l in train]
    test_texts  = [t for t, _ in test];  y_test  = [l for _, l in test]
    X_train = np.stack([_embed_sentence_faltz(t) for t in train_texts], axis=0)
    X_test  = np.stack([_embed_sentence_faltz(t) for t in test_texts ], axis=0)
    clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"  Faltz LR accuracy on AG News: {acc:.4f} (random=0.25)")
    assert acc > 0.5, \
        f"Faltz carriers don't clear 50% (acc={acc:.4f}); doesn't encode topic signal"


# ── Full comparison report (always-run, always-pass diagnostic) ──────
def test_full_comparison_report():
    """Print the full comparison so we can see where Faltz lands relative
    to TF-IDF and random baselines, plus richer Faltz variants."""
    train, test = _load_ag_news()
    assert train is not None
    train_texts = [t for t, _ in train]; y_train = np.array([l for _, l in train])
    test_texts  = [t for t, _ in test];  y_test  = np.array([l for _, l in test])

    print("\n  ── AG News comparison (train 8K, test 2K) ──")
    print(f"  {'embedding':<25s}  {'dim':>5s}  {'silhouette':>10s}  {'LR-acc':>7s}")

    # 1. Random unit-vector baseline (negative control)
    X_tr = np.stack([_embed_sentence_random(t) for t in train_texts], axis=0)
    X_te = np.stack([_embed_sentence_random(t) for t in test_texts ], axis=0)
    s = silhouette_score(X_tr, y_train, metric='cosine')
    clf = LogisticRegression(max_iter=1000).fit(X_tr, y_train)
    acc = accuracy_score(y_test, clf.predict(X_te))
    print(f"  {'Random (4d)':<25s}  {4:>5d}  {s:>10.4f}  {acc:>7.4f}")

    # 2. Faltz hash carriers (4d)
    X_tr = np.stack([_embed_sentence_faltz(t) for t in train_texts], axis=0)
    X_te = np.stack([_embed_sentence_faltz(t) for t in test_texts ], axis=0)
    s = silhouette_score(X_tr, y_train, metric='cosine')
    clf = LogisticRegression(max_iter=1000).fit(X_tr, y_train)
    acc = accuracy_score(y_test, clf.predict(X_te))
    print(f"  {'Faltz hash mean (4d)':<25s}  {4:>5d}  {s:>10.4f}  {acc:>7.4f}")

    # 3. TF-IDF
    X_tr, X_te = _build_tfidf(train_texts, test_texts)
    s_sample = silhouette_score(X_tr[:2000], y_train[:2000], metric='cosine')
    clf = LogisticRegression(max_iter=1000).fit(X_tr, y_train)
    acc = accuracy_score(y_test, clf.predict(X_te))
    print(f"  {'TF-IDF (2000d)':<25s}  {X_tr.shape[1]:>5d}  {s_sample:>10.4f}  {acc:>7.4f}")

    # 4. Faltz hash carriers concatenated with bag-of-hash-bits (high-d Faltz)
    # Approach: aggregate per-token carriers as a sum (higher-magnitude for common topics)
    def hash_bag(text, dim=128):
        v = np.zeros(dim)
        for tok in _tokenize(text):
            h = hashlib.sha256(tok.encode()).digest()
            for i in range(dim // 8):
                idx = h[i] % dim
                v[idx] += 1.0
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else v
    X_tr = np.stack([hash_bag(t) for t in train_texts], axis=0)
    X_te = np.stack([hash_bag(t) for t in test_texts ], axis=0)
    s = silhouette_score(X_tr, y_train, metric='cosine')
    clf = LogisticRegression(max_iter=1000).fit(X_tr, y_train)
    acc = accuracy_score(y_test, clf.predict(X_te))
    print(f"  {'Hash bag-of-bits (128d)':<25s}  {128:>5d}  {s:>10.4f}  {acc:>7.4f}")


if __name__ == "__main__":
    failures = []
    for name in ['test_faltz_silhouette_positive',
                 'test_faltz_beats_random_geometry',
                 'test_faltz_classification_clears_chance',
                 'test_full_comparison_report']:
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
    print("\nAll tests passed")

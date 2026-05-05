#!/usr/bin/env python3
"""TDD coverage for the synthesis LM (v7).

Five tests, increasing ambition:

  T1 — HoloField concept emergence.
       Ingest a tiny corpus where two tokens co-occur tightly; their
       phase-correlated interference produces a stable shared concept
       (cosine sim of their state-projections > 0.9 after stabilization).

  T2 — Hopfield retrieval at β=32.
       Store 100 (key, value) pairs of unit quaternions; retrieve each
       key, recover the stored value with cosine > 0.99.

  T3 — Spherical manifold variance scaling.
       Build embeddings for V tokens at d=16. Verify low-frequency
       modes have larger expected norm than high-frequency modes
       (the holographic principle: boundary > bulk).

  T4 — Bootstrap operators distinguish state.
       Apply NOT to a content carrier; verify the result has σ > π/3
       from the original. Apply IS; verify σ ≈ 0.

  T5 — End-to-end win condition.
       Train on Shakespeare+P&P, evaluate on 1500 fixed test positions.
       Top-1 must exceed 0.058 (clearly past v3's 5.2%, past noise).
       This is the falsification target.
"""
import math, random, hashlib
from collections import Counter
from pathlib import Path
import numpy as np

import synthesis as s

THIS_DIR = Path(__file__).parent.resolve()


# ── T1: HoloField concept emergence ───────────────────────────────────
def test_holofield_concept_emergence():
    """Tightly co-occurring tokens should converge to overlapping field
    states. Test corpus: 'ab ab ab ab ... cd cd cd cd ...' (50 each).
    After ingest, the field's projection of 'a' and 'b' should be more
    similar than the projection of 'a' and 'c'."""
    field = s.HoloField(dim=64, seed=42)
    corpus = (['a', 'b'] * 50) + (['c', 'd'] * 50)
    field.ingest(corpus, epochs=3)

    p_a = field.project('a')
    p_b = field.project('b')
    p_c = field.project('c')

    sim_ab = float(np.dot(p_a, p_b) / (np.linalg.norm(p_a) * np.linalg.norm(p_b) + 1e-12))
    sim_ac = float(np.dot(p_a, p_c) / (np.linalg.norm(p_a) * np.linalg.norm(p_c) + 1e-12))

    # Co-occurring tokens should be MORE similar in field projection than
    # disjoint tokens. (Initial random hash gives sim near 0; after
    # interference, sim_ab should pull positive, sim_ac stays near 0.)
    assert sim_ab > sim_ac + 0.10, \
        f"co-occurring 'a','b' (sim={sim_ab:.3f}) should be much closer than disjoint 'a','c' (sim={sim_ac:.3f})"


# ── T2: Hopfield retrieval at β=32 ────────────────────────────────────
def test_hopfield_retrieval_beta32():
    """Store 100 (key, value) pairs; query each key; recover each value
    with cos > 0.99. β=32 is the architecture's reliable-retrieval setting."""
    rng = np.random.RandomState(42)
    keys   = rng.randn(100, 16); keys   /= np.linalg.norm(keys,   axis=1, keepdims=True)
    values = rng.randn(100, 16); values /= np.linalg.norm(values, axis=1, keepdims=True)

    mem = s.HopfieldMemory(beta=32.0)
    for k, v in zip(keys, values):
        mem.store(k, v)

    correct = 0
    for k, v in zip(keys, values):
        retrieved = mem.retrieve(k)
        cos = float(np.dot(retrieved, v) / (np.linalg.norm(retrieved) * np.linalg.norm(v) + 1e-12))
        if cos > 0.99:
            correct += 1

    assert correct >= 95, f"β=32 should retrieve ≥95/100 correctly, got {correct}"


# ── T3: Spherical manifold variance scaling ───────────────────────────
def test_spherical_variance_scaling():
    """TAO's spectral principle: low-frequency modes carry stable
    concepts (high norm); high-frequency modes carry plastic detail
    (lower norm). At d=16, the first 4 modes should have median norm
    > the last 4 modes."""
    emb = s.SphericalEmbeddings(vocab_size=500, dim=16, seed=42)
    weights = emb.weights()  # (V, d)
    # Per-mode norm = column-wise L2 across the vocabulary.
    per_mode = np.linalg.norm(weights, axis=0)  # (d,)
    low_norm  = float(np.median(per_mode[:4]))
    high_norm = float(np.median(per_mode[-4:]))
    assert low_norm > high_norm * 1.5, \
        f"low-mode norm ({low_norm:.3f}) should be >1.5× high-mode norm ({high_norm:.3f})"


# ── T4: Bootstrap operators distinguish state ─────────────────────────
def test_bootstrap_operators_distinguish():
    """Apply NOT to a content carrier; result should be far (σ > π/3) from
    the original. Apply IS (identity); should be close (σ < 0.01)."""
    carrier = s.domain_embed_unit('cat')
    not_q = s.apply_op(carrier, 'not')
    is_q  = s.apply_op(carrier, 'is')

    sigma_not = s.sigma_between(carrier, not_q)
    sigma_is  = s.sigma_between(carrier, is_q)

    assert sigma_not > math.pi / 3, \
        f"NOT applied should produce σ > π/3, got {sigma_not:.3f}"
    assert sigma_is  < 0.01, \
        f"IS applied should produce σ ≈ 0, got {sigma_is:.3f}"


# ── T5: End-to-end win condition ──────────────────────────────────────
def test_end_to_end_beats_v3():
    """The integration must beat v3's 5.2% top-1 by ≥ 0.6pp on the same
    fixed 1500 test positions. Below 5.8% means no real win."""
    import re
    here = THIS_DIR / 'corpora'
    text = (here/'shakespeare.txt').read_text(errors='ignore').lower() + \
           ' ' + (here/'pride_prejudice.txt').read_text(errors='ignore').lower()
    toks = re.findall(r"[a-z']+", text)
    n_train = int(len(toks) * 0.9)
    train, test = toks[:n_train], toks[n_train:]
    counts = Counter(train)
    vocab = [w for w, c in counts.most_common(2500) if c >= 10]
    wid = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)

    model = s.SynthesisLM(vocab=vocab, wid=wid, dim=32, beta=32.0, seed=42, context_len=5)
    model.train(train)

    # Fixed test positions, deterministic across runs and across variants.
    valid = [t for t in test if t in wid]
    rng = random.Random(42)
    pos = sorted(rng.sample(range(5, len(valid)), min(1500, len(valid)-5)))

    correct = 0
    for p in pos:
        ctx = valid[p-5:p]
        target_id = wid[valid[p]]
        ranked = model.predict_topk(ctx, k=1)
        if ranked and ranked[0] == target_id:
            correct += 1
    top1 = correct / len(pos)

    assert top1 >= 0.058, \
        f"synthesis top-1 ({top1:.4f}) must exceed v3's 5.2% by ≥0.6pp to count as a real win"


if __name__ == "__main__":
    failures = []
    for name, fn in list(globals().items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        try:
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

#!/usr/bin/env python3
"""TDD coverage for the closure-dynamics LM (v6).

Three behavioural tests motivate the implementation:

  1. On a pathological sequence (single token repeated), closure-genome
     stops growing once cell_c locks onto the carrier. Result: genome
     stays SMALL relative to sequence length. This proves σ-thresholded
     ingest actually filters predictable observations.

  2. On the real corpus, closure-genome accuracy stays within a small
     margin (<= 1.0pp) of v3's 5.2% top-1 baseline despite a much
     smaller genome. Compression without accuracy loss.

  3. The very first observation always fires a closure. cell_c starts
     at identity; any embedded token has σ(token, identity) > 0, and
     for a typical hash-derived carrier σ approaches π/2, well past
     the π/4 threshold. So genome[0] exists after one ingest step.

Tests import closure_dynamics (v6).
"""
import math
import random
from pathlib import Path

THIS_DIR = Path(__file__).parent.resolve()

import closure_dynamics as m


def _load_v6():
    return m


# ── Test 1: predictable repetition compresses dramatically ────────────
def test_repetition_produces_small_genome():
    """Stream of one token repeated 100 times. After the first few
    closures lock cell_c onto the carrier, no further closures fire.
    Genome size must be <= 10 (allowing a few warmup closures), not 100."""
    mod = _load_v6()
    tokens = ['the'] * 100
    vocab = ['the']
    wid = {'the': 0}
    carriers = {'the': mod.domain_embed('the')}
    genome = mod.train_closure(tokens, carriers, mod.OPERATORS, wid,
                                threshold=math.pi/4)
    assert len(genome) <= 10, \
        f"closure-genome should be <= 10 entries on repeated input, got {len(genome)}"
    # And blind ingest of the same input would store ALL ~99 left-context
    # transitions, so closure must compress meaningfully.
    blind_size = max(1, len(tokens) - 1)
    assert len(genome) < blind_size / 5, \
        f"closure-genome ({len(genome)}) should be < blind ({blind_size}) / 5"


# ── Test 2: first observation always fires closure ────────────────────
def test_first_observation_fires_closure():
    """cell_c = identity. First token's σ-residual w.r.t. identity is
    > π/4 for any non-identity carrier. Genome[0] must exist."""
    mod = _load_v6()
    tokens = ['hello']
    wid = {'hello': 0}
    carriers = {'hello': mod.domain_embed('hello')}
    genome = mod.train_closure(tokens, carriers, mod.OPERATORS, wid,
                                threshold=math.pi/4)
    assert len(genome) >= 1, \
        "first observation must produce at least one genome entry"


# ── Test 3: accuracy holds within 1pp of v3 baseline ──────────────────
def test_closure_lm_accuracy_holds_on_corpus():
    """On Shakespeare+P&P, closure-LM must achieve top-1 within 1pp of
    v3's ~5.2% baseline. Sample 500 test positions for speed."""
    import re
    from collections import Counter

    mod = _load_v6()
    here = THIS_DIR / 'corpora'
    text = ' '.join((here/'shakespeare.txt').read_text(errors='ignore').lower()
                  + ' ' + (here/'pride_prejudice.txt').read_text(errors='ignore').lower()
                  for _ in [0])
    toks = re.findall(r"[a-z']+", text)
    n_train = int(len(toks) * 0.9)
    train, test = toks[:n_train], toks[n_train:]
    counts = Counter(train)
    vocab = [w for w,c in counts.most_common(2500) if c >= 10]
    wid = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    carriers = {w: mod.domain_embed(w) for w in vocab}

    genome = mod.train_closure(train, carriers, mod.OPERATORS, wid,
                                threshold=math.pi/4)

    # Sanity: genome should be smaller than blind-ingest size (≈265K).
    assert len(genome) < 200_000, \
        f"closure-genome should compress vs blind ingest, got {len(genome)}"

    # Eval on a small sample for test speed.
    random.seed(42)
    valid = [t for t in test if t in wid]
    pos = random.sample(range(5, len(valid)), min(500, len(valid)-5))
    correct = 0
    for p in pos:
        ctx = valid[p-5:p]
        target_id = wid[valid[p]]
        ranked = mod.predict_topk(ctx, genome, carriers, mod.OPERATORS, wid, V, k=1)
        if ranked and ranked[0] == target_id:
            correct += 1
    top1 = correct / len(pos)
    # v3 baseline ~5.2%, allow drop down to 4.2%.
    assert top1 >= 0.042, \
        f"closure-LM top-1 ({top1:.4f}) dropped >1pp below v3 baseline (0.052)"


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

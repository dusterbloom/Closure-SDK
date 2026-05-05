#!/usr/bin/env python3
"""
GeoLLM v8 — SVD-derived S³ embeddings replace hash embeddings
==============================================================
Hypothesis: the gap between closure LM (5.2%) and the rank-4 SVD ceiling (8.1%)
is entirely due to hash embeddings scattering semantically similar tokens.
Fix: compute PPMI → rank-4 SVD → normalize to S³. Use U√Σ as input operators,
V√Σ as output carriers. Separate roles: input rotates context, output is stored
in genome as observation.

Comparison: hash vs SVD vs SVD+multichannel on same corpus/split.
"""
import sys
import math
import hashlib
from collections import Counter
from pathlib import Path
import numpy as np

# ── Quaternion primitives ────────────────────────────────────────────

def hamilton(a, b):
    w1,x1,y1,z1 = a; w2,x2,y2,z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2])

def normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v if n < eps else v / n

def geodesic(a, b):
    d = np.clip(abs(np.dot(a, b)), 0.0, 1.0)
    return math.acos(d)

def slerp(a, b, t):
    dot = np.dot(a, b)
    if dot < 0:
        b = -b
        dot = -dot
    if dot > 0.9995:
        return normalize(a + t * (b - a))
    theta0 = math.acos(np.clip(dot, -1, 1))
    theta = theta0 * t
    s0 = math.cos(theta) - dot * math.sin(theta) / math.sin(theta0)
    s1 = math.sin(theta) / math.sin(theta0)
    return normalize(s0 * a + s1 * b)

I_q = np.array([1.0, 0.0, 0.0, 0.0])

# ── Embedding strategies ─────────────────────────────────────────────

def hash_embed(token_bytes):
    h = hashlib.sha256(token_bytes).digest()
    raw = np.array([
        int.from_bytes(h[0:4], 'little') / 2**32 - 0.5,
        int.from_bytes(h[4:8], 'little') / 2**32 - 0.5,
        int.from_bytes(h[8:12], 'little') / 2**32 - 0.5,
        int.from_bytes(h[12:16], 'little') / 2**32 - 0.5,
    ])
    return normalize(raw)

def build_svd_embeddings(train_tokens, vocab_size):
    bigram = np.zeros((vocab_size, vocab_size))
    for i in range(len(train_tokens) - 1):
        bigram[train_tokens[i], train_tokens[i+1]] += 1

    row_sums = bigram.sum(axis=1, keepdims=True) + 1e-10
    col_sums = bigram.sum(axis=0, keepdims=True) + 1e-10
    total = bigram.sum() + 1e-10
    pmi = np.log2(bigram * total / (row_sums * col_sums) + 1e-10)
    ppmi = np.maximum(pmi, 0)

    U, S, Vt = np.linalg.svd(ppmi, full_matrices=False)
    sqrt_S = np.sqrt(S[:4])

    input_ops = U[:, :4] * sqrt_S[np.newaxis, :]
    output_carriers = Vt[:4, :].T * sqrt_S[np.newaxis, :]

    in_norms = np.linalg.norm(input_ops, axis=1, keepdims=True)
    out_norms = np.linalg.norm(output_carriers, axis=1, keepdims=True)
    input_ops = input_ops / np.maximum(in_norms, 1e-10)
    output_carriers = output_carriers / np.maximum(out_norms, 1e-10)

    return input_ops, output_carriers


def build_svd_ordered_operators(train_tokens, vocab_size):
    """
    Strategy: use SVD to find semantic ordering, then assign well-separated
    Fibonacci-spiral operators in that order. Tokens with similar bigram
    contexts get nearby operators → Hamilton composition produces similar
    cell_c → kNN can generalize. But operators remain maximally separated.
    """
    # Get SVD embeddings for ordering
    svd_input, svd_output = build_svd_embeddings(train_tokens, vocab_size)

    # Sort tokens by first principal component of SVD embedding
    order = np.argsort(svd_input[:, 0])

    # Generate Fibonacci spiral points on S³ (well-separated)
    phi = (1 + math.sqrt(5)) / 2
    spiral_ops = np.zeros((vocab_size, 4))
    for i in range(vocab_size):
        theta1 = 2 * math.pi * i / phi
        theta2 = math.acos(1 - 2 * (i + 0.5) / vocab_size)
        w = math.cos(theta2/2) * math.cos(theta1/2)
        x = math.cos(theta2/2) * math.sin(theta1/2)
        y = math.sin(theta2/2) * math.cos(theta1/2 + math.pi*i/vocab_size)
        z = math.sin(theta2/2) * math.sin(theta1/2 + math.pi*i/vocab_size)
        spiral_ops[i] = normalize(np.array([w, x, y, z]))

    # Assign: token ranked i-th in SVD order gets the i-th spiral operator
    ops = np.zeros((vocab_size, 4))
    carriers = np.zeros((vocab_size, 4))
    for rank, token_id in enumerate(order):
        ops[token_id] = spiral_ops[rank]
        carriers[token_id] = spiral_ops[rank]

    return ops, carriers


def build_svd_direct_fixed(train_tokens, vocab_size):
    """
    Direct SVD embeddings but with guaranteed unit norm (fallback to random
    for degenerate tokens) and separated input/output.
    """
    svd_input, svd_output = build_svd_embeddings(train_tokens, vocab_size)

    # Fix degenerate rows: replace near-zero with random unit quaternions
    rng = np.random.default_rng(42)
    for i in range(vocab_size):
        if np.linalg.norm(svd_input[i]) < 0.1:
            svd_input[i] = normalize(rng.standard_normal(4))
        else:
            svd_input[i] = normalize(svd_input[i])
        if np.linalg.norm(svd_output[i]) < 0.1:
            svd_output[i] = normalize(rng.standard_normal(4))
        else:
            svd_output[i] = normalize(svd_output[i])

    return svd_input, svd_output


# ── Closure LM engine ────────────────────────────────────────────────

THRESHOLD = math.pi / 4

def run_closure_lm(train_tokens, test_tokens, vocab_size, mode='hash',
                   input_ops=None, output_carriers=None, num_channels=1):
    """
    Train genome via closure dynamics, evaluate on test positions.
    mode: 'hash' or 'svd'
    """
    # Build vocab → embedding lookup
    if mode == 'hash':
        vocab_ops = np.array([hash_embed(bytes([t])) for t in range(vocab_size)])
        vocab_carriers = vocab_ops.copy()
    else:
        vocab_ops = input_ops
        vocab_carriers = output_carriers

    # Train: build genome
    genome = []  # list of (context_q, observation_q)
    cell_c = [I_q.copy() for _ in range(num_channels)]

    for t in train_tokens:
        op = vocab_ops[t]
        carrier = vocab_carriers[t]

        for c in range(num_channels):
            cell_c[c] = normalize(hamilton(cell_c[c], op))

        # Predict from fast channel
        if len(genome) > 0:
            dists = np.array([geodesic(cell_c[0], g[0]) for g in genome])
            k = min(5, len(genome))
            top_k = np.argsort(dists)[:k]
            weights = 1.0 / (dists[top_k] + 1e-6)
            pred = normalize(sum(weights[i] * genome[top_k[i]][1]
                                 for i in range(k)) / weights.sum())
        else:
            pred = I_q.copy()

        sigma = geodesic(pred, carrier)
        if sigma > THRESHOLD:
            genome.append((cell_c[0].copy(), carrier.copy()))

        # Slerp update
        if sigma > THRESHOLD:
            alpha_fast = 0.3
        else:
            alpha_fast = 0.05
        cell_c[0] = slerp(cell_c[0], carrier, alpha_fast)

        if num_channels > 1:
            cell_c[1] = slerp(cell_c[1], carrier, 0.01)

    genome_size = len(genome)

    # Test: evaluate top-1 accuracy
    correct = 0
    total = 0
    # Reset state for test
    cell_c = [I_q.copy() for _ in range(num_channels)]

    for idx in range(len(test_tokens) - 1):
        t = test_tokens[idx]
        next_t = test_tokens[idx + 1]
        op = vocab_ops[t]
        carrier = vocab_carriers[t]

        for c in range(num_channels):
            cell_c[c] = normalize(hamilton(cell_c[c], op))

        # Predict next token
        if len(genome) > 0:
            dists = np.array([geodesic(cell_c[0], g[0]) for g in genome])
            k = min(5, len(genome))
            top_k = np.argsort(dists)[:k]
            weights = 1.0 / (dists[top_k] + 1e-6)
            pred = normalize(sum(weights[i] * genome[top_k[i]][1]
                                 for i in range(k)) / weights.sum())
        else:
            pred = I_q.copy()

        # Which token is closest to prediction?
        pred_dists = np.array([geodesic(pred, vocab_carriers[v])
                               for v in range(vocab_size)])
        predicted_token = np.argmin(pred_dists)

        if predicted_token == next_t:
            correct += 1
        total += 1

        # Update state
        sigma = geodesic(pred, carrier)
        alpha = 0.3 if sigma > THRESHOLD else 0.05
        cell_c[0] = slerp(cell_c[0], carrier, alpha)
        if num_channels > 1:
            cell_c[1] = slerp(cell_c[1], carrier, 0.01)

    accuracy = correct / max(total, 1) * 100
    return genome_size, accuracy

# ── Hopf analysis ────────────────────────────────────────────────────

def hopf_decompose(q):
    w, x, y, z = q
    base = np.array([
        2*(x*z + w*y),
        2*(y*z - w*x),
        w*w + z*z - x*x - y*y
    ])
    if (w*w + z*z) > 1e-10:
        fiber = math.atan2(z, w) - math.atan2(x, y) if (x*x + y*y) > 1e-10 else math.atan2(z, w)
    else:
        fiber = 0.0
    return base, fiber

def analyze_hopf(input_ops, vocab_size, token_map):
    print("\n=== Hopf Fibration Analysis ===")
    bases = []
    fibers = []
    for i in range(vocab_size):
        b, f = hopf_decompose(input_ops[i])
        bases.append(b)
        fibers.append(f)
    bases = np.array(bases)
    fibers = np.array(fibers)

    # Classify: space/punctuation vs letters
    func_indices = [i for i in range(vocab_size)
                    if token_map.get(i, '') in ' .,;:!?\'"()-']
    content_indices = [i for i in range(vocab_size)
                       if token_map.get(i, '').isalpha()]

    if func_indices and content_indices:
        func_bases = bases[func_indices]
        content_bases = bases[content_indices]
        func_center = normalize(func_bases.mean(axis=0))
        content_center = normalize(content_bases.mean(axis=0))
        separation = math.acos(np.clip(np.dot(func_center, content_center), -1, 1))
        print(f"  S² base separation (function vs content): {math.degrees(separation):.1f}°")

        func_fiber_std = np.std(fibers[func_indices])
        content_fiber_std = np.std(fibers[content_indices])
        print(f"  S¹ fiber spread — function words: {func_fiber_std:.3f} rad")
        print(f"  S¹ fiber spread — content words:  {content_fiber_std:.3f} rad")
    else:
        print("  (insufficient token classification for Hopf analysis)")

# ── Main ─────────────────────────────────────────────────────────────

def load_corpus():
    candidates = [
        Path('./corpora/shakespeare.txt'),
        Path('/home/peppi/Dev/Closure-SDK/corpora/shakespeare.txt'),
    ]
    for p in candidates:
        if p.exists():
            text = p.read_text(encoding='utf-8', errors='ignore')
            print(f"  Loaded corpus: {p} ({len(text)} chars)")
            return text

    # Fallback: synthetic
    print("  Using synthetic corpus (no real text found)")
    return ("to be or not to be that is the question " * 500 +
            "the quick brown fox jumps over the lazy dog " * 300 +
            "she sells sea shells by the sea shore " * 200 +
            "now is the winter of our discontent made glorious summer " * 150)

def main():
    print("GeoLLM v8 — SVD Embeddings Experiment")
    print("=" * 50)

    text = load_corpus()

    # Tokenize: byte-level, mod vocab_size
    vocab_size = 64
    tokens = np.array([(b - 32) % vocab_size if 32 <= b <= 126 else 0
                       for b in text.encode('ascii', errors='ignore')],
                      dtype=np.int32)

    # Build token map for Hopf analysis
    token_map = {}
    for b in range(32, 127):
        tok = (b - 32) % vocab_size
        if tok not in token_map:
            token_map[tok] = chr(b)

    # Train/test split
    split = int(len(tokens) * 0.8)
    train = tokens[:split]
    test = tokens[split:]
    print(f"  Vocab: {vocab_size} | Train: {len(train)} | Test: {len(test)} tokens")

    # Build embeddings
    print("\nBuilding embeddings...")
    print("  [a] SVD direct (PPMI → rank-4 SVD → S³)...")
    svd_input, svd_output = build_svd_direct_fixed(train, vocab_size)
    print(f"      Norms: input={np.linalg.norm(svd_input, axis=1).mean():.4f}, output={np.linalg.norm(svd_output, axis=1).mean():.4f}")

    print("  [b] SVD-ordered Fibonacci operators...")
    ordered_ops, ordered_carriers = build_svd_ordered_operators(train, vocab_size)
    print(f"      Norms: {np.linalg.norm(ordered_ops, axis=1).mean():.4f}")

    # Run all variants
    train_size = 10000
    max_test = 1500
    test_capped = test[:max_test]
    print(f"\n--- Running experiments (train={train_size}, test={max_test}) ---\n")

    print("[1/4] Hash embeddings (baseline)...")
    g1, a1 = run_closure_lm(train[:train_size], test_capped, vocab_size,
                            mode='hash', num_channels=1)
    print(f"       Genome={g1}, Accuracy={a1:.2f}%")

    print("[2/4] SVD direct (fixed norms)...")
    g2, a2 = run_closure_lm(train[:train_size], test_capped, vocab_size,
                            mode='svd', input_ops=svd_input,
                            output_carriers=svd_output, num_channels=1)
    print(f"       Genome={g2}, Accuracy={a2:.2f}%")

    print("[3/4] SVD-ordered operators...")
    g3, a3 = run_closure_lm(train[:train_size], test_capped, vocab_size,
                            mode='svd', input_ops=ordered_ops,
                            output_carriers=ordered_carriers, num_channels=1)
    print(f"       Genome={g3}, Accuracy={a3:.2f}%")

    print("[4/4] SVD-ordered + multi-channel...")
    g4, a4 = run_closure_lm(train[:train_size], test_capped, vocab_size,
                            mode='svd', input_ops=ordered_ops,
                            output_carriers=ordered_carriers, num_channels=2)
    print(f"       Genome={g4}, Accuracy={a4:.2f}%")

    # Results table
    ceiling = 8.1
    print(f"\n{'='*65}")
    print(f"{'Model':<28} | {'Genome':>7} | {'Top-1':>8} | {'vs Ceiling':>10}")
    print(f"{'-'*28}-+-{'-'*7}-+-{'-'*8}-+-{'-'*10}")
    print(f"{'Hash embeddings':<28} | {g1:>7} | {a1:>7.2f}% | {a1-ceiling:>+9.2f} pp")
    print(f"{'SVD direct (fixed)':<28} | {g2:>7} | {a2:>7.2f}% | {a2-ceiling:>+9.2f} pp")
    print(f"{'SVD-ordered operators':<28} | {g3:>7} | {a3:>7.2f}% | {a3-ceiling:>+9.2f} pp")
    print(f"{'SVD-ordered + multichannel':<28} | {g4:>7} | {a4:>7.2f}% | {a4-ceiling:>+9.2f} pp")
    print(f"{'Ceiling (rank-4 SVD)':<28} | {'N/A':>7} | {ceiling:>7.1f}% | {'baseline':>10}")
    print(f"{'='*65}")

    # Hopf analysis on the ordered operators
    analyze_hopf(ordered_ops, vocab_size, token_map)

    print("\nDone.")

if __name__ == '__main__':
    main()

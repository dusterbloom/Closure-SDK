#!/usr/bin/env python3
"""
GeoLLM v6 — closure dynamics
============================
Faltz's full ingest loop, ported to LM. The genome only stores observations
that SURPRISE the model. Predictable transitions silently update cell_c
without writing.

Algorithm:
  cell_c = IDENTITY
  for each train token t:
      carrier = embed(t)
      total   = ZREAD(genome, query=cell_c)              # what the field says
      known   = cell_c                                    # what model predicts
      sigma_pred = sigma(compose(total, inverse(known)))  # prediction tension
      actual_residual = compose(carrier, inverse(known))
      sigma_actual    = sigma(actual_residual)            # observation gap
      if sigma_actual > pi/4:
          # CLOSE: write to genome, update cell_c toward observation
          genome.append((cell_c.copy(), carrier))
          cell_c = slerp(cell_c, carrier, alpha=0.3)
          cell_c = normalize(cell_c)
      else:
          # No closure: cell_c drifts gently
          cell_c = slerp(cell_c, carrier, alpha=0.05)
          cell_c = normalize(cell_c)

Inference is the same v3 ZREAD with full+phase scoring; we just query
against the (much smaller) closure-genome.
"""
import math
import numpy as np
from geollm_core import (
    I_q, hamilton, normalize, inverse, sigma, slerp,
    axis_angle, hopf_decompose, carrier_from_hopf,
    semantic_base_from_bytes, domain_embed,
    OPERATORS, compose_state, load_corpus,
)

def _zread_predict(state, genome_states, genome_targets,
                   pending_states=None, pending_targets=None,
                   cutoff=math.pi/3):
    """ZREAD soft attention. Includes pending (unrestacked) entries so a
    closure that just fired is visible to the very next step's
    prediction; without this, the genome would silently grow until
    restack and bypass the σ-threshold gate."""
    parts_states = [genome_states]
    parts_targets = [genome_targets]
    if pending_states:
        parts_states.append(np.stack(pending_states, axis=0))
        parts_targets.append(np.stack(pending_targets, axis=0))
    states  = np.concatenate(parts_states,  axis=0) if parts_states  else np.zeros((0,4))
    targets = np.concatenate(parts_targets, axis=0) if parts_targets else np.zeros((0,4))
    if len(states) == 0:
        return I_q.copy()
    dots = np.clip(np.abs(states @ state), -1.0, 1.0)
    gaps = np.arccos(dots)
    w = np.cos(gaps); w[gaps > cutoff] = 0.0
    w = np.clip(w, 0.0, 1.0)
    if w.sum() < 1e-12:
        return I_q.copy()
    avg = (w[:, None] * targets).sum(axis=0) / w.sum()
    return normalize(avg)

# ── Closure dynamics: σ-thresholded ingest, ARCHITECTURE-CORRECT ──────
def train_closure(tokens, carriers, ops, wid, threshold=math.pi/4,
                  context_len=3, restack_every=64):
    """Walk tokens. At each step:
       - state = compose(left context of length `context_len`)
       - predicted = ZREAD(genome, query=state)
       - actual    = embed(token)
       - residual  = compose(actual, inverse(predicted))
       - if σ(residual) > threshold: close (write (state, actual) to genome)

    Returns list of (state, actual_carrier) entries.

    `restack_every` controls how often we re-pack the genome into a numpy
    array for fast ZREAD; small values are slow, large values let the
    genome grow stale before its predictions update. 512 is a reasonable
    default for hundreds of thousands of observations."""
    genome = []
    g_states  = np.zeros((0, 4), dtype=np.float64)
    g_targets = np.zeros((0, 4), dtype=np.float64)
    pending_states  = []
    pending_targets = []

    n = len(tokens)
    for i in range(n):
        target = tokens[i]
        if target not in wid:
            continue
        # Allow partial context at the start so closures can fire from step 0
        # (matches the architecture: cell_c starts at identity, first
        # observation is by definition surprising).
        ctx = tokens[max(0, i-context_len):i]
        state = compose_state(ctx, carriers, ops)
        actual = carriers[target]
        predicted = _zread_predict(state, g_states, g_targets,
                                    pending_states, pending_targets)
        residual = hamilton(actual, inverse(predicted))
        sigma_actual = sigma(residual)
        if sigma_actual > threshold:
            genome.append((state.copy(), actual.copy()))
            pending_states.append(state)
            pending_targets.append(actual)
            if len(pending_states) >= restack_every:
                # Re-pack into numpy for fast ZREAD on subsequent steps.
                g_states  = np.concatenate([g_states,  np.stack(pending_states,  0)], axis=0)
                g_targets = np.concatenate([g_targets, np.stack(pending_targets, 0)], axis=0)
                pending_states.clear(); pending_targets.clear()

    # Flush any remaining pendings (so test introspection of the genome
    # length matches what train_closure returned).
    if pending_states:
        g_states  = np.concatenate([g_states,  np.stack(pending_states,  0)], axis=0)
        g_targets = np.concatenate([g_targets, np.stack(pending_targets, 0)], axis=0)
    return genome

# ── Inference: ZREAD-style population read ────────────────────────────
def predict_topk(ctx, genome, carriers, ops, wid, V, k=10, cutoff=math.pi/3):
    """Score next tokens via cos(σ) over the closure-genome (full channel),
    return ids of the top-k by score. Returns [] if genome empty."""
    if not genome:
        return []
    q = compose_state(ctx, carriers, ops)
    # Build an index from carrier-bytes to wid for fast lookup.
    # Genome stores carriers[w].copy(), so exact bytes match is guaranteed.
    cid_to_wid = {}
    for w, c in carriers.items():
        cid_to_wid[c.tobytes()] = wid[w]
    states = np.stack([g[0] for g in genome], axis=0).astype(np.float64)
    nxt_ids = []
    for _, c in genome:
        wid_match = cid_to_wid.get(c.tobytes())
        if wid_match is None:
            raise KeyError(
                f"genome entry carrier not found in carriers dict; "
                f"genome must only contain carriers from the vocab carriers table"
            )
        nxt_ids.append(wid_match)
    N = np.array(nxt_ids, dtype=np.int64)
    # Full-channel σ
    dots = np.clip(np.abs(states @ q), -1.0, 1.0)
    gaps = np.arccos(dots)
    w_full = np.cos(gaps); w_full[gaps > cutoff] = 0.0
    s = np.bincount(N, weights=np.clip(w_full, 0, 1), minlength=V)
    if s.sum() == 0:
        return []
    return list(np.argsort(-s)[:k])

if __name__ == '__main__':
    print("closure_dynamics.py: run test_closure_dynamics.py for tests.")

#!/usr/bin/env python3
"""
geollm_core.py — shared primitives for GeoLLM experiments v0–v6.

Pulled from 11_closure_dynamics.py (the most complete version).
All functions here are used identically across scripts 04, 06, 07, 08, 09, 10, 11.
"""
import math
import hashlib
from pathlib import Path
import numpy as np

# ── Identity quaternion ───────────────────────────────────────────────
I_q = np.array([1.0, 0.0, 0.0, 0.0])

# ── Quaternion ops ────────────────────────────────────────────────────
def hamilton(a, b):
    w1,x1,y1,z1 = a; w2,x2,y2,z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2])

def normalize(v, eps=1e-12):
    n = np.linalg.norm(v); return v if n < eps else v / n

def inverse(q):
    """Conjugate / inverse of a unit quaternion."""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def sigma(q):
    """Geodesic distance from q to identity, in [0, π]."""
    return math.acos(max(-1.0, min(1.0, q[0])))

def slerp(a, b, t):
    """Spherical linear interpolation on S^3."""
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot < 0:  # take shorter arc
        b = -b; dot = -dot
    if dot > 0.9995:
        return normalize(a + t*(b - a))
    omega = math.acos(dot)
    so = math.sin(omega)
    return (math.sin((1-t)*omega)/so) * a + (math.sin(t*omega)/so) * b

def axis_angle(axis, theta):
    a = np.asarray(axis, dtype=np.float64); a = a / (np.linalg.norm(a) + 1e-12)
    c, s = math.cos(theta/2), math.sin(theta/2)
    return np.array([c, s*a[0], s*a[1], s*a[2]])

# ── Hopf / domain embed ───────────────────────────────────────────────
def hopf_decompose(q):
    w,x,y,z = q
    base = np.array([2*(x*z+w*y), 2*(y*z-w*x), w*w+z*z-x*x-y*y])
    n = np.linalg.norm(base); base = base/n if n > 1e-15 else np.array([0.,0.,1.])
    phase = (math.atan2(z, w) + math.atan2(x, y)) % (2*math.pi)
    return base, phase

def carrier_from_hopf(base, phase):
    base = base / (np.linalg.norm(base) + 1e-12); phase = phase % (2*math.pi)
    bz = max(-1.0, min(1.0, base[2]))
    sin_t = math.sqrt(max(0.0, 1.0 - bz*bz))
    if sin_t < 1e-12:
        if bz >= 0: return np.array([math.cos(phase),0,0,math.sin(phase)])
        return np.array([0, math.sin(phase), math.cos(phase), 0])
    theta = math.acos(bz); delta = math.atan2(base[1], base[0])
    alpha = 0.5*(phase+delta); beta = 0.5*(phase-delta)
    ct, st = math.cos(0.5*theta), math.sin(0.5*theta)
    q = np.array([ct*math.cos(alpha), st*math.sin(beta),
                  st*math.cos(beta),  ct*math.sin(alpha)])
    return q / np.linalg.norm(q)

def semantic_base_from_bytes(data):
    h = hashlib.sha256(data).digest()
    u1 = (int.from_bytes(h[:8],'little')+1)/(2**64+1)
    u2 = (int.from_bytes(h[8:16],'little')+1)/(2**64+1)
    z = 1.0 - 2.0*u1; r = math.sqrt(max(0., 1-z*z)); phi = 2*math.pi*u2
    return np.array([r*math.cos(phi), r*math.sin(phi), z])

def domain_embed(token):
    return carrier_from_hopf(semantic_base_from_bytes(token.encode('utf-8')), 0.0)

# ── Operators (55-entry Zipfian operator table) ───────────────────────
R_i_pi  = axis_angle([1,0,0], math.pi)
R_j_pi  = axis_angle([0,1,0], math.pi)
R_j_pi2 = axis_angle([0,1,0], math.pi/2)
R_k_pi2 = axis_angle([0,0,1], math.pi/2)
R_diag  = axis_angle([1,1,1],  2*math.pi/3)
R_diag2 = axis_angle([1,1,1], -2*math.pi/3)

OPERATORS = {
    'is':I_q,'are':I_q,'was':I_q,'were':I_q,'be':I_q,'am':I_q,'been':I_q,'being':I_q,
    'the':I_q,'a':I_q,'an':I_q,
    'not':R_i_pi,"n't":R_i_pi,'no':R_i_pi,'never':R_i_pi,
    'and':R_j_pi2,'or':R_k_pi2,'but':R_j_pi,
    'of':I_q,'to':I_q,'in':I_q,'on':I_q,'at':I_q,'for':I_q,'with':I_q,
    'by':I_q,'from':I_q,'as':I_q,'into':I_q,'about':I_q,
    'i':I_q,'you':I_q,'he':I_q,'she':I_q,'it':I_q,'we':I_q,'they':I_q,
    'this':I_q,'that':I_q,'these':I_q,'those':I_q,
    'will':R_diag,'would':R_diag2,'shall':R_diag,
    'can':R_diag,'could':R_diag2,'may':R_diag,'might':R_diag2,
    'should':R_diag2,'must':R_diag,
    'have':I_q,'has':I_q,'had':I_q,
    'do':I_q,'does':I_q,'did':I_q,
}

# ── Composition (used identically in 07, 08, 09, 10, 11) ─────────────
def compose_state(ctx_tokens, carriers, ops):
    q = I_q.copy()
    for t in ctx_tokens:
        if t in ops:    q = hamilton(ops[t], q)
        elif t in carriers: q = hamilton(q, carriers[t])
        q = normalize(q)
    return q

# ── Corpus loader ─────────────────────────────────────────────────────
def load_corpus():
    here = Path(__file__).parent / 'corpora'
    parts = []
    for p in [here/'shakespeare.txt', here/'pride_prejudice.txt']:
        if p.exists(): parts.append(p.read_text(errors='ignore').lower())
    return ' '.join(parts)

# ── Fixed test-position sampler (deterministic across all variants) ───
def fixed_test_positions(test_toks, wid, context_len, n_eval, seed=42):
    """Deterministic sample of test positions across all variants."""
    valid = [t for t in test_toks if t in wid]
    if len(valid) <= context_len: return [], []
    import random
    rng = random.Random(seed)
    all_pos = list(range(context_len, len(valid)))
    if len(all_pos) > n_eval: pos = rng.sample(all_pos, n_eval)
    else: pos = all_pos
    return valid, sorted(pos)

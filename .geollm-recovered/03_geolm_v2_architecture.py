"""
GeoLM_v2: The Scalable Geometric Language Model
================================================
Implementing the seven missing pieces that connect basic primitives
(Hamilton product, σ) to a competitive LM architecture.

The thread:
  Transformer attention is the LINEAR APPROXIMATION of Hamilton composition
  in the small-rotation limit. The geometric form is the more complete object.

The seven missing links:
  1. DEPTH — stack N GeoBlocks for iterative refinement
  2. GEOMETRIC ATTENTION — softmax over S³ alignment, rotation by context
  3. CONTEXT-DEPENDENT ROTATION — R(context) ⊗ state ⊗ R(context)*
  4. SLERP RESIDUAL — smooth bypass on the sphere
  5. MULTI-HEAD as (S³)^H — parallel quaternions per token
  6. RESONATE retrieval — nearest-anchor on a 24-cell genome
  7. HOPF DECOMPOSITION — semantic S² + syntactic S¹

Built on the existing primitives:
  - quaternion.py:   hamilton_product, quaternion_normalize, QuaternionLinear
  - quaternion_v2.py: hopf_project, slerp, quaternion_exp, quaternion_log
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Primitives (vectorized for speed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def hamilton(a, b):
    """Hamilton product on the last dim. a, b: (..., 4)."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dim=-1)


def conjugate(q):
    """Quaternion conjugate. q: (..., 4)."""
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)


def normalize_q(q, eps=1e-8):
    """Normalize each quaternion to S³."""
    return F.normalize(q, dim=-1, eps=eps)


def hamilton_h(a, b):
    """Per-head Hamilton: a, b shape (..., H, 4)."""
    return hamilton(a, b)


def slerp(p, q, t, eps=1e-7):
    """
    Spherical linear interpolation on S³.
    p, q: (..., 4) unit quaternions. t: (...) or (..., 1) ∈ [0, 1].
    """
    p = normalize_q(p)
    q = normalize_q(q)
    dot = (p * q).sum(dim=-1, keepdim=True)
    # Take shortest path
    q = torch.where(dot < 0, -q, q)
    dot = dot.abs().clamp(0, 1 - eps)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    if t.dim() < p.dim():
        t = t.unsqueeze(-1)
    s_p = torch.sin((1 - t) * theta) / sin_theta.clamp(min=eps)
    s_q = torch.sin(t * theta) / sin_theta.clamp(min=eps)
    # Fall back to lerp for very small angles
    small = sin_theta < eps
    s_p = torch.where(small, 1 - t, s_p)
    s_q = torch.where(small, t, s_q)
    return normalize_q(s_p * p + s_q * q)


def hopf_decompose(q):
    """
    Hopf fibration: q ∈ S³ → (s2_point, s1_phase).
    s2_point: (..., 3) on S²    — semantic direction
    s1_phase: (..., 1) on S¹    — syntactic phase
    """
    w, x, y, z = q.unbind(-1)
    s2 = torch.stack([
        2 * (x*z + w*y),
        2 * (y*z - w*x),
        w*w + z*z - x*x - y*y,
    ], dim=-1)
    s2 = F.normalize(s2, dim=-1, eps=1e-8)
    phi = torch.atan2(y*w + x*z + 1e-12, w*z - x*y + 1e-12).unsqueeze(-1)
    return s2, phi


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Link 1+5: Multi-head quaternion embedding (S³)^H
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class QuaternionEmbedding(nn.Module):
    """Token → (S³)^H: H quaternion 'heads' per token."""
    def __init__(self, vocab_size, n_heads=8):
        super().__init__()
        self.V = vocab_size
        self.H = n_heads
        # Initialize near identity but random
        emb = torch.randn(vocab_size, n_heads, 4) * 0.1
        emb[..., 0] += 1.0  # bias toward identity quaternion
        self.weight = nn.Parameter(normalize_q(emb))

    def normalized(self):
        return normalize_q(self.weight)  # (V, H, 4)

    def forward(self, ids):
        return self.normalized()[ids]  # (B, T, H, 4)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Link 2+3: Geometric Attention
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GeoAttention(nn.Module):
    """
    Multi-head geometric attention.

    For each query position t and head h:
      1. Get Q, K, V quaternions by rotating x with learned per-head rotations.
      2. Affinity(s, t, h) = ⟨Q_t,h, K_s,h⟩ on S³ (cosine).
      3. Causal-masked softmax over keys → weights w(s).
      4. Compose values via SLERP from identity, weighted by attention:
            update_t,h = ⊗_s SLERP(I, V_s,h, w(s))
         (sequential composition; in practice we approximate via weighted
         average + renormalize, which is the small-rotation limit.)
      5. Apply update as a rotation: x'_t,h = update ⊗ x_t,h ⊗ update*
    """
    def __init__(self, n_heads, dropout=0.0):
        super().__init__()
        self.H = n_heads
        # Per-head learnable rotation quaternions for Q, K, V projections
        # These play the role of W_q, W_k, W_v but as rotations (not linear maps)
        self.r_q = nn.Parameter(self._init_rotations(n_heads))
        self.r_k = nn.Parameter(self._init_rotations(n_heads))
        self.r_v = nn.Parameter(self._init_rotations(n_heads))
        self.dropout = nn.Dropout(dropout)

    def _init_rotations(self, H):
        r = torch.randn(H, 4) * 0.05
        r[..., 0] += 1.0  # near identity
        return normalize_q(r)

    def forward(self, x, mask=None):
        """
        x: (B, T, H, 4)  — sequence of multi-head quaternions
        mask: (T, T) causal mask, optional
        Returns: (B, T, H, 4)
        """
        B, T, H, _ = x.shape

        # Per-head rotations: shape (H, 4) → (1, 1, H, 4)
        r_q = normalize_q(self.r_q).view(1, 1, H, 4)
        r_k = normalize_q(self.r_k).view(1, 1, H, 4)
        r_v = normalize_q(self.r_v).view(1, 1, H, 4)

        # Q = r_q ⊗ x ⊗ r_q*    (rotation of x by r_q)
        Q = hamilton(hamilton(r_q.expand(B, T, H, 4), x), conjugate(r_q.expand(B, T, H, 4)))
        K = hamilton(hamilton(r_k.expand(B, T, H, 4), x), conjugate(r_k.expand(B, T, H, 4)))
        V = hamilton(hamilton(r_v.expand(B, T, H, 4), x), conjugate(r_v.expand(B, T, H, 4)))
        Q, K, V = normalize_q(Q), normalize_q(K), normalize_q(V)

        # Affinity: (B, H, T, T) — cosine similarity between Q and K per head
        # Q: (B, T, H, 4) → (B, H, T, 4), K: (B, H, T, 4) → transpose for matmul
        Qh = Q.permute(0, 2, 1, 3)  # (B, H, T, 4)
        Kh = K.permute(0, 2, 1, 3)  # (B, H, T, 4)
        Vh = V.permute(0, 2, 1, 3)
        affinity = torch.einsum('bhtd,bhsd->bhts', Qh, Kh)  # (B, H, T_q, T_k)
        affinity = affinity / math.sqrt(4)  # √d scaling like transformers

        if mask is not None:
            affinity = affinity.masked_fill(mask == 0, -1e4)

        weights = F.softmax(affinity, dim=-1)  # (B, H, T, T)
        weights = self.dropout(weights)

        # Geometric aggregation: weighted "rotation-toward-V_s" for each token
        # The exact form is sequential SLERP composition. In the small-rotation
        # limit (which is where transformer attention lives), this becomes:
        #     update_t = normalize( Σ_s w(s) * V_s )
        # We use this approximation for tractability and add a SLERP residual
        # at the end to preserve geometric structure.
        update = torch.einsum('bhts,bhsd->bhtd', weights, Vh)  # (B, H, T, 4)
        update = normalize_q(update)

        # Apply update as rotation: x'_t = update_t ⊗ x_t ⊗ update_t*
        x_h = x.permute(0, 2, 1, 3)  # (B, H, T, 4)
        update_conj = conjugate(update)
        x_rot = hamilton(hamilton(update, x_h), update_conj)
        x_rot = normalize_q(x_rot)

        return x_rot.permute(0, 2, 1, 3)  # back to (B, T, H, 4)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GeoFFN — Hamilton-product MLP using QuaternionLinear from existing code
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class QuaternionLinear(nn.Module):
    """
    Linear layer with quaternion structure (as in GeoLLM v1).
    Takes quaternion-stacked input (B, T, H_in*4) and produces (B, T, H_out*4).
    Uses 4 weight matrices to construct a Hamilton-structured matrix that
    has 4× fewer parameters than nn.Linear.
    """
    def __init__(self, in_heads, out_heads, bias=True):
        super().__init__()
        self.in_h = in_heads
        self.out_h = out_heads
        std = 1.0 / math.sqrt(2.0 * in_heads)
        self.W_r = nn.Parameter(torch.randn(out_heads, in_heads) * std)
        self.W_i = nn.Parameter(torch.randn(out_heads, in_heads) * std)
        self.W_j = nn.Parameter(torch.randn(out_heads, in_heads) * std)
        self.W_k = nn.Parameter(torch.randn(out_heads, in_heads) * std)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_heads * 4))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        """x: (..., in_heads, 4) → (..., out_heads, 4)"""
        oh, ih = self.out_h, self.in_h
        H = torch.zeros(oh*4, ih*4, device=x.device, dtype=x.dtype)
        H[0*oh:1*oh, 0*ih:1*ih] =  self.W_r
        H[0*oh:1*oh, 1*ih:2*ih] = -self.W_i
        H[0*oh:1*oh, 2*ih:3*ih] = -self.W_j
        H[0*oh:1*oh, 3*ih:4*ih] = -self.W_k
        H[1*oh:2*oh, 0*ih:1*ih] =  self.W_i
        H[1*oh:2*oh, 1*ih:2*ih] =  self.W_r
        H[1*oh:2*oh, 2*ih:3*ih] =  self.W_k
        H[1*oh:2*oh, 3*ih:4*ih] = -self.W_j
        H[2*oh:3*oh, 0*ih:1*ih] =  self.W_j
        H[2*oh:3*oh, 1*ih:2*ih] = -self.W_k
        H[2*oh:3*oh, 2*ih:3*ih] =  self.W_r
        H[2*oh:3*oh, 3*ih:4*ih] =  self.W_i
        H[3*oh:4*oh, 0*ih:1*ih] =  self.W_k
        H[3*oh:4*oh, 1*ih:2*ih] =  self.W_j
        H[3*oh:4*oh, 2*ih:3*ih] = -self.W_i
        H[3*oh:4*oh, 3*ih:4*ih] =  self.W_r

        # Reshape (..., H, 4) → (..., H*4) → linear → (..., H*4) → (..., H, 4)
        # The (w, x, y, z) channel layout becomes 4 separate slabs of size H
        # for the matrix multiplication.
        shape = x.shape
        x_flat = x.reshape(*shape[:-2], 4 * self.in_h)
        # Permute (..., H*4) into (..., 4*H) by separating w,x,y,z components
        x_re = x.reshape(*shape[:-2], self.in_h, 4)
        x_w, x_x, x_y, x_z = x_re.unbind(-1)
        x_perm = torch.cat([x_w, x_x, x_y, x_z], dim=-1)  # (..., 4*H_in)
        out_perm = F.linear(x_perm, H)  # (..., 4*H_out)
        # Split back
        ow, ox, oy, oz = out_perm.split(self.out_h, dim=-1)
        out = torch.stack([ow, ox, oy, oz], dim=-1)  # (..., H_out, 4)
        if self.bias is not None:
            out = out + self.bias.view(self.out_h, 4)
        return out


class GeoFFN(nn.Module):
    """
    Geometric feed-forward: QuaternionLinear → ReLU → QuaternionLinear,
    then renormalize to S³ per head.
    """
    def __init__(self, n_heads, expand=4):
        super().__init__()
        self.l1 = QuaternionLinear(n_heads, n_heads * expand)
        self.l2 = QuaternionLinear(n_heads * expand, n_heads)

    def forward(self, x):
        # x: (B, T, H, 4)
        h = self.l1(x)
        h = F.relu(h)
        h = self.l2(h)
        return normalize_q(h)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Link 4: GeoBlock with SLERP residuals
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GeoBlock(nn.Module):
    """
    Attention + FFN with SLERP-based residual connections.
    Each residual is a learnable per-head SLERP factor in [0, 1].
    """
    def __init__(self, n_heads, ffn_expand=4, dropout=0.0):
        super().__init__()
        self.attn = GeoAttention(n_heads, dropout=dropout)
        self.ffn = GeoFFN(n_heads, expand=ffn_expand)
        # Learnable per-head SLERP "alpha" — controls residual strength
        self.slerp_attn = nn.Parameter(torch.ones(n_heads) * 0.5)
        self.slerp_ffn = nn.Parameter(torch.ones(n_heads) * 0.5)

    def forward(self, x, mask=None):
        # Attention sublayer
        attn_out = self.attn(x, mask=mask)
        a_t = torch.sigmoid(self.slerp_attn).view(1, 1, -1)  # (1, 1, H)
        x = slerp(x, attn_out, a_t)

        # FFN sublayer
        ffn_out = self.ffn(x)
        f_t = torch.sigmoid(self.slerp_ffn).view(1, 1, -1)
        x = slerp(x, ffn_out, f_t)
        return x


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Link 6: RESONATE — nearest-anchor retrieval on a 24-cell genome
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_24cell():
    """24-cell vertices on S³: the Hurwitz units."""
    verts = []
    for i in range(4):
        for s in [1.0, -1.0]:
            v = [0.0]*4; v[i] = s; verts.append(v)
    h = 0.5
    for s0 in [h, -h]:
        for s1 in [h, -h]:
            for s2 in [h, -h]:
                for s3 in [h, -h]:
                    verts.append([s0, s1, s2, s3])
    return torch.tensor(verts, dtype=torch.float32)  # (24, 4)


class Resonate(nn.Module):
    """
    Retrieval over a learnable codebook of 'anchor' quaternions.
    Initialized to the 24-cell vertices for symmetry.
    Query → softmax-weighted blend of anchor values via SLERP composition.
    """
    def __init__(self, n_heads, n_anchors=64):
        super().__init__()
        cell = make_24cell()  # (24, 4)
        # Tile to fill n_anchors × n_heads
        anchors = cell[torch.randint(24, (n_anchors,))].unsqueeze(0).repeat(n_heads, 1, 1)
        # add small noise to break symmetry
        anchors = anchors + 0.01 * torch.randn_like(anchors)
        self.anchors_k = nn.Parameter(normalize_q(anchors))  # (H, A, 4) keys
        self.anchors_v = nn.Parameter(normalize_q(anchors.clone()))  # values

    def forward(self, x):
        """
        x: (B, T, H, 4) — query quaternions
        Returns: (B, T, H, 4) retrieved quaternions
        """
        B, T, H, _ = x.shape
        keys = normalize_q(self.anchors_k)    # (H, A, 4)
        vals = normalize_q(self.anchors_v)    # (H, A, 4)

        # Affinity: (B, T, H, A) — cosine on S³
        # x: (B, T, H, 4), keys: (H, A, 4)
        affinity = torch.einsum('bthd,had->btha', x, keys)
        weights = F.softmax(affinity * math.sqrt(4), dim=-1)  # (B, T, H, A)

        # Retrieve: weighted composition of values
        retrieved = torch.einsum('btha,had->bthd', weights, vals)
        return normalize_q(retrieved)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Link 7: Hopf head — explicitly decomposes for semantic vs syntactic
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class HopfProjector(nn.Module):
    """
    Decompose final state via Hopf fibration into S² (semantics) + S¹ (syntax).
    Used optionally as inductive bias: separate prediction streams for
    'what word' and 'how it functions'.
    """
    def forward(self, x):
        s2, phi = hopf_decompose(x)
        return s2, phi  # (..., 3), (..., 1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# The full model: GeoLM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GeoLM(nn.Module):
    """
    Scalable Geometric Language Model.

    Total parameters per token:
      vocab_size × n_heads × 4    (embedding)
      + transformer-style blocks but using quaternion structure
      
    For comparison, a standard transformer has roughly:
      vocab_size × d_model         (embedding)
      + n_layers × (4 × d_model² + 8 × d_model²)  (per block)
      
    GeoLM's QuaternionLinear has 4× fewer params at the same expressive power,
    so total params are roughly 4× smaller.
    """
    def __init__(self, vocab_size, n_heads=8, n_layers=6,
                 ffn_expand=4, max_len=512, use_resonate=True,
                 dropout=0.0):
        super().__init__()
        self.V = vocab_size
        self.H = n_heads
        self.L = n_layers
        self.max_len = max_len

        self.embedding = QuaternionEmbedding(vocab_size, n_heads)

        # Positional encoding as learned per-position rotations
        # (each position rotates the embedding by a learned amount per head)
        pos = torch.randn(max_len, n_heads, 4) * 0.05
        pos[..., 0] += 1.0
        self.pos_rotation = nn.Parameter(normalize_q(pos))

        self.blocks = nn.ModuleList([
            GeoBlock(n_heads, ffn_expand=ffn_expand, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.resonate = Resonate(n_heads) if use_resonate else None

        # Causal mask cache
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0),
            persistent=False
        )

    def forward(self, ctx_ids, target_ids=None):
        """
        ctx_ids: (B, T) token IDs
        target_ids: (B, T) for full-sequence training, or (B,) for last-token only
        Returns: loss, logits  if target_ids is given
                 logits        otherwise
        """
        B, T = ctx_ids.shape
        x = self.embedding(ctx_ids)  # (B, T, H, 4)

        # Apply positional rotation: x'_t = pos_t ⊗ x_t ⊗ pos_t*
        pos = normalize_q(self.pos_rotation[:T]).unsqueeze(0)  # (1, T, H, 4)
        pos_e = pos.expand(B, T, self.H, 4)
        x = hamilton(hamilton(pos_e, x), conjugate(pos_e))
        x = normalize_q(x)

        # Causal mask
        mask = self.causal_mask[:, :, :T, :T]

        # Stack of GeoBlocks
        for block in self.blocks:
            x = block(x, mask=mask)

        # Optional: RESONATE retrieval at the top
        # SLERP between attended state and retrieved memory
        if self.resonate is not None:
            r = self.resonate(x)
            x = slerp(x, r, torch.tensor(0.3, device=x.device))

        # Predict via σ to vocabulary
        # For each candidate w in vocab, compute σ(state ⊗ q_w) per head, average
        E = self.embedding.normalized()  # (V, H, 4)
        # state at position t: (B, T, H, 4)
        # composed_w = state_t ⊗ q_w for each w
        # We need: (B, T, V, H, 4) — too large at full vocab
        # Approximation: for each (B, T, H), compute σ for all V via
        #   affinity = state · q_w = first component of state ⊗ q_w (the scalar)
        # In quaternion math, the scalar of (a ⊗ b) = a·b - sum of vector parts
        # Specifically: (a ⊗ b)_w = aw*bw - ax*bx - ay*by - az*bz
        # But conjugate identity is needed for distance, so we use:
        # σ(state ⊗ inv(q_w)) = arccos(|scalar(state ⊗ conj(q_w))|)
        E_conj = conjugate(E)  # (V, H, 4)
        # state: (B, T, H, 4) → (B, T, 1, H, 4)
        state_e = x.unsqueeze(2)  # (B, T, 1, H, 4)
        E_e = E_conj.unsqueeze(0).unsqueeze(0)  # (1, 1, V, H, 4)
        composed = hamilton(state_e, E_e)  # (B, T, V, H, 4)
        composed = normalize_q(composed)
        # Scalar (first component) per head
        scalars = composed[..., 0]  # (B, T, V, H)
        # σ_h = arccos(|scalar_h|), average over heads
        sigmas = torch.acos(scalars.abs().clamp(0, 1 - 1e-6))
        sigma_total = sigmas.mean(dim=-1)  # (B, T, V)

        # Logits: lower σ = higher likelihood
        logits = -sigma_total

        if target_ids is None:
            return logits

        if target_ids.dim() == 1:
            # Predict last token only
            loss = F.cross_entropy(logits[:, -1], target_ids)
        else:
            # Full-sequence (next-token at every position)
            loss = F.cross_entropy(
                logits.reshape(-1, self.V),
                target_ids.reshape(-1)
            )
        return loss, logits


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Riemannian projection helper for training
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def project_grad_to_tangent(model):
    """
    For all unit-quaternion parameters, project gradient to tangent plane
    of S³ (i.e. remove radial component). Call after .backward(), before .step().
    Then renormalize after step.
    """
    quat_param_names = ['embedding.weight', 'pos_rotation',
                         'r_q', 'r_k', 'r_v',
                         'anchors_k', 'anchors_v']
    for name, p in model.named_parameters():
        if any(s in name for s in quat_param_names) and p.grad is not None:
            with torch.no_grad():
                # Reshape to (..., 4) blocks, normalize, project
                shape = p.shape
                if shape[-1] == 4:
                    pn = normalize_q(p.data)
                    radial = (p.grad * pn).sum(dim=-1, keepdim=True) * pn
                    p.grad = p.grad - radial


def renormalize_quat_params(model):
    """Project all quaternion parameters back to S³ after optimizer step."""
    quat_param_names = ['embedding.weight', 'pos_rotation',
                         'r_q', 'r_k', 'r_v',
                         'anchors_k', 'anchors_v']
    for name, p in model.named_parameters():
        if any(s in name for s in quat_param_names):
            with torch.no_grad():
                if p.shape[-1] == 4:
                    p.data = normalize_q(p.data)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sanity check / forward pass test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == '__main__':
    torch.manual_seed(0)

    V = 100
    H = 4
    L = 2
    T = 8
    B = 2

    print(f"GeoLM: V={V}, H={H} heads, L={L} layers, T={T} context, B={B} batch")

    model = GeoLM(vocab_size=V, n_heads=H, n_layers=L,
                   ffn_expand=2, max_len=32)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    # Forward pass
    ctx = torch.randint(0, V, (B, T))
    tgt = torch.randint(0, V, (B, T))
    loss, logits = model(ctx, tgt)
    print(f"Forward OK. Loss: {loss.item():.4f}, logits shape: {logits.shape}")

    # Backward
    loss.backward()
    print(f"Backward OK")

    # Riemannian projection
    project_grad_to_tangent(model)
    print(f"Tangent projection OK")

    # Step (would happen with optimizer, then renormalize)
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.data -= 0.01 * p.grad
    renormalize_quat_params(model)
    print(f"Step + renormalize OK")

    # Verify quaternions are still on S³
    emb = model.embedding.weight
    norms = emb.norm(dim=-1)
    print(f"Embedding quaternion norms after step: "
          f"min={norms.min():.4f}, max={norms.max():.4f}, "
          f"mean={norms.mean():.4f} (should be ~1.0)")

    print("\nArchitecture intact. Ready for training.")

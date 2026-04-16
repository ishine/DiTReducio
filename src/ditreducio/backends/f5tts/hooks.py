"""F5-TTS backend-specific calibration hooks.

Contains only the implementations that are tightly coupled to the F5-TTS
model architecture (attention projection names, batch structure, etc.).
All shared calibration flow is in ``ditreducio.calibration.hooks``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ditreducio.calibration.accessor import TransformerView


# ---------------------------------------------------------------------------
# TransformerView factory
# ---------------------------------------------------------------------------

def make_view(model) -> TransformerView:
    """Create a TransformerView for an F5-TTS CFM model."""
    return TransformerView(
        transformer=model.transformer,
        blocks_fn=lambda t: t.transformer_blocks,
        attn_fn=lambda b: b.attn,
        ff_fn=lambda b: b.ff,
        attn_forward_fn=efficient_attention_forward,
        ff_forward_fn=efficient_ff_forward,
        comparison_fn=lambda x: x,  # F5-TTS uses raw outputs directly
    )


# ---------------------------------------------------------------------------
# F5-TTS specific: pre-calibration hook (diagonal similarity)
# ---------------------------------------------------------------------------

def pre_calibration_hook(module, args, kwargs):
    """Compare model heatmaps with diagonal at each layer and timestep."""
    step = module.step
    x = kwargs["x"]
    mask = kwargs.get("mask", None)

    query = module.to_q(x).to(dtype=torch.bfloat16)
    key = module.to_k(x).to(dtype=torch.bfloat16)

    inner_dim = key.shape[-1]
    attn_weights = query @ key.transpose(-2, -1) / math.sqrt(inner_dim)
    if mask is not None:
        attn_weights = attn_weights.masked_fill(~mask, 0)
    attn_weights = F.softmax(attn_weights, dim=-1)

    _, n, _ = attn_weights.shape
    diagonal_matrix = torch.eye(n, device=attn_weights.device, dtype=attn_weights.dtype)

    attn_weights_cond, attn_weights_uncond = attn_weights.chunk(2, dim=0)
    batch_size = attn_weights_cond.shape[0]
    similarities = []
    for b in range(batch_size):
        attn_mat = attn_weights_cond[b]
        attn_vec = attn_mat.reshape(-1)
        diag_vec = diagonal_matrix.reshape(-1)
        attn_norm = attn_vec / torch.norm(attn_vec)
        diag_norm = diag_vec / torch.norm(diag_vec)
        sim = torch.dot(attn_norm, diag_norm)
        similarities.append(sim.item())

    similarity_ts = sum(similarities) / len(similarities)
    if not hasattr(module, "diagonal_similarities"):
        module.diagonal_similarities = {}
    module.diagonal_similarities[step] = similarity_ts
    module.step += 1


# ---------------------------------------------------------------------------
# F5-TTS specific: efficient forward implementations
# ---------------------------------------------------------------------------

def efficient_attention_forward(
    self,
    x: float,
    c: float = None,
    mask: bool | None = None,
    rope=None,
    c_rope=None,
):
    """Efficient attention with TS/BS strategies for F5-TTS."""
    method = self.steps_method[self.step]

    # TS: Temporal Skipping
    if "TS" in method:
        self.step += 1
        return self.cached_output

    batch_size = x.shape[0]
    bs_mode = getattr(self, "bs_mode", "residual")

    # BS: Branch Skipping
    if "BS" in method:
        batch_size //= 2
        if bs_mode == "uncond_replace":
            x = x[batch_size:]
        else:
            x = x[:batch_size]

    x = self.processor(self, x, mask=mask, rope=rope)

    if "BS" in method:
        if bs_mode in {"cond_replace", "uncond_replace"}:
            x = torch.cat((x, x), dim=0)
        else:
            assert self.cached_output is not None
            x = self.cached_output - self.cached_output[:batch_size] + x

    if self.need_cache_output[self.step]:
        self.cached_output = x

    self.step += 1
    return x


def efficient_ff_forward(self, x):
    """Efficient feed-forward with TS/BS strategies for F5-TTS."""
    method = self.steps_method[self.step]
    if "TS" in method:
        self.step += 1
        return self.cached_output
    elif "NONE" in method or "BS" in method:
        out = self.ff(x)
        if self.need_cache_output[self.step]:
            self.cached_output = out
        self.step += 1
        return out
    else:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# F5-TTS specific: FLOPs hooks
# ---------------------------------------------------------------------------

def calculate_flops_hook(module, args, kwargs):
    """FLOPs tracking hook for F5-TTS attention modules."""
    hidden_states = kwargs["x"]
    batch_size, seq_len, dim = hidden_states.shape

    base_ops = (
        seq_len * seq_len * module.heads * batch_size * dim // module.heads
        + seq_len * dim * batch_size * seq_len
    )
    module.full_ops += base_ops

    method = module.steps_method[module.step]
    if method == "BS":
        base_ops = base_ops / 2
    elif method == "TS":
        base_ops = 0

    module.efficient_ops += base_ops


def calculate_ff_flops_hook(module, args, kwargs):
    """FLOPs tracking hook for F5-TTS feed-forward modules."""
    hidden_states = args[0]
    batch_size, seq_len, dim = hidden_states.shape
    project_in = module.ff[0]
    first_linear = project_in[0]
    inner_dim = first_linear.out_features

    base_ops = (
        batch_size * seq_len * dim * inner_dim
        + batch_size * seq_len * inner_dim * dim
    )
    module.full_ops += base_ops

    method = module.steps_method[module.step]
    if method == "TS":
        base_ops = 0
    elif method == "BS":
        base_ops *= 0.5

    module.efficient_ops += base_ops

"""MegaTTS 3 backend-specific calibration hooks.

Contains only the implementations that are tightly coupled to the MegaTTS 3
model architecture (attention projection names, 3-way batch structure, etc.).
All shared calibration flow is in ``ditreducio.calibration.hooks``.
"""

from __future__ import annotations

import math
import types

import torch
import torch.nn.functional as F

from ditreducio.calibration.accessor import TransformerView


# ---------------------------------------------------------------------------
# TransformerView factory
# ---------------------------------------------------------------------------

def make_view(model) -> TransformerView:
    """Create a TransformerView for a MegaTTS 3 inference model."""
    return TransformerView(
        transformer=model.dit.encoder,
        blocks_fn=lambda t: t.layers,
        attn_fn=lambda b: b.attention,
        ff_fn=lambda b: b.feed_forward,
        attn_forward_fn=efficient_attention_forward,
        ff_forward_fn=efficient_ff_forward,
        comparison_fn=_cfg_weighted_output,
    )


def _cfg_weighted_output(raw_outputs: torch.Tensor) -> torch.Tensor:
    """Apply 3-way CFG weighting to MegaTTS 3 model output."""
    cond_spktxt, condtxt, uncond = raw_outputs.chunk(3, dim=0)
    return uncond + 2.5 * (cond_spktxt - uncond) + 1.6 * (cond_spktxt - condtxt)


# ---------------------------------------------------------------------------
# MegaTTS 3 specific: pre-calibration hook (diagonal similarity)
# ---------------------------------------------------------------------------

def pre_calibration_hook(module, args, kwargs):
    """Compare model heatmaps with diagonal at each layer and timestep."""
    step = module.step
    x = args[0]
    start_pos = args[1] if len(args) > 1 else kwargs.get("start_pos", 0)

    bsz, seqlen, _ = x.shape

    xq = module.wq(x)
    xk = module.wk(x)
    xq_64 = xq.to(torch.float64)
    xk_64 = xk.to(torch.float64)
    scale = 1.0 / math.sqrt(module.head_dim)
    attn_scores = torch.matmul(xq_64, xk_64.transpose(-2, -1)) * scale
    attn_scores_stable = attn_scores - torch.max(attn_scores, dim=-1, keepdim=True)[0]
    attn_weights = F.softmax(attn_scores_stable, dim=-1)

    if torch.isnan(attn_weights).any():
        print(f"Warning: NaN detected after softmax at step {step}")
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    diagonal_matrix = torch.eye(seqlen, device=attn_weights.device, dtype=attn_weights.dtype)

    attn_weights_cond, attn_weights_uncond = attn_weights.chunk(2, dim=0)
    batch_size = attn_weights_cond.shape[0]
    similarities = []

    for b in range(batch_size):
        attn_mat = attn_weights_cond[b]
        attn_vec = attn_mat.reshape(-1)
        diag_vec = diagonal_matrix.reshape(-1)
        attn_norm = attn_vec / (torch.norm(attn_vec) + 1e-6)
        diag_norm = diag_vec / (torch.norm(diag_vec) + 1e-6)
        sim = torch.dot(attn_norm, diag_norm)
        similarities.append(sim.item())

    similarity_ts = sum(similarities) / len(similarities)

    if not hasattr(module, "diagonal_similarities"):
        module.diagonal_similarities = {}
    module.diagonal_similarities[step] = similarity_ts
    module.step += 1


# ---------------------------------------------------------------------------
# MegaTTS 3 specific: efficient forward implementations
# ---------------------------------------------------------------------------

def efficient_attention_forward(self, x: float, start_pos: int, freqs_cis, mask):
    """Efficient attention with TS/BS strategies for MegaTTS 3."""
    from tts.modules.llm_dit.transformer import apply_rotary_emb

    method = self.steps_method[self.step]
    batch_size, seq_len, _ = x.shape
    bs_mode = getattr(self, "bs_mode", "residual")

    if "TS" in method:
        assert self.cached_output is not None
        self.step += 1
        return self.cached_output

    if "BS" in method:
        batch_size //= 3
        if bs_mode == "uncond_replace":
            x = x[batch_size : 2 * batch_size]
            mask = mask[batch_size : 2 * batch_size] if mask is not None else None
        else:
            x = x[:batch_size]
            mask = mask[:batch_size] if mask is not None else None

    query = self.wq(x)
    key = self.wk(x)
    value = self.wv(x)

    query = query.view(batch_size, -1, self.n_local_heads, self.head_dim)
    key = key.view(batch_size, -1, self.n_local_kv_heads, self.head_dim)
    value = value.view(batch_size, -1, self.n_local_kv_heads, self.head_dim)

    query, key = apply_rotary_emb(query, key, freqs_cis=freqs_cis)

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    output = F.scaled_dot_product_attention(
        query, key, value, mask[:, None, None, :], is_causal=False
    )
    output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

    x = self.wo(output)

    if "BS" in method:
        if bs_mode in {"cond_replace", "uncond_replace"}:
            x = torch.cat((x, x, x), dim=0)
        else:
            x = self.cached_output - self.cached_output[:batch_size] + x

    if self.need_cache_output[self.step]:
        self.cached_output = x

    self.step += 1
    return x


def efficient_ff_forward(self, x):
    """Efficient feed-forward with TS/BS strategies for MegaTTS 3."""
    method = self.steps_method[self.step]
    if "TS" in method:
        self.step += 1
        return self.cached_output
    elif "BS" in method:
        batch_single = x.shape[0] // 3
        bs_mode = getattr(self, "bs_mode", "residual")
        if bs_mode == "uncond_replace":
            branch = x[batch_single : 2 * batch_single]
        else:
            branch = x[:batch_single]

        out_branch = self.w2(F.silu(self.w1(branch)))
        if bs_mode in {"cond_replace", "uncond_replace"}:
            full_output = torch.cat((out_branch, out_branch, out_branch), dim=0)
        else:
            full_output = (
                self.cached_output - self.cached_output[:batch_single] + out_branch
            )
        if self.need_cache_output[self.step]:
            self.cached_output = full_output
        self.step += 1
        return full_output
    elif "NONE" in method:
        out = self.w2(F.silu(self.w1(x)))
        if self.need_cache_output[self.step]:
            self.cached_output = out
        self.step += 1
        return out
    else:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# MegaTTS 3 specific: FLOPs hooks
# ---------------------------------------------------------------------------

def calculate_flops_hook(module, args, kwargs):
    """FLOPs tracking hook for MegaTTS 3 attention modules."""
    hidden_states = kwargs["x"]
    batch_size, seq_len, dim = hidden_states.shape

    base_ops = (
        seq_len
        * seq_len
        * module.n_local_heads
        * batch_size
        * dim
        // module.n_local_heads
        + seq_len * dim * batch_size * seq_len
    )
    module.full_ops += base_ops

    method = module.steps_method[module.step]
    if method == "BS":
        base_ops = base_ops / 3
    elif method == "TS":
        base_ops = 0

    module.efficient_ops += base_ops


def calculate_ff_flops_hook(module, args, kwargs):
    """FLOPs tracking hook for MegaTTS 3 feed-forward modules."""
    hidden_states = args[0]
    batch_size, seq_len, dim = hidden_states.shape
    inner_dim = module.w1.out_features

    base_ops = (
        batch_size * seq_len * dim * inner_dim
        + batch_size * seq_len * inner_dim
        + batch_size * seq_len * inner_dim * dim
    )
    module.full_ops += base_ops

    method = module.steps_method[module.step]
    if method == "TS":
        base_ops = 0
    elif method == "BS":
        base_ops = base_ops // 2

    module.efficient_ops += base_ops

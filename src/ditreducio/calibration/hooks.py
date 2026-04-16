"""Shared calibration hooks for DiTReducio.

All calibration flow functions that were previously duplicated across
backends/f5tts and backends/megatts3 are unified here.  Backend-specific
differences (attribute names, batch structure, efficient-forward implementations)
are abstracted via :class:`TransformerView`.
"""

from __future__ import annotations

import json
import types
from typing import Any

import torch

from ditreducio.calibration.accessor import TransformerView
from ditreducio.calibration.metrics import compression_loss
from ditreducio.calibration.timer import cuda_timer


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------

def build_need_cache_output(method_table: list[list[str]]) -> list[list[bool]]:
    """Build a flag table indicating which (layer, step) pairs need cached output.

    For any step that applies TS or BS, the last preceding non-TS step must
    cache its output so the TS/BS step can reuse it.
    """
    n_layers = len(method_table)
    if n_layers == 0:
        return []
    n_steps = len(method_table[0])
    flag_table = [[False for _ in range(n_steps)] for _ in range(n_layers)]

    for i in range(n_layers):
        last_non_ts = 0
        for j in range(n_steps):
            method = method_table[i][j]
            if method != "NONE":
                flag_table[i][last_non_ts] = True
            if method != "TS":
                last_non_ts = j
    return flag_table


# ---------------------------------------------------------------------------
# Reset / preparation helpers
# ---------------------------------------------------------------------------

def calibration_reset(view: TransformerView, steps: int = 32) -> None:
    """Reset calibration state on all blocks after a calibration phase."""
    for block in view.blocks():
        attn = view.attn(block)
        ff = view.ff(block)
        attn.step = 0
        attn.total_latency = 0.0
        attn.need_cache_output = [False] * steps
        attn.cached_output = None
        ff.need_cache_output = [False] * steps
        ff.step = 0
        ff.total_latency = 0.0
        ff.cached_output = None


def calibration_reset_step(view: TransformerView) -> None:
    """Reset step counters and caches (but keep need_cache_output intact)."""
    for block in view.blocks():
        attn = view.attn(block)
        ff = view.ff(block)
        attn.step = 0
        attn.cached_output = None
        ff.step = 0
        ff.cached_output = None


def eval_reset(view: TransformerView, steps: int = 32) -> None:
    """Reset for evaluation: same as calibration_reset."""
    for block in view.blocks():
        attn = view.attn(block)
        ff = view.ff(block)
        attn.step = 0
        attn.need_cache_output = [False] * steps
        attn.cached_output = None
        ff.need_cache_output = [False] * steps
        ff.step = 0
        ff.cached_output = None


def dit_reset_hook(module, input, output, view: TransformerView) -> None:
    """Forward hook that resets step counters after each full transformer pass."""
    for block in view.blocks():
        attn = view.attn(block)
        ff = view.ff(block)
        attn.step = 0
        attn.cached_output = None
        ff.step = 0
        ff.cached_output = None


# ---------------------------------------------------------------------------
# calibration_preparation
# ---------------------------------------------------------------------------

def calibration_preparation(
    view: TransformerView,
    steps: int = 32,
    method_path: str | None = None,
    is_method_init: bool = True,
    debugging: bool = False,
) -> None:
    """Prepare all blocks for calibration or speedup inference.

    Sets up ``steps_method``, ``need_cache_output``, ``cached_output``,
    ``bs_mode``, and replaces ``forward`` with efficient implementations.
    """
    bs_mode = getattr(view.transformer, "bs_mode", "residual")

    if method_path is None:
        for i, block in enumerate(view.blocks()):
            attn = view.attn(block)
            ff = view.ff(block)
            # attention
            attn.block_id = i
            attn.step = 0
            attn.total_latency = 0.0
            attn.full_ops = 0
            attn.efficient_ops = 0
            if debugging:
                attn.forward = types.MethodType(
                    cuda_timer(view.attn_forward_fn), attn
                )
            else:
                attn.forward = types.MethodType(view.attn_forward_fn, attn)
            if is_method_init:
                attn.steps_method = ["NONE"] * steps
            attn.need_cache_output = [False] * steps
            attn.cached_output = None
            attn.bs_mode = bs_mode
            # feed-forward
            if is_method_init:
                ff.steps_method = ["NONE"] * steps
            ff.need_cache_output = [False] * steps
            ff.full_ops = 0
            ff.efficient_ops = 0
            ff.block_id = i
            ff.step = 0
            ff.total_latency = 0.0
            if debugging:
                ff.forward = types.MethodType(
                    cuda_timer(view.ff_forward_fn), ff
                )
            else:
                ff.forward = types.MethodType(view.ff_forward_fn, ff)
            ff.cached_output = None
            ff.bs_mode = bs_mode
    else:
        method_data = json.loads(open(method_path).read())
        saved_methods = method_data["methods"]
        saved_need_cached = method_data["need_cached_output"]

        for i, (methods, need_cached, block) in enumerate(
            zip(saved_methods, saved_need_cached, view.blocks())
        ):
            attn = view.attn(block)
            attn.steps_method = methods
            attn.block_id = i
            attn.step = 0
            attn.total_latency = 0.0
            attn.full_ops = 0
            attn.efficient_ops = 0
            if debugging:
                attn.forward = types.MethodType(
                    cuda_timer(view.attn_forward_fn), attn
                )
            else:
                attn.forward = types.MethodType(view.attn_forward_fn, attn)
            attn.need_cache_output = need_cached
            attn.cached_output = None
            attn.bs_mode = bs_mode

            ff = view.ff(block)
            ff.steps_method = methods
            ff.block_id = i
            ff.need_cache_output = need_cached
            ff.full_ops = 0
            ff.efficient_ops = 0
            ff.step = 0
            ff.total_latency = 0.0
            if debugging:
                ff.forward = types.MethodType(
                    cuda_timer(view.ff_forward_fn), ff
                )
            else:
                ff.forward = types.MethodType(view.ff_forward_fn, ff)
            ff.cached_output = None
            ff.bs_mode = bs_mode


# ---------------------------------------------------------------------------
# Pre-calibration check (diagonal similarity exploration)
# ---------------------------------------------------------------------------

def pre_calibration_check(
    view: TransformerView,
    pre_cal_hook_fn: callable,
    steps: int = 32,
) -> list:
    """Explore which (block, step) pairs have high diagonal similarity.

    Returns a list of hooks that should be removed after the exploration pass.
    """
    print("Pre Calibration Exploring for transformer!!!")
    calibration_reset(view)
    for block in view.blocks():
        view.attn(block).need_cache_output = [False] * steps
        view.ff(block).need_cache_output = [False] * steps
    hooks = []
    for block in view.blocks():
        hooks.append(
            view.attn(block).register_forward_pre_hook(
                pre_cal_hook_fn, with_kwargs=True
            )
        )
    return hooks


# ---------------------------------------------------------------------------
# Pre-calibration (greedy TS search for ts_first blocks)
# ---------------------------------------------------------------------------

def pre_calibration(
    view: TransformerView,
    steps: int = 32,
    threshold: float = 0.1,
) -> Any:
    """Run pre-calibration: try TS on ts_first blocks."""
    print("Pre Calibration for transformer!!!")

    loss_thresholds = _build_loss_thresholds(view, steps, threshold)
    calibration_preparation(view)

    hook_fn = _make_pre_calibration_hook(view)
    hook = view.transformer.register_forward_pre_hook(hook_fn, with_kwargs=True)
    view.transformer.loss_thresholds = loss_thresholds
    return hook


# ---------------------------------------------------------------------------
# Calibration (greedy TS/BS search for remaining blocks)
# ---------------------------------------------------------------------------

def calibration(
    view: TransformerView,
    steps: int = 32,
    threshold: float = 0.1,
) -> Any:
    """Run calibration: greedy search for TS/BS on non-ts_first blocks."""
    print("Calibration for transformer!!!")

    loss_thresholds = _build_loss_thresholds(view, steps, threshold)
    calibration_preparation(view, is_method_init=False)

    hook_fn = _make_calibration_hook(view)
    hook = view.transformer.register_forward_pre_hook(hook_fn, with_kwargs=True)
    view.transformer.loss_thresholds = loss_thresholds
    return hook


# ---------------------------------------------------------------------------
# Speedup (load saved method table and apply)
# ---------------------------------------------------------------------------

def speedup(
    view: TransformerView,
    delta: float | None = None,
    steps: int = 32,
    methods_path: str = "methods",
) -> None:
    """Load a saved method table and prepare the model for accelerated inference."""
    assert delta is not None, "delta should be set"
    print("Speedup for transformer!!!")
    path = f"{methods_path}/{steps}_{delta}.json"
    calibration_preparation(view, steps=steps, method_path=path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_loss_thresholds(
    view: TransformerView, steps: int, delta: float
) -> list[list[float]]:
    """Build per-(step, block) loss thresholds with linear scaling."""
    blocks = view.blocks()
    n_blocks = len(blocks)
    thresholds = []
    for _ in range(steps):
        step_thresholds = [
            (blocki + 1) / n_blocks * delta for blocki in range(n_blocks)
        ]
        thresholds.append(step_thresholds)
    return thresholds


def _make_pre_calibration_hook(view: TransformerView):
    """Create the pre-calibration forward pre-hook (closure over view)."""

    def hook(model, args, kwargs):
        blocks = view.blocks()
        now_stepi = view.attn(blocks[0]).step
        print(f"Pre Calibration Step: {now_stepi}")

        for block in blocks:
            attn = view.attn(block)
            ff = view.ff(block)
            attn.forward = types.MethodType(
                cuda_timer(view.attn_forward_fn), attn
            )
            attn.need_cache_output[now_stepi] = False
            ff.need_cache_output[now_stepi] = False

        raw_outputs = model.forward(*args, **kwargs)
        raw_outputs = view.comparison_fn(raw_outputs)

        for blocki, block in enumerate(blocks):
            if now_stepi == 0:
                continue
            attn = view.attn(block)
            assert hasattr(attn, "ts_first"), "attn.ts_first not found"
            if attn.ts_first[now_stepi] is False:
                continue
            elif attn.ts_first[now_stepi] is True:
                method_candidates = ["TS"]

            selected_method = "NONE"
            for method in method_candidates:
                view.attn(block).steps_method[now_stepi] = method
                view.ff(block).steps_method[now_stepi] = method

                for block_ in blocks:
                    view.attn(block_).step = now_stepi
                    view.ff(block_).step = now_stepi
                efficient_outputs = model.forward(*args, **kwargs)
                efficient_outputs = view.comparison_fn(efficient_outputs)

                loss = compression_loss(raw_outputs, efficient_outputs)
                threshold = model.loss_thresholds[now_stepi][blocki]

                if loss < threshold:
                    selected_method = method
                    break

            view.attn(block).steps_method[now_stepi] = selected_method
            view.ff(block).steps_method[now_stepi] = selected_method
            del loss, efficient_outputs

        del raw_outputs

        for block_ in blocks:
            view.attn(block_).step = now_stepi
            view.ff(block_).step = now_stepi

        for block in blocks:
            view.attn(block).need_cache_output[now_stepi] = True
            view.ff(block).need_cache_output[now_stepi] = True

    return hook


def _make_calibration_hook(view: TransformerView):
    """Create the calibration forward pre-hook (closure over view)."""

    def hook(model, args, kwargs):
        blocks = view.blocks()
        now_stepi = view.attn(blocks[0]).step
        print(f"Calibration Step: {now_stepi}")

        for block in blocks:
            attn = view.attn(block)
            ff = view.ff(block)
            attn.forward = types.MethodType(
                cuda_timer(view.attn_forward_fn), attn
            )
            attn.need_cache_output[now_stepi] = False
            ff.need_cache_output[now_stepi] = False

        raw_outputs = model.forward(*args, **kwargs)

        nots = getattr(model, "nots", False)
        nobs = getattr(model, "nobs", False)

        for blocki, block in enumerate(blocks):
            if now_stepi == 0:
                continue
            attn = view.attn(block)
            assert hasattr(attn, "ts_first"), "attn.ts_first not found"
            if attn.steps_method[now_stepi] == "TS":
                continue

            if not nots and not nobs:
                method_candidates = ["TS", "BS"]
            elif nots:
                method_candidates = ["BS"]
            elif nobs:
                method_candidates = ["TS"]

            selected_method = "NONE"
            for method in method_candidates:
                view.attn(block).steps_method[now_stepi] = method
                view.ff(block).steps_method[now_stepi] = method

                for block_ in blocks:
                    view.attn(block_).step = now_stepi
                    view.ff(block_).step = now_stepi
                efficient_outputs = model.forward(*args, **kwargs)

                loss = compression_loss(raw_outputs, efficient_outputs)
                threshold = model.loss_thresholds[now_stepi][blocki]

                if loss < threshold:
                    selected_method = method
                    break

            view.attn(block).steps_method[now_stepi] = selected_method
            view.ff(block).steps_method[now_stepi] = selected_method
            del loss, efficient_outputs

        del raw_outputs

        for block_ in blocks:
            view.attn(block_).step = now_stepi
            view.ff(block_).step = now_stepi

        for block in blocks:
            view.attn(block).need_cache_output[now_stepi] = True
            view.ff(block).need_cache_output[now_stepi] = True

    return hook

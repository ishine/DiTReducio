"""
Real-time FLOPs tracker for F5-TTS inference.
Tracks actual FLOPs during model inference rather than theoretical calculations.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import time


class FLOPsTracker:
    """
    Real-time FLOPs tracker that hooks into model inference.
    """

    def __init__(self):
        self.total_flops = 0
        self.attention_flops = 0
        self.ff_flops = 0
        self.embedding_flops = 0
        self.other_flops = 0

        self.hooks = []
        self.start_time = None
        self.end_time = None

        # Per-block statistics
        self.block_stats = {}

    def reset(self):
        """Reset all statistics."""
        self.total_flops = 0
        self.attention_flops = 0
        self.ff_flops = 0
        self.embedding_flops = 0
        self.other_flops = 0
        self.block_stats = {}
        self.start_time = None
        self.end_time = None

    def start_timing(self):
        """Start timing the inference."""
        self.start_time = time.time()

    def end_timing(self):
        """End timing the inference."""
        self.end_time = time.time()

    def get_inference_time(self):
        """Get total inference time in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0

    def calculate_attention_flops(self, module, x):
        """
        Calculate actual attention FLOPs based on input shape and method.

        Args:
            module: Attention module
            x: Input tensor

        Returns:
            FLOPs count
        """
        if not isinstance(x, tuple):
            x = (x,)

        hidden_states = x[0]
        batch_size, seq_len, dim = hidden_states.shape

        # Get current method from calibration info
        method = (
            getattr(module, "steps_method", ["NONE"])[0]
            if hasattr(module, "step")
            else "NONE"
        )
        current_step = getattr(module, "step", 0)
        if hasattr(module, "steps_method") and current_step < len(module.steps_method):
            method = module.steps_method[current_step]

        # Base FLOPs calculation
        heads = getattr(module, "heads", 16)
        dim_head = dim // heads
        inner_dim = heads * dim_head

        # Q, K, V projections
        qkv_ops = 3 * batch_size * seq_len * dim * inner_dim

        # Attention scores computation
        attn_ops = batch_size * heads * seq_len * seq_len * dim_head

        # Softmax operations (approximate)
        softmax_ops = batch_size * heads * seq_len * seq_len

        # Attention @ Value
        attn_value_ops = batch_size * heads * seq_len * seq_len * dim_head

        # Output projection
        out_ops = batch_size * seq_len * inner_dim * dim

        total_ops = qkv_ops + attn_ops + softmax_ops + attn_value_ops + out_ops

        # Adjust for optimization methods
        if method == "TS":
            # Time-step sharing: no computation
            total_ops = 0
        elif method == "BS":
            # Batch sharing: compute only half batch
            total_ops = total_ops // 2

        return total_ops

    def calculate_ff_flops(self, module, x):
        """
        Calculate actual feed-forward FLOPs based on input shape and method.

        Args:
            module: Feed-forward module
            x: Input tensor

        Returns:
            FLOPs count
        """
        if not isinstance(x, tuple):
            x = (x,)

        hidden_states = x[0]
        batch_size, seq_len, dim = hidden_states.shape

        # Get FF configuration
        if hasattr(module, "ff") and len(module.ff) > 0:
            first_linear = module.ff[0]
            if hasattr(first_linear, "out_features"):
                inner_dim = first_linear.out_features
            else:
                inner_dim = dim * 4  # default expansion
        else:
            inner_dim = dim * 4

        # Calculate base operations
        linear1_ops = batch_size * seq_len * dim * inner_dim
        gelu_ops = batch_size * seq_len * inner_dim
        linear2_ops = batch_size * seq_len * inner_dim * dim

        total_ops = linear1_ops + gelu_ops + linear2_ops

        # Get current method and adjust
        method = (
            getattr(module, "steps_method", ["NONE"])[0]
            if hasattr(module, "step")
            else "NONE"
        )
        current_step = getattr(module, "step", 0)
        if hasattr(module, "steps_method") and current_step < len(module.steps_method):
            method = module.steps_method[current_step]

        if method == "TS":
            total_ops = 0
        elif method == "BS":
            total_ops = total_ops // 2

        return total_ops

    def calculate_embedding_flops(self, module, x):
        """
        Calculate embedding FLOPs.

        Args:
            module: Embedding module
            x: Input tensor

        Returns:
            FLOPs count
        """
        if not isinstance(x, tuple):
            x = (x,)

        # For embeddings, FLOPs are relatively simple
        input_tensor = x[0]
        batch_size = input_tensor.shape[0] if len(input_tensor.shape) > 0 else 1
        seq_len = input_tensor.shape[1] if len(input_tensor.shape) > 1 else 1

        # Embedding lookup (mainly memory operations, small FLOPs)
        embed_dim = getattr(module, "embedding_dim", 512)
        ops = batch_size * seq_len * embed_dim

        return ops

    def attention_flops_hook(self, module, args, kwargs):
        """Hook function for attention modules."""
        # Extract input from kwargs since args might be empty
        if args and len(args) > 0:
            input_data = args
        elif kwargs:
            # PyTorch passes inputs in kwargs when using with_kwargs=True
            # Look for common parameter names that might contain the input
            input_data = kwargs.get("input", None)
            if input_data is None:
                input_data = kwargs.get("hidden_states", None)
            if input_data is None:
                # If all else fails, take the first value from kwargs
                input_data = next(iter(kwargs.values())) if kwargs else ()
        else:
            input_data = ()

        flops = self.calculate_attention_flops(module, input_data)
        self.attention_flops += flops
        self.total_flops += flops

        # Track per-block statistics
        block_id = getattr(module, "block_id", 0)
        if block_id not in self.block_stats:
            self.block_stats[block_id] = {"attention": 0, "ff": 0}
        self.block_stats[block_id]["attention"] += flops

    def ff_flops_hook(self, module, args, kwargs):
        """Hook function for feed-forward modules."""
        # Extract input from kwargs since args might be empty
        if args and len(args) > 0:
            input_data = args
        elif kwargs:
            # PyTorch passes inputs in kwargs when using with_kwargs=True
            # Look for common parameter names that might contain the input
            input_data = kwargs.get("input", None)
            if input_data is None:
                input_data = kwargs.get("hidden_states", None)
            if input_data is None:
                # If all else fails, take the first value from kwargs
                input_data = next(iter(kwargs.values())) if kwargs else ()
        else:
            input_data = ()

        flops = self.calculate_ff_flops(module, input_data)
        self.ff_flops += flops
        self.total_flops += flops

        # Track per-block statistics
        block_id = getattr(module, "block_id", 0)
        if block_id not in self.block_stats:
            self.block_stats[block_id] = {"attention": 0, "ff": 0}
        self.block_stats[block_id]["ff"] += flops

    def embedding_flops_hook(self, module, args, kwargs):
        """Hook function for embedding modules."""
        # Extract input from kwargs since args might be empty
        if args and len(args) > 0:
            input_data = args
        elif kwargs:
            # PyTorch passes inputs in kwargs when using with_kwargs=True
            # Look for common parameter names that might contain the input
            input_data = kwargs.get("input", None)
            if input_data is None:
                input_data = kwargs.get("hidden_states", None)
            if input_data is None:
                # If all else fails, take the first value from kwargs
                input_data = next(iter(kwargs.values())) if kwargs else ()
        else:
            input_data = ()

        flops = self.calculate_embedding_flops(module, input_data)
        self.embedding_flops += flops
        self.total_flops += flops

    def register_hooks(self, model):
        """Register FLOPs tracking hooks to the model."""
        self.reset()

        # Register hooks for attention modules
        for block in model.transformer.transformer_blocks:
            hook = block.attn.register_forward_pre_hook(
                self.attention_flops_hook, with_kwargs=True
            )
            self.hooks.append(hook)

            # Register hook for feed-forward
            hook = block.ff.register_forward_pre_hook(
                self.ff_flops_hook, with_kwargs=True
            )
            self.hooks.append(hook)

        # Register hooks for embedding modules if they exist
        hook = model.transformer.text_embed.register_forward_pre_hook(
            self.embedding_flops_hook, with_kwargs=True
        )
        self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_summary(self):
        """Get a summary of FLOPs statistics."""
        inference_time = self.get_inference_time()

        summary = {
            "total_flops_g": self.total_flops / 1e9,
            "attention_flops_g": self.attention_flops / 1e9,
            "ff_flops_g": self.ff_flops / 1e9,
            "embedding_flops_g": self.embedding_flops / 1e9,
            "other_flops_g": self.other_flops / 1e9,
            "inference_time_s": inference_time,
            "flops_per_second": self.total_flops / inference_time
            if inference_time > 0
            else 0,
            "block_stats": self.block_stats,
        }

        # Calculate percentages
        if self.total_flops > 0:
            summary["attention_percentage"] = (
                self.attention_flops / self.total_flops
            ) * 100
            summary["ff_percentage"] = (self.ff_flops / self.total_flops) * 100
            summary["embedding_percentage"] = (
                self.embedding_flops / self.total_flops
            ) * 100
            summary["other_percentage"] = (self.other_flops / self.total_flops) * 100
        else:
            summary.update(
                {
                    "attention_percentage": 0,
                    "ff_percentage": 0,
                    "embedding_percentage": 0,
                    "other_percentage": 0,
                }
            )

        return summary

    def print_summary(self):
        """Print FLOPs summary in a readable format."""
        summary = self.get_summary()

        print("\n" + "=" * 50)
        print("F5-TTS Real-time FLOPs Analysis")
        print("=" * 50)
        print(f"Total FLOPs: {summary['total_flops_g']:.2f} GFLOPs")
        print(f"Inference Time: {summary['inference_time_s']:.3f} seconds")
        print(f"Throughput: {summary['flops_per_second'] / 1e9:.2f} GFLOPs/s")

        print(f"\nFLOPs Breakdown:")
        print(
            f"  Attention: {summary['attention_flops_g']:.2f} GFLOPs ({summary['attention_percentage']:.1f}%)"
        )
        print(
            f"  Feed-Forward: {summary['ff_flops_g']:.2f} GFLOPs ({summary['ff_percentage']:.1f}%)"
        )
        print(
            f"  Embeddings: {summary['embedding_flops_g']:.2f} GFLOPs ({summary['embedding_percentage']:.1f}%)"
        )
        print(
            f"  Other: {summary['other_flops_g']:.2f} GFLOPs ({summary['other_percentage']:.1f}%)"
        )

        if summary["block_stats"]:
            print(f"\nPer-Block Statistics:")
            for block_id, stats in summary["block_stats"].items():
                total_block_flops = stats["attention"] + stats["ff"]
                print(
                    f"  Block {block_id}: {total_block_flops / 1e9:.2f} GFLOPs "
                    f"(Attention: {stats['attention'] / 1e9:.2f} GFLOPs, "
                    f"FF: {stats['ff'] / 1e9:.2f} GFLOPs)"
                )
        print("=" * 50)


# Global tracker instance
_global_tracker = None


def get_flops_tracker():
    """Get the global FLOPs tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = FLOPsTracker()
    return _global_tracker


def start_flops_tracking(model):
    """Start FLOPs tracking for the given model."""
    tracker = get_flops_tracker()
    tracker.register_hooks(model)
    tracker.start_timing()
    return tracker


def end_flops_tracking():
    """End FLOPs tracking and return summary."""
    tracker = get_flops_tracker()
    tracker.end_timing()
    tracker.remove_hooks()
    return tracker.get_summary()

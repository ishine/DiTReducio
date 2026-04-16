"""Uniform accessor for transformer model internals.

Hides backend-specific attribute names (F5-TTS uses transformer_blocks/attn/ff;
MegaTTS 3 uses layers/attention/feed_forward) behind a single interface so
shared calibration code can work with either backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List


@dataclass
class TransformerView:
    """Uniform view over a transformer model for calibration hooks.

    Attributes:
        transformer: The transformer module (model.transformer or model.dit.encoder).
        blocks_fn: transformer -> list of blocks.
        attn_fn: block -> attention module.
        ff_fn: block -> feed-forward module.
        attn_forward_fn: Standalone efficient_attention_forward function.
        ff_forward_fn: Standalone efficient_ff_forward function.
        comparison_fn: raw forward output -> tensor used for compression loss.
            F5-TTS uses identity; MegaTTS 3 applies CFG weighting.
    """

    transformer: Any
    blocks_fn: Callable[[Any], List[Any]]
    attn_fn: Callable[[Any], Any]
    ff_fn: Callable[[Any], Any]
    attn_forward_fn: Callable
    ff_forward_fn: Callable
    comparison_fn: Callable[[Any], Any]

    def blocks(self) -> list:
        return self.blocks_fn(self.transformer)

    def attn(self, block: Any) -> Any:
        return self.attn_fn(block)

    def ff(self, block: Any) -> Any:
        return self.ff_fn(block)

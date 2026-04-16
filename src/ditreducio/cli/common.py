from __future__ import annotations

import argparse
import logging

from ditreducio.ablation.presets import apply_preset
from ditreducio.core.config import load_app_config
from ditreducio.core.registry import build_adapter


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--backend", choices=["f5tts", "megatts3"], help="Override backend from config"
    )
    parser.add_argument(
        "--delta", type=float, default=None, help="Override compression threshold"
    )
    parser.add_argument(
        "--preset",
        choices=[
            "full",
            "no_ws",
            "no_pre",
            "bs_cond_replace",
            "bs_uncond_replace",
            "only_ts",
            "only_bs",
        ],
        default=None,
        help="Ablation preset override",
    )
    parser.add_argument(
        "--track-flops", action="store_true", help="Enable backend FLOPs tracker"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print backend command without running"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")


def prepare_adapter(args: argparse.Namespace):
    config = load_app_config(config_path=args.config, backend_override=args.backend)
    method, calibration = apply_preset(config.method, config.calibration, args.preset)
    config.method = method
    config.calibration = calibration
    return build_adapter(config)

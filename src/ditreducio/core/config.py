from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ditreducio.core.types import AppConfig
from ditreducio.core.types import CalibrationConfig
from ditreducio.core.types import InferenceConfig
from ditreducio.core.types import MethodConfig
from ditreducio.core.types import PathsConfig
from ditreducio.core.types import RuntimeConfig


def _required_path(data: dict[str, Any], key: str) -> Path:
    value = data.get(key)
    if not value:
        raise ValueError(f"Missing required paths.{key} in config file")
    return Path(value).expanduser().resolve()


def _load_backend_args(data: dict[str, Any], backend: str) -> dict[str, Any]:
    backend_args = dict(data.get("backend_args", {}))
    backend_specific = data.get(backend, {})
    if isinstance(backend_specific, dict):
        backend_args.update(backend_specific)
    return backend_args


def load_app_config(
    config_path: str | Path, backend_override: str | None = None
) -> AppConfig:
    config_file = Path(config_path).expanduser().resolve()
    raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid config structure in {config_file}")

    backend = (backend_override or raw.get("backend") or "").strip().lower()
    if backend not in {"f5tts", "megatts3"}:
        raise ValueError("backend must be one of: f5tts, megatts3")

    paths_data = raw.get("paths", {})
    if not isinstance(paths_data, dict):
        raise ValueError("paths must be a mapping")

    backend_entry_raw = paths_data.get("backend_entry")
    backend_entry = None
    if backend_entry_raw:
        backend_entry = Path(backend_entry_raw).expanduser().resolve()

    paths = PathsConfig(
        backend_code_root=_required_path(paths_data, "backend_code_root"),
        backend_ckpt_root=_required_path(paths_data, "backend_ckpt_root"),
        calibration_audio_root=_required_path(paths_data, "calibration_audio_root"),
        strategy_output_root=_required_path(paths_data, "strategy_output_root"),
        inference_output_root=_required_path(paths_data, "inference_output_root"),
        backend_entry=backend_entry,
    )

    runtime_data = raw.get("runtime", {})
    inference_data = raw.get("inference", {})
    calibration_data = raw.get("calibration", {})
    method_data = raw.get("method", {})

    runtime = RuntimeConfig(
        device=runtime_data.get("device", "cuda"),
        dtype=runtime_data.get("dtype", "bfloat16"),
        seed=int(runtime_data.get("seed", 888)),
    )
    inference = InferenceConfig(
        nfe_step=int(inference_data.get("nfe_step", 32)),
        cfg_strength=float(inference_data.get("cfg_strength", 2.0)),
        sway_sampling_coef=float(inference_data.get("sway_sampling_coef", -1.0)),
        speed=float(inference_data.get("speed", 1.0)),
        p_w=float(inference_data.get("p_w", 1.6)),
        t_w=float(inference_data.get("t_w", 2.5)),
    )
    calibration = CalibrationConfig(
        delta=float(calibration_data.get("delta", 0.2)),
        enable_precheck=bool(calibration_data.get("enable_precheck", True)),
        enable_precalibration=bool(calibration_data.get("enable_precalibration", True)),
        top_ratio=float(calibration_data.get("top_ratio", 0.1)),
    )

    bs_mode = method_data.get("bs_mode", "residual")
    if bs_mode not in {"residual", "cond_replace", "uncond_replace"}:
        raise ValueError(
            "method.bs_mode must be residual, cond_replace, or uncond_replace"
        )

    method = MethodConfig(
        enable_ts=bool(method_data.get("enable_ts", True)),
        enable_bs=bool(method_data.get("enable_bs", True)),
        enable_ws=bool(method_data.get("enable_ws", True)),
        bs_mode=bs_mode,
    )

    return AppConfig(
        backend=backend,
        paths=paths,
        runtime=runtime,
        inference=inference,
        calibration=calibration,
        method=method,
        backend_args=_load_backend_args(raw, backend),
    )

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PathsConfig:
    backend_code_root: Path
    backend_ckpt_root: Path
    calibration_audio_root: Path
    strategy_output_root: Path
    inference_output_root: Path
    backend_entry: Path | None = None


@dataclass(slots=True)
class RuntimeConfig:
    device: str = "cuda"
    dtype: str = "bfloat16"
    seed: int = 888


@dataclass(slots=True)
class InferenceConfig:
    nfe_step: int = 32
    cfg_strength: float = 2.0
    sway_sampling_coef: float = -1.0
    speed: float = 1.0
    p_w: float = 1.6
    t_w: float = 2.5


@dataclass(slots=True)
class CalibrationConfig:
    delta: float = 0.2
    enable_precheck: bool = True
    enable_precalibration: bool = True
    top_ratio: float = 0.1


@dataclass(slots=True)
class MethodConfig:
    enable_ts: bool = True
    enable_bs: bool = True
    enable_ws: bool = True
    bs_mode: str = "residual"


@dataclass(slots=True)
class AppConfig:
    backend: str
    paths: PathsConfig
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    backend_args: dict[str, Any] = field(default_factory=dict)

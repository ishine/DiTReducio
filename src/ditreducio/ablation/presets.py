from __future__ import annotations

from dataclasses import replace

from ditreducio.core.types import CalibrationConfig
from ditreducio.core.types import MethodConfig


VALID_PRESETS = {
    "full",
    "no_ws",
    "no_pre",
    "bs_cond_replace",
    "bs_uncond_replace",
    "only_ts",
    "only_bs",
}


def apply_preset(
    method: MethodConfig,
    calibration: CalibrationConfig,
    preset: str | None,
) -> tuple[MethodConfig, CalibrationConfig]:
    if preset is None:
        return method, calibration
    if preset not in VALID_PRESETS:
        available = ", ".join(sorted(VALID_PRESETS))
        raise ValueError(f"Unsupported preset {preset}. Available: {available}")

    out_method = replace(method)
    out_calibration = replace(calibration)

    if preset == "full":
        out_method.enable_ts = True
        out_method.enable_bs = True
        out_method.enable_ws = True
        out_method.bs_mode = "residual"
        out_calibration.enable_precheck = True
        out_calibration.enable_precalibration = True
        return out_method, out_calibration

    if preset == "no_ws":
        out_method.enable_ws = False
        return out_method, out_calibration

    if preset == "no_pre":
        out_calibration.enable_precheck = False
        out_calibration.enable_precalibration = False
        return out_method, out_calibration

    if preset == "bs_cond_replace":
        out_method.bs_mode = "cond_replace"
        return out_method, out_calibration

    if preset == "bs_uncond_replace":
        out_method.bs_mode = "uncond_replace"
        return out_method, out_calibration

    if preset == "only_ts":
        out_method.enable_ts = True
        out_method.enable_bs = False
        out_method.enable_ws = False
        return out_method, out_calibration

    out_method.enable_ts = False
    out_method.enable_bs = True
    out_method.enable_ws = False
    out_calibration.enable_precheck = False
    out_calibration.enable_precalibration = False
    return out_method, out_calibration

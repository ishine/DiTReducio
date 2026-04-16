from __future__ import annotations

from pathlib import Path

from ditreducio.backends.base import BackendAdapter


class F5TTSAdapter(BackendAdapter):
    name = "f5tts"

    def _entry(self) -> Path:
        if self.config.paths.backend_entry is not None:
            return self._resolve_script_path(self.config.paths.backend_entry)
        return self._resolve_script_path(
            self._project_root()
            / "src"
            / "ditreducio"
            / "backends"
            / "f5tts"
            / "cli.py"
        )

    def _workspace(self) -> Path:
        workspace = self.config.paths.strategy_output_root / "f5tts"
        (workspace / "methods").mkdir(parents=True, exist_ok=True)
        return workspace

    def _output_dir(self) -> Path:
        path = self.config.paths.inference_output_root / "f5tts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _build_common(self, delta: float | None, track_flops: bool) -> list[str]:
        args = self.config.backend_args
        method = self.config.method
        calibration = self.config.calibration

        command = [
            self._python_bin(),
            str(self._entry()),
            "--output_dir",
            str(self._output_dir()),
            "--nfe_step",
            str(self.config.inference.nfe_step),
            "--cfg_strength",
            str(self.config.inference.cfg_strength),
            "--sway_sampling_coef",
            str(self.config.inference.sway_sampling_coef),
            "--speed",
            str(self.config.inference.speed),
            "--device",
            str(self.config.runtime.device),
            "--seed",
            str(self.config.runtime.seed),
            "--methods_path",
            str(self._workspace() / "methods"),
            "--bs-mode",
            method.bs_mode,
        ]

        if "ref_audio" in args:
            command.extend(["--ref_audio", str(args["ref_audio"])])
        if "config" in args:
            command.extend(["--config", str(args["config"])])
        if "ref_text" in args:
            command.extend(["--ref_text", str(args["ref_text"])])
        if "gen_text" in args:
            command.extend(["--gen_text", str(args["gen_text"])])
        if "model" in args:
            command.extend(["--model", str(args["model"])])
        if "model_cfg" in args:
            command.extend(["--model_cfg", str(args["model_cfg"])])
        if "ckpt_file" in args:
            command.extend(["--ckpt_file", str(args["ckpt_file"])])
        if "vocab_file" in args:
            command.extend(["--vocab_file", str(args["vocab_file"])])
        if "vocoder_local_path" in args:
            command.extend(["--vocoder_local_path", str(args["vocoder_local_path"])])
        if bool(args.get("load_vocoder_from_local", False)):
            command.append("--load_vocoder_from_local")

        if delta is not None:
            command.extend(["-d", str(delta)])
        if not method.enable_ts:
            command.append("--nots")
        if not method.enable_bs:
            command.append("--nobs")
        if not calibration.enable_precheck or not calibration.enable_precalibration:
            command.append("--nopre")
        if track_flops:
            command.append("--track_flops")
        return command

    def calibrate(
        self, delta: float | None, track_flops: bool = False, dry_run: bool = False
    ) -> None:
        run_delta = self.config.calibration.delta if delta is None else delta
        command = self._build_common(delta=run_delta, track_flops=track_flops)
        command.append("-q")
        self._run_command(command=command, cwd=self._workspace(), dry_run=dry_run)

    def infer(
        self, delta: float | None, track_flops: bool = False, dry_run: bool = False
    ) -> None:
        run_delta = self.config.calibration.delta if delta is None else delta
        command = self._build_common(delta=run_delta, track_flops=track_flops)
        self._run_command(command=command, cwd=self._workspace(), dry_run=dry_run)

from __future__ import annotations

from pathlib import Path

from ditreducio.backends.base import BackendAdapter


class MegaTTS3Adapter(BackendAdapter):
    name = "megatts3"

    def _entry(self) -> Path:
        if self.config.paths.backend_entry is not None:
            return self._resolve_script_path(self.config.paths.backend_entry)
        return self._resolve_script_path(
            self._project_root()
            / "src"
            / "ditreducio"
            / "backends"
            / "megatts3"
            / "cli.py"
        )

    def _workspace(self) -> Path:
        workspace = self.config.paths.strategy_output_root / "megatts3"
        (workspace / "methods").mkdir(parents=True, exist_ok=True)
        return workspace

    def _output_dir(self) -> Path:
        path = self.config.paths.inference_output_root / "megatts3"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _bool_text(self, value: bool) -> str:
        return "True" if value else "False"

    def _build_common(self, delta: float | None, track_flops: bool) -> list[str]:
        args = self.config.backend_args
        method = self.config.method
        calibration = self.config.calibration

        input_wav = args.get("input_wav")
        if not input_wav:
            raise ValueError("megatts3 config requires backend_args.input_wav")
        input_text = args.get("input_text")
        if not input_text:
            raise ValueError("megatts3 config requires backend_args.input_text")

        command = [
            self._python_bin(),
            str(self._entry()),
            "--input_wav",
            str(input_wav),
            "--input_text",
            str(input_text),
            "--output_dir",
            str(self._output_dir()),
            "--time_step",
            str(self.config.inference.nfe_step),
            "--p_w",
            str(self.config.inference.p_w),
            "--t_w",
            str(self.config.inference.t_w),
            "--seed",
            str(self.config.runtime.seed),
            "--methods_path",
            str(self._workspace() / "methods"),
            "--bs-mode",
            method.bs_mode,
        ]

        if not method.enable_ts:
            command.extend(["--nots", "True"])
        if not method.enable_bs:
            command.extend(["--nobs", "True"])
        if (not calibration.enable_precheck) or (not calibration.enable_precalibration):
            command.extend(["--nopre", "True"])

        latent_file = args.get("latent_file")
        if latent_file:
            command.extend(["--latent_file", str(latent_file)])

        if delta is not None:
            command.extend(["-d", str(delta)])
        if track_flops:
            command.append("--track_flops")
        return command

    def calibrate(
        self, delta: float | None, track_flops: bool = False, dry_run: bool = False
    ) -> None:
        run_delta = self.config.calibration.delta if delta is None else delta
        command = self._build_common(delta=run_delta, track_flops=track_flops)
        command.append("-q")
        self._run_command(
            command=command,
            cwd=self.config.paths.backend_code_root,
            dry_run=dry_run,
        )

    def infer(
        self, delta: float | None, track_flops: bool = False, dry_run: bool = False
    ) -> None:
        run_delta = self.config.calibration.delta if delta is None else delta
        command = self._build_common(delta=run_delta, track_flops=track_flops)
        self._run_command(
            command=command,
            cwd=self.config.paths.backend_code_root,
            dry_run=dry_run,
        )

from __future__ import annotations

import logging
import os
import subprocess
import sys
from abc import ABC
from abc import abstractmethod
from pathlib import Path

from ditreducio.core.types import AppConfig

LOGGER = logging.getLogger(__name__)


class BackendAdapter(ABC):
    name: str

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    @abstractmethod
    def calibrate(
        self, delta: float | None, track_flops: bool = False, dry_run: bool = False
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def infer(
        self, delta: float | None, track_flops: bool = False, dry_run: bool = False
    ) -> None:
        raise NotImplementedError

    def _python_bin(self) -> str:
        override = self.config.backend_args.get("python_bin")
        if override:
            return str(override)

        candidates = [
            self.config.paths.backend_code_root / ".venv" / "bin" / "python",
            self._project_root() / ".venv" / "bin" / "python",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        return sys.executable

    def _project_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    def _resolve_script_path(self, *candidates: Path) -> Path:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        joined = "\n".join(str(item) for item in candidates)
        raise FileNotFoundError(f"No backend entry file found. Tried:\n{joined}")

    def _backend_env(self) -> dict[str, str]:
        env = os.environ.copy()
        backend_root = self.config.paths.backend_code_root
        backend_src = backend_root / "src"
        project_root = self._project_root()
        project_src = project_root / "src"
        search_paths = [str(backend_root), str(project_src), str(project_root)]
        if backend_src.exists():
            search_paths.insert(1, str(backend_src))
        existing = env.get("PYTHONPATH", "")
        base = ":".join(search_paths)
        env["PYTHONPATH"] = base if not existing else f"{base}:{existing}"
        env.setdefault("TORCHAUDIO_USE_BACKEND_DISPATCHER", "0")
        return env

    def _run_command(self, command: list[str], cwd: Path, dry_run: bool) -> None:
        command_str = " ".join(command)
        LOGGER.info("%s command: %s", self.name, command_str)
        if dry_run:
            return
        result = subprocess.run(
            command,
            cwd=cwd,
            env=self._backend_env(),
            check=False,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            LOGGER.info(result.stdout.rstrip())
        if result.returncode != 0:
            message = (
                result.stderr.rstrip() if result.stderr else "backend command failed"
            )
            raise RuntimeError(message)

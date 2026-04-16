from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def strategy_filename(steps: int, delta: float | None) -> str:
    return f"{steps}_{delta}.json"


def strategy_path(root: Path, backend: str, steps: int, delta: float | None) -> Path:
    methods_dir = root / backend / "methods"
    methods_dir.mkdir(parents=True, exist_ok=True)
    return methods_dir / strategy_filename(steps=steps, delta=delta)


def load_strategy_table(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_strategy_table(path: Path, table: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, indent=2), encoding="utf-8")

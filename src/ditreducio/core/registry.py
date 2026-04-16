from __future__ import annotations

from typing import Type

from ditreducio.backends.base import BackendAdapter
from ditreducio.backends.f5tts_adapter import F5TTSAdapter
from ditreducio.backends.megatts3_adapter import MegaTTS3Adapter
from ditreducio.core.types import AppConfig


_BACKENDS: dict[str, Type[BackendAdapter]] = {
    "f5tts": F5TTSAdapter,
    "megatts3": MegaTTS3Adapter,
}


def build_adapter(config: AppConfig) -> BackendAdapter:
    adapter_cls = _BACKENDS.get(config.backend)
    if adapter_cls is None:
        available = ", ".join(sorted(_BACKENDS.keys()))
        raise ValueError(f"Unknown backend: {config.backend}. Available: {available}")
    return adapter_cls(config)

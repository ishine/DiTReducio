"""CUDA timing decorator for profiling module forward passes."""

from __future__ import annotations

import time

import torch


def cuda_timer(func):
    """Decorator that measures wall-clock or CUDA-elapsed time of a method.

    Accumulates latency into ``self.total_latency`` when that attribute exists.
    """

    def wrapper(self, *args, **kwargs):
        use_cuda = hasattr(self, "total_latency") and torch.cuda.is_available()
        if use_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        elif hasattr(self, "total_latency"):
            start_time = time.perf_counter()

        result = func(self, *args, **kwargs)

        if use_cuda:
            end_event.record()
            torch.cuda.synchronize()
            self.total_latency += start_event.elapsed_time(end_event) / 1000.0
        elif hasattr(self, "total_latency"):
            self.total_latency += time.perf_counter() - start_time

        return result

    return wrapper

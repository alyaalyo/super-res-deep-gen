"""Device selection, timing and peak-memory measurement."""

from __future__ import annotations

import time
from contextlib import contextmanager

import torch


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple MPS"
    return "CPU"


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class Meter:
    """Wall-clock seconds and peak accelerator memory for one call.

    On CUDA the peak is ``max_memory_allocated`` (tensor memory, what the plan
    reports for the T4). MPS exposes no peak counter, so we report the driver
    allocation after the call as an approximation; on CPU memory is not
    measured.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.seconds = float("nan")
        self.peak_mb = float("nan")

    @contextmanager
    def measure(self):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        synchronize(self.device)
        t0 = time.perf_counter()
        yield self
        synchronize(self.device)
        self.seconds = time.perf_counter() - t0
        if self.device.type == "cuda":
            self.peak_mb = torch.cuda.max_memory_allocated(self.device) / 2**20
        elif self.device.type == "mps":
            self.peak_mb = torch.mps.driver_allocated_memory() / 2**20

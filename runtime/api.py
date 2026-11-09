"""Unified runtime API — the 'CUDA alternative' surface.

One codebase, multiple backends: devices (gpu/cpu/npu), tiered memory pools,
streams, kernels. Linux backends: Vulkan / ROCm. Windows: DirectML / ONNX RT.
Production core moves to Rust/C++ after the prototype proves out; this Python
API is the contract everything else implements.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional


class DeviceKind(str, Enum):
    GPU = "gpu"
    CPU = "cpu"
    NPU = "npu"


@dataclass
class Device:
    kind: DeviceKind
    name: str
    index: int = 0
    memory_bytes: int = 0
    backend: str = "fake"

    @property
    def id(self) -> str:
        return f"{self.kind.value}{self.index}"


_KERNELS: Dict[str, Callable] = {}


def register_kernel(name: str):
    """Register a compute kernel under a name; backends bind implementations."""
    def deco(fn: Callable):
        _KERNELS[name] = fn
        return fn
    return deco


def kernel(name: str) -> Callable:
    return _KERNELS[name]


class Tier(str, Enum):
    VRAM = "vram"
    DRAM = "dram"
    NVME = "nvme"


@dataclass
class Tensor:
    id: int
    size: int
    tier: Tier
    pool: "MemoryPool"

    def free(self):
        self.pool._release(self)


class MemoryPool:
    """Tiered pool VRAM -> DRAM -> NVMe(file-backed). allocate() fills the
    hottest tier with space, so hot weights naturally stay fast."""

    def __init__(self, vram_bytes: int = 0, dram_bytes: int = 0, nvme_path: Optional[str] = None):
        self.caps = {Tier.VRAM: vram_bytes, Tier.DRAM: dram_bytes}
        self.nvme_path = nvme_path
        self._lock = threading.Lock()
        self._used = {Tier.VRAM: 0, Tier.DRAM: 0}
        self._next_id = 1

    def allocate(self, size: int, prefer: Tier = Tier.VRAM) -> Tensor:
        order = [Tier.VRAM, Tier.DRAM] if prefer != Tier.NVME else [Tier.DRAM, Tier.VRAM]
        with self._lock:
            for tier in order:
                if self._used[tier] + size <= self.caps[tier]:
                    self._used[tier] += size
                    return self._make(size, tier)
            return self._make(size, Tier.NVME)

    def _make(self, size: int, tier: Tier) -> Tensor:
        t = Tensor(id=self._next_id, size=size, tier=tier, pool=self)
        self._next_id += 1
        return t

    def _release(self, t: Tensor):
        with self._lock:
            if t.tier in self._used:
                self._used[t.tier] -= t.size

    def usage(self) -> dict:
        return {t.value: {"used": self._used[t], "cap": self.caps[t]} for t in (Tier.VRAM, Tier.DRAM)}


class Stream:
    """Ordered queue of kernel launches on a device; synchronize() drains it."""

    def __init__(self, device: Device):
        self.device = device
        self._q: List = []
        self._lock = threading.Lock()

    def submit(self, kernel_name: str, *args) -> int:
        with self._lock:
            self._q.append((kernel_name, args, time.time()))
            return len(self._q) - 1

    def synchronize(self):
        while True:
            with self._lock:
                if not self._q:
                    return
                name, args, _ = self._q.pop(0)
            kernel(name)(*args)

    def __len__(self):
        return len(self._q)

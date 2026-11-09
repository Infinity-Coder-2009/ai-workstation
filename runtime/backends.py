"""Backend registry. Each backend probes the OS for real devices; the fake
backend lets the whole API run on any machine (CI, dev laptops) so we can
build and test before the pilot hardware arrives."""
from __future__ import annotations

import shutil

from .api import Device, DeviceKind


class Backend:
    name = "base"

    def devices(self) -> list:
        return []

    @staticmethod
    def _have(*tools) -> bool:
        return all(shutil.which(t) for t in tools)


class FakeBackend(Backend):
    name = "fake"

    def devices(self):
        return [
            Device(DeviceKind.GPU, "Fake GPU", 0, 24 << 30, "fake"),
            Device(DeviceKind.CPU, "Fake CPU", 0, 128 << 30, "fake"),
            Device(DeviceKind.NPU, "Fake NPU", 0, 0, "fake"),
        ]


class VulkanBackend(Backend):  # Linux + Windows GPU path
    name = "vulkan"

    def devices(self):
        if not self._have("vulkaninfo"):
            return []
        # TODO: parse `vulkaninfo --summary` for real GPUs
        return []


class RocmBackend(Backend):  # Linux AMD/instinct path
    name = "rocm"

    def devices(self):
        if not self._have("rocm-smi"):
            return []
        return []


class DirectMLBackend(Backend):  # Windows GPU + NPU via ONNX Runtime DML EP
    name = "directml"

    def devices(self):
        import platform
        if platform.system() != "Windows":
            return []
        # TODO: enumerate via onnxruntime DML provider
        return []


class NpuBackend(Backend):  # OpenVINO (Linux/Win) / QNN / WindowsML
    name = "npu"

    def devices(self):
        # TODO: probe OpenVINO device list; small aux models only
        return []


_BACKENDS = [VulkanBackend(), RocmBackend(), DirectMLBackend(), NpuBackend(), FakeBackend()]


def list_devices() -> list:
    out = []
    for b in _BACKENDS:
        try:
            out.extend(b.devices())
        except Exception:
            continue
    return out


def get_device(dev_id: str) -> Device:
    for d in list_devices():
        if d.id == dev_id:
            return d
    raise KeyError(f"device not found: {dev_id}")

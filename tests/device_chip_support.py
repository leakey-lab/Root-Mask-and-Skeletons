"""Stand-ins for the device chip's tests (``test_device_chip.py``, ``test_device_check.py``).

- :class:`FakeRuntime`: ``app.inference.runtime`` before and after torch resolved it, so
  torch is never loaded (install ``FakeRuntime().module`` as ``sys.modules[RUNTIME]``);
- :data:`GPU` / :data:`CPU`: the driver's device guesses;
- :class:`Line` and :class:`GuessLine`: stand-in status lines that record what a
  ``DeviceCheck`` tells them (``GuessLine`` also takes a guess).

Qt-free; no asserts.
"""

from __future__ import annotations

import types

from app.inference.device_probe import DeviceGuess

RUNTIME = "app.inference.runtime"
GPU = DeviceGuess(
    "GPU", name="Fake RTX 9000", cuda_version="12.8", note="Named by the NVIDIA driver."
)
CPU = DeviceGuess("CPU", note="PyTorch 2.8.0+cpu is a CPU-only build: inference runs on the CPU.")


class FakeRuntime:
    """``app.inference.runtime`` before and after torch resolved it."""

    def __init__(self, *, resolved: bool = False, mask_weights: bool = True) -> None:
        self.resolved = resolved
        self.mask_weights = mask_weights
        self.module = types.ModuleType(RUNTIME)
        self.module.describe_runtime = self.describe

    def describe(self, resolve: bool = False):
        if resolve:
            self.resolved = True
        text = (
            "GPU · Fake RTX 9000 · CUDA 12.8 · TF32" if self.resolved else "Device not detected yet"
        )
        if not self.mask_weights:
            text += " · mask model missing"
        return types.SimpleNamespace(
            device="GPU" if self.resolved else "unknown",
            text=text,
            models_ok=self.mask_weights,
            mask_weights=self.mask_weights,
            skeleton_weights=True,
        )


class Line:
    """Stand-in status line: records the chip states ``(state, detail)``; ``busy`` reports a
    running task."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.busy = False

    def set_device_check(self, state, detail: str = "") -> None:
        self.calls.append((state, detail))

    def state(self) -> str:
        return "working" if self.busy else "idle"


class GuessLine:
    """Stand-in status line that also takes a guess: ``("check", state, detail)`` and
    ``("guess", guess)``."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def set_device_check(self, state, detail: str = "") -> None:
        self.calls.append(("check", state, detail))

    def set_device_guess(self, guess) -> None:
        self.calls.append(("guess", guess))

    def state(self) -> str:
        return "idle"

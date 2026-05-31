"""
Shared inference runtime auto-tuner.

Single source of truth for *how* both inference paths (skeleton_inference.py and
mask_generation_handler.py) talk to the hardware: which device, which autocast
dtype, whether to use channels_last memory format, whether to pin host memory,
how many DataLoader workers, and whether torch.compile is safe to attempt.

Design rules (all enforced here so the two paths can never drift):
  * Detect the runtime environment at first call -- never hardcode a GPU index.
    Always fall back to CPU when CUDA is unavailable.
  * amp_dtype: bfloat16 when CUDA supports it, else float16 on CUDA, else None
    (CPU runs full fp32 -- autocast is intentionally disabled there).
  * channels_last and pin_memory are CUDA-only.
  * num_workers defaults to 0: a Windows/QThread-safe policy made explicit so
    DataLoader never deadlocks or pays spawn cost inside a Qt worker thread.
  * should_compile is gated behind torch>=2 *and* an importable triton (often
    missing on Windows). torch.compile usage must always degrade gracefully.

Public API:
  get_runtime() -> RuntimeConfig          (cached, lru_cache(maxsize=1))
  get_skeleton_model(device) -> SkeletonModel  (cached singleton, maxsize=1)
  warmup(model=None) -> None              (optional latency pre-pay)
"""
from __future__ import annotations

import functools
import importlib.util
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal probes
# ---------------------------------------------------------------------------
def _torch_major_ge_2() -> bool:
    """True when the installed torch is version 2 or newer."""
    try:
        major = int(torch.__version__.split(".", 1)[0])
    except (ValueError, IndexError):
        # Dev / nightly strings should still parse; if not, be conservative.
        return hasattr(torch, "compile")
    return major >= 2


def _triton_available() -> bool:
    """True when triton is importable (required backend for torch.compile)."""
    try:
        return importlib.util.find_spec("triton") is not None
    except (ImportError, ValueError):
        return False


def _detect_amp_dtype(is_cuda: bool) -> Optional[torch.dtype]:
    """Choose the autocast dtype for the detected device.

    bfloat16 on CUDA when supported (better numerical range, no loss scaling),
    float16 on CUDA otherwise, and None on CPU (full fp32, autocast disabled).
    """
    if not is_cuda:
        return None
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except (RuntimeError, AssertionError):
        # Some stacks raise if queried without a usable device; fall back to fp16.
        pass
    return torch.float16


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved, immutable per-machine inference runtime configuration.

    Attributes:
        device:            torch.device to run inference on (cuda or cpu).
        amp_dtype:         autocast dtype (bfloat16/float16) or None on CPU.
        use_channels_last: apply channels_last memory format (CUDA only).
        pin_memory:        pin DataLoader host memory (CUDA only).
        num_workers:       DataLoader workers (0 = Windows/QThread-safe policy).
        should_compile:    torch.compile is safe to attempt (torch>=2 + triton).
        device_name:       human-readable accelerator name for logging.
        bf16_supported:    whether the CUDA device advertises bf16 support.
        compute_capability:(major, minor) CUDA capability, or None on CPU.
    """

    device: torch.device
    amp_dtype: Optional[torch.dtype]
    use_channels_last: bool
    pin_memory: bool
    num_workers: int
    should_compile: bool
    device_name: str = "cpu"
    bf16_supported: bool = False
    compute_capability: Optional[tuple[int, int]] = None

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    def autocast(self):
        """Return an autocast context manager appropriate for this runtime.

        On CUDA this enables mixed precision with the chosen amp_dtype; on CPU
        it is a no-op (enabled=False) so call sites can wrap inference
        unconditionally:  with rt.autocast(): out = model(x)
        """
        enabled = self.amp_dtype is not None
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype if enabled else None,
            enabled=enabled,
        )


# ---------------------------------------------------------------------------
# Cached runtime detection
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def get_runtime() -> RuntimeConfig:
    """Detect and cache the inference runtime for this machine.

    Called once per process (lru_cache(maxsize=1)). Auto-detects CUDA vs CPU,
    selects the autocast dtype, and applies global CUDA performance switches.
    Never hardcodes a GPU index and always falls back to CPU.
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        device = torch.device("cuda")
        try:
            device_name = torch.cuda.get_device_name(device)
        except (RuntimeError, AssertionError):
            device_name = "cuda"
        try:
            cc = torch.cuda.get_device_capability(device)
            compute_capability: Optional[tuple[int, int]] = (int(cc[0]), int(cc[1]))
        except (RuntimeError, AssertionError):
            compute_capability = None

        # Global perf switches -- safe to set once, only meaningful on CUDA.
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    else:
        device = torch.device("cpu")
        device_name = "cpu"
        compute_capability = None

    amp_dtype = _detect_amp_dtype(is_cuda)
    bf16_supported = amp_dtype is torch.bfloat16

    should_compile = bool(
        _torch_major_ge_2() and _triton_available()
    )

    config = RuntimeConfig(
        device=device,
        amp_dtype=amp_dtype,
        use_channels_last=is_cuda,
        pin_memory=is_cuda,
        # 0 workers: explicit Windows/QThread-safe policy (avoids spawn deadlocks).
        num_workers=0,
        should_compile=should_compile,
        device_name=device_name,
        bf16_supported=bf16_supported,
        compute_capability=compute_capability,
    )

    logger.info(
        "Inference runtime resolved: device=%s (%s) amp_dtype=%s channels_last=%s "
        "pin_memory=%s num_workers=%d should_compile=%s cc=%s",
        config.device,
        config.device_name,
        config.amp_dtype,
        config.use_channels_last,
        config.pin_memory,
        config.num_workers,
        config.should_compile,
        config.compute_capability,
    )
    return config


def maybe_compile(model: "torch.nn.Module") -> "torch.nn.Module":
    """torch.compile the model when supported, else return it unchanged.

    Guarded by both the runtime's should_compile gate and a try/except so a
    missing triton backend (common on Windows) degrades gracefully.
    """
    rt = get_runtime()
    if not rt.should_compile:
        return model
    try:
        compiled = torch.compile(model)
        logger.info("torch.compile applied to %s", type(model).__name__)
        return compiled
    except Exception as exc:  # noqa: BLE001 -- compile must never break inference
        logger.warning("torch.compile unavailable/failed (%s); using eager model", exc)
        return model


# ---------------------------------------------------------------------------
# Skeleton model singleton factory
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def get_skeleton_model(device: Optional[torch.device] = None):
    """Return a cached singleton SkeletonModel for skeleton inference.

    The heavy generator network and its checkpoint are loaded exactly once per
    process. ``device`` is accepted for explicitness/testability; when omitted
    the auto-detected runtime device is used. The model is placed on the
    runtime device, set to eval, and (on CUDA) converted to channels_last.

    Imported lazily to avoid a circular import with skeleton_inference, which
    imports this module.
    """
    from app.inference.skeleton_inference import SkeletonModel  # local import

    rt = get_runtime()
    target = device if device is not None else rt.device

    gpu_ids = [target.index or 0] if target.type == "cuda" else []
    model = SkeletonModel(gpu_ids=gpu_ids)

    # Move the underlying network to the resolved device and apply layout policy.
    net = model.netG
    if rt.use_channels_last and target.type == "cuda":
        try:
            net.to(memory_format=torch.channels_last)
        except (RuntimeError, ValueError) as exc:
            logger.debug("channels_last not applied to skeleton model: %s", exc)

    net.eval()
    logger.info("Skeleton model singleton created on %s", target)
    return model


def warmup(model=None) -> None:
    """Pre-pay one-time CUDA/cuDNN/compile latency with a dummy forward pass.

    No-op on CPU. Safe to call multiple times. Any failure is swallowed -- a
    warmup is an optimization, never a correctness requirement.
    """
    rt = get_runtime()
    if not rt.is_cuda:
        return

    try:
        if model is None:
            model = get_skeleton_model(rt.device)
        net = model.netG
        # LOAD_SIZE/INPUT_NC mirror skeleton_inference's preprocessing (256, RGB).
        dummy = torch.zeros(1, 3, 256, 256, device=rt.device)
        if rt.use_channels_last:
            dummy = dummy.to(memory_format=torch.channels_last)
        with torch.inference_mode(), rt.autocast():
            net(dummy)
        torch.cuda.synchronize()
        logger.info("Skeleton model warmup complete on %s", rt.device)
    except Exception as exc:  # noqa: BLE001 -- warmup is best-effort only
        logger.debug("Warmup skipped/failed: %s", exc)


__all__ = [
    "RuntimeConfig",
    "get_runtime",
    "maybe_compile",
    "get_skeleton_model",
    "warmup",
]

"""Stand-ins and folders for the generation tests (Qt-free; no checkpoint, no GPU).

- Folders of photos: ``_images`` (random RGB photos with the given names, some of them not an
  image), ``_make_images`` (tiny photos for the batch engine) and its ``NAMES``.
- The batch engine's pieces: ``_load``, ``_step`` (the grey mean of each row: every row is
  computed alone, so any batch composition gives the same pixels), ``_spec``, ``_outputs``.
- Stub models: ``_StubMaskModel`` (a seeded 1x1 convolution and a sigmoid; it counts its
  ``calls`` and ``items``, runs ``on_forward()`` in each forward, and with ``oom`` set every
  forward runs out of GPU memory), ``_StubSkeletonModel`` (a seeded 1x1 convolution and tanh),
  ``_Brightness`` / ``_Skeleton`` (bright pixels are foreground) with ``_mean``; and
  ``_read_all`` (name -> bytes of a folder's outputs).
"""

from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image

from app.inference.batch_engine import BatchSpec


def _images(folder, names, *, corrupt=(), size=(64, 48)):
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(1)
    for name in names:
        path = os.path.join(folder, name)
        if name in corrupt:
            with open(path, "wb") as fh:
                fh.write(b"this is not an image")
        else:
            arr = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
            Image.fromarray(arr).save(path)
    return folder


class _StubMaskModel(torch.nn.Module):
    """Tiny deterministic stand-in for ResNetSkeleton (sigmoid output, 1 channel)."""

    def __init__(self, on_forward=None):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(3, 1, 1)
        self.on_forward = on_forward
        self.oom = False
        self.calls = 0
        self.items = 0

    def forward(self, x):
        if self.oom:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (stub)")
        self.calls += 1
        self.items += x.shape[0]
        if self.on_forward:
            self.on_forward()
        return torch.sigmoid(self.conv(x))


class _StubSkeletonModel:
    def __init__(self):
        torch.manual_seed(0)
        self.net = torch.nn.Conv2d(3, 3, 1)

    def run_batch(self, x):
        with torch.inference_mode():
            return torch.tanh(self.net(x))


def _read_all(folder, suffix=".png"):
    return {
        n: open(os.path.join(folder, n), "rb").read()
        for n in sorted(os.listdir(folder))
        if n.endswith(suffix)
    }


def _make_images(folder, names, *, corrupt=()):
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(0)
    for name in names:
        path = os.path.join(folder, name)
        if name in corrupt:
            with open(path, "wb") as fh:
                fh.write(b"\xff\xd8\xff\xe0 truncated jpeg")
        else:
            Image.fromarray(rng.integers(0, 255, (6, 8, 3), dtype=np.uint8)).save(path)
    return [os.path.join(folder, n) for n in sorted(names)]


def _load(path):
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _step(x):
    return (x.mean(1) * 255).to(torch.uint8).numpy()


def _spec(out_dir, step=_step, save=None):
    def default_save(arr, path):
        stem = os.path.splitext(os.path.basename(path))[0]
        Image.fromarray(arr, mode="L").save(os.path.join(out_dir, stem + ".png"))

    os.makedirs(out_dir, exist_ok=True)
    return BatchSpec(load=_load, step=step, save=save or default_save)


def _outputs(folder):
    out = {}
    for name in sorted(os.listdir(folder)):
        with open(os.path.join(folder, name), "rb") as fh:
            out[name] = fh.read()
    return out


NAMES = [f"img{i:02d}.png" for i in range(7)]


class _Brightness(torch.nn.Module):
    """Mask model stand-in: foreground where the input is bright."""

    def forward(self, x):
        return torch.sigmoid(x.mean(1, keepdim=True) * 50)


class _Skeleton:
    def run_batch(self, x):
        return torch.tanh(x * 5)


def _mean(path):
    with Image.open(path) as im:
        return float(np.asarray(im).mean())

"""Stand-ins of the generation runs: a mask model that marks bright pixels, a skeleton model,
and the mean grey of an output file.

A COPY of ``test_robust_generation_stem_clash.py``'s (that module is frozen in wave 1;
wave 2 imports them from here and drops its own), for ``test_stem_case_and_formats.py``.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


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

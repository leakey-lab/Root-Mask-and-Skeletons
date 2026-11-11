"""The skeleton network (``EnhancedResnetGenerator`` in ``app/inference/skeleton_inference.py``).

Promises checked here, on the CPU, without a checkpoint: built as ``SkeletonModel`` builds it,
the generator holds exactly the layout of the published checkpoint
(``tests/data/model_checkpoint_layout.json``) and loads it strictly; it maps a 256 x 256 RGB
input in [-1, 1] to a 256 x 256 RGB image in [-1, 1], deterministically in ``eval()`` (Dropout
off). Loading, norm statistics and the output files: ``test_skeleton_inference.py``,
``test_skeleton_norm_stats.py``; what the published weights compute: ``test_checkpoints.py``.
"""

from __future__ import annotations

import json
import os

import torch

from app.inference import skeleton_inference as si

LAYOUT_PIN = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "model_checkpoint_layout.json"
)


def _generator() -> si.EnhancedResnetGenerator:
    """The generator with ``SkeletonModel._load_network``'s arguments."""
    return si.EnhancedResnetGenerator(
        si.INPUT_NC,
        si.OUTPUT_NC,
        si.NGF,
        norm_layer=si.get_norm_layer(norm_type=si.NORM),
        use_dropout=si.USE_DROPOUT,
        n_blocks=9,
    )


def test_the_generator_has_the_published_checkpoints_layout_and_loads_it_strictly():
    with open(LAYOUT_PIN, encoding="utf-8") as fh:
        recorded = json.load(fh)["models"]["skeleton"]["entries"]
    with torch.device("meta"):
        meta = _generator()

    layout = {name: [list(t.shape), str(t.dtype)] for name, t in meta.state_dict().items()}
    assert layout == recorded
    state = {
        name: torch.empty(shape, dtype=getattr(torch, dtype.removeprefix("torch.")), device="meta")
        for name, (shape, dtype) in recorded.items()
    }
    meta.load_state_dict(state, strict=True, assign=True)


def test_a_256_input_gives_a_256_rgb_image_in_minus_one_to_one():
    torch.manual_seed(0)
    net = _generator().eval()
    x = torch.rand(1, 3, si.LOAD_SIZE, si.LOAD_SIZE) * 2 - 1  # the transform's range

    with torch.inference_mode():
        first, second = net(x), net(x)

    assert first.shape == (1, si.OUTPUT_NC, si.LOAD_SIZE, si.LOAD_SIZE)
    assert float(first.abs().max()) <= 1.0  # tanh
    assert torch.equal(first, second)  # Dropout is off in eval()

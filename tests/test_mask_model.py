"""The mask network (``app/mask_model/model.py``: ``ResNetSkeleton``, ``DualAttention``).

The network that makes every root area: a ResNet-18 encoder, dilated convolutions with dual
attention at 1/8, 1/16 and 1/32 of the photo, transposed convolutions back to its size and
a sigmoid. Promises checked here, on the CPU, without a checkpoint and without the network:
- built as the app builds it, it holds exactly the layout of the published checkpoint
  (``tests/data/model_checkpoint_layout.json``) and loads it strictly;
- it maps a photo tensor of the model's size (480 x 640) to one probability in (0, 1) per
  pixel; in ``eval()`` it is deterministic and a photo's output does not depend on the other
  photos of its batch;
- the app's step (``make_mask_spec``) runs it without gradients and writes 255 where the
  probability exceeds the threshold, else 0;
- ``DualAttention`` computes what it is meant to: attention over the positions and over the
  channels, each a softmax over what is attended to (an independent einsum reference).
Building it never downloads anything (``pretrained=False``): any attempt fails the test.
What the published weights compute: ``test_checkpoints.py``.
"""

from __future__ import annotations

import json
import os
import urllib.request

import numpy as np
import pytest
import torch
import torchvision.models._api as torchvision_api

from app.inference import mask_inference
from app.mask_model.model import DualAttention, ResNetSkeleton

LAYOUT_PIN = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "model_checkpoint_layout.json"
)
H, W = mask_inference.MASK_INPUT_SIZE


@pytest.fixture(scope="module", autouse=True)
def no_downloads():
    """A network attempt (torchvision's pretrained weights) fails at once instead of waiting;
    module-wide, so the module's network is built under it too."""

    def refuse(*_args, **_kwargs):
        raise AssertionError("building the mask network tried to download something")

    with pytest.MonkeyPatch.context() as patch:
        # torchvision asks for its weights here (also when they are cached); torch.hub
        # downloads through its own reference to urlopen.
        patch.setattr(torchvision_api, "load_state_dict_from_url", refuse)
        patch.setattr(torch.hub, "load_state_dict_from_url", refuse)
        patch.setattr(torch.hub, "urlopen", refuse)
        patch.setattr(urllib.request, "urlopen", refuse)
        yield


@pytest.fixture(scope="module")
def net(no_downloads):
    """The network from a fixed seed, in eval mode (about 0.35 GB, built once)."""
    torch.manual_seed(0)
    return ResNetSkeleton(num_classes=1, pretrained=False).eval()


def test_the_network_has_the_published_checkpoints_layout_and_loads_it_strictly():
    with open(LAYOUT_PIN, encoding="utf-8") as fh:
        recorded = json.load(fh)["models"]["mask"]["entries"]
    with torch.device("meta"):  # as runtime.get_mask_model builds it: no weights allocated
        meta = ResNetSkeleton(num_classes=1, pretrained=False)

    layout = {name: [list(t.shape), str(t.dtype)] for name, t in meta.state_dict().items()}
    assert layout == recorded
    state = {
        name: torch.empty(shape, dtype=getattr(torch, dtype.removeprefix("torch.")), device="meta")
        for name, (shape, dtype) in recorded.items()
    }
    meta.load_state_dict(state, strict=True, assign=True)  # raises on any missing / extra key


def test_a_photo_sized_input_gives_one_probability_per_pixel(net):
    torch.manual_seed(1)
    x = torch.rand(1, 3, H, W)

    with torch.inference_mode():
        y = net(x)

    assert y.shape == (1, 1, H, W) and y.dtype == torch.float32
    assert float(y.min()) > 0.0 and float(y.max()) < 1.0
    # Also where the network is sure there is no root (strongly negative scores): a
    # probability is never negative.
    bias = net.mask_output.bias.detach().clone()
    try:
        with torch.no_grad():
            net.mask_output.bias.sub_(5.0)
        with torch.inference_mode():
            sure_not = net(x)
    finally:
        with torch.no_grad():
            net.mask_output.bias.copy_(bias)
    assert 0.0 < float(sure_not.min()) and float(sure_not.max()) < 0.05


def test_in_eval_mode_the_output_is_deterministic_and_independent_of_the_batch(net):
    torch.manual_seed(2)
    x = torch.rand(2, 3, 96, 128)

    with torch.inference_mode():
        first, second = net(x), net(x)
        alone = torch.cat([net(x[i : i + 1]) for i in range(2)])

    assert torch.equal(first, second)
    # The CPU convolutions round differently at another batch size (~6e-8 here); BatchNorm
    # uses its running statistics, so no row's statistics leak into another.
    assert torch.allclose(first, alone, rtol=0, atol=1e-6)


def test_the_app_step_thresholds_the_network_output_without_gradients(net, tmp_path, cpu_runtime):
    torch.manual_seed(3)
    x = torch.rand(1, 3, H, W)
    with torch.no_grad():
        probability = net(x).squeeze(1).numpy()
    threshold = float(np.median(probability))  # a threshold that splits this output
    grad_modes = []
    hook = net.register_forward_hook(
        lambda *_: grad_modes.append((torch.is_inference_mode_enabled(), torch.is_grad_enabled()))
    )
    try:
        written = mask_inference.make_mask_spec(net, str(tmp_path), threshold=threshold).step(x)
    finally:
        hook.remove()

    assert written.dtype == np.uint8 and written.shape == (1, H, W)
    assert np.array_equal(written, np.where(probability > threshold, 255, 0).astype(np.uint8))
    assert set(np.unique(written)) == {0, 255}
    assert grad_modes == [(True, False)]


def _reference_dual_attention(block: DualAttention, x: torch.Tensor) -> torch.Tensor:
    """Position attention (each position attends over every position) and channel attention
    (each channel attends over every channel), written with einsum, then the fusion conv."""
    b, c, h, w = x.shape
    q = block.query_conv(x).reshape(b, -1, h * w)
    k = block.key_conv(x).reshape(b, -1, h * w)
    v = block.value_conv(x).reshape(b, c, h * w)
    weights = torch.softmax(torch.einsum("bki,bkj->bij", q, k), dim=2)  # over the keys j
    position = torch.einsum("bcj,bij->bci", v, weights).reshape(b, c, h, w)
    flat = x.reshape(b, c, h * w)
    affinity = torch.softmax(torch.einsum("bcn,bdn->bcd", flat, flat), dim=2)  # over channels d
    channel = torch.einsum("bcd,bdn->bcn", affinity, flat).reshape(b, c, h, w)
    fused = torch.cat([block.gamma * position + x, block.beta * channel + x], dim=1)
    return block.fusion_conv(fused)


def test_dual_attention_attends_over_positions_and_channels():
    torch.manual_seed(4)
    block = DualAttention(16).eval()
    with torch.no_grad():  # both start at 0, which would hide the attention entirely
        block.gamma.fill_(0.7)
        block.beta.fill_(-0.4)
    x = torch.randn(2, 16, 5, 6) * 0.3

    with torch.no_grad():
        got, expected = block(x), _reference_dual_attention(block, x)

    assert torch.allclose(got, expected, rtol=0, atol=1e-5)
    assert not torch.allclose(got, block.fusion_conv(torch.cat([x, x], dim=1)), atol=1e-3)

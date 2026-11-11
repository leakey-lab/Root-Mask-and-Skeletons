"""Mask generation (``app/inference/mask_inference.py``), Qt-free, CPU stub models.

Promises: one image per stem, unreadable inputs recorded and the run goes on (PI-03); the
output equals the plain one-image-at-a-time loop (``_legacy_masks``, the reference of the
batched path: resize to 480 x 640, normalise, ``> MASK_THRESHOLD`` -> 255, 8-bit L PNG);
``skip_stems`` are never written or run (RB-01), compared without case (O7); the model loads
lazily with its stage; the run manifest names the threshold and the overwrite policy (FR-02);
the fixed policy pads the tail, ``auto`` does not (O4); the CPU batch is small (R2). The real
network and its checkpoint: ``test_mask_model.py``, ``test_checkpoints.py``.
"""

from __future__ import annotations

import hashlib
import io
import json
import os

import numpy as np
import pytest
import torch
from PIL import Image

from app import config, provenance
from app.config import MASK_THRESHOLD
from app.inference import mask_inference, runtime
from tests.generation_stubs_support import _images, _read_all, _StubMaskModel


def _legacy_masks(input_dir, model):
    """The pre-refactor MaskGenerationThread loop, for byte equality."""
    tf = mask_inference.mask_transform()
    out = {}
    for name in sorted(os.listdir(input_dir)):
        try:
            img = Image.open(os.path.join(input_dir, name)).convert("RGB")
        except OSError:
            continue
        with torch.inference_mode():
            mask = model(tf(img).unsqueeze(0))[0]
        mask_np = mask.squeeze().float().cpu().numpy()
        binary = (mask_np > MASK_THRESHOLD).astype(np.uint8) * 255
        buf = io.BytesIO()
        Image.fromarray(binary, mode="L").save(buf, "PNG")
        out[os.path.splitext(name)[0] + ".png"] = buf.getvalue()
    return out


def test_mask_generation_records_unreadable_images_and_matches_legacy(tmp_path, cpu_runtime):
    src = _images(
        str(tmp_path / "imgs"), ["a.jpg", "b.png", "garbage.jpg", "c.jpeg"], corrupt={"garbage.jpg"}
    )
    out = str(tmp_path / "imgs" / "mask")
    model = _StubMaskModel().eval()
    report = mask_inference.generate_masks(src, out, model=model, batch_size=2)

    assert report.failed_names == ["garbage.jpg"]
    assert sorted(report.written) == ["a.jpg", "b.png", "c.jpeg"]
    assert not os.path.exists(os.path.join(out, "garbage.png"))
    assert _read_all(out) == _legacy_masks(src, model)
    with Image.open(os.path.join(out, "a.png")) as im:
        assert im.mode == "L"


def test_mask_generation_skips_existing_stems_without_inference(tmp_path, cpu_runtime):
    src = _images(str(tmp_path / "imgs"), ["a.jpg", "b.jpg", "c.jpg"])
    out = str(tmp_path / "imgs" / "mask")
    os.makedirs(out)
    traced = os.path.join(out, "a.png")
    Image.new("RGBA", (64, 48), (255, 255, 255, 255)).save(traced)
    before = open(traced, "rb").read()
    model = _StubMaskModel().eval()

    report = mask_inference.generate_masks(
        src, out, model=model, batch_size=4, skip_stems=frozenset({"a"}), batch_policy="auto"
    )
    assert open(traced, "rb").read() == before
    assert report.skipped == ["a"]
    assert sorted(report.written) == ["b.jpg", "c.jpg"]
    assert model.items == 2


def test_mask_run_writes_provenance_manifest(tmp_path, cpu_runtime, monkeypatch):
    weights = tmp_path / "w.pth"
    weights.write_bytes(b"fake weights")
    monkeypatch.setattr(runtime, "mask_weights_path", lambda: str(weights))
    src = _images(str(tmp_path / "imgs"), ["a.jpg", "bad.jpg"], corrupt={"bad.jpg"})
    out = str(tmp_path / "imgs" / "mask")
    mask_inference.generate_masks(
        src, out, model=_StubMaskModel().eval(), batch_size=2, policy="keep_existing"
    )
    with open(provenance.run_manifest_path(out), encoding="utf-8") as fh:
        record = json.load(fh)
    assert record["kind"] == "mask_generation"
    assert record["models"]["mask"]["sha256"] == hashlib.sha256(b"fake weights").hexdigest()
    assert record["parameters"]["mask_threshold"] == MASK_THRESHOLD
    assert record["parameters"]["overwrite_policy"] == "keep_existing"
    assert record["outputs"]["n_written"] == 1
    assert record["outputs"]["failed"][0]["input"] == "bad.jpg"
    assert record["runtime"]["inference"]["precision"] == "fp32"
    assert record["inputs"]["names_sha256"] == provenance.names_digest(["a.jpg", "bad.jpg"])


def test_mask_run_loads_model_lazily_with_a_stage(tmp_path, cpu_runtime, monkeypatch):
    stub = _StubMaskModel().eval()
    monkeypatch.setattr(runtime, "get_mask_model", lambda device=None: stub)
    stages = []
    src = _images(str(tmp_path / "imgs"), ["a.jpg"])
    mask_inference.generate_masks(
        src, str(tmp_path / "imgs" / "mask"), on_stage=stages.append, batch_policy="auto"
    )
    assert stages == [mask_inference.STAGE_LOADING, mask_inference.STAGE_RUNNING]
    assert stub.items == 1


def _manifest(out):
    with open(provenance.run_manifest_path(out), encoding="utf-8") as fh:
        return json.load(fh)


def test_a_fixed_mask_run_pads_its_tail(tmp_path, cpu_runtime):
    src = _images(str(tmp_path / "imgs"), [f"i{k}.jpg" for k in range(5)])
    model = _StubMaskModel()
    mask_inference.generate_masks(
        src, str(tmp_path / "imgs" / "mask"), model=model, batch_size=2, batch_policy="fixed"
    )
    assert (model.calls, model.items) == (3, 6)


def test_the_default_policy_comes_from_the_environment(tmp_path, cpu_runtime, monkeypatch):
    monkeypatch.setenv(runtime.BATCH_POLICY_ENV, "auto")
    src = _images(str(tmp_path / "imgs"), [f"i{k}.jpg" for k in range(5)])
    out = str(tmp_path / "imgs" / "mask")
    model = _StubMaskModel()

    mask_inference.generate_masks(src, out, model=model, batch_size=2)

    assert (model.calls, model.items) == (3, 5)  # auto: the short tail is not padded
    assert _manifest(out)["parameters"]["batch_policy"] == "auto"


@pytest.mark.parametrize(("device", "auto_size"), [("cpu", 4), ("cuda", config.MASK_BATCH_SIZE)])
def test_the_mask_default_size_follows_the_policy(monkeypatch, device, auto_size):
    fake = runtime.RuntimeConfig(
        device=torch.device(device),
        use_channels_last=False,
        pin_memory=False,
        io_threads=1,
        cudnn_benchmark=False,
        cudnn_deterministic=True,
        precision="fp32",
    )
    monkeypatch.setattr(runtime, "get_runtime", lambda: fake)
    assert mask_inference.default_mask_batch_size(policy="fixed") == 4
    assert mask_inference.default_mask_batch_size(policy="auto") == auto_size


class _StubMask(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 1)
        self.sizes = []

    def forward(self, x):
        self.sizes.append(x.shape[0])
        return torch.sigmoid(self.conv(x))


def test_mask_generation_uses_the_cpu_batch_on_cpu(tmp_path, monkeypatch):
    from app.inference import mask_inference

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    runtime.get_runtime.cache_clear()
    src = tmp_path / "imgs"
    src.mkdir()
    for i in range(10):
        Image.fromarray(np.full((48, 64, 3), 20 * i, np.uint8)).save(src / f"T1_L{i}.png")
    model = _StubMask()
    try:
        report = mask_inference.generate_masks(str(src), str(tmp_path / "mask"), model=model)
    finally:
        runtime.get_runtime.cache_clear()
    assert report.batch_size == config.MASK_BATCH_SIZE_CPU
    assert max(model.sizes) <= config.MASK_BATCH_SIZE_CPU


class _ConstantMaskModel(torch.nn.Module):
    """Outputs exactly ``MASK_THRESHOLD`` on the left half, the next float above on the right."""

    def forward(self, x):
        out = torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), MASK_THRESHOLD)
        out[..., x.shape[3] // 2 :] = torch.nextafter(
            torch.tensor(MASK_THRESHOLD), torch.tensor(1.0)
        )
        return out


def test_a_probability_equal_to_the_threshold_is_background(tmp_path, cpu_runtime):
    # 255 where the output *exceeds* the threshold (the area counts those pixels).
    src = _images(str(tmp_path / "imgs"), ["a.jpg"])
    out = str(tmp_path / "imgs" / "mask")

    mask_inference.generate_masks(src, out, model=_ConstantMaskModel(), batch_policy="auto")

    with Image.open(os.path.join(out, "a.png")) as im:
        mask = np.asarray(im)
    half = mask.shape[1] // 2
    assert (mask[:, :half] == 0).all() and (mask[:, half:] == 255).all()


def test_a_skipped_stem_keeps_the_mask_of_an_image_named_in_another_case(tmp_path, cpu_runtime):
    # O7: skip_stems compare without case, whoever calls generate_masks.
    src = _images(str(tmp_path / "imgs"), ["A.jpg", "b.jpg"])
    out = str(tmp_path / "imgs" / "mask")
    model = _StubMaskModel().eval()

    report = mask_inference.generate_masks(
        src, out, model=model, skip_stems=frozenset({"a"}), batch_policy="auto"
    )

    assert report.skipped == ["A"] and report.written == ["b.jpg"]
    assert sorted(n for n in os.listdir(out) if n.endswith(".png")) == ["b.png"]
    assert model.items == 1  # the model never ran on A

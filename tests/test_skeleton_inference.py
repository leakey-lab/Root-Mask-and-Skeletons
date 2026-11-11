"""Skeleton generation (``app/inference/skeleton_inference.py``), Qt-free, CPU stub models.

Promises: unreadable inputs recorded and the run goes on; the output equals the plain
one-image-at-a-time loop (``_legacy_skeletons``, the reference: bicubic 256 x 256, the
generator, ``(x + 1) / 2 * 255`` truncated, bicubic resize to 341 x 256 with the validated
axis order); ``run_inference`` keeps its legacy contract; ``SkeletonModel`` loads the
checkpoint at ``runtime.skeleton_weights_path()`` strictly and names it when it is missing;
the fixed skeleton batch never asks the GPU (O4). Norm statistics: ``test_skeleton_norm_stats.py``.
"""

from __future__ import annotations

import io
import json
import os

import numpy as np
import pytest
import torch
from PIL import Image

from app import config, provenance
from app.inference import runtime
from app.inference import skeleton_inference as si
from tests.generation_stubs_support import _images, _read_all, _StubSkeletonModel


def _legacy_skeletons(input_dir, model):
    tf = si.get_transform(grayscale=False)
    out = {}
    for path in si.make_dataset(input_dir):
        try:
            img = Image.open(path).convert("RGB")
        except OSError:
            continue
        # the former per-image tensor2im: numpy fp32 (x + 1) / 2 * 255 -> uint8
        chw = model.run_batch(tf(img).unsqueeze(0))[0].cpu().float().numpy()
        arr = ((np.transpose(chw, (1, 2, 0)) + 1) / 2.0 * 255.0).astype(np.uint8)
        img_pil = Image.fromarray(arr)
        h, w, _ = arr.shape
        img_pil = img_pil.resize((int(h / si.ASPECT_RATIO), w), Image.BICUBIC)
        buf = io.BytesIO()
        img_pil.save(buf, "PNG")
        stem = os.path.splitext(os.path.basename(path))[0]
        out[f"{stem}_fake.png"] = buf.getvalue()
    return out


def test_skeleton_run_continues_past_unreadable_images(tmp_path, cpu_runtime):
    src = _images(str(tmp_path / "imgs"), ["a.jpg", "bad.jpg", "c.jpg"], corrupt={"bad.jpg"})
    out = str(tmp_path / "imgs" / "skeletons")
    model = _StubSkeletonModel()
    report = si.run_skeleton_generation(src, out, model=model, batch_size=2)
    assert report.failed_names == ["bad.jpg"]
    assert sorted(report.written) == ["a.jpg", "c.jpg"]
    assert _read_all(out) == _legacy_skeletons(src, model)
    with open(provenance.run_manifest_path(out), encoding="utf-8") as fh:
        assert json.load(fh)["kind"] == "skeleton_generation"


def test_run_inference_keeps_its_contract(tmp_path, cpu_runtime, monkeypatch):
    model = _StubSkeletonModel()
    monkeypatch.setattr(runtime, "get_skeleton_model", lambda device=None: model)
    src = _images(str(tmp_path / "imgs"), ["a.jpg", "b.jpg"])
    out = str(tmp_path / "imgs" / "skeletons")
    ticks = []
    result = si.run_inference(
        src, out, progress_callback=lambda d, t: ticks.append((d, t)), batch_size=1
    )
    assert result == out
    assert sorted(_read_all(out)) == ["a_fake.png", "b_fake.png"]
    assert ticks[-1] == (2, 2)


def test_the_skeleton_model_loads_its_checkpoint_from_the_runtime_path(tmp_path, monkeypatch):
    torch.manual_seed(0)
    net = si.EnhancedResnetGenerator(
        si.INPUT_NC,
        si.OUTPUT_NC,
        si.NGF,
        norm_layer=si.get_norm_layer(norm_type=si.NORM),
        use_dropout=si.USE_DROPOUT,
        n_blocks=9,
    )
    saved = {key: value.clone() for key, value in net.state_dict().items()}
    path = tmp_path / "latest_net_G.pth"
    torch.save(net.state_dict(), path)
    monkeypatch.setattr(runtime, "skeleton_weights_path", lambda: str(path))

    model = si.SkeletonModel(gpu_ids=[])

    loaded = model.netG.state_dict()
    assert model.device == torch.device("cpu") and not model.netG.training
    assert loaded.keys() == saved.keys()
    assert all(torch.equal(loaded[key], saved[key]) for key in saved)
    assert not any(t.is_meta for t in [*model.netG.parameters(), *model.netG.buffers()])
    assert model.run_batch(torch.zeros(1, 3, 64, 64)).shape == (1, 3, 64, 64)


def test_a_missing_skeleton_checkpoint_names_its_path(tmp_path, monkeypatch):
    missing = str(tmp_path / "nope_net_G.pth")
    monkeypatch.setattr(runtime, "skeleton_weights_path", lambda: missing)
    with pytest.raises(
        RuntimeError, match="Failed to load skeleton model weights from .*nope_net_G"
    ):
        si.SkeletonModel(gpu_ids=[])


def test_fixed_skeleton_batch_never_asks_the_gpu(monkeypatch):
    monkeypatch.setattr(config, "get_skeleton_batch_size", lambda refresh=False: pytest.fail())
    assert si.default_skeleton_batch_size("cuda", policy="fixed") == 64
    assert si.default_skeleton_batch_size("cpu", policy="fixed") == 8
    monkeypatch.setattr(config, "get_skeleton_batch_size", lambda refresh=False: 104)
    assert si.default_skeleton_batch_size("cuda", policy="auto") == 104
    assert si.default_skeleton_batch_size("cpu", policy="auto") == 8


def test_a_folder_without_images_writes_nothing(tmp_path, cpu_runtime):
    src = tmp_path / "imgs"
    src.mkdir()
    (src / "notes.txt").write_text("not a photo")
    out = str(src / "skeletons")

    report = si.run_skeleton_generation(str(src), out, model=_StubSkeletonModel())

    assert report.total == 0 and report.written == [] and report.failed == []
    assert report.status == "completed"
    assert os.listdir(out) == []  # no output, no run manifest: nothing changed

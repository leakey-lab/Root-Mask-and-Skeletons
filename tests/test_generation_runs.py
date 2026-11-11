"""What the two generation runs share (``generate_masks``, ``run_skeleton_generation``).

Both take the Library's images (PNG / JPEG, one per stem; folders without a shared stem get
the inputs they always got), record the batch policy and a fixed run's free GPU memory in the
manifest (O4), refuse an unknown policy, stop on out of memory under ``fixed`` with a failed
run record, check free memory only under ``fixed`` and before any forward, give the runtime's
torch-thread count to the engine (S12-01), and end a run with failures with one bounded log
line (C6-02). CPU stub models, no checkpoint, no GPU.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import pytest
import torch
from PIL import Image

from app import config, provenance
from app.inference import mask_inference, runtime
from app.inference import mask_inference as mi
from app.inference import skeleton_inference as si
from app.inference.batch_engine import (
    FIXED_OOM_MESSAGE,
    FixedBatchMemoryTooLow,
    FixedBatchOutOfMemory,
)
from tests.generation_stubs_support import (
    _images,
    _StubMaskModel,
    _StubSkeletonModel,
)


def _manifest(out):
    with open(provenance.run_manifest_path(out), encoding="utf-8") as fh:
        return json.load(fh)


def _pngs(out):
    return sorted(name for name in os.listdir(out) if name.endswith(".png"))


class _OomMaskModel(_StubMaskModel):
    """Succeeds on forward 1, then runs out of GPU memory on every later forward."""

    def forward(self, x):
        if self.calls >= 1:
            self.calls += 1
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (stub)")
        return super().forward(x)


class _OomSkeletonModel(_StubSkeletonModel):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def run_batch(self, x):
        self.calls += 1
        if self.calls > 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (stub)")
        return super().run_batch(x)


def _run(kind, src, out, model, **kwargs):
    if kind == "mask":
        return mask_inference.generate_masks(src, out, model=model, **kwargs)
    return si.run_skeleton_generation(src, out, model=model, **kwargs)


def _model(kind):
    return _StubMaskModel().eval() if kind == "mask" else _StubSkeletonModel()


@pytest.mark.parametrize("kind", ["mask", "skeleton"])
def test_a_fixed_stop_is_a_failed_run_record(tmp_path, cpu_runtime, kind):
    src = _images(str(tmp_path / "imgs"), [f"i{k}.jpg" for k in range(5)])
    out = str(tmp_path / "imgs" / "out")
    model = _OomMaskModel() if kind == "mask" else _OomSkeletonModel()

    with pytest.raises(FixedBatchOutOfMemory):
        _run(kind, src, out, model, batch_size=2, batch_policy="fixed")

    record = _manifest(out)
    assert record["outputs"]["status"] == "failed"
    assert record["outputs"]["error"] == "FixedBatchOutOfMemory: " + FIXED_OOM_MESSAGE.format(n=2)
    assert record["parameters"]["batch_policy"] == "fixed"
    assert record["outputs"]["oom_events"] == []
    assert len(_pngs(out)) == 2


@pytest.mark.parametrize("kind", ["mask", "skeleton"])
@pytest.mark.parametrize("policy", ["auto", "fixed"])
def test_the_manifest_records_the_batch_policy(tmp_path, cpu_runtime, kind, policy):
    src = _images(str(tmp_path / "imgs"), [f"i{k}.jpg" for k in range(3)])
    out = str(tmp_path / "imgs" / "out")
    model = _StubMaskModel() if kind == "mask" else _StubSkeletonModel()

    report = _run(kind, src, out, model, batch_size=2, batch_policy=policy)

    record = _manifest(out)
    assert record["parameters"]["batch_policy"] == policy
    assert "free_vram_gib" not in record["parameters"]
    assert record["outputs"]["free_vram_gib"] is None  # a CPU run measures no GPU memory
    assert len(report.written) == 3


@pytest.mark.parametrize("kind", ["mask", "skeleton"])
def test_an_unknown_explicit_policy_is_refused(tmp_path, cpu_runtime, kind):
    src = _images(str(tmp_path / "imgs"), ["a.jpg"])
    with pytest.raises(ValueError, match="bogus"):
        _run(kind, src, str(tmp_path / "imgs" / "out"), None, batch_policy="bogus")


@pytest.mark.parametrize("kind", ["mask", "skeleton"])
def test_a_fixed_run_checks_memory_and_records_it(tmp_path, cpu_runtime, monkeypatch, kind):
    calls = []

    def check(model, batch_size, device_type, **_mode):  # skeleton: norm_stats=
        calls.append((model, batch_size, device_type))
        return 7.4567

    monkeypatch.setattr(runtime, "check_fixed_batch_memory", check)
    src = _images(str(tmp_path / "imgs"), [f"i{k}.jpg" for k in range(3)])
    out = str(tmp_path / "imgs" / "out")
    model = _StubMaskModel() if kind == "mask" else _StubSkeletonModel()

    _run(kind, src, out, model, batch_size=2, batch_policy="fixed")

    assert calls == [(kind, 2, "cpu")]
    record = _manifest(out)
    assert record["outputs"]["free_vram_gib"] == 7.46  # next to batch_shapes / oom_events
    assert "free_vram_gib" not in record["parameters"]


@pytest.mark.parametrize("kind", ["mask", "skeleton"])
def test_an_auto_run_never_checks_memory(tmp_path, cpu_runtime, monkeypatch, kind):
    monkeypatch.setattr(runtime, "check_fixed_batch_memory", lambda *a, **k: pytest.fail("checked"))
    src = _images(str(tmp_path / "imgs"), ["a.jpg"])
    model = _StubMaskModel() if kind == "mask" else _StubSkeletonModel()
    _run(kind, src, str(tmp_path / "imgs" / "out"), model, batch_size=2, batch_policy="auto")


@pytest.mark.parametrize("kind", ["mask", "skeleton"])
def test_a_memory_stop_runs_no_forward(tmp_path, cpu_runtime, monkeypatch, kind):
    def check(model, batch_size, device_type, **_mode):  # skeleton: norm_stats=
        raise FixedBatchMemoryTooLow(batch_size, free_gib=1.0, needed_gib=3.6)

    monkeypatch.setattr(runtime, "check_fixed_batch_memory", check)
    src = _images(str(tmp_path / "imgs"), [f"i{k}.jpg" for k in range(3)])
    out = str(tmp_path / "imgs" / "out")
    model = _OomMaskModel() if kind == "mask" else _OomSkeletonModel()

    with pytest.raises(FixedBatchMemoryTooLow):
        _run(kind, src, out, model, batch_size=2, batch_policy="fixed")

    assert model.calls == 0
    assert _pngs(out) == []


@pytest.mark.parametrize("kind", ["mask", "skeleton"])
@pytest.mark.parametrize(("device_type", "threads"), [("cpu", None), ("cuda", 1)])
def test_a_run_passes_the_runtimes_torch_threads(tmp_path, monkeypatch, kind, device_type, threads):
    rt = runtime.RuntimeConfig(
        device=torch.device(device_type),
        use_channels_last=False,
        pin_memory=False,
        io_threads=2,
        cudnn_benchmark=False,
        cudnn_deterministic=True,
        precision="fp32",
    )
    monkeypatch.setattr(runtime, "get_runtime", lambda: rt)
    seen = []

    def fake_run_batches(paths, spec, **kwargs):
        seen.append(kwargs["torch_threads"])
        return kwargs["report"]

    monkeypatch.setattr(mask_inference if kind == "mask" else si, "run_batches", fake_run_batches)
    src = _images(str(tmp_path / "imgs"), ["a.jpg"])
    _run(kind, src, str(tmp_path / "out"), _model(kind), batch_size=1, batch_policy="auto")
    assert seen == [threads]


@pytest.mark.parametrize(("kind", "label"), [("mask", "Mask"), ("skeleton", "Skeleton")])
def test_a_run_with_failures_ends_with_one_bounded_line(tmp_path, cpu_runtime, caplog, kind, label):
    bad = [f"bad{i:02d}.jpg" for i in range(8)]
    src = _images(str(tmp_path / "imgs"), bad + ["good.jpg"], corrupt=set(bad))
    out = str(tmp_path / "imgs" / "out")
    with caplog.at_level(logging.WARNING):
        report = _run(kind, src, out, _model(kind), batch_size=2)
    assert report.failed_names == bad
    [line] = [r.getMessage() for r in caplog.records if "input(s) failed" in r.getMessage()]
    assert line.startswith(f"{label} generation: 8 input(s) failed: bad00.jpg: ")
    assert "bad04.jpg" in line and "bad05.jpg" not in line and "; and 3 more; " in line
    assert line.endswith(f"every one is listed in {provenance.run_manifest_path(out)}")


def test_inputs_without_a_shared_stem_are_unchanged(tmp_path):
    src = tmp_path / "imgs"
    src.mkdir()
    names = ["B.JPG", "a.jpeg", "c.png", "d.jpg", "e.PNG", "notes.txt"]
    for name in names:
        (src / name).write_bytes(b"x")
    legacy_masks = sorted(
        n for n in os.listdir(src) if n.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    legacy_skeletons = [
        os.path.join(str(src), n) for n in sorted(os.listdir(src)) if config.is_image_file(n)
    ]

    all_names, todo, skipped = mi.list_inputs(str(src))
    assert all_names == legacy_masks
    assert todo == [os.path.join(str(src), n) for n in legacy_masks] and skipped == []
    assert si.make_dataset(str(src)) == legacy_skeletons


@pytest.mark.parametrize("kind", ["mask", "skeleton"])
def test_the_output_files_have_the_models_grid_and_format(tmp_path, cpu_runtime, kind):
    # A photo of any size gives a 640 x 480 8-bit mask of 0 / 255 and a 341 x 256 RGB
    # skeleton (config.SKELETON_OUTPUT_SIZE): the grids Measure's pixel pitch is taken from.
    src = _images(str(tmp_path / "imgs"), ["a.jpg"], size=(100, 70))
    out = str(tmp_path / "imgs" / "out")

    _run(kind, src, out, _model(kind), batch_size=1)

    name = "a.png" if kind == "mask" else "a_fake.png"
    with Image.open(os.path.join(out, name)) as im:
        size, mode, pixels = im.size, im.mode, np.asarray(im)
    if kind == "mask":
        assert (size, mode) == ((640, 480), "L")
        assert set(np.unique(pixels)) <= {0, 255}
    else:
        assert (size, mode) == (config.SKELETON_OUTPUT_SIZE, "RGB")

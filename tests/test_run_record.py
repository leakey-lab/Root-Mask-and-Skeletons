"""The run manifest of a generation run that fails, is cancelled or cannot describe itself.

R7: a run that raises after writing outputs replaces ``_sprouts_run.json`` with its partial
outcome (``status`` "failed", the error, ``n_written`` = the files on disk); a run that wrote
nothing keeps the previous record; the run's own exception always propagates unchanged. R2:
releasing the CUDA cache never replaces the run's error nor fails a finished run. R5: a record
that cannot be built is replaced by a minimal honest one, and if even that cannot be written
the outdated record is removed. The no-op runs and the history: ``test_run_record_history.py``.
CPU stub models, no checkpoints.
"""

from __future__ import annotations

import errno
import json
import logging
import os

import pytest
import torch

from app import provenance
from app.inference import mask_inference, run_record, runtime
from app.inference import skeleton_inference as si
from app.inference.batch_engine import BatchReport
from tests.generation_stubs_support import _images, _StubMaskModel, _StubSkeletonModel

STALE = b'{"kind": "an older run into this folder"}\n'


ERROR_TEXT = "cuDNN error: CUDNN_STATUS_INTERNAL_ERROR (simulated)"


KINDS = ["mask", "skeleton"]


class _MaskModel(_StubMaskModel):
    """Stub mask model that raises a non-out-of-memory error on forward ``fail_on``."""

    def __init__(self, fail_on=None):
        super().__init__()
        self.fail_on = fail_on
        self.error = RuntimeError(ERROR_TEXT)

    def forward(self, x):
        if self.calls + 1 == self.fail_on:
            raise self.error
        return super().forward(x)


class _SkeletonModel(_StubSkeletonModel):
    """Stub skeleton model that raises a non-out-of-memory error on batch ``fail_on``."""

    def __init__(self, fail_on=None):
        super().__init__()
        self.fail_on = fail_on
        self.calls = 0
        self.error = RuntimeError(ERROR_TEXT)

    def run_batch(self, x):
        if self.calls + 1 == self.fail_on:
            raise self.error
        self.calls += 1
        return super().run_batch(x)


def _model(kind, fail_on=None):
    return _MaskModel(fail_on) if kind == "mask" else _SkeletonModel(fail_on)


def _run(kind, src, out, model=None, **kwargs):
    if kind == "mask":
        return mask_inference.generate_masks(src, out, model=model, batch_size=2, **kwargs)
    return si.run_skeleton_generation(src, out, model=model, batch_size=2, **kwargs)


def _folder_with_stale_manifest(path):
    os.makedirs(path)
    with open(provenance.run_manifest_path(path), "wb") as fh:
        fh.write(STALE)
    return str(path)


def _manifest_bytes(out):
    with open(provenance.run_manifest_path(out), "rb") as fh:
        return fh.read()


def _manifest(out):
    return json.loads(_manifest_bytes(out).decode("utf-8"))


def _pngs(out):
    return sorted(name for name in os.listdir(out) if name.endswith(".png"))


@pytest.mark.parametrize("kind", KINDS)
def test_a_run_that_fails_mid_run_replaces_the_stale_manifest(
    tmp_path, cpu_runtime, monkeypatch, kind
):
    released = []
    monkeypatch.setattr(runtime, "release_cuda_cache", lambda: released.append(1))
    src = _images(str(tmp_path / "imgs"), [f"i{k}.jpg" for k in range(5)])
    out = _folder_with_stale_manifest(tmp_path / "imgs" / "out")
    model = _model(kind, fail_on=2)

    with pytest.raises(RuntimeError) as caught:
        _run(kind, src, out, model)

    assert caught.value is model.error  # the run's own error, unchanged
    record = _manifest(out)
    assert record["kind"] == (mask_inference.RUN_KIND if kind == "mask" else si.RUN_KIND)
    outputs = record["outputs"]
    assert outputs["status"] == "failed"
    assert outputs["error"] == f"RuntimeError: {ERROR_TEXT}"
    assert outputs["n_written"] == len(_pngs(out)) == 2  # the first batch of two
    assert outputs["n_inputs"] == 5 and outputs["cancelled"] is False
    assert released == [1]


def test_a_failed_mask_run_still_records_its_skipped_stems(tmp_path, cpu_runtime):
    src = _images(str(tmp_path / "imgs"), [f"i{k}.jpg" for k in range(5)])
    out = _folder_with_stale_manifest(tmp_path / "imgs" / "mask")

    with pytest.raises(RuntimeError):
        _run("mask", src, out, _MaskModel(fail_on=2), skip_stems=frozenset({"i0"}))

    outputs = _manifest(out)["outputs"]
    assert outputs["status"] == "failed"
    assert outputs["skipped"] == ["i0"] and outputs["n_skipped"] == 1
    assert outputs["n_written"] == len(_pngs(out)) == 2


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("stage", ["first batch", "model load"])
def test_a_run_that_fails_before_writing_keeps_the_previous_manifest(
    tmp_path, cpu_runtime, monkeypatch, kind, stage
):
    src = _images(str(tmp_path / "imgs"), [f"i{k}.jpg" for k in range(3)])
    out = _folder_with_stale_manifest(tmp_path / "imgs" / "out")
    if stage == "first batch":
        model = _model(kind, fail_on=1)
        error = model.error
    else:
        model, error = None, RuntimeError("weights missing (simulated)")

        def fail_to_load(device=None):
            raise error

        getter = "get_mask_model" if kind == "mask" else "get_skeleton_model"
        monkeypatch.setattr(runtime, getter, fail_to_load)

    with pytest.raises(RuntimeError) as caught:
        _run(kind, src, out, model)

    assert caught.value is error
    assert _pngs(out) == []
    assert _manifest_bytes(out) == STALE  # still describes the outputs in the folder


@pytest.mark.parametrize("kind", KINDS)
def test_a_normal_run_is_completed_without_an_error(tmp_path, cpu_runtime, kind):
    src = _images(str(tmp_path / "imgs"), [f"i{k}.jpg" for k in range(3)])
    out = _folder_with_stale_manifest(tmp_path / "imgs" / "out")

    report = _run(kind, src, out, _model(kind))

    outputs = _manifest(out)["outputs"]
    assert outputs["status"] == report.status == "completed"
    assert outputs["error"] is None and report.error is None
    assert outputs["n_written"] == len(_pngs(out)) == 3


@pytest.mark.parametrize("kind", KINDS)
def test_a_cancelled_run_is_cancelled(tmp_path, cpu_runtime, kind):
    src = _images(str(tmp_path / "imgs"), [f"i{k}.jpg" for k in range(5)])
    out = _folder_with_stale_manifest(tmp_path / "imgs" / "out")
    model = _model(kind)

    report = _run(kind, src, out, model, should_stop=lambda: model.calls >= 1)

    outputs = _manifest(out)["outputs"]
    assert outputs["status"] == report.status == "cancelled"
    assert outputs["error"] is None
    assert outputs["n_written"] == len(_pngs(out)) == 2


STICKY = "CUDA error: an illegal memory access was encountered (simulated, sticky)"


def _sticky_cuda(monkeypatch):
    """From now on CUDA looks initialised and every cache release fails (sticky error).

    Armed after the stub model is built: its ``torch.manual_seed`` would otherwise
    try to seed the (absent) CUDA generators.
    """
    calls = []

    def empty_cache():
        calls.append(1)
        raise RuntimeError(STICKY)

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    return calls


@pytest.mark.parametrize("kind", KINDS)
def test_a_failing_cache_release_never_replaces_the_run_error(
    tmp_path,
    cpu_runtime,
    monkeypatch,
    kind,
):
    src = _images(str(tmp_path / "imgs"), [f"i{k}.jpg" for k in range(5)])
    out = _folder_with_stale_manifest(tmp_path / "imgs" / "out")
    model = _model(kind, fail_on=2)
    sticky_cuda = _sticky_cuda(monkeypatch)

    with pytest.raises(RuntimeError) as caught:
        _run(kind, src, out, model)

    assert caught.value is model.error  # the run's own error, not the cleanup's
    assert sticky_cuda == [1]  # the release was still attempted, once
    outputs = _manifest(out)["outputs"]
    assert outputs["status"] == "failed" and outputs["error"] == f"RuntimeError: {ERROR_TEXT}"


@pytest.mark.parametrize("kind", KINDS)
def test_a_failing_cache_release_never_fails_a_finished_run(
    tmp_path,
    cpu_runtime,
    monkeypatch,
    kind,
):
    src = _images(str(tmp_path / "imgs"), [f"i{k}.jpg" for k in range(3)])
    out = _folder_with_stale_manifest(tmp_path / "imgs" / "out")
    model = _model(kind)
    sticky_cuda = _sticky_cuda(monkeypatch)

    report = _run(kind, src, out, model)

    assert report.status == "completed" and len(report.written) == len(_pngs(out)) == 3
    outputs = _manifest(out)["outputs"]  # the finished run's manifest was written
    assert outputs["status"] == "completed" and outputs["n_written"] == 3
    assert sticky_cuda == [1]


RECORD_KEYS = {
    "schema",
    "kind",
    "created_utc",
    "app",
    "git_commit",
    "calibration",
    "runtime",
    "models",
    "inputs",
    "parameters",
    "outputs",
}


def _broken_build_record(*args, **kwargs):
    raise KeyError("provenance bug (simulated)")


def _run_kind(kind):
    return mask_inference.RUN_KIND if kind == "mask" else si.RUN_KIND


@pytest.mark.parametrize("kind", KINDS)
def test_a_record_that_cannot_be_built_is_replaced_by_a_minimal_one(
    tmp_path,
    cpu_runtime,
    monkeypatch,
    caplog,
    kind,
):
    monkeypatch.setattr(provenance, "build_record", _broken_build_record)
    src = _images(str(tmp_path / "imgs"), ["a.jpg", "bad.jpg", "c.jpg"], corrupt={"bad.jpg"})
    out = _folder_with_stale_manifest(tmp_path / "imgs" / "out")

    with caplog.at_level(logging.WARNING):
        report = _run(kind, src, out, _model(kind))

    assert report.status == "completed" and len(_pngs(out)) == 2  # the run still succeeds
    record = _manifest(out)  # no longer the older run's record
    assert RECORD_KEYS <= set(record)
    assert record["schema"] == provenance.SCHEMA and record["kind"] == _run_kind(kind)
    assert record["created_utc"].endswith("Z")
    assert record["outputs"] == report.summary()  # the outcome is fully known
    assert record["outputs"]["failed"] == [{"input": "bad.jpg", "reason": report.failed[0][1]}]
    assert "provenance bug (simulated)" in record["provenance_error"]
    assert record["models"] == {} and record["git_commit"] is None  # not described: empty
    assert "provenance bug (simulated)" in caplog.text


@pytest.mark.parametrize("kind", KINDS)
def test_a_failed_run_whose_record_cannot_be_built_still_says_it_failed(
    tmp_path,
    cpu_runtime,
    monkeypatch,
    kind,
):
    monkeypatch.setattr(provenance, "build_record", _broken_build_record)
    src = _images(str(tmp_path / "imgs"), [f"i{k}.jpg" for k in range(5)])
    out = _folder_with_stale_manifest(tmp_path / "imgs" / "out")
    model = _model(kind, fail_on=2)

    with pytest.raises(RuntimeError) as caught:
        _run(kind, src, out, model)

    assert caught.value is model.error  # the run's own error, unchanged
    outputs = _manifest(out)["outputs"]
    assert outputs["status"] == "failed" and outputs["error"] == f"RuntimeError: {ERROR_TEXT}"
    assert outputs["n_written"] == len(_pngs(out)) == 2


@pytest.mark.parametrize("kind", KINDS)
def test_when_no_record_can_be_written_the_outdated_one_is_removed(
    tmp_path,
    cpu_runtime,
    monkeypatch,
    caplog,
    kind,
):
    attempts = []

    def disk_full(path, record):
        attempts.append(record)
        raise OSError(errno.ENOSPC, "No space left on device (simulated)", path)

    monkeypatch.setattr(provenance, "_write_json", disk_full)
    src = _images(str(tmp_path / "imgs"), [f"i{k}.jpg" for k in range(3)])
    out = _folder_with_stale_manifest(tmp_path / "imgs" / "out")

    with caplog.at_level(logging.WARNING):
        report = _run(kind, src, out, _model(kind))

    assert report.status == "completed" and len(_pngs(out)) == 3
    assert len(attempts) == 2 and "provenance_error" in attempts[1]  # full, then minimal
    assert not os.path.exists(provenance.run_manifest_path(out))  # never the older run's
    assert "No space left on device" in caplog.text


def test_an_outdated_record_that_cannot_be_removed_is_logged(tmp_path, monkeypatch, caplog):
    out = _folder_with_stale_manifest(tmp_path / "out")

    def denied(*args, **kwargs):
        raise PermissionError(errno.EACCES, "Access is denied (simulated)")

    monkeypatch.setattr(provenance, "_write_json", denied)
    monkeypatch.setattr(run_record.os, "remove", denied)
    with caplog.at_level(logging.WARNING):
        path = run_record.write_run_manifest(
            "mask_generation",
            input_dir=str(tmp_path),
            output_dir=out,
            input_names=["a.jpg"],
            report=BatchReport(total=1, written=["a.jpg"]),
            models={},
            parameters={},
        )

    assert path is None  # never raises: the outputs exist
    with open(provenance.run_manifest_path(out), "rb") as fh:
        assert fh.read() == STALE
    assert "Could not remove the outdated run manifest" in caplog.text


def test_a_record_that_can_be_built_is_written_in_full(tmp_path):
    out = str(tmp_path / "out")
    os.makedirs(out)
    report = BatchReport(total=1, written=["a.jpg"])

    path = run_record.write_run_manifest(
        "mask_generation",
        input_dir=str(tmp_path),
        output_dir=out,
        input_names=["a.jpg"],
        report=report,
        models={},
        parameters={"batch_size": 1},
    )

    record = _manifest(out)
    assert path == provenance.run_manifest_path(out)
    assert "provenance_error" not in record
    assert record["parameters"] == {"batch_size": 1} and record["outputs"] == report.summary()
    assert record["inputs"]["n_images"] == 1


def test_the_minimal_record_has_every_record_key():
    outputs = BatchReport(total=2, written=["a.jpg"], failed=[("b.jpg", "OSError: x")]).summary()
    record = provenance.minimal_record("skeleton_generation", outputs=outputs, reason="boom")

    assert set(record) == RECORD_KEYS | {"provenance_error"}
    assert record["outputs"] == outputs and record["provenance_error"] == "boom"
    assert record["app"] == provenance.app_info()
    assert record["calibration"] == provenance.calibration_info()

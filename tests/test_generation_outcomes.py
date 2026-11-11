"""How a generation run ends, for both handlers (``app/handlers/generation_base.py``).

R6: a run in which every input failed (out of memory even on single images, or every photo
unreadable) is a failure: ``task_failed``, the one non-modal report (UXS-02) that claims no
other outputs, nothing chains, no skeleton results load. A partial failure and a run with
nothing to do still complete and chain; kept masks are not failures. A finished run's toast
has one "Open folder" button that opens the folder it wrote; a run with failures reports
them instead. Offscreen Qt, CPU stub models, no checkpoints.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from PyQt6 import sip
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtWidgets import QWidget

from app.feedback import RecordingFeedback
from app.handlers import generate_skeleton_handler as gsh
from app.handlers import generation_base
from app.handlers import mask_generation_handler as mgh
from app.handlers.generation_base import show_failure_report
from app.handlers.job_registry import JobRegistry
from app.inference.batch_engine import BatchReport
from app.inference.mask_origin import OverwritePolicy
from tests.generation_support import generation_project
from tests.generation_support import wait_for_thread as _wait
from tests.qt_teardown import flush_deleted


@pytest.fixture
def project(tmp_path, qapp, monkeypatch):
    """A folder of 3 images the image manager does not list, CPU runtime, stub models and a
    fake main window."""
    yield from generation_project(tmp_path, monkeypatch, list_images=False)


def _messages(feedback, name):
    return [c.args[0] for c in feedback.calls if c.name == name]


def _toast_kinds(feedback):
    return [c.args[1] for c in feedback.calls if c.name == "notify"]


def _unreadable(folder, *names):
    for name in names:
        (folder / name).write_bytes(b"not an image")


def test_mask_run_where_every_image_failed_is_a_failure(project, qapp, monkeypatch):
    monkeypatch.setenv("SPROUTS_BATCH_POLICY", "auto")  # (O4) halving on out of memory
    project.stub.oom = True  # [oom-always]: even a single image runs out of memory
    chained = []
    handler = mgh.MaskGenerationHandler(project.window)
    assert handler.generate_masks(on_success=lambda: chained.append(True)) is True
    thread = handler.generation_thread
    _wait(thread, qapp)

    assert thread.report.written == [] and len(thread.report.failed) == 3
    fb = project.window.feedback
    assert "task_finished" not in fb.names()
    assert _messages(fb, "task_failed") == [
        "Mask generation failed: none of 3 images could be processed"
    ]
    assert "success" not in _toast_kinds(fb)
    assert not fb.busy
    # UXS-02: the one non-modal report lists every input with its reason; no modal box
    assert [kind for kind, _ in project.dialogs] == ["report"]
    _parent, title, _out, failed = project.dialogs[0][1]
    assert title == "Mask generation"
    assert sorted(name for name, _ in failed) == ["a.jpg", "b.jpg", "c.jpg"]
    assert all("out of memory" in reason for _, reason in failed)
    assert project.report_kwargs == [{"nothing_written": True}]  # no "Other outputs" line
    # nothing chains: Generate must not start the skeleton pass
    assert thread.completed is False
    assert chained == []
    assert project.window.outputs_changed == [True]


def test_skeleton_run_where_every_image_failed_is_a_failure(project, qapp):
    _unreadable(project.folder, "a.jpg", "b.jpg", "c.jpg")
    handler = gsh.GenerateSkeletonHandler(project.window)
    assert handler.generate_skeleton() is True
    thread = handler._thread
    _wait(thread, qapp)

    assert thread.report.written == [] and len(thread.report.failed) == 3
    fb = project.window.feedback
    assert "task_finished" not in fb.names()
    assert _messages(fb, "task_failed") == [
        "Skeleton generation failed: none of 3 images could be processed"
    ]
    assert "success" not in _toast_kinds(fb)
    assert [kind for kind, _ in project.dialogs] == ["report"]
    _parent, title, _out, failed = project.dialogs[0][1]
    assert title == "Skeleton generation"
    assert sorted(name for name, _ in failed) == ["a.jpg", "b.jpg", "c.jpg"]
    assert project.report_kwargs == [{"nothing_written": True}]
    assert thread.completed is False
    assert project.window.loaded == []  # no results to show


def test_the_report_of_a_run_that_wrote_nothing_claims_no_other_outputs(qapp, tmp_path):
    parent = QWidget()
    failed = [("a.jpg", "UnidentifiedImageError: x"), ("b.jpg", "ItemOutOfMemory: y")]
    box = show_failure_report(
        parent, "Mask generation", str(tmp_path), failed, nothing_written=True
    )
    try:
        assert not box.isModal()
        assert "Other outputs" not in box.informativeText()
        assert "No output was written" in box.informativeText()
        assert str(tmp_path) in box.informativeText()
        assert box.detailedText().splitlines() == [f"{name}: {reason}" for name, reason in failed]
    finally:
        box.close()
        parent.deleteLater()
        flush_deleted()


def test_the_report_box_shows_modeless_and_frees_itself_once_closed(qapp, tmp_path):
    """One non-modal box (UXS-02) under the window, deleted once closed (app.dialogs)."""
    parent = QWidget()
    box = show_failure_report(parent, "Mask generation", str(tmp_path), [("a.jpg", "x")])
    try:
        assert box.isVisible() and not box.isModal()
        assert box.windowModality() == Qt.WindowModality.NonModal
        assert box.testAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        assert box.parent() is parent
        box.close()
        flush_deleted()
        assert sip.isdeleted(box)
    finally:
        parent.deleteLater()
        flush_deleted()


def test_mask_partial_failure_still_completes_and_chains(project, qapp):
    _unreadable(project.folder, "zz_bad.jpg")
    chained = []
    handler = mgh.MaskGenerationHandler(project.window)
    handler.generate_masks(on_success=lambda: chained.append(True))
    thread = handler.generation_thread
    _wait(thread, qapp)

    fb = project.window.feedback
    assert "task_failed" not in fb.names()
    assert _messages(fb, "task_finished") == ["Generated 3 masks"]
    assert ("1 images could not be processed", "warn") in [
        c.args for c in fb.calls if c.name == "notify"
    ]
    assert [kind for kind, _ in project.dialogs] == ["report"]
    assert project.report_kwargs == [{}]  # other outputs were written
    assert thread.completed is True
    assert chained == [True]
    assert project.window.outputs_changed == [True]


def test_skeleton_partial_failure_still_completes_and_loads_results(project, qapp):
    _unreadable(project.folder, "zz_bad.jpg")
    handler = gsh.GenerateSkeletonHandler(project.window)
    handler.generate_skeleton()
    thread = handler._thread
    _wait(thread, qapp)

    fb = project.window.feedback
    assert "task_failed" not in fb.names()
    assert _messages(fb, "task_finished") == ["Generated 3 skeletons"]
    assert [kind for kind, _ in project.dialogs] == ["report"]
    assert project.report_kwargs == [{}]
    assert thread.completed is True
    assert project.window.loaded == [os.path.normpath(str(project.folder / "skeletons"))]


def test_a_run_that_kept_masks_is_not_a_run_where_every_input_failed(project, qapp):
    # Keep-existing: a and b keep their masks, and the only image attempted (c) fails.
    # The kept inputs did not fail, so this is a partial failure: Generate still chains
    # the skeleton pass (it reads the images, not the masks).
    mask_dir = project.folder / "mask"
    mask_dir.mkdir()
    for stem in ("a", "b"):
        Image.fromarray(np.zeros((48, 64), np.uint8)).save(mask_dir / f"{stem}.png")
    _unreadable(project.folder, "c.jpg")
    chained = []
    handler = mgh.MaskGenerationHandler(
        project.window, confirm_overwrite=lambda e, t: OverwritePolicy.KEEP_EXISTING
    )
    handler.generate_masks(on_success=lambda: chained.append(True))
    thread = handler.generation_thread
    _wait(thread, qapp)

    assert thread.report.written == [] and thread.report.failed_names == ["c.jpg"]
    fb = project.window.feedback
    assert "task_failed" not in fb.names()
    assert _messages(fb, "task_finished") == ["Generated 0 masks, kept 2 existing predictions"]
    assert ("1 images could not be processed", "warn") in [
        c.args for c in fb.calls if c.name == "notify"
    ]
    assert [kind for kind, _ in project.dialogs] == ["report"]
    assert project.report_kwargs == [{}]  # the kept masks are the other outputs
    assert thread.completed is True
    assert chained == [True]


def test_a_run_with_nothing_to_do_still_completes_and_chains(project, qapp):
    mask_dir = project.folder / "mask"
    mask_dir.mkdir()
    for stem in ("a", "b", "c"):
        Image.fromarray(np.zeros((48, 64), np.uint8)).save(mask_dir / f"{stem}.png")
    chained = []
    handler = mgh.MaskGenerationHandler(
        project.window, confirm_overwrite=lambda e, t: OverwritePolicy.KEEP_EXISTING
    )
    handler.generate_masks(on_success=lambda: chained.append(True))
    thread = handler.generation_thread
    _wait(thread, qapp)

    fb = project.window.feedback
    assert "task_failed" not in fb.names()
    assert _messages(fb, "task_finished") == ["Generated 0 masks, kept 3 existing predictions"]
    assert "success" in _toast_kinds(fb)
    assert project.dialogs == []
    assert thread.completed is True
    assert chained == [True]


def _report(written=3, failed=()):
    report = BatchReport(total=written + len(failed))
    report.written = [f"{i}.jpg" for i in range(written)]
    report.failed = list(failed)
    return report


def _window(root):
    return SimpleNamespace(
        image_manager=SimpleNamespace(original_folder=str(root), images={}),
        on_outputs_changed=lambda: None,
        load_results=lambda _folder: None,
    )


def _mask_done(root, report):
    feedback = RecordingFeedback()
    handler = mgh.MaskGenerationHandler(_window(root), feedback=feedback, jobs=JobRegistry())
    handler._report_failures = lambda *_a: None  # its dialog is tested elsewhere
    out = str(root / "mask")
    handler._on_finished(SimpleNamespace(report=report), out, [])
    return feedback, out


def _skeleton_done(root, report):
    feedback = RecordingFeedback()
    handler = gsh.GenerateSkeletonHandler(_window(root), feedback=feedback, jobs=JobRegistry())
    handler._report_failures = lambda *_a: None
    out = str(root / "skeletons")
    handler._on_finished(SimpleNamespace(report=report), out)
    return feedback, out


def _notices(feedback):
    return [call for call in feedback.calls if call.name == "notify"]


@pytest.fixture
def opened(monkeypatch):
    urls: list[QUrl] = []
    monkeypatch.setattr(generation_base, "QDesktopServices", SimpleNamespace(openUrl=urls.append))
    return urls


@pytest.mark.parametrize("done", [_mask_done, _skeleton_done], ids=["mask", "skeleton"])
def test_the_done_toast_opens_the_output_folder(tmp_path, opened, done):
    feedback, out = done(tmp_path / "proj", _report(3))

    (notice,) = _notices(feedback)
    action = notice.kwargs["action"]
    assert action.label == "Open folder"
    assert opened == []  # nothing opens until the button is pressed

    action.run()
    assert opened == [QUrl.fromLocalFile(out)]


@pytest.mark.parametrize("done", [_mask_done, _skeleton_done], ids=["mask", "skeleton"])
def test_a_run_with_failures_shows_no_folder_button(tmp_path, done):
    feedback, _out = done(tmp_path / "proj", _report(2, failed=[("bad.jpg", "unreadable")]))
    assert all("action" not in call.kwargs for call in _notices(feedback))


@pytest.mark.parametrize(
    ("done", "message"),
    [(_mask_done, "Generated 1,234 masks"), (_skeleton_done, "Generated 1,234 skeletons")],
    ids=["mask", "skeleton"],
)
def test_the_done_message_counts_with_thousands_separators(tmp_path, done, message):
    feedback, _out = done(tmp_path / "proj", _report(1234))
    assert [c.args[0] for c in feedback.calls if c.name == "task_finished"] == [message]

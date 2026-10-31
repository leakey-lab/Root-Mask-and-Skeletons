"""A finished Generate's toast opens the folder it wrote (2026-10-31 capsule toasts).

The done toast of mask and of skeleton generation carries one "Open folder" button; it
shows the output folder in the system file manager and closes the toast. A run that left
failures reports them instead, with no button.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PyQt6.QtCore import QUrl

from app.feedback import RecordingFeedback
from app.handlers import generate_skeleton_handler as gsh
from app.handlers import generation_base
from app.handlers import mask_generation_handler as mgh
from app.handlers.job_registry import JobRegistry
from app.inference.batch_engine import BatchReport


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

"""Mask / skeleton generation handlers (offscreen Qt, CPU stub models, no checkpoints).

RB-01 / PI-04 / FR-04 / UXS-01: existing predictions -> one prompt (WG-T7: Keep / Replace
    predictions, gt/ is never touched); Cancel starts nothing (so generate_all chains nothing).
PG-04 / PI-11: importing the handlers or building them loads neither torch nor a model.
PI-12 / UXS-02 / AR-07: progress through main_window.feedback, no modal success box.
PI-08 / UI-20: Cancel stops the thread, also one pressed during the last batch (wf2 R1: then
    nothing chains); RB-07: refuses politely while another job runs.
S11-10: a missing checkpoint is refused before anything starts; Generate names the mask
    model first, in one box.
R6: the scan of existing masks shows a wait cursor; R7: a button's ``checked`` flag chains
    nothing; a cancelled or failed skeleton run still refreshes the window.
CQ-04: one HandlerDependencies mixin supplies ``feedback`` / ``jobs``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMessageBox, QPushButton

from app import config
from app.handlers import generate_skeleton_handler as gsh
from app.handlers import mask_generation_handler as mgh
from app.handlers.job_registry import JobRegistry
from app.inference import runtime
from app.inference.mask_origin import OverwritePolicy
from tests.generation_support import generation_project
from tests.generation_support import wait_for_thread as _wait
from tests.window_support import app as app  # fixture
from tests.window_support import make_window as make_window  # fixture
from tests.window_support import mw as mw  # fixture
from tests.window_support import settings as settings  # fixture


class _Jobs:
    def __init__(self, accept=True):
        self.accept = accept
        self.started = []

    def try_start(self, name, thread):
        if not self.accept:
            return False
        self.started.append(name)
        thread.start()
        return True

    def running_name(self):
        return "Measuring root length"


@pytest.fixture
def project(tmp_path, qapp, monkeypatch):
    """A folder of 3 images, CPU runtime, stub models and a fake main window."""
    yield from generation_project(tmp_path, monkeypatch)


def _save_traced(path):
    Image.new("RGBA", (64, 48), (255, 255, 255, 255)).save(path)


def test_handler_modules_do_not_import_torch():
    code = (
        "import sys, app.handlers.mask_generation_handler, "
        "app.handlers.generate_skeleton_handler; print('torch' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )
    assert out.stdout.strip() == "False"


def test_constructing_the_handler_loads_no_model(project, monkeypatch):
    calls = []
    monkeypatch.setattr(runtime, "get_mask_model", lambda device=None: calls.append(1))
    mgh.MaskGenerationHandler(project.window)
    assert calls == []


def test_cancel_in_the_prompt_starts_nothing(project, qapp):
    (project.folder / "mask").mkdir()
    _save_traced(project.folder / "mask" / "a.png")
    handler = mgh.MaskGenerationHandler(project.window, confirm_overwrite=lambda e, t: None)
    assert handler.generate_masks() is False
    assert handler.generation_thread is None
    assert project.stub.items == 0
    assert "task_started" not in project.window.feedback.names()


def test_no_existing_masks_means_no_prompt(project, qapp):
    def confirm(existing, total):
        raise AssertionError("no prompt expected")

    handler = mgh.MaskGenerationHandler(project.window, confirm_overwrite=confirm)
    assert handler.generate_masks() is True
    _wait(handler.generation_thread, qapp)
    assert sorted(os.listdir(project.folder / "mask")) == [
        "_sprouts_run.json",
        "_sprouts_runs.jsonl",
        "a.png",
        "b.png",
        "c.png",
    ]


def test_progress_goes_through_feedback_without_modal_success(project, qapp):
    handler = mgh.MaskGenerationHandler(project.window)
    handler.generate_masks()
    _wait(handler.generation_thread, qapp)
    fb = project.window.feedback
    labels = [c.args[0] for c in fb.calls if c.name == "task_started"]
    assert labels == ["Loading mask model", "Generating masks"]
    assert all(c.kwargs.get("cancelable") for c in fb.calls if c.name == "task_started")
    assert "task_progress" in fb.names() and "task_finished" in fb.names()
    assert "success" in [
        c.args[1] if len(c.args) > 1 else c.kwargs.get("kind")
        for c in fb.calls
        if c.name == "notify"
    ]
    assert project.dialogs == []
    assert not fb.busy


def test_missing_weights_are_refused_before_starting(project, monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "mask_weights_path", lambda: str(tmp_path / "none.pth"))
    handler = mgh.MaskGenerationHandler(project.window)
    assert handler.generate_masks() is False
    assert handler.generation_thread is None
    assert [d[0] for d in project.dialogs] == ["warning"]


def test_missing_skeleton_weights_are_refused_before_starting(project, monkeypatch, tmp_path):
    # S11-10: refused at once with a box naming the file, as a missing mask model is; before,
    # the worker raised only once the run had started.
    missing = str(tmp_path / "none.pth")
    monkeypatch.setattr(runtime, "skeleton_weights_path", lambda: missing)
    handler = gsh.GenerateSkeletonHandler(project.window)
    assert handler.generate_skeleton() is False
    assert handler._thread is None
    assert [kind for kind, _ in project.dialogs] == ["warning"]
    _parent, title, text = project.dialogs[0][1]
    assert title == "Skeleton model missing"
    assert text == (
        f"Skeleton generation is unavailable: the model weights were not found at\n{missing}"
    )
    assert not (project.folder / "skeletons").exists()
    assert "task_started" not in project.window.feedback.names()


def _weights(monkeypatch, tmp_path, *, mask: bool, skeleton: bool) -> dict[str, str]:
    """Point both checkpoint paths into ``tmp_path``; each file exists only when asked."""
    from app.inference import runtime

    paths = {}
    for name, exists in (("mask", mask), ("skeleton", skeleton)):
        path = tmp_path / f"{name}.pth"
        if exists:
            path.write_bytes(b"w")
        paths[name] = str(path)
        monkeypatch.setattr(runtime, f"{name}_weights_path", lambda p=str(path): p)
    return paths


@pytest.mark.parametrize(
    ("mask", "skeleton", "title"),
    [(True, False, "Skeleton model missing"), (False, False, "Mask model missing")],
)
def test_generate_all_refuses_at_once_without_both_models(
    mw, monkeypatch, tmp_path, mask, skeleton, title
):
    # S11-10: a missing skeleton model is said before the mask pass, not after it; with both
    # missing, the mask model is named first, in one box.
    _weights(monkeypatch, tmp_path, mask=mask, skeleton=skeleton)
    started, boxes = [], []
    mw.mask_generation_handler.generate_masks = lambda on_success=None: started.append(1)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: boxes.append(a[1]))

    assert mw.generate_all() is False

    assert started == [] and boxes == [title]


@pytest.mark.parametrize("kind", ["mask", "skeleton"])
def test_a_failed_run_refreshes_the_window_then_shows_one_box(
    project, qapp, monkeypatch, caplog, kind
):
    # L2b: both kinds end a failed run the same way: one log wording, the window re-reads
    # the outputs written before the error, then the one error box.
    def unavailable(device=None):
        raise runtime.ModelUnavailableError("weights not found: x.pth")

    project.window.on_outputs_changed = lambda: project.dialogs.append(("refresh", ()))
    if kind == "mask":
        monkeypatch.setattr(runtime, "get_mask_model", unavailable)
        handler = mgh.MaskGenerationHandler(project.window)
        assert handler.generate_masks() is True
        thread, title = handler.generation_thread, "Mask generation"
    else:
        monkeypatch.setattr(runtime, "get_skeleton_model", unavailable)
        handler = gsh.GenerateSkeletonHandler(project.window)
        assert handler.generate_skeleton() is True
        thread, title = handler._thread, "Skeleton generation"
    with caplog.at_level("ERROR"):
        _wait(thread, qapp)

    assert [k for k, _ in project.dialogs] == ["refresh", "critical"]
    assert project.dialogs[1][1][1] == f"{title} failed"
    assert "weights not found: x.pth" in project.dialogs[1][1][2]
    logged = [r for r in caplog.records if r.name == type(handler).__module__]
    assert [r.getMessage().split(":")[0] for r in logged] == [f"{title} failed"]
    assert [c.args[0] for c in project.window.feedback.calls if c.name == "task_failed"] == [
        f"{title} failed"
    ]
    assert project.window.loaded == []


def test_busy_job_registry_refuses_politely(project, qapp):
    project.window.jobs = _Jobs(accept=False)
    handler = mgh.MaskGenerationHandler(project.window)
    assert handler.generate_masks() is False
    assert handler.generation_thread is None
    notes = [c for c in project.window.feedback.calls if c.name == "notify"]
    assert notes and "Measuring root length" in notes[-1].args[0]
    assert project.dialogs == []


def test_jobs_registry_runs_the_thread(project, qapp):
    project.window.jobs = _Jobs(accept=True)
    handler = mgh.MaskGenerationHandler(project.window)
    assert handler.generate_masks() is True
    _wait(handler.generation_thread, qapp)
    assert project.window.jobs.started == ["Generating masks"]


def test_on_success_runs_after_the_job_registry_is_free(project, qapp):
    """generate_all chains skeletons via on_success: the mask job must be cleared first,
    otherwise main_window.jobs refuses the skeleton job."""
    from app.handlers.job_registry import JobRegistry

    registry = JobRegistry()
    project.window.jobs = registry
    busy_at_callback = []
    handler = mgh.MaskGenerationHandler(project.window)
    assert handler.generate_masks(on_success=lambda: busy_at_callback.append(registry.is_busy()))
    _wait(handler.generation_thread, qapp)
    assert busy_at_callback == [False]


def test_on_success_is_not_called_after_cancel_or_failure(project, qapp, monkeypatch):
    monkeypatch.setattr(config, "MASK_BATCH_SIZE", 1)
    calls = []
    handler = mgh.MaskGenerationHandler(
        project.window, confirm_overwrite=lambda e, t: OverwritePolicy.OVERWRITE_ALL
    )
    project.stub.on_forward = lambda: handler.generation_thread.cancel()
    handler.generate_masks(on_success=lambda: calls.append("chained"))
    _wait(handler.generation_thread, qapp)

    def unavailable(device=None):
        raise runtime.ModelUnavailableError("missing")

    monkeypatch.setattr(runtime, "get_mask_model", unavailable)
    handler.generate_masks(on_success=lambda: calls.append("chained"))
    _wait(handler.generation_thread, qapp)
    assert calls == []


def test_cancel_stops_the_run_and_keeps_finished_masks(project, qapp, monkeypatch):
    monkeypatch.setenv("SPROUTS_BATCH_POLICY", "auto")  # (O4) MASK_BATCH_SIZE sizes auto runs
    monkeypatch.setattr(config, "MASK_BATCH_SIZE", 1)
    handler = mgh.MaskGenerationHandler(project.window)
    project.stub.on_forward = lambda: handler.generation_thread.cancel()
    finished = []
    handler.generate_masks()
    handler.generation_thread.finished.connect(lambda *a: finished.append(a))
    _wait(handler.generation_thread, qapp)
    assert finished == []  # generate_all chains on finished: a cancelled run must not chain
    written = [n for n in os.listdir(project.folder / "mask") if n.endswith(".png")]
    assert 1 <= len(written) < 3
    fb = project.window.feedback
    finished_msgs = [c.args[0] for c in fb.calls if c.name == "task_finished"]
    assert finished_msgs[-1].startswith("Mask generation cancelled after")
    toasts = [c.args for c in fb.calls if c.name == "notify"]
    assert toasts[-1] == (f"{finished_msgs[-1]}; finished masks were kept", "info")
    assert not fb.busy


def _spin(qapp, predicate, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _last_batch_cancel(thread):
    """Stand-in inference: Cancel arrives while the only batch runs; all images done."""

    def run(*args, **kwargs):
        thread.cancel()
        from app.inference.batch_engine import BatchReport

        return BatchReport(total=8, written=[f"{i}.png" for i in range(8)])

    return run


@pytest.mark.parametrize("kind", ["mask", "skeleton"])
def test_cancel_in_the_last_batch_is_a_cancel_and_chains_nothing(qapp, monkeypatch, kind):
    from app.handlers import generate_skeleton_handler as gsh
    from app.handlers import mask_generation_handler as mgh
    from app.handlers.generation_base import GenerationHandlerBase
    from app.inference import mask_inference, skeleton_inference

    if kind == "mask":
        thread = mgh.MaskGenerationThread("in", "out")
        monkeypatch.setattr(mask_inference, "generate_masks", _last_batch_cancel(thread))
    else:
        thread = gsh.SkeletonGeneratorThread("in", "out")
        monkeypatch.setattr(
            skeleton_inference, "run_skeleton_generation", _last_batch_cancel(thread)
        )
    window = SimpleNamespace(jobs=JobRegistry(), feedback=None)
    handler = GenerationHandlerBase(window)
    calls, cancelled = [], []
    thread.cancelled.connect(lambda: cancelled.append(True))
    assert handler._start(thread, on_success=lambda: calls.append("chained"))
    assert _spin(qapp, lambda: not window.jobs.is_busy())
    _spin(qapp, lambda: False, timeout=0.2)
    assert not thread.completed and cancelled == [True] and calls == []


def test_skeleton_handler_loads_results_and_reports(project, qapp):
    handler = gsh.GenerateSkeletonHandler(project.window)
    assert handler.generate_skeleton() is True
    _wait(handler._thread, qapp)
    out = os.path.normpath(os.path.join(str(project.folder), "skeletons"))
    assert project.window.loaded == [out]
    assert sorted(n for n in os.listdir(out) if n.endswith(".png")) == [
        "a_fake.png",
        "b_fake.png",
        "c_fake.png",
    ]
    fb = project.window.feedback
    assert [c.args[0] for c in fb.calls if c.name == "task_started"] == ["Generating skeletons"]
    assert project.dialogs == []


def test_skeleton_failures_get_one_report(project, qapp):
    (project.folder / "zz_bad.jpg").write_bytes(b"not an image")
    handler = gsh.GenerateSkeletonHandler(project.window)
    handler.generate_skeleton()
    _wait(handler._thread, qapp)
    reports = [d for d in project.dialogs if d[0] == "report"]
    assert len(reports) == 1 and "zz_bad.jpg" in str(reports[0][1])
    assert project.window.loaded  # results still load


class _Model:
    """Stub skeleton model: ``on_batch(n)`` runs before batch ``n`` (1-based)."""

    def __init__(self, on_batch):
        self.on_batch = on_batch
        self.batches = 0

    def run_batch(self, x):
        self.batches += 1
        self.on_batch(self.batches)
        return torch.tanh(x)


def _skeletons(folder):
    out = folder / "skeletons"
    return sorted(n for n in os.listdir(out) if n.endswith("_fake.png")) if out.is_dir() else []


def test_a_cancelled_skeleton_run_refreshes_the_window(project, qapp, monkeypatch):  # noqa: F811
    monkeypatch.setenv("SPROUTS_BATCH_POLICY", "auto")  # (O4) the fixture sizes auto batches
    handler = gsh.GenerateSkeletonHandler(project.window)

    def cancel_after_the_first_batch(n):
        if n == 1:
            handler._thread.cancel()

    model = _Model(cancel_after_the_first_batch)
    monkeypatch.setattr(runtime, "get_skeleton_model", lambda device=None: model)
    assert handler.generate_skeleton() is True
    _wait(handler._thread, qapp)

    assert 1 <= len(_skeletons(project.folder)) < 3  # the first batch was kept
    fb = project.window.feedback
    cancelled = [c.args[0] for c in fb.calls if c.name == "task_finished"][-1]
    assert cancelled.startswith("Skeleton generation cancelled after")
    toasts = [c.args for c in fb.calls if c.name == "notify"]
    assert toasts[-1] == (f"{cancelled}; finished skeletons were kept", "info")
    assert project.window.outputs_changed == [True]  # badges / display follow the disk
    assert project.window.loaded == []  # a cancelled run loads no results


def test_a_failed_skeleton_run_refreshes_the_window(project, qapp, monkeypatch):  # noqa: F811
    monkeypatch.setenv("SPROUTS_BATCH_POLICY", "auto")  # (O4) the fixture sizes auto batches

    def fail_on_the_second_batch(n):
        if n == 2:
            raise RuntimeError("CUDA error: unspecified launch failure (simulated)")

    model = _Model(fail_on_the_second_batch)
    monkeypatch.setattr(runtime, "get_skeleton_model", lambda device=None: model)
    handler = gsh.GenerateSkeletonHandler(project.window)
    assert handler.generate_skeleton() is True
    _wait(handler._thread, qapp)

    assert len(_skeletons(project.folder)) == 2  # the first batch of two is on disk
    assert [kind for kind, _args in project.dialogs] == ["critical"]
    assert project.window.outputs_changed == [True]
    assert project.window.loaded == []


def _cursor_shape():
    cursor = QApplication.overrideCursor()
    return None if cursor is None else cursor.shape()


def test_the_existing_mask_scan_shows_a_wait_cursor(project, qapp, monkeypatch):  # noqa: F811
    (project.folder / "mask").mkdir()
    _save_traced(project.folder / "mask" / "a.png")
    during_scan, during_prompt = [], []
    real_scan = mgh.scan_existing_masks

    def spy_scan(layout, stems):
        during_scan.append(_cursor_shape())
        return real_scan(layout, stems)

    def confirm(existing, total):
        during_prompt.append(_cursor_shape())
        return OverwritePolicy.KEEP_EXISTING

    monkeypatch.setattr(mgh, "scan_existing_masks", spy_scan)
    handler = mgh.MaskGenerationHandler(project.window, confirm_overwrite=confirm)
    assert handler.generate_masks() is True
    _wait(handler.generation_thread, qapp)

    assert during_scan == [Qt.CursorShape.WaitCursor]
    assert during_prompt == [None]  # the user answers the prompt with a normal cursor
    assert _cursor_shape() is None


def test_a_failed_scan_restores_the_cursor(project, qapp, monkeypatch):  # noqa: F811
    def unreadable(layout, stems):
        raise PermissionError(13, "Access is denied (simulated)")

    monkeypatch.setattr(mgh, "scan_existing_masks", unreadable)
    handler = mgh.MaskGenerationHandler(project.window)

    assert handler.generate_masks() is False
    assert _cursor_shape() is None
    assert [kind for kind, _args in project.dialogs] == ["critical"]


def test_the_generate_masks_button_click_chains_nothing(project, qapp, monkeypatch):  # noqa: F811
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *exc_info: errors.append(exc_info[1]))
    handler = mgh.MaskGenerationHandler(project.window)
    button = QPushButton("Generate Masks")
    button.clicked.connect(handler.generate_masks)  # as app/gui/ui_panels.py wires it

    button.click()  # clicked(checked=False) -> generate_masks(False)
    _wait(handler.generation_thread, qapp)

    assert handler.generation_thread.completed
    assert errors == []  # no TypeError from QTimer.singleShot(0, False) in a slot
    assert len(project.window.outputs_changed) == 1
    button.deleteLater()


def test_handlers_share_one_dependencies_mixin():
    from app.feedback import NullFeedback
    from app.handlers.deps import HandlerDependencies
    from app.handlers.generation_base import GenerationHandlerBase
    from app.handlers.skeleton_handler import SkeletonHandler

    assert issubclass(GenerationHandlerBase, HandlerDependencies)
    assert issubclass(SkeletonHandler, HandlerDependencies)
    bare = HandlerDependencies(SimpleNamespace())
    assert isinstance(bare.feedback, NullFeedback) and bare.feedback is bare.feedback
    assert isinstance(bare.jobs, JobRegistry) and bare.jobs is bare.jobs
    injected = JobRegistry()
    window = SimpleNamespace(feedback="fb", jobs=injected)
    assert HandlerDependencies(window).feedback == "fb"
    assert HandlerDependencies(window).jobs is injected
    assert HandlerDependencies(window, jobs=JobRegistry()).jobs is not injected


def test_a_skeleton_folder_that_cannot_be_made_is_said_and_starts_nothing(project, qapp):
    (project.folder / "skeletons").write_bytes(b"a file where the folder should be")
    handler = gsh.GenerateSkeletonHandler(project.window)

    assert handler.generate_skeleton() is False

    [(kind, (_parent, title, text))] = project.dialogs
    assert (kind, title) == ("critical", "Error")
    assert text.startswith("Could not create output directory:\n")
    assert os.path.normpath(str(project.folder / "skeletons")) in text
    assert handler._thread is None and "task_started" not in project.window.feedback.names()


def test_generate_masks_without_a_folder_asks_for_one_and_starts_nothing(project, qapp):
    project.window.image_manager = SimpleNamespace(images={}, original_folder=None)
    handler = mgh.MaskGenerationHandler(project.window)

    assert handler.generate_masks() is False

    assert [(kind, args[1:]) for kind, args in project.dialogs] == [
        ("warning", ("No images", "Load an image folder first."))
    ]
    assert handler.generation_thread is None
    assert "task_started" not in project.window.feedback.names()

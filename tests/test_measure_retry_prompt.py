"""RB-16: Measure's Retry prompt is freed once answered.

The non-modal Retry prompt of Measure (``SkeletonHandler._offer_retry``) left one
QMessageBox child on its long-lived parent (the main window) per prompt until the window
closed; it is now deleted once answered, and answering an older prompt keeps the newer one.
(Generate's overwrite prompt: ``tests/test_generate_predictions_only.py``; the task progress
hide timer: ``tests/test_task_progress.py``; the F1 sheet and About dialog:
``tests/test_main_window_wiring.py``.)
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEvent
from PyQt6.QtWidgets import QMessageBox, QWidget

from app.feedback import RecordingFeedback
from app.handlers.job_registry import JobRegistry
from app.handlers.skeleton_handler import SkeletonHandler
from tests.qt_teardown import flush_deleted

_Button = QMessageBox.StandardButton


@pytest.fixture
def window(qapp):
    """A long-lived parent, as the main window is for every prompt."""
    widget = QWidget()
    yield widget
    widget.deleteLater()
    del widget
    flush_deleted()


def _flush_deferred_deletes() -> None:
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _boxes(parent: QWidget) -> list[QMessageBox]:
    return parent.findChildren(QMessageBox)


class _IdleGeneration:
    def is_busy(self) -> bool:
        return False


def _measure_handler(window, monkeypatch):
    window.image_manager = SimpleNamespace(original_folder=None, images={})
    window.feedback = RecordingFeedback()
    window.jobs = JobRegistry()
    handler = SkeletonHandler(window, generation=_IdleGeneration())
    retried = []
    monkeypatch.setattr(handler, "retry_measure", retried.append)
    return handler, retried


@pytest.mark.parametrize("answer", [_Button.Retry, _Button.Cancel])
def test_retry_prompt_is_freed_once_answered(window, monkeypatch, answer):
    handler, retried = _measure_handler(window, monkeypatch)
    handler._retry_jobs["length"] = object()
    handler._offer_retry("length", "root_lengths.csv is open in another program")
    box = handler._retry_box
    assert _boxes(window) == [box]
    box.button(answer).click()
    _flush_deferred_deletes()
    assert sip.isdeleted(box)
    assert _boxes(window) == []
    assert handler._retry_box is None
    if answer == _Button.Retry:
        assert retried == ["length"]
    else:
        assert retried == [] and "length" not in handler._retry_jobs


def test_answering_an_older_retry_prompt_keeps_the_newer_one(window, monkeypatch):
    handler, retried = _measure_handler(window, monkeypatch)
    handler._retry_jobs.update(length=object(), area=object())
    handler._offer_retry("length", "root_lengths.csv is open in another program")
    first = handler._retry_box
    handler._offer_retry("area", "root_areas.csv is open in another program")
    second = handler._retry_box

    first.button(_Button.Cancel).click()  # the older prompt is answered first
    _flush_deferred_deletes()
    assert sip.isdeleted(first)
    assert handler._retry_box is second  # still the open prompt, never the freed one
    assert _boxes(window) == [second]

    second.button(_Button.Retry).click()
    _flush_deferred_deletes()
    assert handler._retry_box is None
    assert _boxes(window) == []
    assert retried == ["area"] and "length" not in handler._retry_jobs

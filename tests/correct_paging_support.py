"""Paging helpers of the Correct page tests: a neighbour provider, the wait for the
read-ahead, its keys and a spy on the photo decodes of the GUI thread.

Moved here from ``test_correct_editor_paging.py``, whose helpers other Correct, read-ahead
and window test modules borrowed.
"""

from __future__ import annotations

import os
import threading
import time

from PyQt6.QtTest import QTest

import app.enhancement as enhancement
from app.gui.correct.paging import Neighbours
from app.io_utils import path_key


def ring(*paths):
    """A provider over ``paths`` in library order: the next image, then the previous one."""

    def neighbours(path):
        keys = [os.path.normcase(os.path.abspath(p)) for p in paths]
        index = keys.index(os.path.normcase(os.path.abspath(path)))
        return [paths[(index + 1) % len(paths)], paths[index - 1]]

    return neighbours


def read_ahead_done(editor, timeout: float = 5.0) -> list[str]:
    """Wait for the read-ahead timer and its jobs; the kept entries (normalised paths)."""
    QTest.qWait(Neighbours.DELAY_MS + 30)
    deadline = time.perf_counter() + timeout
    # On a loaded machine the paging timer can fire after the fixed wait: wait it out.
    while editor.paging._timer.isActive() and time.perf_counter() < deadline:
        QTest.qWait(10)
    ahead = editor.paging._ahead
    while ahead is not None and time.perf_counter() < deadline:
        if all(job.done() for job in ahead._jobs.values()):
            break
        time.sleep(0.005)
    return editor.paging.pending()


def key(path: str) -> str:
    """The read-ahead's key of ``path`` (``io_utils.path_key``: it folds case on macOS too)."""
    return path_key(path)


class GuiDecodes:
    """Counts photo decodes made on the calling (GUI) thread."""

    def __init__(self, monkeypatch) -> None:
        self.paths: list[str] = []
        gui, real = threading.current_thread(), enhancement.imread_unicode

        def spy(path, *args, **kwargs):
            if threading.current_thread() is gui:
                self.paths.append(str(path))
            return real(path, *args, **kwargs)

        monkeypatch.setattr(enhancement, "imread_unicode", spy)


class PhotoDecodes:
    """Every photo decode, on any thread: ``(file name, on the GUI thread)`` (moved here from
    ``test_correct_library_photo_sharing.py``)."""

    def __init__(self, monkeypatch) -> None:
        import app.gui.display_controller as display_module  # its first use loads it

        self.seen: list[tuple[str, bool]] = []
        gui = threading.current_thread()
        for module in (enhancement, display_module):
            real = module.imread_unicode

            def spy(path, *args, _real=real, **kwargs):
                self.seen.append((os.path.basename(str(path)), threading.current_thread() is gui))
                return _real(path, *args, **kwargs)

            monkeypatch.setattr(module, "imread_unicode", spy)

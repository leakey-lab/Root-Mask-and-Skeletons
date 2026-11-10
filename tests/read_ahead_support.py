"""Stand-ins of the ReadAhead tests (``app.gui.correct.read_ahead``), Qt-free: an executor
that runs each job at once, a payload that reports its size, and a photo on disk.

Moved here from ``test_correct_read_ahead.py``, whose helpers
``test_correct_read_ahead_budget.py`` borrowed.
"""

from __future__ import annotations

from concurrent.futures import Executor, Future

import cv2
import numpy as np


class InlineExecutor(Executor):
    """Runs each job at once on the calling thread (a finished Future comes back)."""

    def __init__(self) -> None:
        self.submitted = 0

    def submit(self, fn, /, *args, **kwargs):
        self.submitted += 1
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # noqa: BLE001 - the future carries it
            future.set_exception(exc)
        return future


class Payload:
    """A layer payload that reports its size."""

    def __init__(self, nbytes: int) -> None:
        self.nbytes = nbytes


def photo(tmp_path, name: str = "a.png", w: int = 40, h: int = 30, value: int = 90) -> str:
    path = str(tmp_path / name)
    cv2.imwrite(path, np.full((h, w, 3), value, np.uint8))
    return path

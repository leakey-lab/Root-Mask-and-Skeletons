"""The slot-error guard of conftest.py (``tests/guards.record_slot_errors``).

PyQt hands an exception raised in a Qt slot to ``sys.excepthook``, which only prints it
under pytest. The guard records it, the test fails at teardown, and with ``sources`` only an
exception whose traceback passes through one of those files counts (at a path-separator
boundary). The other guards are one ``monkeypatch.setattr`` each and have no test of their
own.
"""

import os
import sys

import pytest
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from tests import guards


def _exc_info_from(filename: str):
    """``(kind, value, tb)`` of an error raised and caught in code of ``filename`` only; its
    message is ``filename``."""
    source = "def capture():\n    try:\n        raise ValueError(NAME)\n"
    source += "    except ValueError:\n        return sys.exc_info()\n"
    namespace = {"sys": sys, "NAME": filename}
    exec(compile(source, filename, "exec"), namespace)  # its frames name ``filename``
    return namespace["capture"]()


@pytest.fixture
def sources_in_module(monkeypatch):
    """This module names the files whose slot errors count, as a module would."""
    names = (os.path.join("pkg", "thing.py"),)
    monkeypatch.setattr(sys.modules[__name__], "SLOT_ERRORS_FROM", names, raising=False)


def test_slot_errors_records_an_exception_raised_in_a_qt_slot(app, slot_errors):
    def slot():
        raise RuntimeError("raised in a slot")

    QTimer.singleShot(0, slot)
    for _ in range(3):
        QApplication.processEvents()

    assert [str(error) for error in slot_errors] == ["raised in a slot"]
    slot_errors.clear()  # seen: the test passes


def test_a_recorded_slot_error_fails_the_test_at_teardown(monkeypatch):
    clean = guards.record_slot_errors(monkeypatch)
    next(clean)
    with pytest.raises(StopIteration):
        next(clean)  # nothing recorded: the teardown passes

    failing = guards.record_slot_errors(monkeypatch)
    errors = next(failing)
    sys.excepthook(*_exc_info_from("anywhere.py"))
    assert len(errors) == 1
    with pytest.raises(AssertionError):
        next(failing)


def test_sources_limit_the_recording_to_tracebacks_through_those_files(monkeypatch, capsys):
    recorder = guards.record_slot_errors(monkeypatch, (os.path.join("pkg", "thing.py"), "solo.py"))
    errors = next(recorder)
    recorded = [
        os.path.join("root", "pkg", "thing.py"),  # ends with the entry
        os.path.join("root", "solo.py"),  # the file-name entry
        os.path.join("pkg", "thing.py"),  # the entry itself
    ]
    printed = [
        os.path.join("root", "notpkg", "thing.py"),  # no separator before the entry
        os.path.join("root", "notsolo.py"),
        os.path.join("root", "pkg", "other.py"),
    ]
    for filename in recorded + printed:
        sys.excepthook(*_exc_info_from(filename))

    assert [str(error) for error in errors] == recorded
    err = capsys.readouterr().err
    assert all(f"ValueError: {name}" in err for name in printed)
    assert not any(f"ValueError: {name}\n" in err for name in recorded)
    errors.clear()
    with pytest.raises(StopIteration):
        next(recorder)


def test_slot_errors_reads_the_modules_sources(sources_in_module, slot_errors, capsys):
    sys.excepthook(*_exc_info_from(os.path.join("root", "pkg", "thing.py")))
    sys.excepthook(*_exc_info_from(os.path.join("root", "elsewhere.py")))

    assert [str(error) for error in slot_errors] == [os.path.join("root", "pkg", "thing.py")]
    assert "elsewhere.py" in capsys.readouterr().err
    slot_errors.clear()

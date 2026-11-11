"""Guard of the Correct workspace package (``app.gui.correct``): no god class.

Every ``app/gui/correct/*.py`` stays at or under 450 lines (``CLAUDE.md``: the ratchet
test). The layer contract (``missing_members``) is tested in
``test_correct_layer_protocol.py``; the Qt-free modules of the package in
``test_qt_free_core.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CORRECT = ROOT / "app" / "gui" / "correct"
MAX_LINES = 450


@pytest.mark.parametrize("path", sorted(CORRECT.glob("*.py")), ids=lambda path: path.name)
def test_no_correct_module_grows_past_the_line_cap(path):
    lines = len(path.read_text(encoding="utf-8").splitlines())

    assert lines <= MAX_LINES, f"{path.name} has {lines} lines: split it (the cap is {MAX_LINES})"

"""Import-smoke test: every module and package under ``app/`` imports cleanly.

This is the primary guard for dependency upgrades (dash, PyQt6, numpy, pandas, plotly,
etc.) where the behaviour tests do not reach every module: a broken import after an upgrade
fails here, named. The modules are found on disk, so a new module is covered without editing
this file.

GUI modules only need a QApplication for *instantiation*, not import, so importing is safe
in a headless/offscreen environment. Handlers are imported lazily in ``MainWindow.__init__``
(the old handler -> editor -> main_window import cycle is gone), so every module imports
cleanly on its own.
"""

import importlib
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

APP = Path(__file__).resolve().parents[1] / "app"


def app_modules() -> list[str]:
    """The dotted name of every module and package under ``app/``."""
    names = set()
    for path in APP.rglob("*.py"):
        parts = path.relative_to(APP.parent).with_suffix("").parts
        names.add(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
    return sorted(names)


APP_MODULES = app_modules()


def test_the_module_list_is_read_from_disk():
    assert {"app", "app.gui", "app.gui.widgets.fonts", "app.gui.correct.editor"} <= set(APP_MODULES)
    assert len(APP_MODULES) > 150


@pytest.mark.parametrize("modname", APP_MODULES)
def test_module_imports(modname):
    importlib.import_module(modname)

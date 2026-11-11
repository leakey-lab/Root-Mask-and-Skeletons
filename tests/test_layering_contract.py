"""Layering contract (CQ-05): lower layers never import upward.

* ``app.inference``, ``app.visualization`` and the layer-neutral modules
  (``app/{dialogs,feedback,settings,workers}.py``) import neither ``app.gui`` nor
  ``app.handlers``;
* ``app.handlers`` does not import ``app.gui`` (the window imports the handlers);
* every import counts, relative ones too (``from ..gui import panels``), and
  ``from X import y`` counts as ``X.y``;
* run in a fresh interpreter, the Qt-free ``UiFeedback`` port (``app.feedback``), the metric
  thread and the two handlers the window builds load no GUI module, also not transitively.

The file-stamp helpers' one home is tested in ``test_single_sources.py``.
"""

from __future__ import annotations

import ast
import subprocess
import sys

import pytest

from tests.source_scan_support import ROOT, imported_names, names_in, rel

APP = ROOT / "app"
GUI_OR_HANDLERS = ("app.gui", "app.handlers")
NEUTRAL = ("dialogs.py", "feedback.py", "settings.py", "workers.py")

#: rule id -> (the files it covers, the packages they must not import)
RULES = {
    "inference": (lambda: sorted((APP / "inference").rglob("*.py")), GUI_OR_HANDLERS),
    "visualization": (lambda: sorted((APP / "visualization").rglob("*.py")), GUI_OR_HANDLERS),
    "neutral": (lambda: [APP / name for name in NEUTRAL], GUI_OR_HANDLERS),
    "handlers": (lambda: sorted((APP / "handlers").rglob("*.py")), ("app.gui",)),
}


def _above(names, forbidden: tuple[str, ...]) -> list[str]:
    """The names of ``names`` inside one of the ``forbidden`` packages."""
    return sorted(n for n in names if any(n == f or n.startswith(f + ".") for f in forbidden))


@pytest.mark.parametrize("rule", sorted(RULES))
def test_no_module_imports_a_layer_above_it(rule: str) -> None:
    files, forbidden = RULES[rule]
    paths = files()
    assert paths and all(path.is_file() for path in paths), f"{rule}: no file to check"
    offenders = {
        rel(path): bad for path in paths if (bad := _above(imported_names(path), forbidden))
    }
    assert offenders == {}


def test_the_import_reader_resolves_aliases_and_relative_imports() -> None:
    source = (
        "from app.gui import panels\n"
        "from .job_registry import JobRegistry\n"
        "from .. import feedback as port\n"
        "from ..gui.correct import layer\n"
        "import app.handlers.deps as deps\n"
    )
    found = names_in(ast.parse(source), ("app", "handlers"))
    assert {
        "app.gui.panels",
        "app.handlers.job_registry",
        "app.feedback",
        "app.gui.correct.layer",
        "app.handlers.deps",
    } <= found
    assert _above(found, ("app.gui",)) == [
        "app.gui",
        "app.gui.correct",
        "app.gui.correct.layer",
        "app.gui.panels",
    ]


def _loaded_after_import(module: str) -> set[str]:
    code = (
        f"import sys, importlib; importlib.import_module({module!r}); "
        "print('\\n'.join(sorted(sys.modules)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return set(out.split())


def test_feedback_port_stays_qt_free() -> None:
    loaded = _loaded_after_import("app.feedback")
    assert not any(m == "PyQt6" or m.startswith("PyQt6.") for m in loaded)
    assert not _above((m for m in loaded if m.startswith("app")), GUI_OR_HANDLERS)


def test_metric_thread_does_not_load_handlers_or_gui() -> None:
    loaded = _loaded_after_import("app.inference.metric_thread")
    assert not _above((m for m in loaded if m.startswith("app")), GUI_OR_HANDLERS)


@pytest.mark.parametrize(
    "module", ["app.handlers.skeleton_handler", "app.handlers.mask_generation_handler"]
)
def test_the_handlers_load_no_gui_module(module: str) -> None:
    assert _above(_loaded_after_import(module), ("app.gui",)) == []

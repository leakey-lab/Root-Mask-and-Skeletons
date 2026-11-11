"""The Qt-free core loads neither PyQt6 nor torch (``CLAUDE.md``, "Qt-free core").

The modules ``CLAUDE.md`` and their own docstrings call Qt-free (the editing and imaging
cores, the mask and label stores, the read-ahead caches, the Correct page's documents and
files, the feedback port, the key texts ...) are imported in a fresh interpreter; a module
whose import brings in PyQt6 or torch fails the test, named. So a headless script, a worker
or a document test never starts Qt or torch by the way, and the documents stay testable
without a QApplication.

``app``, ``app.config``, ``app.project_layout``, ``app.naming``, ``app.io_utils``,
``app.user_paths``, ``app.provenance``, ``app.models`` and ``app.calibration`` are A1's
stricter probe (``test_config.py::test_foundation_modules_import_no_heavy_stack``, which
also forbids cv2).

A module of the list that is not on disk fails too: a rename must update the list, not skip.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEAVY = ("PyQt6", "torch")

QT_FREE = (
    # editing and imaging cores
    "app.editing.journal",
    "app.editing.history",
    "app.editing.geometry",
    "app.editing.planes",
    "app.editing.thinning",
    "app.imaging",
    "app.imaging.pyramid",
    "app.imaging.skeleton_vectors",
    # stores and caches
    "app.masks.store",
    "app.enhancement",
    "app.read_ahead_cache",
    "app.gt.registry",
    "app.gt.store",
    "app.gt.export",
    "app.gt.files",
    # ports, tokens, start-up helpers
    "app.design.tokens",
    "app.feedback",
    "app.gc_policy",
    "app.qt_environment",
    "app.inference.device_probe",
    # Qt-free parts of app.gui
    "app.gui.key_text",
    "app.gui.stage_summary",
    "app.gui.display_read_ahead",
    # the Correct page's Qt-free modules
    "app.gui.correct.layer",
    "app.gui.correct.session",
    "app.gui.correct.save_gate",
    "app.gui.correct.stamps",
    "app.gui.correct.mask_document",
    "app.gui.correct.mask_files",
    "app.gui.correct.skeleton_document",
    "app.gui.correct.skeleton_records",
    "app.gui.correct.classes_document",
    "app.gui.correct.classes_paint",
    "app.gui.correct.classes_files",
    "app.gui.correct.read_ahead",
    "app.gui.correct.read_back",
)

# One interpreter imports the modules in turn and reports, per module, the heavy packages its
# import added and whether it is missing. A heavy package loaded by an earlier module hides a
# later module's own import of it, so a red run is repeated one module per interpreter.
_PROBE = """
import importlib, json, sys
heavy = tuple(sys.argv[1].split(","))
report = {}
for name in sys.argv[2:]:
    before = {m.split(".")[0] for m in sys.modules}
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name == name or name.startswith(f"{error.name}."):
            report[name] = "missing"
            continue
        raise
    added = sorted({m.split(".")[0] for m in sys.modules} - before & set(heavy))
    if added:
        report[name] = added
print(json.dumps(report))
"""


def _heavy_imports(modules) -> dict:
    """``{module: [heavy packages its import added] or "missing"}`` for the offenders."""
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, ",".join(HEAVY), *modules],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def offenders(modules) -> dict:
    """Every module of ``modules`` that is missing or loads a heavy package: one interpreter
    when all are clean, one per module otherwise (so every offender is named)."""
    found = _heavy_imports(modules)
    if not found:
        return {}
    return {name: bad for name in modules if (bad := _heavy_imports([name]).get(name))}


def test_the_qt_free_modules_load_neither_pyqt_nor_torch():
    assert offenders(QT_FREE) == {}, "these modules must not import PyQt6 or torch"


def test_the_probe_names_a_heavy_import_and_a_missing_module():
    # The scan cannot pass by seeing nothing: app.settings is built on PyQt6, and a module
    # that is not there is reported, not skipped.
    found = offenders(("app.editing.journal", "app.settings", "app.no_such_module"))
    assert found == {"app.settings": ["PyQt6"], "app.no_such_module": "missing"}

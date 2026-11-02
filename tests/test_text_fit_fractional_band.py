"""Text fit at fractional glyph advances: a label that picks one of several forms paints the
first form that fits whole, and an elided name always has its tooltip (guard of the text-fit
patches, Qt toolkit item 4).

``QFontMetrics.horizontalAdvance`` rounds a text's width to whole pixels while ``elidedText``
compares the fractional width, so a form chosen by the rounded width can be cut by up to
half a pixel: "T1 · L44 · 2024.07.…" painted where "T1 · L44" fits. That band exists where
glyph advances are fractional: Windows at 125 / 150 % scaling (laptops), DejaVu on Linux,
CoreText on macOS. ``text_fit.label_advance`` (the fractional width, rounded up) is the
rule; the metrics bar's name cell and the class panel's names used the rounded one until
C5-02 (owner question #14) was fixed.

The band needs the native font engine: a child process on the Windows platform at 125 %
(``QT_SCALE_FACTOR``) measures the labels; no window is shown. Skipped off Windows.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCALE = "1.25"
NAMES = (
    "24EFCAM1_T001_L044_2024.07.18_111530_001_SMM",
    "24EFCAM2_T017_L103_2024.08.02_093015_002_SMM",
    "FIELD_T12_L7_2023.06.30_120000",
    "T5_L21_2024.07.18_101010_x",
)
CLASS_NAMES = ("Ignore", "Nodule", "Coarse roots", "Mycorrhizal hyphae", "Dead root")
CHILD_TIMEOUT_S = 120
DUMP_MARGIN_S = 20  # the child dumps its stacks this long before the test gives up
STDERR_TAIL = 6000  # characters of the child's stderr a failure shows

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="the native Windows platform")


# --------------------------------------------------------------------------- the child
def _stage(text: str) -> None:
    """Say on stderr how far the child got (the test shows it when the child hangs)."""
    print(f"stage: {text}", file=sys.stderr, flush=True)


def run_child(out: str) -> None:
    """Sweep the labels' widths; write the forms that broke the rule to ``out`` (JSON)."""
    import faulthandler

    # A hang dumps every thread's stack to stderr and ends the child before the test's
    # timeout, so the test can show where it hung (Windows CI hung here on 2026-11-02).
    faulthandler.dump_traceback_later(CHILD_TIMEOUT_S - DUMP_MARGIN_S, exit=True)
    _stage("start")
    sys.path.insert(0, str(ROOT))
    # First: the app's environment and logging, before Qt loads.
    import main as app_main

    # isort: split
    from PyQt6.QtWidgets import QApplication

    record: dict = {}
    try:
        _stage("main imported")
        app = QApplication.instance() or QApplication(["sprouts-text-fit"])
        _stage("application made")
        app_main.apply_stylesheet(app)
        _stage("stylesheet applied")
        from app.gui.class_panel import _NameLabel
        from app.gui.metrics_bar import MetricsBar
        from app.gui.widgets.text_fit import fit_forms, label_advance

        record["device_pixel_ratio"] = app.devicePixelRatio()
        bar = MetricsBar()
        _stage("metrics bar made")
        cell = bar._name_value
        cell.ensurePolished()
        cut_cell, cut_shared = [], []
        for name in NAMES:
            _stage(f"name cell: {name}")
            cell.setText(name)
            forms = cell._text_forms()
            for width in range(20, 500):
                cell.resize(width, 24)
                room = cell.contentsRect().width()
                metrics = cell.fontMetrics()
                whole = next((f for f in forms if label_advance(metrics, f) <= room), None)
                if whole is None:
                    continue
                if cell._elided() != whole:
                    cut_cell.append([name, room, cell._elided()])
                if fit_forms(forms, metrics, room) != whole:
                    cut_shared.append([name, room])
        label = _NameLabel()
        label.ensurePolished()
        untipped = []
        for name in CLASS_NAMES:
            _stage(f"class name: {name}")
            label.setText(name)
            for width in range(10, 200):
                label.resize(width, 20)
                if (label.elided_text() != label.text()) != label.is_elided():
                    untipped.append([name, label.contentsRect().width(), label.elided_text()])
        record.update(cut_cell=cut_cell, cut_shared=cut_shared, untipped=untipped)
        bar.deleteLater()
        label.deleteLater()
    except BaseException:  # noqa: BLE001 - reported to the test, which fails on it
        record["error"] = traceback.format_exc()
    finally:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=1)
        _stage("record written")
        sys.stdout.flush()
        os._exit(0)


# --------------------------------------------------------------------------- the tests
@pytest.fixture(scope="module")
def band(tmp_path_factory) -> dict:
    tmp_path = tmp_path_factory.mktemp("text_fit_band")
    out = tmp_path / "band.json"
    env = dict(os.environ)
    env.pop("QT_QPA_PLATFORM", None)  # the native font engine
    env.update(
        QT_SCALE_FACTOR=SCALE,
        SPROUTS_CONFIG_DIR=str(tmp_path / "config"),
        SPROUTS_LOG_DIR=str(tmp_path / "logs"),
        PYTHONIOENCODING="utf-8",
    )
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from tests.test_text_fit_fractional_band import run_child; run_child(sys.argv[2])"
    )
    try:
        done = subprocess.run(
            [sys.executable, "-c", code, str(ROOT), str(out)],
            cwd=str(tmp_path),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=CHILD_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        # The child's own stack dump comes first; this is a hang outside Python (e.g. at
        # its exit, after the record: then the record exists).
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode()
        pytest.fail(
            f"the child hung for {CHILD_TIMEOUT_S} s (record written: {out.exists()}); "
            f"its stderr:\n{stderr[-STDERR_TAIL:]}"
        )
    if not out.exists():
        stderr = done.stderr or ""
        if "Timeout (" in stderr:  # faulthandler's dump: the child hung, here is where
            pytest.fail(f"the child hung; its stderr:\n{stderr[-STDERR_TAIL:]}")
        pytest.skip(
            f"the native platform could not start (exit {done.returncode}): {stderr[-STDERR_TAIL:]}"
        )
    record = json.loads(out.read_text(encoding="utf-8"))
    assert "error" not in record, record["error"]
    return record


def test_the_shared_rule_paints_the_first_form_that_fits_whole(band):
    assert band["device_pixel_ratio"] == float(SCALE)
    assert band["cut_shared"] == []


def test_the_name_cell_paints_the_first_form_that_fits_whole(band):
    assert band["cut_cell"] == []


def test_an_elided_class_name_always_has_its_tooltip(band):
    assert band["untipped"] == []

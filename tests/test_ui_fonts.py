"""The Qt UI draws in IBM Plex (owner item #46 iii).

The stylesheets name "IBM Plex Sans" and "IBM Plex Mono" first, but Qt cannot read the
``.woff2`` files of ``app/assets/fonts/`` (the dashboards'), so each OS drew its own font.
The TTFs in ``resources/fonts/`` ship unmodified with their OFL licence and are registered
before the stylesheet loads (``main.apply_stylesheet`` -> ``widgets.fonts.register_fonts``).

The Qt checks run ``main.main`` in a child process: registering fonts in the suite's
QApplication would change the text widths of every later test (on Windows the offscreen
platform has no other font).
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from app.design import tokens
from app.gui.widgets import fonts

REPO_ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = REPO_ROOT / "resources" / "fonts"
THIRD_PARTY = REPO_ROOT / "app" / "assets" / "THIRD_PARTY.md"
#: CSS weights Plex Sans ships here, and the names a stylesheet may use for them; 800 draws
#: in Bold (the heaviest shipped).
DRAWN_WEIGHTS = {"normal", "bold", "500", "600", "700", "800"}


def test_the_shipped_fonts_are_the_ones_registered():
    on_disk = sorted(path.name for path in FONT_DIR.glob("*.ttf"))
    assert on_disk == sorted(fonts.FONT_FILES)
    assert Path(fonts.font_dir()) == FONT_DIR


def test_the_ofl_licence_is_next_to_the_fonts():
    text = (FONT_DIR / "IBM-Plex-OFL-1.1.txt").read_text(encoding="utf-8")
    assert 'Copyright © 2017 IBM Corp. with Reserved Font Name "Plex"' in text
    assert "SIL OPEN FONT LICENSE Version 1.1" in text


@pytest.mark.parametrize("name", fonts.FONT_FILES)
def test_each_font_has_the_bytes_third_party_md_records(name):
    rows = [
        line
        for line in THIRD_PARTY.read_text(encoding="utf-8").splitlines()
        if line.startswith(f"| `resources/fonts/{name}` |")
    ]
    assert len(rows) == 1, f"no row for {name} in app/assets/THIRD_PARTY.md"
    recorded = re.findall(r"`([0-9a-f]{64})`", rows[0])
    assert hashlib.sha256((FONT_DIR / name).read_bytes()).hexdigest() == recorded[0]


def test_the_stylesheets_name_the_shipped_families_first():
    assert tokens.SANS.startswith(f'"{fonts.FAMILIES[0]}"')
    assert tokens.MONO.startswith(f'"{fonts.FAMILIES[1]}"')
    qss = (REPO_ROOT / "resources" / "themes" / "dark_theme.qss").read_text(encoding="utf-8")
    named = re.findall(r'font:\s*[\d.]+pt\s+"([^"]+)"', qss)
    assert set(named) == set(fonts.FAMILIES)


def test_every_weight_the_ui_asks_for_is_shipped():
    """A new weight (a Light 300, say) needs its TTF: Qt would fake it from another."""
    sources = [*(REPO_ROOT / "app" / "gui").rglob("*.py")]
    sources += [*(REPO_ROOT / "resources" / "themes").glob("*.qss")]
    asked = {
        weight.lower()
        for path in sources
        for weight in re.findall(r"font-weight:\s*(\w+)", path.read_text(encoding="utf-8"))
    }
    assert asked and asked <= DRAWN_WEIGHTS, asked - DRAWN_WEIGHTS


_DRIVER = """
import sys
sys.path.insert(0, {repo!r})
import main
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QFontDatabase, QFontInfo
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow
from app.design import tokens
from app.gui.widgets import fonts

def added_ids():
    # The application's own fonts (their ids count up from 0); a system font is not one.
    return [i for i in range(64) if QFontDatabase.applicationFontFamilies(i)]

def added_families():
    return sorted({{f for i in added_ids() for f in QFontDatabase.applicationFontFamilies(i)}})

_set_style_sheet = QApplication.setStyleSheet

def set_style_sheet(self, sheet):
    print("AT_QSS", "|".join(added_families()))  # what is registered when the QSS loads
    _set_style_sheet(self, sheet)

QApplication.setStyleSheet = set_style_sheet

class StubWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # What main.main registered: this driver registers nothing before the labels.
        print("FAMILIES", "|".join(added_families()))
        cases = {{
            "plain": "",
            "semibold": "font-weight: 600;",
            "bold": "font-weight: bold;",
            "mono": f"font-family: {{tokens.MONO}};",
            "mono600": f"font-family: {{tokens.MONO}}; font-weight: 600;",
        }}
        for name, sheet in cases.items():
            label = QLabel("Root length 17.94 mm", self)
            label.setStyleSheet(sheet)
            label.ensurePolished()
            info = QFontInfo(label.font())
            print("FONT", name, info.family(), info.styleName(), sep="|")
        before = len(added_ids())
        fonts.register_fonts()
        print("AGAIN", before, len(added_ids()))
        QTimer.singleShot(50, QApplication.instance().quit)

main.MainWindow = StubWindow
print("RC", main.main(["sprouts"]))
"""


@pytest.fixture(scope="module")
def started(tmp_path_factory) -> dict[str, list[str]]:
    """What ``main.main`` registered and how labels resolve under its stylesheet, from a
    child process on the offscreen platform: ``{"FAMILIES": [...], "AT_QSS": [[...], ...],
    "AGAIN": [count, count], "plain": [family, style], ...}``. Families are those of the
    application's own fonts, so a Plex installed on the system does not count."""
    tmp_path = tmp_path_factory.mktemp("ui_fonts")
    env = dict(
        os.environ,
        QT_QPA_PLATFORM="offscreen",
        PYTHONDONTWRITEBYTECODE="1",
        SPROUTS_LOG_DIR=str(tmp_path / "logs"),
        SPROUTS_CONFIG_DIR=str(tmp_path / "config"),
    )
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_DRIVER.format(repo=str(REPO_ROOT)))],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    assert "RC 0" in lines, proc.stdout + proc.stderr
    record: dict[str, list] = {"AT_QSS": []}
    for line in lines:
        if line.startswith("FAMILIES "):
            record["FAMILIES"] = line.removeprefix("FAMILIES ").split("|")
        elif line.startswith("AT_QSS "):
            record["AT_QSS"].append(line.removeprefix("AT_QSS ").split("|"))
        elif line.startswith("AGAIN "):
            record["AGAIN"] = [int(n) for n in line.split()[1:]]
        elif line.startswith("FONT|"):
            _tag, name, *font = line.split("|")
            record[name] = font
    return record


def test_main_registers_both_families(started):
    assert set(fonts.FAMILIES) <= set(started["FAMILIES"])


def test_the_fonts_are_registered_before_the_stylesheet_loads(started):
    assert started["AT_QSS"], "main.main loaded no stylesheet"
    for registered in started["AT_QSS"]:
        assert set(fonts.FAMILIES) <= set(registered)


def test_registering_again_adds_nothing(started):
    assert started["AGAIN"] == [len(fonts.FONT_FILES)] * 2


def test_a_font_qt_cannot_load_is_logged_and_skipped(app, tmp_path, monkeypatch, caplog):
    (tmp_path / "Broken.ttf").write_bytes(b"not a font")
    monkeypatch.setattr(fonts, "FONT_FILES", ("Missing.ttf", "Broken.ttf"))
    monkeypatch.setattr(fonts, "font_dir", lambda: str(tmp_path))
    with caplog.at_level("WARNING", logger=fonts.__name__):
        assert fonts.register_fonts() == ()
    skipped = [r.getMessage() for r in caplog.records if r.name == fonts.__name__]
    assert len(skipped) == 2
    assert skipped[0].endswith("Missing.ttf") and skipped[1].endswith("Broken.ttf")


@pytest.mark.parametrize(
    ("case", "family", "style"),
    [
        ("plain", "IBM Plex Sans", "Regular"),
        ("semibold", "IBM Plex Sans", "SemiBold"),
        ("bold", "IBM Plex Sans", "Bold"),
        ("mono", "IBM Plex Mono", "Regular"),
        ("mono600", "IBM Plex Mono", "SemiBold"),
    ],
)
def test_the_stylesheet_draws_in_plex(started, case, family, style):
    assert started[case] == [family, style]

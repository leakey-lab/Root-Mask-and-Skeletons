"""The status line's device chip (``app/gui/shell_chrome.py``: ``runtime_status``,
``StatusLine``, ``runtime_chip_text``).

The chip names the inference device without loading torch. Before torch has resolved the
runtime it reads "Checking GPU…" while the start-up check runs, then the driver's guess
("GPU · <name> · CUDA 12.8" / "CPU", why in the tooltip), or "Device unknown" in the warning
colour when the check failed (reason in the tooltip); once torch has resolved the runtime,
its full line wins. A check state wins over a guess; the missing-model warnings are always
kept, and kept whole when the chip is short of room (V10): the device's details give way
first (vendor words, then precision and CUDA, then the name), "GPU" / "CPU" stay, and a
status such as "Device not detected yet" gives way to the warnings. The tooltip keeps the
full line. The runtime module is a fake: torch is never loaded.
"""

from __future__ import annotations

import sys
import types

import pytest
from PyQt6.QtGui import QFontMetrics

from app.gui import canvas_gl, shell_chrome
from app.gui.shell_chrome import (
    DEVICE_CHECK_FAILED,
    DEVICE_CHECKING,
    INFO_MAX_WIDTH,
    StatusLine,
    runtime_chip_text,
    runtime_status,
)
from app.gui.widgets import tokens
from app.inference.device_probe import DeviceGuess
from tests.device_chip_support import CPU, GPU, RUNTIME, FakeRuntime
from tests.qt_teardown import delete_widgets, flush_deleted


@pytest.fixture(autouse=True)
def guards(no_prompts, slot_errors):
    """No prompt and no exception in a Qt slot; the widgets a test deleteLater()s go first,
    so the slot check sees what their deletion raises."""
    yield
    flush_deleted()


@pytest.fixture
def canvas(monkeypatch):
    """The canvas path is fixed: the chip's tooltip ends with its line."""
    monkeypatch.setattr(canvas_gl, "canvas_line", lambda: ("OpenGL · Fake GPU", False))


@pytest.fixture
def runtime(monkeypatch):
    fake = FakeRuntime()
    monkeypatch.setitem(sys.modules, RUNTIME, fake.module)
    import app.inference as inference_package

    monkeypatch.setattr(inference_package, "runtime", fake.module, raising=False)
    return fake


@pytest.fixture
def line(app):
    made = [shell_chrome.StatusLine()]
    yield made[0]
    delete_widgets(made, close=False)


def shown(line) -> str:
    """The chip's full text (the label itself is elided to INFO_MAX_WIDTH)."""
    return line.info.toolTip().splitlines()[0].removeprefix("Inference runtime: ")


# ------------------------------------------------------------------ the chip text
def test_an_unresolved_device_reads_checking_while_the_check_runs(runtime):
    assert runtime_status("checking") == (DEVICE_CHECKING, True)
    assert runtime_status() == ("Device not detected yet", True)  # no check: as before


def test_checking_keeps_the_missing_model_warning(monkeypatch):
    fake = FakeRuntime(mask_weights=False)
    monkeypatch.setitem(sys.modules, RUNTIME, fake.module)
    assert runtime_status("checking") == (f"{DEVICE_CHECKING} · mask model missing", False)


def test_a_failed_check_reads_device_unknown(runtime):
    assert runtime_status("failed")[0] == DEVICE_CHECK_FAILED


def test_a_resolved_device_wins_over_the_check_state(runtime):
    runtime.describe(resolve=True)
    assert runtime_status("checking")[0] == "GPU · Fake RTX 9000 · CUDA 12.8 · TF32"


def test_checking_before_the_runtime_module_is_loaded(monkeypatch):
    monkeypatch.delitem(sys.modules, RUNTIME, raising=False)
    assert runtime_status("checking") == (DEVICE_CHECKING, True)
    assert RUNTIME not in sys.modules  # the chip never imports it


def test_the_chip_text_before_the_runtime_module_is_loaded(monkeypatch):
    monkeypatch.delitem(sys.modules, RUNTIME, raising=False)
    assert shell_chrome.runtime_status(None, GPU) == ("GPU · Fake RTX 9000 · CUDA 12.8", True)
    assert RUNTIME not in sys.modules


# ------------------------------------------------------------------ the status line
def test_the_statusline_state_api(app, runtime, monkeypatch):
    monkeypatch.setattr(canvas_gl, "canvas_line", lambda: ("OpenGL · Fake GPU", False))
    line = shell_chrome.StatusLine()
    line.set_device_check("checking")
    assert line.info.text() == DEVICE_CHECKING
    line.set_device_check("failed", "no CUDA driver")
    assert line.info.text() == DEVICE_CHECK_FAILED
    assert "no CUDA driver" in line.info.toolTip()
    runtime.describe(resolve=True)
    line.set_device_check(None)
    assert line.info.toolTip() == (
        "Inference runtime: GPU · Fake RTX 9000 · CUDA 12.8 · TF32\nCanvas: OpenGL · Fake GPU"
    )
    assert tokens.WARN not in line.info.styleSheet()
    line.deleteLater()


def test_a_failed_check_turns_the_chip_to_the_warning_colour(runtime, line):
    line.set_device_check(shell_chrome.FAILED, "DLL load failed")
    assert line.info.text() == DEVICE_CHECK_FAILED
    assert "DLL load failed" in line.info.toolTip()
    assert tokens.WARN in line.info.styleSheet()


def test_the_chip_names_the_guessed_gpu_with_why_in_the_tooltip(runtime, canvas, line):
    line.set_device_check(shell_chrome.CHECKING)
    assert line.info.text() == DEVICE_CHECKING
    line.set_device_guess(GPU)
    assert shown(line) == "GPU · Fake RTX 9000 · CUDA 12.8"
    assert line.info.text().startswith("GPU")
    assert line.info.toolTip() == (
        "Inference runtime: GPU · Fake RTX 9000 · CUDA 12.8\nNamed by the NVIDIA driver."
        "\nCanvas: OpenGL · Fake GPU"
    )
    assert tokens.WARN not in line.info.styleSheet()


def test_a_cpu_guess_reads_cpu(runtime, line):
    line.set_device_guess(CPU)
    assert line.info.text() == "CPU"
    assert "CPU-only build" in line.info.toolTip()


def test_the_guess_keeps_the_missing_model_warning(monkeypatch, line):
    fake = FakeRuntime(mask_weights=False)
    monkeypatch.setitem(sys.modules, RUNTIME, fake.module)
    line.set_device_guess(GPU)
    assert shown(line) == "GPU · Fake RTX 9000 · CUDA 12.8 · mask model missing"
    assert tokens.WARN in line.info.styleSheet()


def test_the_resolved_runtime_replaces_the_guess(runtime, canvas, line):
    line.set_device_guess(GPU)
    runtime.describe(resolve=True)  # torch loaded (the Generate warm-up, or a job)
    line.refresh_runtime()
    assert shown(line) == "GPU · Fake RTX 9000 · CUDA 12.8 · TF32"
    assert line.info.toolTip() == (
        "Inference runtime: GPU · Fake RTX 9000 · CUDA 12.8 · TF32\nCanvas: OpenGL · Fake GPU"
    )


def test_a_failure_after_the_guess_wins_and_drops_its_note(runtime, line):
    line.set_device_guess(GPU)
    line.set_device_check(shell_chrome.FAILED, "DLL load failed while importing _C")
    assert line.info.text() == DEVICE_CHECK_FAILED
    assert "Named by the NVIDIA driver" not in line.info.toolTip()
    assert "DLL load failed" in line.info.toolTip()


def test_an_unknown_guess_keeps_the_default_text_and_says_why(runtime, line):
    unknown = DeviceGuess("unknown", note="The PyTorch build could not be read.")
    line.set_device_guess(unknown)
    assert line.info.text() == "Device not detected yet"
    assert "could not be read" in line.info.toolTip()


# ------------------------------------------------------------------ fitting the chip (V10)
BOTH = "mask model missing · skeleton model missing"
GPU_LINE = "GPU · NVIDIA GeForce RTX 5070 Ti · CUDA 12.8 · TF32"


class CharMetrics:
    """6 px a character; eliding keeps a prefix and "…" (as Qt's ElideRight)."""

    def horizontalAdvance(self, text: str) -> int:  # noqa: N802 - QFontMetrics API
        return 6 * len(text)

    def elidedText(self, text: str, _mode, width: int) -> str:  # noqa: N802 - QFontMetrics API
        if 6 * len(text) <= width:
            return text
        keep = width // 6 - 1
        return text[:keep] + "…" if keep >= 0 else ""


def chip(text: str) -> str:
    return runtime_chip_text(text, CharMetrics(), 320)  # 53 characters


@pytest.mark.parametrize(
    "text",
    ["CPU", "GPU · RTX 4080 · CUDA 12.8 · TF32", "Device not detected yet · mask model missing"],
)
def test_a_line_that_fits_is_shown_as_it_is(text):
    assert chip(text) == text


def test_the_walkthrough_line_keeps_both_warnings_whole():
    """'GPU · NVIDIA GeForce RTX 5…g · skeleton model missing' before."""
    assert chip(f"GPU · NVIDIA GeForce RTX 5070 Ti · CUDA 12.8 · {BOTH}") == f"GPU · {BOTH}"


def test_the_vendor_words_go_first_then_the_details_from_the_end():
    assert chip("GPU · NVIDIA GeForce RTX 5070 Ti Laptop GPU · CUDA 12.8 · TF32") == (
        "GPU · RTX 5070 Ti Laptop GPU · CUDA 12.8 · TF32"
    )
    assert chip(f"{GPU_LINE} · mask model missing") == (
        "GPU · RTX 5070 Ti · CUDA 12.8 · mask model missing"
    )
    assert chip(f"{GPU_LINE} · skeleton model missing") == (
        "GPU · RTX 5070 Ti · skeleton model missing"
    )


def test_a_long_name_is_shortened_before_it_is_dropped():
    shown = chip(
        "GPU · NVIDIA RTX 6000 Ada Generation Workstation · CUDA 12.8 · mask model missing"
    )
    assert shown == "GPU · RTX 6000 Ada Generation W… · mask model missing"
    assert chip("GPU · " + "X" * 400) == "GPU · " + "X" * 46 + "…"


def test_cpu_and_gpu_stay_and_a_status_gives_way_to_the_warnings():
    assert chip(f"CPU · FP32 · {BOTH}") == f"CPU · {BOTH}"
    assert chip(f"Device not detected yet · {BOTH}") == BOTH
    assert chip(f"Checking GPU… · {BOTH}") == BOTH
    narrow = runtime_chip_text(f"GPU · RTX 4080 · {BOTH}", CharMetrics(), 100)
    assert narrow == f"GPU · {BOTH}"  # wider than asked: the kind and the warnings stay whole


class FixedRuntime:
    """``app.inference.runtime`` with both weights missing (a fresh install)."""

    def __init__(self, text: str, device: str = "GPU") -> None:
        self.module = types.ModuleType(RUNTIME)
        summary = types.SimpleNamespace(
            device=device,
            text=text,
            models_ok=False,
            mask_weights=False,
            skeleton_weights=False,
        )
        self.module.describe_runtime = lambda resolve=False: summary


@pytest.fixture
def make_line(app, monkeypatch):
    made = []

    def make(text: str, device: str = "GPU", guess=None) -> StatusLine:
        monkeypatch.setitem(sys.modules, RUNTIME, FixedRuntime(text, device).module)
        line = StatusLine()  # its first refresh already shows the line
        if guess is not None:
            line.set_device_guess(guess)
        made.append(line)
        return line

    yield make
    delete_widgets(made, close=False)


def check_chip(line: StatusLine, full: str, warnings: str, kind: str | None) -> str:
    """The chip shows the warnings whole (and ``kind`` first), no wider than the warnings
    with the kind need when they need more than INFO_MAX_WIDTH; the tooltip has it all."""
    shown = line.info.text()
    line.info.ensurePolished()  # its stylesheet font (the chip's monospace)
    width = QFontMetrics(line.info.font()).horizontalAdvance
    assert shown.endswith(warnings), shown
    if kind is not None:
        assert shown.startswith(kind + " · "), shown
    least = f"{kind} · {warnings}" if kind else warnings
    assert width(shown) <= max(INFO_MAX_WIDTH, width(least)), shown
    assert line.info.toolTip().splitlines()[0] == f"Inference runtime: {full}"
    return shown


def test_the_resolved_runtime_keeps_both_warnings(make_line):
    full = f"{GPU_LINE} · {BOTH}"
    line = make_line(full)
    check_chip(line, full, BOTH, "GPU")
    assert line.info.property("models_ok") is False


def test_the_driver_guess_keeps_both_warnings(make_line):
    guess = DeviceGuess("GPU", name="NVIDIA GeForce RTX 5070 Ti", cuda_version="12.8")
    line = make_line(f"Device not detected yet · {BOTH}", device="unknown", guess=guess)
    full = f"GPU · NVIDIA GeForce RTX 5070 Ti · CUDA 12.8 · {BOTH}"
    check_chip(line, full, BOTH, "GPU")


def test_an_unresolved_device_gives_way_to_the_warnings(make_line):
    full = f"Device not detected yet · {BOTH}"
    line = make_line(full, device="unknown")
    shown = check_chip(line, full, BOTH, None)
    assert "…" not in shown  # dropped, not cut


def test_one_warning_keeps_as_much_of_the_device_as_fits(make_line):
    full = f"{GPU_LINE} · skeleton model missing"
    line = make_line(full)
    shown = check_chip(line, full, "skeleton model missing", "GPU")
    assert "NVIDIA GeForce" not in shown  # the vendor words went first


def test_the_chip_measures_with_its_own_font_from_the_first_refresh(make_line):
    full = f"{GPU_LINE} · {BOTH}"
    line = make_line(full)
    line.info.ensurePolished()
    metrics = QFontMetrics(line.info.font())
    assert line.info.text() == runtime_chip_text(full, metrics)
    assert shell_chrome.INFO_MAX_WIDTH == INFO_MAX_WIDTH

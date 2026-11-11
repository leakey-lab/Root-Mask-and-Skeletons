"""The Correct layers keep their opacity and visibility across app restarts (QSettings;
KNOWN_ISSUES #6), and LayerPrefs writes only what changed (X-RUNTIME F6: every stage switch
used to write all four values and flush the INI file, 22-58 ms). Every test gives the page
its own INI file: the user's settings are never read or written.
"""

from __future__ import annotations

import pytest
from PyQt6.QtCore import QEvent, QSettings
from PyQt6.QtWidgets import QApplication

from app.gui.correct.editor import CorrectEditor
from app.gui.correct.layer import MASK, SKELETON
from app.gui.correct.layer_prefs import LayerPrefs
from app.settings import MemorySettings
from tests.correct_support import new_project, settle
from tests.qt_teardown import delete_widgets

pytestmark = pytest.mark.usefixtures("correct_guards")


@pytest.fixture
def ini(tmp_path):
    return str(tmp_path / "settings.ini")


@pytest.fixture
def pages(app):
    made: list[CorrectEditor] = []

    def make(settings=None, image=None, base=None) -> CorrectEditor:
        editor = CorrectEditor(settings=settings)
        editor.resize(1100, 800)
        editor.show()
        settle(editor)
        if image is not None:
            editor.load_image(image, base)
            settle(editor)
        made.append(editor)
        return editor

    yield make
    delete_widgets(made, prepare=lambda editor: editor.layers[SKELETON].cancel_scheduled_render())


def settings_at(path: str) -> QSettings:
    return QSettings(path, QSettings.Format.IniFormat)


def restart(editor: CorrectEditor) -> None:
    """What quitting the app does to the page: it is hidden with its window."""
    editor.hide()
    settle()


def test_opacity_and_visibility_come_back_after_a_restart(pages, ini, tmp_path):
    project = new_project(tmp_path)
    first = pages(settings_at(ini), project.a, project.folder)
    first.controls.opacity_slider.slider.setValue(35)  # the mask is the edited layer
    first.set_layer_visible(SKELETON, False)
    restart(first)

    second = pages(settings_at(ini), project.b, project.folder)

    mask, skeleton = second.layers[MASK], second.layers[SKELETON]
    assert mask.opacity() == pytest.approx(0.35) and mask.is_visible()
    assert not skeleton.is_visible()
    assert skeleton.opacity() == pytest.approx(skeleton.DEFAULT_OPACITY)
    assert second.controls.opacity_slider.slider.value() == 35
    assert not second.layer_panel.rows[SKELETON].eye_button.isChecked()
    assert second.layer_panel.rows[MASK].eye_button.isChecked()
    assert mask.item.opacity() == pytest.approx(0.35)  # drawn as restored
    assert not skeleton.item.isVisible()


def test_a_restored_hidden_layer_is_shown_again_to_be_edited(pages, ini, tmp_path):
    project = new_project(tmp_path)
    first = pages(settings_at(ini))
    first.set_layer_visible(MASK, False)
    restart(first)
    second = pages(settings_at(ini), project.a, project.folder)
    notices = []
    second.notice.connect(lambda message, kind: notices.append(message))

    second.clear_active_layer()  # the mask, hidden: shown first

    assert second.layers[MASK].is_visible() and "Mask shown" in notices


def test_the_values_are_stored_under_the_correct_group(pages, ini):
    first = pages(settings_at(ini))
    first.set_active_layer(SKELETON)
    first.controls.opacity_slider.slider.setValue(50)
    restart(first)

    stored = settings_at(ini)
    assert float(stored.value("correct/skeleton/opacity")) == pytest.approx(0.5)
    assert stored.value("correct/mask/visible") in (True, "true")


@pytest.mark.parametrize(
    "opacity, visible",
    [("abc", "maybe"), ("7.5", "2"), ("-1", ""), ("nan", None)],
)
def test_unreadable_values_leave_the_defaults(pages, ini, opacity, visible):
    stored = settings_at(ini)
    stored.setValue("correct/mask/opacity", opacity)
    if visible is not None:
        stored.setValue("correct/mask/visible", visible)
    stored.sync()

    page = pages(settings_at(ini))

    mask = page.layers[MASK]
    assert mask.opacity() == pytest.approx(mask.DEFAULT_OPACITY) and mask.is_visible()


def test_a_page_without_settings_on_the_offscreen_platform_keeps_nothing(pages, monkeypatch):
    made = []
    monkeypatch.setattr(
        "app.settings.QSettings",
        lambda *args, **kwargs: made.append(args) or pytest.fail("QSettings made"),
    )

    page = pages()
    page.set_layer_visible(MASK, False)
    restart(page)

    assert made == [] and isinstance(LayerPrefs.for_app(None).settings, MemorySettings)
    again = pages()  # a fresh store: nothing of the first page was kept
    assert again.layers[MASK].is_visible()


class Layer:
    """The part of an edit layer LayerPrefs reads and sets."""

    def __init__(self, layer_id: str, opacity: float = 0.6, visible: bool = True) -> None:
        self.layer_id = layer_id
        self._opacity, self._visible = opacity, visible

    def opacity(self) -> float:
        return self._opacity

    def set_opacity(self, value: float) -> None:
        self._opacity = max(0.0, min(1.0, float(value)))

    def is_visible(self) -> bool:
        return self._visible

    def set_visible(self, visible: bool) -> None:
        self._visible = bool(visible)


class Recording:
    """A QSettings stand-in over a real INI QSettings that records writes and flushes."""

    def __init__(self, path: str) -> None:
        self.inner = QSettings(path, QSettings.Format.IniFormat)
        self.writes: list[tuple[str, object]] = []
        self.syncs = 0

    def value(self, key, *args):
        return self.inner.value(key, *args)

    def setValue(self, key, value) -> None:  # noqa: N802 - QSettings' name
        self.writes.append((key, value))
        self.inner.setValue(key, value)

    def sync(self) -> None:
        self.syncs += 1
        self.inner.sync()

    def status(self):
        return self.inner.status()


def stored_prefs(ini: str, **values) -> None:
    settings = QSettings(ini, QSettings.Format.IniFormat)
    for key, value in values.items():
        settings.setValue(key.replace("__", "/"), value)
    settings.sync()


# ---------------------------------------------------------------------------- LayerPrefs
def test_a_store_with_nothing_changed_writes_and_flushes_nothing(ini):
    stored_prefs(
        ini,
        correct__mask__opacity=0.35,
        correct__mask__visible=True,
        correct__skeleton__opacity=0.85,
        correct__skeleton__visible=False,
    )
    layers = [Layer(MASK), Layer(SKELETON, 0.85)]
    settings = Recording(ini)
    prefs = LayerPrefs(settings)
    prefs.restore(layers)

    prefs.store(layers)

    assert settings.writes == [] and settings.syncs == 0


def test_only_the_value_that_changed_is_written_then_flushed_once(ini):
    stored_prefs(ini, correct__mask__opacity=0.35, correct__mask__visible=True)
    layers = [Layer(MASK), Layer(SKELETON, 0.85)]
    settings = Recording(ini)
    prefs = LayerPrefs(settings)
    prefs.restore(layers)
    prefs.store(layers)  # the skeleton had nothing stored yet: its two values are written
    settings.writes.clear()
    settings.syncs = 0

    layers[0].set_opacity(0.5)
    prefs.store(layers)

    assert settings.writes == [("correct/mask/opacity", 0.5)] and settings.syncs == 1
    settings.writes.clear()
    prefs.store(layers)  # written once: the next store has nothing to write
    assert settings.writes == [] and settings.syncs == 1
    assert float(QSettings(ini, QSettings.Format.IniFormat).value("correct/mask/opacity")) == 0.5


def test_the_first_store_writes_every_value_nothing_was_stored_for(ini):
    layers = [Layer(MASK), Layer(SKELETON, 0.85, visible=False)]
    settings = Recording(ini)
    prefs = LayerPrefs(settings)
    prefs.restore(layers)

    prefs.store(layers)

    assert sorted(key for key, _value in settings.writes) == [
        "correct/mask/opacity",
        "correct/mask/visible",
        "correct/skeleton/opacity",
        "correct/skeleton/visible",
    ]
    assert settings.syncs == 1


def test_an_unreadable_stored_value_is_written_again(ini):
    stored_prefs(ini, correct__mask__opacity="abc", correct__mask__visible="maybe")
    layers = [Layer(MASK)]
    settings = Recording(ini)
    prefs = LayerPrefs(settings)
    prefs.restore(layers)

    prefs.store(layers)

    assert settings.writes == [("correct/mask/opacity", 0.6), ("correct/mask/visible", True)]


def test_without_a_restore_every_value_is_written_as_before(ini):
    settings = Recording(ini)

    LayerPrefs(settings).store([Layer(MASK)])

    assert len(settings.writes) == 2 and settings.syncs == 1


# ------------------------------------------------------------------------------ the page
def test_hiding_the_page_again_without_changes_writes_nothing(app, ini):
    recording = Recording(ini)
    editor = CorrectEditor(settings=recording)  # LayerPrefs needs only these four methods
    try:
        editor.resize(900, 700)
        editor.show()
        settle(editor)
        editor.hide()  # the first hide stores the defaults (nothing was stored yet)
        settle()
        assert recording.syncs == 1
        recording.writes.clear()
        recording.syncs = 0

        for _switch in range(3):  # stage switches back and forth
            editor.show()
            settle(editor)
            editor.hide()
            settle()
        assert recording.writes == [] and recording.syncs == 0

        editor.show()
        settle(editor)
        editor.set_layer_visible(SKELETON, False)
        editor.hide()
        settle()
        assert recording.writes == [("correct/skeleton/visible", False)]
        assert recording.syncs == 1
    finally:
        editor.layers[SKELETON].cancel_scheduled_render()
        editor.close()
        editor.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.mark.parametrize("stored", [0.0, 1.0])
def test_the_bounds_of_the_opacity_come_back(pages, ini, stored):
    """A layer kept fully transparent (or opaque) comes back so, not at its default."""
    settings = settings_at(ini)
    settings.setValue("correct/mask/opacity", stored)
    settings.sync()

    page = pages(settings_at(ini))

    assert page.layers[MASK].opacity() == pytest.approx(stored)

"""The settings file (#45): one INI file in the user config folder on every OS, and the
one-time import of the values the app kept in the native store before it.

The native store (the registry on Windows, a plist on macOS, ``~/.config/LeakeyLab/SPROUTS.conf``
on Linux) is never opened here: each test hands :func:`app.settings.ini_settings` a stand-in
(:class:`FakeNative`, or an INI file in ``tmp_path``) and a settings file in ``tmp_path``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from PyQt6.QtCore import QByteArray, QSettings

import app.settings as factory
from app import APP_NAME, ORGANIZATION_NAME
from app.gui.correct.layer_prefs import LayerPrefs
from app.gui.device_check import device_check_enabled
from app.gui.recent_projects import RecentProjectsStore
from app.gui.window_state import WindowStateStore
from app.settings import IMPORT_KEY, MemorySettings, app_settings, ini_settings, native_settings
from app.user_paths import SETTINGS_FILE_NAME, user_settings_file

INI = QSettings.Format.IniFormat
NO_ERROR = QSettings.Status.NoError

# A recent-projects list as the app writes it: JSON with commas, quotes, backslashes and a
# non-ASCII folder name (the INI writer must quote and escape it, and read it back whole).
RECENT = (
    '[{"name": "Racine été", "path": "D:\\\\data\\\\Racine été", "count": 40, '
    '"ts": "2026-10-30T09:15:00"}, {"name": "b", "path": "E:/b", "count": null, "ts": ""}]'
)
GEOMETRY = QByteArray(bytes(range(256)) * 2)  # every byte value, NULs included
SPLITTER = QByteArray(b"\x00\x00\x00\xff\x00\x00\x00\x01\x00\x00\x00\x02\x01\x2c")

# What each native store hands back for the app's keys (probed on Windows 2026-10-31: every
# value REG_SZ text but the two QByteArray states, kept as REG_BINARY). A plist keeps the
# written types; Linux's native store is an INI file already (text, QByteArray).
TEXT_VALUES = {
    "recent_projects": RECENT,
    "window/geometry": GEOMETRY,
    "window/body_splitter": SPLITTER,
    "window/last_folder": "D:/2024 EF image data",
    "correct/mask/opacity": "0.95",
    "correct/mask/visible": "true",
    "correct/skeleton/opacity": "0.85",
    "correct/skeleton/visible": "false",
    "correct/classes/opacity": "0.6",
    "correct/classes/visible": "true",
    "device_check/at_startup": "false",
}
PLIST_VALUES = {
    **TEXT_VALUES,
    "correct/mask/opacity": 0.95,
    "correct/mask/visible": True,
    "correct/skeleton/opacity": 0.85,
    "correct/skeleton/visible": False,
    "correct/classes/opacity": 0.6,
    "correct/classes/visible": True,
    "device_check/at_startup": 0,  # a registry DWORD a user set by hand reads as int too
}
NATIVE_VALUES = {"windows": TEXT_VALUES, "macos": PLIST_VALUES, "linux": TEXT_VALUES}


class FakeNative:
    """A native store that can only be read: any write fails the test."""

    def __init__(self, values=None, status=NO_ERROR, name="HKEY_CURRENT_USER\\Software\\x"):
        self.values = dict(values or {})
        self._status = status
        self._name = name

    def allKeys(self):  # noqa: N802 - Qt API name
        return list(self.values)

    def value(self, key, default=None):
        return self.values.get(key, default)

    def status(self):
        return self._status

    def fileName(self):  # noqa: N802 - Qt API name
        return self._name

    def _write(self, *args, **kwargs):
        pytest.fail("the import wrote to the native store")

    setValue = remove = clear = sync = _write  # noqa: N815 - Qt API names


class Opened:
    """``native`` for :func:`ini_settings` that counts how often the native store is opened."""

    def __init__(self, store):
        self.store, self.count = store, 0

    def __call__(self):
        self.count += 1
        return self.store


def reopened(path: Path) -> QSettings:
    """``path`` read back from disk: a copy under a new name, so Qt parses the file afresh
    instead of serving its in-process cache of ``path``."""
    copy = path.with_name(f"copy-{len(list(path.parent.iterdir()))}.ini")
    shutil.copyfile(path, copy)
    return QSettings(str(copy), INI)


class Layer:
    """The parts of an :class:`~app.gui.correct.layer.EditLayer` that LayerPrefs restores."""

    def __init__(self, layer_id):
        self.layer_id, self.opacity, self.visible = layer_id, None, None

    def set_opacity(self, value):
        self.opacity = value

    def set_visible(self, value):
        self.visible = value


class Shown:
    """The parts of an :class:`~app.gui.correct.layer.EditLayer` that LayerPrefs stores."""

    def __init__(self, layer_id, opacity, visible):
        self.layer_id, self._opacity, self._visible = layer_id, opacity, visible

    def opacity(self):
        return self._opacity

    def is_visible(self):
        return self._visible


def restored_layers(store) -> dict[str, tuple[float | None, bool | None]]:
    layers = [Layer("mask"), Layer("skeleton"), Layer("classes")]
    LayerPrefs(store).restore(layers)
    return {layer.layer_id: (layer.opacity, layer.visible) for layer in layers}


# ------------------------------------------------------------------- the one-time import
def test_first_open_copies_the_native_values_and_records_it(tmp_path):
    path = tmp_path / "settings.ini"
    native = FakeNative(TEXT_VALUES)

    store = ini_settings(str(path), native=lambda: native)

    assert store.format() == INI and store.status() == NO_ERROR
    assert path.is_file()  # flushed at once: the import is on disk before the app goes on
    back = reopened(path)
    for key, value in TEXT_VALUES.items():
        assert back.value(key) == value, key
    assert back.contains(IMPORT_KEY)
    assert int(back.value("native_import/keys")) == len(TEXT_VALUES)
    assert back.value("native_import/source") == native.fileName()


def test_the_import_runs_once(tmp_path):
    path = str(tmp_path / "settings.ini")
    opened = Opened(FakeNative({"window/last_folder": "D:/a"}))
    ini_settings(path, native=opened)
    assert opened.count == 1

    opened.store.values["recent_projects"] = RECENT  # set by an older version run since
    again = ini_settings(path, native=opened)
    fresh = reopened(Path(path))  # a later start: the file alone decides

    assert opened.count == 1  # the native store is not even opened again
    assert again.value("window/last_folder") == "D:/a"
    assert not again.contains("recent_projects") and not fresh.contains("recent_projects")
    assert fresh.contains(IMPORT_KEY)


def test_values_already_in_the_file_win(tmp_path):
    path = tmp_path / "settings.ini"
    hand_made = QSettings(str(path), INI)  # e.g. a file copied from another computer
    hand_made.setValue("window/last_folder", "E:/mine")
    hand_made.sync()

    store = ini_settings(str(path), native=lambda: FakeNative(TEXT_VALUES))

    assert store.value("window/last_folder") == "E:/mine"
    assert store.value("correct/mask/opacity") == "0.95"  # what the file lacked is filled in
    assert int(store.value("native_import/keys")) == len(TEXT_VALUES) - 1


def test_the_native_store_is_only_read(tmp_path):
    """Rollback stays possible: an older version still finds every value where it was."""
    native_path = tmp_path / "native.conf"  # the Linux native store is an INI file
    native = QSettings(str(native_path), INI)
    for key, value in TEXT_VALUES.items():
        native.setValue(key, value)
    native.sync()
    before = native_path.read_bytes()

    ini_settings(str(tmp_path / "settings.ini"), native=lambda: native)
    ini_settings(str(tmp_path / "other.ini"), native=lambda: FakeNative(TEXT_VALUES))

    assert native_path.read_bytes() == before
    assert sorted(native.allKeys()) == sorted(TEXT_VALUES)


def test_an_unreadable_native_store_is_tried_again_next_time(tmp_path):
    path = str(tmp_path / "settings.ini")
    broken = FakeNative(TEXT_VALUES, status=QSettings.Status.FormatError)

    store = ini_settings(path, native=lambda: broken)
    assert not store.contains(IMPORT_KEY) and store.allKeys() == []

    store = ini_settings(path, native=lambda: FakeNative(TEXT_VALUES))  # readable again
    assert store.contains(IMPORT_KEY) and store.value("window/last_folder")


def test_a_failing_native_store_never_stops_the_app(tmp_path, caplog):
    def explode():
        raise OSError("no registry")

    store = ini_settings(str(tmp_path / "settings.ini"), native=explode)

    assert not store.contains(IMPORT_KEY)
    store.setValue("window/last_folder", "D:/a")  # the file works all the same
    assert store.value("window/last_folder") == "D:/a"
    assert "native settings" in caplog.text


def test_missing_values_are_not_copied(tmp_path):
    store = ini_settings(
        str(tmp_path / "settings.ini"),
        native=lambda: FakeNative({"window/last_folder": "D:/a", "gone": None}),
    )
    assert store.contains("window/last_folder") and not store.contains("gone")
    assert int(store.value("native_import/keys")) == 1


def test_the_native_store_is_the_users_app_store_without_fallbacks(monkeypatch):
    """Only the user's own app values (``HKCU\\Software\\LeakeyLab\\SPROUTS``), not the
    machine-wide or organisation-wide fallbacks that the native store also reads."""

    class Recorder:
        made: list = []

        def __init__(self, *args):
            self.made.append(args)
            self.fallbacks = True

        def setFallbacksEnabled(self, on):  # noqa: N802 - Qt API name
            self.fallbacks = on

    monkeypatch.setattr(factory, "QSettings", Recorder)
    store = native_settings()
    assert Recorder.made == [(ORGANIZATION_NAME, APP_NAME)] and store.fallbacks is False


# ------------------------------------------------------------ resetting the settings
def native_import_section(text: str) -> str:
    """``text`` (a settings file) cut to its ``[native_import]`` section, as a user would
    cut it by hand: the header and its lines, up to the next section."""
    kept, inside = [], False
    for line in text.splitlines():
        if line.startswith("["):
            inside = line.strip() == "[native_import]"
        if inside:
            kept.append(line)
    return "\n".join(kept) + "\n"


def test_deleting_the_file_imports_the_native_values_again(tmp_path):
    """The import is recorded in the file only: without the file, the next start copies
    the old native values back. So deleting the file is no reset where an older version
    left values (the docs must not call it one)."""
    path = tmp_path / "settings.ini"
    native = FakeNative({"window/last_folder": "D:/old", "recent_projects": RECENT})
    store = ini_settings(str(path), native=lambda: native)
    store.setValue("window/last_folder", "E:/new")
    store.sync()
    del store

    path.unlink()  # the user deletes the file to reset the settings
    opened = Opened(native)
    later = ini_settings(str(path), native=opened)

    assert opened.count == 1  # the native store is read again
    assert later.value("window/last_folder") == "D:/old"  # the old value is back
    assert RecentProjectsStore(later).entries()  # and the old recent projects


def test_keeping_only_the_native_import_section_resets_the_settings(tmp_path):
    """The reset the docs give: delete every line of the file but its ``[native_import]``
    section. The import does not run again and every store reads its default."""
    path = tmp_path / "settings.ini"
    store = ini_settings(str(path), native=lambda: FakeNative(TEXT_VALUES))
    store.setValue("recent_projects", RECENT)
    store.sync()
    reset = tmp_path / "reset.ini"  # the cut file, as the next start reads it
    reset.write_text(native_import_section(path.read_text(encoding="utf-8")), encoding="utf-8")

    opened = Opened(FakeNative(TEXT_VALUES))
    later = ini_settings(str(reset), native=opened)

    assert opened.count == 0  # the native store is not opened: no old value comes back
    assert sorted(later.allKeys()) == [IMPORT_KEY, "native_import/keys", "native_import/source"]
    assert RecentProjectsStore(later).entries() == []
    assert WindowStateStore(later).last_folder() == ""
    assert later.value("window/geometry") is None
    assert restored_layers(later) == {
        "mask": (None, None),
        "skeleton": (None, None),
        "classes": (None, None),
    }
    assert device_check_enabled(later, {}) is True


# ------------------------------------------------------------- every type reads back
@pytest.mark.parametrize("platform", sorted(NATIVE_VALUES))
def test_every_imported_value_reads_back_from_the_file(app, tmp_path, platform):
    """Each key the app keeps, as each native store types it, read by the code that reads
    it: from the file on disk, as at the next start."""
    from PyQt6.QtWidgets import QWidget

    from tests.qt_teardown import delete_widgets

    path = tmp_path / "settings.ini"
    ini_settings(str(path), native=lambda: FakeNative(NATIVE_VALUES[platform]))
    back = reopened(path)

    entries = RecentProjectsStore(back).entries()
    assert [(e.name, e.path, e.count) for e in entries] == [
        ("Racine été", "D:\\data\\Racine été", 40),
        ("b", "E:/b", None),
    ]
    assert back.value("window/geometry") == GEOMETRY
    assert back.value("window/body_splitter") == SPLITTER
    assert WindowStateStore(back).last_folder() == "D:/2024 EF image data"
    assert restored_layers(back) == {
        "mask": (0.95, True),
        "skeleton": (0.85, False),
        "classes": (0.6, True),
    }
    assert device_check_enabled(back, {}) is False

    # A real window's geometry, imported, restores a window.
    widgets = [QWidget(), QWidget()]
    try:
        widgets[0].setGeometry(40, 50, 640, 480)
        native = dict(NATIVE_VALUES[platform], **{"window/geometry": widgets[0].saveGeometry()})
        geometry_path = tmp_path / "geometry.ini"
        ini_settings(str(geometry_path), native=lambda: FakeNative(native))
        assert WindowStateStore(reopened(geometry_path)).restore(widgets[1]) is True
        assert widgets[1].size() == widgets[0].size()
    finally:
        delete_widgets(widgets)


def test_what_the_stores_write_reads_back_from_the_file(tmp_path):
    """The types the stores write (float, bool, QByteArray, JSON text) survive the file."""
    path = tmp_path / "settings.ini"
    store = ini_settings(str(path), native=lambda: FakeNative())
    shown = [Shown("mask", 0.95, True), Shown("skeleton", 0.1, False), Shown("classes", 1.0, True)]
    LayerPrefs(store).store(shown)
    recents = RecentProjectsStore(store)
    recents.add("D:\\data\\Racine été", 40)
    recents.add("E:/b, c", 3)  # a comma in a folder name
    store.setValue("window/geometry", GEOMETRY)
    WindowStateStore(store).set_last_folder("")  # cleared: empty text
    store.sync()

    back = reopened(path)
    assert restored_layers(back) == {
        "mask": (0.95, True),
        "skeleton": (0.1, False),
        "classes": (1.0, True),
    }
    assert [e.path for e in RecentProjectsStore(back).entries()] == [
        "E:/b, c",
        "D:\\data\\Racine été",
    ]
    assert back.value("window/geometry") == GEOMETRY
    assert WindowStateStore(back).last_folder() == ""


# ------------------------------------------------------------------- where the file is
def test_the_file_is_settings_ini_in_the_user_config_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("SPROUTS_CONFIG_DIR", str(tmp_path / "cfg"))
    assert SETTINGS_FILE_NAME == "settings.ini"
    assert user_settings_file() == os.path.join(str(tmp_path / "cfg"), "settings.ini")
    assert not (tmp_path / "cfg").exists()  # asking for the path creates nothing


def test_on_linux_the_file_shares_the_one_config_folder(monkeypatch, tmp_path):
    """One folder: the settings next to the Enhance defaults in ``~/.config/sprouts``, not
    a second ``~/.config/LeakeyLab``."""
    from app import user_paths

    monkeypatch.delenv("SPROUTS_CONFIG_DIR", raising=False)
    monkeypatch.setattr(user_paths.sys, "platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    assert user_settings_file() == os.path.join(
        str(tmp_path / ".config"), "sprouts", "settings.ini"
    )


def test_on_a_screen_the_app_store_is_the_file_with_the_import(app, monkeypatch, tmp_path):
    folder = tmp_path / "cfg"  # missing: made on the first write
    monkeypatch.setenv("SPROUTS_CONFIG_DIR", str(folder))
    monkeypatch.setattr(factory, "headless", lambda: False)
    opened = Opened(FakeNative({"window/last_folder": "D:/a"}))
    monkeypatch.setattr(factory, "native_settings", opened)

    first, second = app_settings(), app_settings()

    assert isinstance(first, QSettings) and first.format() == INI
    assert Path(first.fileName()) == folder / "settings.ini"
    assert (folder / "settings.ini").is_file()
    assert second.value("window/last_folder") == "D:/a" and opened.count == 1
    first.setValue("recent_projects", "[]")  # every store shares the one file
    assert second.value("recent_projects") == "[]"


def test_headless_still_gives_memory_and_opens_no_file(monkeypatch, tmp_path):
    monkeypatch.setenv("SPROUTS_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setattr(factory, "headless", lambda: True)
    monkeypatch.setattr(factory, "native_settings", lambda: pytest.fail("native opened"))
    assert isinstance(app_settings(), MemorySettings)
    assert not (tmp_path / "cfg").exists()

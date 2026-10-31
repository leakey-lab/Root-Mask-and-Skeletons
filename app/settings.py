"""The app's settings store: one factory, the only place that builds a ``QSettings``.

:func:`app_settings` gives the user's settings file: ``settings.ini`` in the user config
folder (:func:`app.user_paths.user_settings_file`: ``%APPDATA%/SPROUTS`` on Windows,
``~/Library/Application Support/SPROUTS`` on macOS, ``~/.config/sprouts`` on Linux), one INI
file on every OS (#45). On a headless platform (``offscreen``: tests, scripts, audit agents)
it gives a fresh :class:`MemorySettings` instead, so a headless window never reads or writes
the user's settings (C5-04, X8). The window state, the recent projects, the Correct layers'
opacity and visibility and the device-check switch default to it; a caller may hand its
own store (the tests' INI files).

Before the file, the settings lived in the native store, ``QSettings(ORGANIZATION_NAME,
APP_NAME)``: the registry on Windows (``HKCU\\Software\\LeakeyLab\\SPROUTS``), a plist on
macOS, ``~/.config/LeakeyLab/SPROUTS.conf`` on Linux. The first time the file is opened, the
native values it lacks are copied into it, and the import is recorded in the file
(:data:`IMPORT_KEY`), so it runs once. The native store is only read: an older version still
finds its values there.

Build every store here (``tests/test_settings_factory.py`` checks that no other module
builds one): ``QSettings(organization, application)`` ignores ``setDefaultFormat`` and
``setPath``, so nothing at the Qt level could redirect a store built elsewhere.

Layer-neutral, like :mod:`app.workers` (PyQt6 only): ``app.handlers`` and ``app.gui`` may
both import it.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from datetime import datetime
from typing import Any

from PyQt6.QtCore import QCoreApplication, QSettings
from PyQt6.QtGui import QGuiApplication

from app import APP_NAME, ORGANIZATION_NAME
from app.user_paths import user_settings_file

logger = logging.getLogger(__name__)

__all__ = [
    "HEADLESS_PLATFORMS",
    "IMPORT_KEY",
    "MemorySettings",
    "app_settings",
    "headless",
    "import_native",
    "ini_settings",
    "native_settings",
]

#: Qt platforms without a screen: they never use the user's settings.
HEADLESS_PLATFORMS = frozenset({"offscreen"})

#: Written to the settings file by the import of the native values (when it ran); next to
#: it ``native_import/source`` (where they came from) and ``native_import/keys`` (how many
#: were copied). Present: the import never runs again.
IMPORT_KEY = "native_import/date"

_NO_ERROR = QSettings.Status.NoError


class MemorySettings:
    """The ``QSettings`` calls the stores make, over a dict that lives as long as this
    object: nothing is persisted."""

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}

    def value(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def setValue(self, key: str, value: Any) -> None:  # noqa: N802 - Qt API name
        self._values[key] = value

    def sync(self) -> None:
        """Nothing to write."""

    def status(self) -> QSettings.Status:
        return _NO_ERROR


def headless() -> bool:
    """True on a platform without a screen (``offscreen``: tests, scripts, agents).

    Before a ``QGuiApplication`` exists, the platform it would start on
    (``QT_QPA_PLATFORM``): Qt then reports its built-in default instead.
    """
    if isinstance(QCoreApplication.instance(), QGuiApplication):
        name = QGuiApplication.platformName()
    else:
        name = os.environ.get("QT_QPA_PLATFORM", "").split(":", 1)[0].strip()
    return name in HEADLESS_PLATFORMS


def native_settings() -> QSettings:
    """The native store the settings lived in before the file (read by the import only).

    Without fallbacks: only the user's own values of the app, not the machine-wide
    (``HKLM``) or organisation-wide ones the native store would also read.
    """
    store = QSettings(ORGANIZATION_NAME, APP_NAME)
    store.setFallbacksEnabled(False)
    return store


def import_native(store: QSettings, native: Any) -> int:
    """Copy the values of ``native`` that ``store`` lacks into ``store`` (a value already in
    the file wins), record the import and flush; return how many were copied.

    ``native`` is only read. When it cannot be read, nothing is copied or recorded, so the
    next start tries again.
    """
    if native.status() != _NO_ERROR:
        logger.warning("Could not read the native settings (%s)", native.status().name)
        return 0
    copied = 0
    for key in native.allKeys():
        value = native.value(key)
        if value is None or store.contains(key):
            continue
        store.setValue(key, value)
        copied += 1
    store.setValue(IMPORT_KEY, datetime.now().isoformat(timespec="seconds"))
    store.setValue("native_import/source", native.fileName())
    store.setValue("native_import/keys", copied)
    store.sync()
    if store.status() == _NO_ERROR:
        logger.info("Imported %d native settings into %s", copied, store.fileName())
    else:
        logger.warning("Could not write the settings file %s", store.fileName())
    return copied


def ini_settings(path: str, native: Callable[[], Any] | None = None) -> QSettings:
    """The settings file at ``path`` (INI format); the first time, with the values of the
    store ``native()`` returns (default :func:`native_settings`) imported into it.

    A failing native store never stops the app: the file is returned all the same.
    """
    store = QSettings(path, QSettings.Format.IniFormat)
    if not store.contains(IMPORT_KEY):
        try:
            import_native(store, (native or native_settings)())
        except Exception:  # noqa: BLE001 - a broken native store must not stop the app
            logger.warning("Could not import the native settings", exc_info=True)
    return store


def app_settings() -> QSettings | MemorySettings:
    """The user's settings file (:func:`ini_settings` at
    :func:`~app.user_paths.user_settings_file`); a fresh :class:`MemorySettings` on a
    headless platform (see the module docstring)."""
    if headless():
        return MemorySettings()
    return ini_settings(user_settings_file())

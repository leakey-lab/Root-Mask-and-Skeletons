"""``app.user_paths``: per-user folders for settings, data and logs (never the repo), and
the settings file in the config folder.
"""

from __future__ import annotations

import os

import pytest

from app import user_paths
from app.user_paths import (
    SETTINGS_FILE_NAME,
    user_config_dir,
    user_data_dir,
    user_log_dir,
    user_settings_file,
)

_ENV = (
    "SPROUTS_CONFIG_DIR",
    "SPROUTS_DATA_DIR",
    "SPROUTS_LOG_DIR",
    "APPDATA",
    "LOCALAPPDATA",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
)


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for var in _ENV:
        monkeypatch.delenv(var, raising=False)
    home = tmp_path / "home"
    monkeypatch.setattr(user_paths, "_home", lambda: str(home))
    return home


def test_config_override_is_created_on_demand(clean_env, monkeypatch, tmp_path):
    target = tmp_path / "cfg"
    monkeypatch.setenv("SPROUTS_CONFIG_DIR", str(target))
    assert user_config_dir(ensure=False) == str(target)
    assert not target.exists()
    assert user_config_dir() == str(target)
    assert target.is_dir()


def test_config_override_also_hosts_data_and_logs(clean_env, monkeypatch, tmp_path):
    monkeypatch.setenv("SPROUTS_CONFIG_DIR", str(tmp_path / "cfg"))
    assert user_data_dir(ensure=False) == str(tmp_path / "cfg" / "data")
    assert user_log_dir(ensure=False) == str(tmp_path / "cfg" / "logs")
    monkeypatch.setenv("SPROUTS_LOG_DIR", str(tmp_path / "elsewhere"))
    assert user_log_dir() == str(tmp_path / "elsewhere")
    assert (tmp_path / "elsewhere").is_dir()


def test_windows_locations(clean_env, monkeypatch, tmp_path):
    monkeypatch.setattr(user_paths.sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    assert user_config_dir(ensure=False) == os.path.join(str(tmp_path / "Roaming"), "SPROUTS")
    assert user_data_dir(ensure=False) == os.path.join(str(tmp_path / "Local"), "SPROUTS")
    assert user_log_dir(ensure=False) == os.path.join(str(tmp_path / "Local"), "SPROUTS", "Logs")


def test_windows_without_appdata_falls_back_to_profile(clean_env, monkeypatch):
    monkeypatch.setattr(user_paths.sys, "platform", "win32")
    expected = os.path.join(str(clean_env), "AppData", "Roaming", "SPROUTS")
    assert user_config_dir(ensure=False) == expected


def test_macos_locations(clean_env, monkeypatch):
    monkeypatch.setattr(user_paths.sys, "platform", "darwin")
    support = os.path.join(str(clean_env), "Library", "Application Support", "SPROUTS")
    assert user_config_dir(ensure=False) == support
    assert user_data_dir(ensure=False) == support
    assert user_log_dir(ensure=False) == os.path.join(str(clean_env), "Library", "Logs", "SPROUTS")


def test_linux_xdg_locations(clean_env, monkeypatch, tmp_path):
    monkeypatch.setattr(user_paths.sys, "platform", "linux")
    home = str(clean_env)
    assert user_config_dir(ensure=False) == os.path.join(home, ".config", "sprouts")
    assert user_data_dir(ensure=False) == os.path.join(home, ".local", "share", "sprouts")
    assert user_log_dir(ensure=False) == os.path.join(home, ".local", "state", "sprouts", "log")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert user_config_dir(ensure=False) == os.path.join(str(tmp_path / "xdg"), "sprouts")


def test_default_location_is_outside_the_repository(clean_env):
    repo = os.path.dirname(os.path.dirname(os.path.abspath(user_paths.__file__)))
    path = os.path.abspath(user_config_dir(ensure=False))
    assert not os.path.normcase(path).startswith(os.path.normcase(repo) + os.sep)


# ------------------------------------------------------------------ the settings file (#45)


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

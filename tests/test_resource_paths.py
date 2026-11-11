"""Resource paths (``resources/resource_utils.py``) and the modules that resolve through it.

Every bundled file (the theme, icons, the dashboards' assets, the checkpoints folder) is
found from any working directory, and inside a PyInstaller bundle from ``sys._MEIPASS``.
``get_resource_path`` stays an alias: the CI workflow's packaging check imports it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from resources.resource_utils import get_resource_path, project_root, resource_path

REPO_ROOT = Path(os.path.abspath(__file__)).parent.parent


def test_resource_path_is_cwd_independent(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    qss = resource_path("resources", "themes", "dark_theme.qss")
    assert os.path.isabs(qss)
    assert os.path.isfile(qss)
    assert os.path.isfile(resource_path("resources/app_icon.ico"))


def test_resource_path_accepts_joined_or_split_parts():
    assert resource_path("resources/themes/dark_theme.qss") == resource_path(
        "resources", "themes", "dark_theme.qss"
    )
    assert resource_path(Path("resources") / "themes") == resource_path("resources", "themes")


def test_resource_path_uses_the_pyinstaller_bundle_when_frozen(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert project_root() == tmp_path
    assert resource_path("resources", "icons") == os.path.join(str(tmp_path), "resources", "icons")


def test_project_root_is_the_repository_in_development():
    assert project_root() == REPO_ROOT


def test_get_resource_path_remains_an_alias():
    assert get_resource_path is resource_path


def test_config_project_root_is_the_shared_resolver():
    from app import config

    assert config.PROJECT_ROOT == str(project_root())
    assert config.CHECKPOINTS_DIR == os.path.join(str(project_root()), "checkpoints")


def test_dash_assets_dir_is_resolved_by_resource_utils():
    from app.visualization import dash_shell

    assert dash_shell.ASSETS_DIR == resource_path("app", "assets")
    assert os.path.isfile(os.path.join(dash_shell.ASSETS_DIR, "style.css"))


def test_icon_paths_do_not_depend_on_the_working_directory(tmp_path, monkeypatch):
    from app.gui.ui_panels import get_icon_path

    icons = os.listdir(resource_path("resources", "icons"))
    name = next(n for n in icons if os.path.isfile(resource_path("resources", "icons", n)))
    monkeypatch.chdir(tmp_path)
    assert get_icon_path(name) == os.path.join(str(project_root()), "resources", "icons", name)

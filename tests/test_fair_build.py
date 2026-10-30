"""The Windows bundle can be rebuilt from the repository (FAIR4RS A1/R1; FR-13).

``scripts/build_support.py`` (imported by ``scripts/build_executable.spec``) is
tested directly. The spec itself is executed against recording stand-ins for
PyInstaller's ``Analysis``/``PYZ``/``EXE``/``COLLECT``, so no bundle is built.
The build scripts are run from a throw-away copy with a stub interpreter.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import app

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


@pytest.fixture(scope="module")
def bs():
    """scripts/build_support.py loaded by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("build_support", SCRIPTS / "build_support.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_required_tree(bs, root: Path) -> None:
    """Placeholder files/folders for every required bundle source under ``root``."""
    for source, _dest in bs.REQUIRED_DATAS:
        path = root / source
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
        else:
            path.mkdir(parents=True, exist_ok=True)
            (path / "placeholder.txt").write_bytes(b"x")


# ------------------------------------------------------------------ datas


def test_repo_datas_exist_and_include_theme_icons_and_dash_assets(bs):
    datas, _warnings = bs.resolve_datas(ROOT)
    sources = {Path(src).resolve() for src, _dest in datas}
    assert all(Path(src).exists() for src, _dest in datas)
    for folder in ("resources/themes", "resources/icons", "app/assets"):
        assert (ROOT / folder).resolve() in sources, folder


def test_missing_required_source_stops_the_build_naming_it(bs, tmp_path):
    _make_required_tree(bs, tmp_path)
    shutil.rmtree(tmp_path / "resources" / "icons")
    with pytest.raises(bs.BuildConfigError, match="resources.icons"):
        bs.resolve_datas(tmp_path)


def test_missing_model_weights_warn_and_are_left_out(bs, tmp_path):
    _make_required_tree(bs, tmp_path)
    datas, warnings = bs.resolve_datas(tmp_path)
    assert len(warnings) == len(bs.MODEL_DATAS) == 2
    assert not [src for src, _dest in datas if "checkpoints" in src]


def test_present_model_weights_are_bundled_at_their_runtime_path(bs, tmp_path):
    _make_required_tree(bs, tmp_path)
    weights = (
        "checkpoints/mask_weights/best_mask_model_V5.pth",
        "checkpoints/skeletonizer/latest_net_G.pth",
    )
    for rel in weights:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"weights")
    datas, warnings = bs.resolve_datas(tmp_path)
    assert warnings == []
    bundled = {Path(src).relative_to(tmp_path).as_posix(): dest for src, dest in datas}
    assert bundled[weights[0]] == "checkpoints/mask_weights"
    assert bundled[weights[1]] == "checkpoints/skeletonizer"


# ------------------------------------------------- torchvision op libraries


def test_torchvision_op_libraries_bundles_every_top_level_extension(bs, tmp_path):
    """torchvision loads _C_stable / image_stable by path; PyInstaller's hook still names
    the pre-0.29 _C, so a frozen import torchvision failed without them."""
    import importlib.machinery

    suffix = importlib.machinery.EXTENSION_SUFFIXES[-1]
    for name in ("_C_stable", "image_stable"):
        (tmp_path / f"{name}{suffix}").write_bytes(b"x")
    (tmp_path / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / f"nested{suffix}").write_bytes(b"x")  # only the top level is loaded
    found = bs.torchvision_op_libraries(tmp_path)
    assert [(Path(src).name, dest) for src, dest in found] == [
        (f"_C_stable{suffix}", "torchvision"),
        (f"image_stable{suffix}", "torchvision"),
    ]


def test_installed_torchvision_op_library_is_found(bs):
    """The environment the bundle is built from: its torchvision ops library is collected."""
    import importlib.util

    if importlib.util.find_spec("torchvision") is None:
        pytest.skip("torchvision not installed")
    names = {Path(src).stem.split(".")[0] for src, _dest in bs.torchvision_op_libraries()}
    assert names & {"_C", "_C_stable"}, names


# ------------------------------------------------------------- version info


@pytest.mark.parametrize(
    "version,expected",
    [
        ("3.0.0", (3, 0, 0, 0)),
        ("3.1", (3, 1, 0, 0)),
        ("3.1.0rc1", (3, 1, 0, 0)),
        ("10.2.3.4", (10, 2, 3, 4)),
    ],
)
def test_version_tuple_is_four_integers_for_windows(bs, version, expected):
    assert bs.version_tuple(version) == expected


def test_windows_version_resource_carries_name_and_version(bs):
    pytest.importorskip("PyInstaller.utils.win32.versioninfo")
    info = bs.windows_version_info()
    ffi = info.ffi
    assert (ffi.fileVersionMS, ffi.fileVersionLS) == (3 << 16 | 0, 0 << 16 | 0)
    strings = {s.name: s.val for table in info.kids[0].kids for s in table.kids}
    assert strings["ProductName"] == app.APP_NAME
    assert strings["ProductVersion"] == strings["FileVersion"] == app.__version__
    assert strings["OriginalFilename"] == f"{app.APP_NAME}.exe"


# --------------------------------------------------------------- build_info


def test_build_info_records_version_commit_and_zoned_time(bs, tmp_path):
    info = bs.collect_build_info(ROOT)
    assert info["name"] == app.APP_NAME and info["version"] == app.__version__
    assert dt.datetime.fromisoformat(info["built_at"]).tzinfo is not None
    assert info["python"] == ".".join(map(str, sys.version_info[:3]))
    path = bs.write_build_info(tmp_path / "nested" / "build_info.json", info)
    assert json.loads(path.read_text(encoding="utf-8")) == info


def test_frozen_app_provenance_reads_the_commit_from_build_info(bs, tmp_path):
    """A frozen app has no .git: app.provenance must find the commit in build_info.json."""
    provenance = pytest.importorskip("app.provenance")
    info = bs.collect_build_info(ROOT)
    if info["git_commit"] is None:
        pytest.skip("not a git checkout")
    frozen_root = tmp_path / "_internal"  # sys._MEIPASS of a one-folder bundle
    bs.write_build_info(frozen_root / bs.BUILD_INFO_FILENAME, info)
    assert provenance.git_commit(str(frozen_root)) == info["git_commit"]


def test_build_info_without_git_has_no_commit(bs, monkeypatch):
    def _no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(bs.subprocess, "run", _no_git)
    info = bs.collect_build_info(ROOT)
    assert info["git_commit"] is None and info["git_dirty"] is None


# --------------------------------------------------------------------- spec


class _Recorder:
    """Stand-in for a PyInstaller build target: records its arguments."""

    calls: dict = {}

    def __init__(self, *args, **kwargs):
        type(self).calls[type(self).__name__] = (args, kwargs)
        self.pure = self.zipped_data = self.scripts = []
        self.binaries = self.zipfiles = self.datas = []


def test_spec_wires_repo_paths_icon_version_and_build_info(bs, tmp_path, monkeypatch):
    pytest.importorskip("PyInstaller.utils.win32.versioninfo")
    monkeypatch.setattr(sys, "path", list(sys.path))  # the spec prepends scripts/ and the root
    _Recorder.calls = {}
    namespace = {
        "__file__": str(SCRIPTS / "build_executable.spec"),
        "SPECPATH": str(SCRIPTS),
        "workpath": str(tmp_path / "work"),
        "DISTPATH": str(tmp_path / "dist"),
        **{name: type(name, (_Recorder,), {}) for name in ("Analysis", "PYZ", "EXE", "COLLECT")},
    }
    source = (SCRIPTS / "build_executable.spec").read_text(encoding="utf-8")
    try:
        exec(compile(source, "build_executable.spec", "exec"), namespace)
    finally:
        sys.modules.pop("build_support", None)

    (scripts,), analysis = _Recorder.calls["Analysis"]
    assert [Path(s) for s in scripts] == [ROOT / "main.py"]
    datas = analysis["datas"]
    assert all(Path(src).exists() for src, _dest in datas)
    assert (str(tmp_path / "work" / "build_info.json"), ".") in datas
    assert "app.gt.registry" in analysis["hiddenimports"]
    assert analysis["binaries"] == bs.torchvision_op_libraries()

    _args, exe = _Recorder.calls["EXE"]
    assert exe["name"] == app.APP_NAME
    assert Path(exe["icon"]).is_file()
    strings = {s.name: s.val for table in exe["version"].kids[0].kids for s in table.kids}
    assert strings["FileVersion"] == app.__version__
    assert _Recorder.calls["COLLECT"][1]["name"] == app.APP_NAME


def test_spec_and_helpers_are_not_git_ignored():
    """A fresh clone must contain what it takes to rebuild the bundle."""
    if shutil.which("git") is None:
        pytest.skip("git not available")
    paths = ["scripts/build_executable.spec", "scripts/build_support.py"]
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", *paths], cwd=ROOT, capture_output=True, text=True
    )
    if result.returncode == 128:
        pytest.skip(f"not a git checkout: {result.stderr.strip()}")
    assert result.stdout.split() == [], "ignored by .gitignore"


# ------------------------------------------------------------ build scripts

_STUB_OK_VERSION = '@if "%2"=="PyInstaller" if "%3"=="--version" exit /b 0\r\n@exit /b 1\r\n'
# Every call succeeds and appends its arguments to args.txt next to the stub.
_STUB_RECORD = '@>>"%~dp0args.txt" echo %*\r\n@exit /b 0\r\n'


def _run_build_script(bs, root: Path, script: str, stub_text: str, env_extra=None):
    """Run a copy of ``script`` in a throw-away tree at ``root`` with a stub interpreter."""
    _make_required_tree(bs, root)
    (root / "main.py").write_text("", encoding="utf-8")
    (root / "scripts").mkdir(exist_ok=True)
    shutil.copy2(SCRIPTS / script, root / "scripts" / script)
    stub = root / "fake_python.bat"
    stub.write_text(stub_text, encoding="ascii")
    env = {k: v for k, v in os.environ.items() if not k.startswith("SPROUTS_BUILD_")}
    env.update(SPROUTS_BUILD_PYTHON=str(stub), SPROUTS_BUILD_NO_PAUSE="1", **(env_extra or {}))
    if script.endswith(".ps1"):
        cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File"]
        cmd += [str(root / "scripts" / script), "-AllowMissingModels"]
    else:
        cmd = ["cmd", "/c", str(root / "scripts" / script), "/allow-missing-models"]
    # cwd is the throw-away copy too: nothing a script does may reach the real repository.
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        cwd=root,
        stdin=subprocess.DEVNULL,
    )


@pytest.mark.skipif(sys.platform != "win32", reason="Windows build scripts")
@pytest.mark.parametrize("script", ["build.bat", "build.ps1"])
def test_build_script_exits_nonzero_when_pyinstaller_fails(bs, tmp_path, script):
    """A failed PyInstaller run must end the script with an error, never 'success'."""
    result = _run_build_script(bs, tmp_path, script, _STUB_OK_VERSION)  # every build fails
    assert result.returncode != 0
    assert "successfully" not in result.stdout.lower()
    assert "failed" in (result.stdout + result.stderr).lower()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows build scripts")
@pytest.mark.parametrize("script", ["build.bat", "build.ps1"])
@pytest.mark.parametrize("custom", [False, True], ids=["repo-folders", "env-folders"])
def test_build_script_sends_the_output_folders_to_pyinstaller(bs, tmp_path, script, custom):
    """dist/ and build/ in the repository by default; SPROUTS_BUILD_DISTPATH and
    SPROUTS_BUILD_WORKPATH move them (a build outside the checkout, U2)."""
    root = tmp_path / "repo"
    root.mkdir()
    dist, work = (tmp_path / "out dist", tmp_path / "out work") if custom else (None, None)
    env = {"SPROUTS_BUILD_DISTPATH": str(dist), "SPROUTS_BUILD_WORKPATH": str(work)}
    result = _run_build_script(bs, root, script, _STUB_RECORD, env if custom else None)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = (root / "args.txt").read_text(encoding="ascii").splitlines()
    build = [line for line in calls if "--distpath" in line]
    assert len(build) == 1, calls
    expected_dist, expected_work = (dist, work) if custom else (root / "dist", root / "build")
    assert f'--distpath "{expected_dist}"' in build[0] or f"--distpath {expected_dist}" in build[0]
    assert f'--workpath "{expected_work}"' in build[0] or f"--workpath {expected_work}" in build[0]
    assert str(expected_dist / "SPROUTS" / "SPROUTS.exe") in result.stdout


def test_citation_is_bundled_at_the_bundle_root(bs):
    assert dict(bs.REQUIRED_DATAS)["CITATION.cff"] == "."
    resolved, _warnings = bs.resolve_datas(ROOT)
    assert any(Path(s).name == "CITATION.cff" and d == "." for s, d in resolved)


def test_licence_files_are_bundled_at_the_bundle_root(bs):
    datas = dict(bs.REQUIRED_DATAS)
    assert datas["LICENSE"] == "." and datas["THIRD_PARTY_NOTICES.md"] == "."
    resolved, _warnings = bs.resolve_datas(ROOT)
    assert {Path(s).name for s, d in resolved if d == "."} >= {"LICENSE", "THIRD_PARTY_NOTICES.md"}

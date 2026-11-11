"""The GT writers: ``write_tracing`` (``gt/<stem>.png``) and ``save_registry``
(``gt/classes.json``) through the shared atomic writer of ``app.io_utils`` (R8).

The saved bytes are pinned against the writers as they were (a science pin: the training
labels), a write is durable (one fsync of the file), goes through a hidden
``.<name>.<8 hex>.tmp`` that the stale-temp sweep knows, keeps the old file and leaves no
temp file when it fails, and a file another program holds open raises ``FileLockedError``
after the lock retry. (Moved from ``test_stale_temp_sweep.py``, ``test_gt_store.py``,
``test_gt_registry.py`` and ``test_gt_editor_store_equivalence.py``.)
"""

from __future__ import annotations

import errno
import io
import json
import os
import re

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

from app import io_utils
from app.config import GT_BACKGROUND_ID, GT_META_CHUNK, GT_UNKNOWN_COLOR
from app.gt.registry import default_registry, save_registry
from app.gt.store import (
    Tracing,
    empty_tracing,
    gt_path,
    write_tracing,
)
from app.io_utils import FileLockedError

STEM = "24EFCAM1_T001_L045_2024.07.18_111618_001_SMM"


def _sharing_violation(path) -> PermissionError:
    exc = PermissionError(errno.EACCES, "The process cannot access the file", str(path))
    exc.winerror = 32  # ERROR_SHARING_VIOLATION
    return exc


def _hex_to_rgb(color):
    return [int(color[i : i + 2], 16) for i in (1, 3, 5)]


def _pre_r8_save_registry(project_dir, registry):
    """``app.gt.registry.save_registry`` before R8, verbatim (text mode: CRLF on Windows)."""
    path = os.path.join(project_dir, "gt", "classes.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        fh.write(json.dumps(registry.to_dict(), indent=2))
    os.replace(path + ".tmp", path)
    return path


def _tracings():
    cycle = np.resize(np.array([0, 1, 2, 3, 255], np.uint8), (30, 40))
    unknown = np.zeros((12, 9), np.uint8)
    unknown[2:5, 3:7] = 7  # an id no class owns: GT_UNKNOWN_COLOR
    return {
        "classes": Tracing(cycle),
        "empty": empty_tracing(6, 7),
        "unknown_id": Tracing(unknown),
    }


def test_save_registry_bytes_are_unchanged(tmp_path):
    registry = default_registry().add(key="dead_root", name="Dead root", color="#9aa0a6")
    old = _pre_r8_save_registry(str(tmp_path / "old"), registry)
    new = save_registry(str(tmp_path / "new"), registry)
    with open(old, "rb") as a, open(new, "rb") as b:
        assert b.read() == a.read()
    assert os.listdir(os.path.dirname(new)) == ["classes.json"]


def test_gt_saves_are_durable(tmp_path, monkeypatch):
    synced = []
    # The one sync point: os.fsync, or the F_BARRIERFSYNC of macOS (X1-05).
    monkeypatch.setattr(io_utils, "_fsync", synced.append)
    monkeypatch.setattr(io_utils, "_sync_folder", lambda folder: None)  # file fsyncs only
    write_tracing(str(tmp_path / "gt" / "a.png"), empty_tracing(4, 4), default_registry())
    assert len(synced) == 1
    save_registry(str(tmp_path), default_registry())
    assert len(synced) == 2


def test_gt_writes_use_the_shared_temp_name(tmp_path, monkeypatch):
    swapped = []
    real_replace = os.replace

    def spy(src, dst):
        swapped.append((os.path.basename(src), os.path.basename(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(io_utils.os, "replace", spy)
    write_tracing(gt_path(str(tmp_path / f"{STEM}.jpg")), empty_tracing(4, 4), default_registry())
    save_registry(str(tmp_path), default_registry())

    assert [dst for _, dst in swapped] == [f"{STEM}.png", "classes.json"]
    for src, dst in swapped:  # ".<name>.<8 hex>.tmp", never "<name>.tmp"
        assert re.fullmatch(rf"\.{re.escape(dst)}\.[0-9a-f]{{8}}\.tmp", src), src
    assert sorted(os.listdir(tmp_path / "gt")) == sorted([f"{STEM}.png", "classes.json"])


@pytest.mark.parametrize("which", ["tracing", "registry"])
def test_a_gt_file_held_open_raises_file_locked_error(tmp_path, monkeypatch, which):
    registry = default_registry()
    path = str(tmp_path / "gt" / ("a.png" if which == "tracing" else "classes.json"))

    def save(ids):
        if which == "tracing":
            write_tracing(path, Tracing(ids), registry)
        else:
            save_registry(str(tmp_path), registry.add(key="k", name="K", color="#010203"))

    save(np.zeros((4, 4), np.uint8))
    with open(path, "rb") as fh:
        before = fh.read()
    attempts = []

    def locked(src, dst):
        attempts.append(dst)
        raise _sharing_violation(dst)

    monkeypatch.setattr(io_utils.os, "replace", locked)
    monkeypatch.setattr(io_utils.time, "sleep", lambda s: None)
    with pytest.raises(FileLockedError, match="open in another program"):
        save(np.ones((4, 4), np.uint8))
    assert len(attempts) == len(io_utils.REPLACE_RETRY_DELAYS) + 1  # the lock retry
    with open(path, "rb") as fh:
        assert fh.read() == before
    assert os.listdir(tmp_path / "gt") == [os.path.basename(path)]


def _reference_png(tracing, registry):
    """The bytes ``write_tracing`` wrote before R8 (verbatim, but for the ``reviewed`` list
    gone since 2026-11-01, and into memory instead of ``<path>.tmp``): the saved labels' pin."""
    ids = np.ascontiguousarray(tracing.ids, dtype=np.uint8)
    palette = [0, 0, 0] * 256
    unknown = _hex_to_rgb(GT_UNKNOWN_COLOR)
    for i in np.unique(ids).tolist():
        if i != GT_BACKGROUND_ID:
            palette[3 * i : 3 * i + 3] = unknown
    for c in registry.classes:
        palette[3 * c.id : 3 * c.id + 3] = _hex_to_rgb(c.color)
    image = Image.fromarray(ids, mode="P")
    image.putpalette(palette)
    meta = {
        "v": 1,
        "classes": {str(c.id): c.key for c in registry.classes},
    }  # the "reviewed" list it also wrote is gone (2026-11-01)
    info = PngImagePlugin.PngInfo()
    info.add_text(GT_META_CHUNK, json.dumps(meta))
    buf = io.BytesIO()
    image.save(buf, format="PNG", pnginfo=info, transparency=GT_BACKGROUND_ID)
    return buf.getvalue()


def _registry():
    return default_registry().add(key="hypha", name="Hypha", color="#3fa7ff").retire("nodule")


def _random_tracing(rng, h=41, w=57):
    values = np.array([0, 1, 2, 3, 4, 5, 9, 200, 255], np.uint8)
    ids = rng.choice(values, size=(h, w)).astype(np.uint8)
    keys = ["root", "nodule", "root_tip", "hypha", "weed", "ignore", None]
    file_classes = {}
    for v in values.tolist():
        key = keys[int(rng.integers(0, len(keys)))]
        if v and key is not None:
            file_classes[v] = key
    return Tracing(ids, file_classes=file_classes)


def _byte_cases() -> dict:
    """``name -> (tracing, registry)`` of every pinned file: three fixed tracings (classes,
    empty, an id no class owns) under a registry with an added class, and six seeded random
    ones (unknown ids, file maps, a retired class) under another."""
    added = default_registry().add(key="dead_root", name="Dead root", color="#9aa0a6")
    cases = {kind: (tracing, added) for kind, tracing in _tracings().items()}
    for seed in range(6):
        cases[f"seed_{seed}"] = (_random_tracing(np.random.default_rng(200 + seed)), _registry())
    return cases


@pytest.mark.parametrize("case", sorted(_byte_cases()))
def test_write_tracing_writes_the_reference_bytes(tmp_path, case):
    """The saved labels' bytes are those of the writer as it was before R8 (the reference:
    np.unique for the palette, Pillow's PNG), with no temp file left behind."""
    tracing, registry = _byte_cases()[case]
    path = tmp_path / "gt" / "a.png"
    write_tracing(str(path), tracing, registry)
    assert path.read_bytes() == _reference_png(tracing, registry)
    assert os.listdir(path.parent) == ["a.png"]


def test_failed_write_removes_tmp_and_keeps_old_file(tmp_path, monkeypatch):
    path = str(tmp_path / "gt" / "a.png")
    write_tracing(path, empty_tracing(8, 8), default_registry())
    before = open(path, "rb").read()

    def _boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", _boom)
    ids = np.ones((8, 8), np.uint8)
    with pytest.raises(OSError, match="disk full"):
        write_tracing(path, Tracing(ids=ids), default_registry())
    assert os.listdir(tmp_path / "gt") == ["a.png"]
    assert open(path, "rb").read() == before


def test_save_registry_failure_leaves_no_tmp_file(tmp_path, monkeypatch):
    import os

    def _boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(OSError, match="disk full"):
        save_registry(str(tmp_path), default_registry())
    assert os.listdir(tmp_path / "gt") == []

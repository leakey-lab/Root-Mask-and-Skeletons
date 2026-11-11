"""The pre-Generate scan of ``mask/`` and the overwrite policy (``app/inference/mask_origin.py``).

``mask/`` holds predictions only (WG-T7): the scan lists the folder once and opens no file
(W5-T5: it used to read every mask's PNG header), counts the stems that have a mask whatever
the file holds, limits them to the folder's images, and reports a mask named in another case
under its image's stem (O7: "Keep existing" must skip that image). Keep skips the existing
stems, Replace none. Qt-free.
"""

from __future__ import annotations

import builtins
import os

import numpy as np
import pytest
from PIL import Image

from app.inference import mask_origin
from app.inference.mask_origin import ExistingMasks, OverwritePolicy, scan_existing_masks
from app.project_layout import ProjectLayout


def _save(path: str, mode: str) -> str:
    size = (8, 6)
    if mode == "L":
        Image.fromarray(np.full((6, 8), 255, np.uint8), mode="L").save(path)
    elif mode == "RGBA":
        Image.new("RGBA", size, (255, 255, 255, 255)).save(path)
    elif mode == "RGB":
        Image.new("RGB", size, (255, 255, 255)).save(path)
    else:
        raise ValueError(mode)
    return path


def _project(tmp_path, masks: dict[str, str]) -> ProjectLayout:
    for stem in ("a", "b", "c", "d"):
        Image.new("RGB", (8, 6)).save(tmp_path / f"{stem}.jpg")
    os.makedirs(tmp_path / "mask")
    for stem, mode in masks.items():
        _save(str(tmp_path / "mask" / f"{stem}.png"), mode)
    return ProjectLayout(str(tmp_path))


def test_scan_existing_masks_counts_every_mask_whatever_its_kind(tmp_path):
    layout = _project(tmp_path, {"a": "RGBA", "b": "L", "c": "RGB"})
    found = scan_existing_masks(layout)
    assert found.stems == frozenset({"a", "b", "c"})
    assert found.count == 3 and bool(found)


def test_scan_existing_masks_limits_to_given_stems(tmp_path):
    layout = _project(tmp_path, {"a": "RGBA", "b": "L", "zz_orphan": "L"})
    found = scan_existing_masks(layout, stems={"a", "b", "c", "d"})
    assert found.stems == frozenset({"a", "b"})


def test_scan_existing_masks_without_mask_folder_is_empty(tmp_path):
    found = scan_existing_masks(ProjectLayout(str(tmp_path)))
    assert found == ExistingMasks.empty()
    assert not found and found.count == 0


def test_overwrite_policy_skip_sets(tmp_path):
    found = scan_existing_masks(_project(tmp_path, {"a": "RGBA", "b": "L"}))
    assert OverwritePolicy.KEEP_EXISTING.skip_stems(found) == frozenset({"a", "b"})
    assert OverwritePolicy.OVERWRITE_ALL.skip_stems(found) == frozenset()


def test_the_scan_opens_no_mask_file(tmp_path, monkeypatch):
    mask_dir = tmp_path / "mask"
    mask_dir.mkdir()
    for name in ("a.png", "B.PNG", "c.jpg", "notes.txt"):
        Image.new("L", (8, 6)).save(mask_dir / name, "PNG")
    (mask_dir / "a_folder.png").mkdir()  # not a file: never a mask
    opened = []

    def spy(real):
        # os.open and the builtin open (PIL, plain reads) both record what they open
        return lambda p, *a, **k: opened.append(p) or real(p, *a, **k)

    monkeypatch.setattr(os, "open", spy(os.open))
    monkeypatch.setattr(builtins, "open", spy(builtins.open))

    found = scan_existing_masks(ProjectLayout(str(tmp_path)))

    assert found.stems == frozenset({"a", "B", "c"}) and found.count == 3
    assert opened == []


@pytest.mark.parametrize("limit_to_images", [False, True])
def test_a_folder_is_counted_by_name(tmp_path, limit_to_images):
    # 30 files: the count rule is the same at any size; the test above proves no file opens.
    mask_dir = tmp_path / "mask"
    mask_dir.mkdir()
    for i in range(30):
        (mask_dir / f"T{i:05d}.png").write_bytes(b"")  # unreadable as a PNG: still a mask
    layout = ProjectLayout(str(tmp_path))
    wanted = {f"T{i:05d}" for i in range(0, 30, 3)}

    found = scan_existing_masks(layout, wanted if limit_to_images else None)

    assert found.count == (len(wanted) if limit_to_images else 30)


def test_a_mask_named_in_another_case_counts_for_its_image(tmp_path):
    """O7: ``mask/a.png`` is the mask of ``A.jpg``; "Keep existing" must skip that image."""
    mask_dir = tmp_path / "mask"
    mask_dir.mkdir()
    for stem in ("A", "B"):
        (tmp_path / f"{stem}.jpg").write_bytes(b"")
    (mask_dir / "a.png").write_bytes(b"")
    layout = ProjectLayout(str(tmp_path))

    found = scan_existing_masks(layout, layout.scan_images())

    assert found.stems == frozenset({"A"})


def test_masks_named_as_their_images_are_not_remapped(tmp_path, monkeypatch):
    """Every mask named as one of the images (nearly always): the count needs no remap."""
    mask_dir = tmp_path / "mask"
    mask_dir.mkdir()
    for i in range(300):
        (tmp_path / f"T{i:05d}.jpg").write_bytes(b"")
        if i % 3:
            (mask_dir / f"T{i:05d}.png").write_bytes(b"")
    layout = ProjectLayout(str(tmp_path))
    remaps = []
    real = mask_origin.key_to_stems
    monkeypatch.setattr(mask_origin, "key_to_stems", lambda *a: remaps.append(a) or real(*a))

    found = scan_existing_masks(layout, layout.scan_images())

    assert found.stems == frozenset(f"T{i:05d}" for i in range(300) if i % 3)
    assert remaps == []
    # A mask of no image: remapped, then left out of the count.
    (mask_dir / "orphan.png").write_bytes(b"")
    assert scan_existing_masks(layout, layout.scan_images()) == found and len(remaps) == 1

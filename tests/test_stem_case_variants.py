"""Images whose names differ only in case are one image in every stage (O7), PNG/JPEG only (O6).

Stems compare without case on every platform (``app.naming.stem_key``): ``A.png`` and ``a.jpg``
are one image, the lowest ``(extension rank, file name)`` wins, and the hidden one gets no
mask, skeleton or Library row. Before, a case-insensitive disk gave three "written" for two
output files, and *Keep existing masks* replaced a hand-corrected mask whose name differed in
case. Every reader takes PNG and JPEG only (``config.IMAGE_EXTENSIONS``): a mixed folder gave 3
Library images but 7 skeletons.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.inference import mask_inference as mi
from app.inference import skeleton_inference as si
from app.inference.mask_origin import OverwritePolicy, scan_existing_masks
from app.naming import image_stem
from app.project_layout import ProjectLayout
from tests.generation_stubs_support import _Brightness, _mean, _Skeleton
from tests.qt_wait import wait_until as _wait
from tests.window_project_support import mw  # noqa: F401 - fixture

pytestmark = pytest.mark.usefixtures("no_temp_sweep")


def _images(folder, spec):
    """``spec``: ``{file name: grey level}``; every image 64x48 RGB."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, level in spec.items():
        Image.new("RGB", (64, 48), (level, level, level)).save(folder / name)
    return str(folder)


def _pngs(folder):
    return sorted(n for n in os.listdir(folder) if n.endswith(".png"))


def _case_sensitive(folder) -> bool:
    (folder / "Probe").write_bytes(b"")
    try:
        return not (folder / "probe").exists()
    finally:
        (folder / "Probe").unlink()


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("L", (8, 6)).save(path, "PNG")
    return path


def test_case_variant_stems_give_one_image_and_one_mask_and_skeleton(tmp_path, cpu_runtime):
    # One run: the winner is decided by (extension rank, file name), not by listing order,
    # so repeating the run on the same names adds nothing.
    src = _images(tmp_path / "imgs", {"A.png": 255, "a.jpg": 0, "b.jpg": 120})
    out = str(tmp_path / "mask")
    skel = str(tmp_path / "skel")

    scan = ProjectLayout(src).scan_image_folder()
    assert sorted(scan.images) == ["A", "b"] and scan.hidden == (("a.jpg", "A.png"),)

    report = mi.generate_masks(src, out, model=_Brightness(), batch_size=8)
    assert sorted(report.written) == ["A.png", "b.jpg"] and report.total == 2
    assert len(_pngs(out)) == 2 and _mean(os.path.join(out, "A.png")) == 255

    skeletons = si.run_skeleton_generation(src, skel, model=_Skeleton(), batch_size=8)
    assert sorted(skeletons.written) == ["A.png", "b.jpg"]


@pytest.mark.parametrize("mask_name", ["A.png", "a.png"])
def test_a_manual_mask_survives_keep_existing_whatever_its_case(
    tmp_path,
    cpu_runtime,
    mask_name,
):
    src = _images(tmp_path / "imgs", {"A.png": 255, "a.jpg": 0, "b.jpg": 120})
    layout = ProjectLayout(src)
    os.makedirs(layout.mask_dir)
    # A hand-corrected mask, as the Correct editor writes it: RGBA, white on transparent.
    pixels = np.zeros((48, 64, 4), np.uint8)
    pixels[10:30, 10:30] = 255
    hand_mask = os.path.join(layout.mask_dir, mask_name)
    Image.fromarray(pixels, "RGBA").save(hand_mask)
    before = Path(hand_mask).read_bytes()

    existing = scan_existing_masks(layout, layout.scan_images())
    assert existing.stems == frozenset({"A"})

    report = mi.generate_masks(
        src,
        layout.mask_dir,
        model=_Brightness(),
        batch_size=8,
        skip_stems=OverwritePolicy.KEEP_EXISTING.skip_stems(existing),
    )
    assert report.written == ["b.jpg"] and Path(hand_mask).read_bytes() == before


def test_mixed_formats_give_the_same_inputs_to_library_masks_and_skeletons(tmp_path):
    names = ("a.png", "b.jpg", "c.JPEG", "d.tif", "e.bmp", "f.ppm", "g.TIFF")
    src = _images(tmp_path / "imgs", dict.fromkeys(names, 90))

    scan = ProjectLayout(src).scan_image_folder()
    stems = sorted(scan.images)
    assert stems == ["a", "b", "c"]
    assert sorted(os.path.splitext(n)[0] for n in mi.list_inputs(src)[0]) == stems
    assert sorted(image_stem(p) for p in si.make_dataset(src)) == stems
    assert scan.unsupported == ("d.tif", "e.bmp", "f.ppm", "g.TIFF")


def test_the_library_finds_a_mask_whose_name_differs_in_case(mw, tmp_path):  # noqa: F811 - fixture
    folder = tmp_path / "p"
    _images(folder, {"A.png": 200, "b.png": 100})
    (folder / "mask").mkdir()
    Image.new("L", (64, 48), 255).save(folder / "mask" / "a.png")

    mw.open_path(str(folder))
    assert _wait(lambda: mw.image_manager.images and mw.library_index.leaves())
    assert mw.image_manager.has_mask("A")


def test_the_overlay_lists_the_skeleton_folder_once_per_miss(tmp_path, monkeypatch):
    if not _case_sensitive(tmp_path):
        pytest.skip("needs a case-sensitive file system (runs on the ubuntu CI)")
    from app.gui.image_manager import ImageManager

    layout = ProjectLayout(_images(tmp_path / "p", {"A.png": 90}))
    os.makedirs(layout.skeletons_dir)
    manager = ImageManager()
    manager.original_folder = layout.root
    manager.on_loading_finished(layout.scan_images(), {}, {}, False)
    manager.processed_base_path = layout.skeletons_dir
    manager.current_view_mode = "Overlay"
    listings = []
    real = os.scandir
    monkeypatch.setattr(os, "scandir", lambda d: listings.append(d) or real(d))

    assert manager.get_fake_image_path("A") is None
    assert manager.get_fake_image_path("A") is None
    assert len(listings) == 1
    skeleton = _touch(os.path.join(layout.skeletons_dir, "a_fake.png"))
    manager.refresh_outputs()
    manager.processed_base_path = layout.skeletons_dir
    assert manager.get_fake_image_path("A") == skeleton

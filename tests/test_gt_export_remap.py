"""The ids and counts the dataset export writes (``app/gt/export.py``).

``remap_ids``: background 0, active classes 1..N in registry order, Ignore, retired and unknown
ids 255; the manifest's pixel counts are those of the exported ids and ``classes_present``
lists the classes with pixels. ``remap_ids`` and the counts were ``lut[ids]`` and
``np.bincount``; they are now ``cv2.LUT`` and the store's histogram (about 10x faster each,
2026-10-26): the plain versions are kept here as the reference, on random planes of every
shape. A retired class exports as Ignore and a restored one gets its id again (O24); the
manifest records the EXIF orientation tag, never applies it.
"""

from __future__ import annotations

import csv
import struct

import numpy as np
import pytest
from PIL import Image

from app.config import GT_BACKGROUND_ID, GT_IGNORE_ID
from app.gt import align_to_registry, present_keys, store
from app.gt.export import (
    ExportOptions,
    _export_stem,
    _pixel_counts,
    export_dataset,
    remap_ids,
)
from app.gt.registry import default_registry
from app.gt.store import (
    Tracing,
    gt_path,
    write_tracing,
)
from tests.gt_support import make_ids as _make_ids

_VALUES = np.array([0, 1, 2, 3, 4, 5, 9, 200, 254, 255], np.uint8)
W, H = 640, 480
NODULE_PIXELS = 59 * 15  # 885
_STEMS = ("plot_T1_L1_2024.01.01_120000", "plot_T2_L1_2024.01.01_120000")


def _reference_remap(ids, registry):
    lut = np.full(256, GT_IGNORE_ID, dtype=np.uint8)
    lut[GT_BACKGROUND_ID] = GT_BACKGROUND_ID
    for new_id, c in enumerate(registry.labelled(), start=1):
        lut[c.id] = new_id
    return lut[ids].astype(np.uint8)


def _reference_counts(remapped, registry):
    counts = np.bincount(remapped.ravel(), minlength=GT_IGNORE_ID + 1)
    row = {"px_background": int(counts[GT_BACKGROUND_ID])}
    for new_id, c in enumerate(registry.labelled(), start=1):
        row[f"px_{c.key}"] = int(counts[new_id])
    row["px_ignore"] = int(counts[GT_IGNORE_ID])
    return row


def _registries():
    base = default_registry()
    five = base.add(key="hypha", name="Hypha", color="#3fa7ff").add(
        key="dead", name="Dead", color="#111111"
    )
    return [base, five, five.retire("nodule"), base.retire("root").retire("root_tip")]


def _planes(rng):
    """Random id planes: C-ordered, strided, transposed, tiny and empty ones."""
    big = rng.choice(_VALUES, size=(97, 131))
    yield big
    yield big[::3, 1::2]  # a strided view
    yield big.T  # Fortran-ordered
    yield rng.choice(_VALUES, size=(1, 1))
    yield rng.choice(_VALUES, size=(1, 9))
    yield rng.choice(_VALUES, size=(9, 1))
    yield np.zeros((0, 5), np.uint8)
    yield np.zeros((480, 640), np.uint8)


@pytest.mark.parametrize("seed", range(6))
def test_remap_and_counts_match_the_original(seed):
    rng = np.random.default_rng(seed)
    for registry in _registries():
        for ids in _planes(rng):
            expected = _reference_remap(ids, registry)
            remapped = remap_ids(ids, registry)
            assert remapped.dtype == np.uint8 and remapped.shape == ids.shape
            assert np.array_equal(remapped, expected)
            assert not np.shares_memory(remapped, ids)
            assert _pixel_counts(remapped, registry) == _reference_counts(expected, registry)


def test_other_shapes_and_types_take_the_original_lookup():
    registry = default_registry()
    rng = np.random.default_rng(7)
    for ids in (
        rng.choice(_VALUES, size=50),  # 1-D
        rng.choice(_VALUES, size=(6, 7, 2)),  # 3-D
        rng.choice(_VALUES, size=(6, 7)).astype(np.int64),  # not uint8
    ):
        remapped = remap_ids(ids, registry)
        assert remapped.dtype == np.uint8 and remapped.shape == ids.shape
        assert np.array_equal(remapped, _reference_remap(ids, registry))


def test_counts_over_two_to_the_24_pixels_stay_exact():
    """Over 2**24 pixels cv2.calcHist's float32 counts would round: np.bincount counts them.

    The background count is odd and over 2**24 (float32 holds only even numbers there), so a
    count that went through float32 is off by one."""
    ids = np.zeros((4097, 4097), np.uint8)  # 16,785,409 px
    ids[0, :2] = 1  # root: 2 px
    registry = default_registry()
    assert ids.size > store._EXACT_HIST_MAX
    counts = _pixel_counts(remap_ids(ids, registry), registry)
    assert counts == _reference_counts(ids, registry)
    assert counts["px_background"] == 16_785_407 > 2**24 and counts["px_root"] == 2


@pytest.mark.parametrize("seed", range(4))
def test_the_manifest_row_lists_the_classes_present_as_before(seed, tmp_path):
    """``classes_present`` from the counts equals the original ``present_keys`` test, for
    files with unknown, retired and Ignore ids."""
    rng = np.random.default_rng(100 + seed)
    h, w = 41, 57
    photo = str(tmp_path / "photo.png")
    Image.new("RGB", (w, h)).save(photo)
    for i, registry in enumerate(_registries()):
        for _ in range(3):
            values = rng.choice(_VALUES, size=int(rng.integers(1, len(_VALUES))), replace=False)
            ids = rng.choice(values, size=(h, w)).astype(np.uint8)
            aligned, _warnings, _changed = align_to_registry(Tracing(ids), registry)
            folder = tmp_path / f"out{i}_{rng.integers(1 << 30)}"
            (folder / "masks").mkdir(parents=True)
            (folder / "images").mkdir()
            row = _export_stem(
                "plot_T1_L1",
                photo,
                aligned.ids,
                registry,
                str(folder / "masks"),
                str(folder / "images"),
            )
            present = present_keys(aligned, registry)
            expected = ";".join(c.key for c in registry.labelled() if c.key in present)
            assert row["classes_present"] == expected
            remapped = _reference_remap(aligned.ids, registry)
            counts = _reference_counts(remapped, registry)
            assert {k: v for k, v in row.items() if k.startswith("px_")} == counts


def test_remap_ids_defaults():
    registry = default_registry()
    ids = _make_ids((4, 4), [0, 1, 2, 3, 255])

    remapped = remap_ids(ids, registry)

    assert np.all(remapped[ids == 0] == 0)
    assert np.all(remapped[ids == 1] == 1)  # root
    assert np.all(remapped[ids == 2] == 2)  # nodule
    assert np.all(remapped[ids == 3] == 3)  # root_tip
    assert np.all(remapped[ids == 255] == GT_IGNORE_ID)


def test_remap_ids_retired_and_unknown_to_ignore():
    registry = default_registry().add(key="dead_root", name="Dead root", color="#111111")
    registry = registry.retire("dead_root")
    ids = _make_ids((4, 4), [1, 4, 9])

    remapped = remap_ids(ids, registry)

    assert np.all(remapped[ids == 1] == 1)
    assert np.all(remapped[ids == 4] == GT_IGNORE_ID)  # retired
    assert np.all(remapped[ids == 9] == GT_IGNORE_ID)  # unknown


def _ids() -> np.ndarray:
    """640 x 480 ids: a root bar, 885 nodule pixels, a root tip block and an Ignore block."""
    ids = np.zeros((H, W), np.uint8)
    ids[100:110, 50:400] = 1  # root: 10 x 350
    ids[200:215, 300:359] = 2  # nodule: 15 x 59 = 885
    ids[300:304, 100:125] = 3  # root tip: 4 x 25
    ids[400:410, 500:520] = GT_IGNORE_ID  # ignore: 10 x 20
    return ids


@pytest.fixture
def gt_project_with_nodules(tmp_path):
    """Two traced 640 x 480 photos; the second is a JPEG stored with EXIF Orientation = 6.

    Returns ``(project_dir, images)``; both photos have class labels.
    """
    project = tmp_path / "project"
    project.mkdir()
    registry = default_registry()
    images: dict[str, str] = {}
    first = project / f"{_STEMS[0]}.png"
    Image.new("RGB", (W, H), (10, 20, 30)).save(first)
    second = project / f"{_STEMS[1]}.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", (W, H), (10, 20, 30)).save(second, exif=exif.tobytes())
    for stem, path in zip(_STEMS, (first, second), strict=True):
        images[stem] = str(path)
        write_tracing(gt_path(str(path)), Tracing(_ids()), registry)
    return str(project), images


def _export(project_dir, images, registry, out_dir):
    result = export_dataset(project_dir, images, registry, ExportOptions(str(out_dir)))
    assert result.exported == 2 and not result.errors
    return result


def _manifest(out_dir) -> list[dict[str, str]]:
    with open(out_dir / "manifest.csv", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _exported_mask(out_dir, stem) -> np.ndarray:
    with Image.open(out_dir / "masks" / f"{stem}.png") as mask:
        return np.asarray(mask)


def test_export_after_restore_gives_the_class_its_own_id_again(gt_project_with_nodules, tmp_path):
    project_dir, images = gt_project_with_nodules
    retired = default_registry().retire("nodule")
    nodule = _ids() == 2

    _export(project_dir, images, retired, tmp_path / "retired")
    assert np.all(_exported_mask(tmp_path / "retired", _STEMS[0])[nodule] == GT_IGNORE_ID)

    _export(project_dir, images, retired.restore("nodule"), tmp_path / "restored")
    assert np.all(_exported_mask(tmp_path / "restored", _STEMS[0])[nodule] == 2)


def test_the_manifest_has_per_class_pixel_counts_and_exif_orientation(
    gt_project_with_nodules, tmp_path
):
    project_dir, images = gt_project_with_nodules
    _export(project_dir, images, default_registry(), tmp_path / "out")
    rows = _manifest(tmp_path / "out")
    assert list(rows[0]) == [
        "stem",
        "source_image",
        "split",
        "tube",
        "position",
        "date",
        "classes_present",
        "exif_orientation",
        "px_background",
        "px_root",
        "px_nodule",
        "px_root_tip",
        "px_ignore",
    ]
    assert int(rows[0]["px_nodule"]) == NODULE_PIXELS
    assert sum(int(rows[0][c]) for c in rows[0] if c.startswith("px_")) == 640 * 480
    assert rows[0]["exif_orientation"] == "" and rows[1]["exif_orientation"] == "6"


def test_the_counts_follow_the_exported_ids(gt_project_with_nodules, tmp_path):
    """A retired class counts as ignore, the rest move up: the counts are of the remapped ids."""
    project_dir, images = gt_project_with_nodules
    _export(project_dir, images, default_registry().retire("nodule"), tmp_path / "out")
    row = _manifest(tmp_path / "out")[0]
    assert list(row)[8:] == ["px_background", "px_root", "px_root_tip", "px_ignore"]
    assert int(row["px_root"]) == 10 * 350
    assert int(row["px_root_tip"]) == 4 * 25
    assert int(row["px_ignore"]) == NODULE_PIXELS + 10 * 20
    assert int(row["px_background"]) + 3500 + 100 + NODULE_PIXELS + 200 == 640 * 480


def test_the_exif_tag_is_recorded_and_never_applied(gt_project_with_nodules, tmp_path):
    project_dir, images = gt_project_with_nodules
    _export(project_dir, images, default_registry(), tmp_path / "out")
    exported = tmp_path / "out" / "images" / f"{_STEMS[1]}.jpg"
    with Image.open(exported) as image:
        assert image.size == (W, H)  # the stored grid
    assert _exported_mask(tmp_path / "out", _STEMS[1]).shape == (H, W)


def _corrupt_exif_jpeg(path) -> None:
    """A 640 x 480 JPEG whose APP1 (EXIF) segment is garbage."""
    Image.new("RGB", (W, H), (10, 20, 30)).save(path)
    data = path.read_bytes()
    garbage = b"Exif\x00\x00" + b"II*\x00\xff\xff\xff\x7f" + b"\xde\xad\xbe\xef" * 8
    segment = b"\xff\xe1" + (len(garbage) + 2).to_bytes(2, "big") + garbage
    path.write_bytes(data[:2] + segment + data[2:])  # right after SOI


@pytest.mark.filterwarnings("ignore:Corrupt EXIF data:UserWarning")
def test_a_corrupt_exif_block_leaves_the_column_blank_and_the_export_completes(
    gt_project_with_nodules, tmp_path, monkeypatch
):
    project_dir, images = gt_project_with_nodules
    broken = tmp_path / "project" / f"{_STEMS[1]}.jpg"
    _corrupt_exif_jpeg(broken)
    _export(project_dir, images, default_registry(), tmp_path / "out")
    assert _manifest(tmp_path / "out")[1]["exif_orientation"] == ""

    # Whatever Pillow raises while reading the tag (here a struct.error) is a blank, too.
    def boom(self):
        raise struct.error("bad EXIF")

    monkeypatch.setattr(Image.Image, "getexif", boom)
    _export(project_dir, images, default_registry(), tmp_path / "out2")
    assert [row["exif_orientation"] for row in _manifest(tmp_path / "out2")] == ["", ""]

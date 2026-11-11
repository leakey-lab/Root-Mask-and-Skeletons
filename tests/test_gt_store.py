"""
Tests for the GT palette-PNG store (``app/gt/store.py``).

Covers the write/read round trip (ids, ``file_classes``; no ``reviewed`` list is written,
and an older file's list is ignored), the
palette + transparency written into the PNG, path derivation (multi-dot
stems, non-ASCII paths), atomic writes, error handling for the wrong PNG
mode, a corrupt metadata chunk, a file Pillow refuses as too large and a broken
PNG chunk, ``align_to_registry`` (spec §1.4), and ``present_keys``.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

from app.config import GT_BACKGROUND_ID, GT_IGNORE_ID, GT_META_CHUNK
from app.gt import store
from app.gt.export import remap_ids
from app.gt.registry import Registry, default_registry
from app.gt.store import (
    GtFileError,
    Tracing,
    align_to_registry,
    empty_tracing,
    gt_path,
    histogram,
    present_keys,
    read_tracing,
    write_tracing,
)
from app.gui.correct import classes_paint
from tests.gt_goldens_support import write_legacy_tracing
from tests.gt_support import break_a_chunk, write_header_bomb

_VALUES = np.array([0, 1, 2, 3, 4, 5, 9, 200, 254, 255], np.uint8)


def _ids_with_values(shape, values):
    """A deterministic (h, w) uint8 array whose pixels cycle through ``values``."""
    h, w = shape
    flat = np.array([values[i % len(values)] for i in range(h * w)], dtype=np.uint8)
    return flat.reshape(h, w)


def test_round_trip_ids_and_chunk(tmp_path):
    ids = _ids_with_values((30, 40), [0, 1, 2, 3, 255])
    tracing = Tracing(ids=ids)
    registry = default_registry()
    path = str(tmp_path / "gt" / "a.png")

    write_tracing(path, tracing, registry)
    result = read_tracing(path)

    assert result is not None
    assert np.array_equal(result.ids, ids)
    assert result.file_classes == {1: "root", 2: "nodule", 3: "root_tip", 255: "ignore"}
    with Image.open(path) as image:  # the id map only: no "reviewed" list (2026-11-01)
        meta = json.loads(image.text[GT_META_CHUNK])
    assert meta == {
        "v": 1,
        "classes": {"1": "root", "2": "nodule", "3": "root_tip", "255": "ignore"},
    }


def test_an_older_file_with_a_reviewed_list_reads_the_same(tmp_path):
    """A file written before 2026-11-01 lists its "checked" classes in the chunk: the list is
    read past (no field keeps it) and the ids and the id map read as from a new file."""
    ids = _ids_with_values((30, 40), [0, 1, 2, 3, 255])
    registry = default_registry()
    old, new = str(tmp_path / "old.png"), str(tmp_path / "new.png")
    write_legacy_tracing(old, Tracing(ids), registry, {"root", "nodule"})
    write_tracing(new, Tracing(ids), registry)
    with Image.open(old) as image:
        assert json.loads(image.text[GT_META_CHUNK])["reviewed"] == ["nodule", "root"]

    result = read_tracing(old)

    assert np.array_equal(result.ids, ids)
    assert result.file_classes == read_tracing(new).file_classes
    assert not hasattr(result, "reviewed")


def test_reviewed_is_no_longer_a_field():
    """The second positional field was ``reviewed``: an old call fails instead of passing
    the set as the id map."""
    with pytest.raises(TypeError):
        Tracing(np.zeros((2, 2), np.uint8), frozenset({"root"}))


def test_palette_matches_registry(tmp_path):
    ids = _ids_with_values((10, 10), [0, 1, 255])
    tracing = Tracing(ids=ids)
    registry = default_registry()
    path = str(tmp_path / "gt" / "a.png")

    write_tracing(path, tracing, registry)

    with Image.open(path) as im:
        im.load()
        assert im.mode == "P"
        palette = im.getpalette()
        assert palette[3:6] == [0x56, 0xB4, 0xE9]  # root
        ignore = registry.by_key("ignore")
        ig_id = ignore.id
        expected_ignore_rgb = [int(ignore.color[i : i + 2], 16) for i in (1, 3, 5)]
        assert palette[3 * ig_id : 3 * ig_id + 3] == expected_ignore_rgb
        assert im.info["transparency"] == 0


def test_gt_path_multi_dot_and_unicode(tmp_path):
    src = os.path.join("somedir", "plötz.T1.L2.png")
    assert gt_path(src) == os.path.join("somedir", "gt", "plötz.T1.L2.png")

    project_dir = tmp_path / "ünï"
    project_dir.mkdir()
    image_path = str(project_dir / "a.b.png")
    path = gt_path(image_path)
    assert path == str(project_dir / "gt" / "a.b.png")

    ids = _ids_with_values((5, 5), [0, 1])
    registry = default_registry()
    write_tracing(path, Tracing(ids=ids), registry)
    result = read_tracing(path)
    assert result is not None
    assert np.array_equal(result.ids, ids)


def test_empty_tracing_still_written(tmp_path):
    path = str(tmp_path / "gt" / "a.png")
    write_tracing(path, empty_tracing(6, 7), default_registry())
    result = read_tracing(path)
    assert result is not None
    assert result.ids.shape == (6, 7)
    assert np.all(result.ids == 0)


def test_missing_returns_none(tmp_path):
    path = str(tmp_path / "gt" / "does_not_exist.png")
    assert read_tracing(path) is None


def test_rgb_file_raises(tmp_path):
    path = tmp_path / "rgb.png"
    Image.new("RGB", (5, 5), color=(1, 2, 3)).save(path, format="PNG")
    with pytest.raises(GtFileError):
        read_tracing(str(path))


def test_16bit_file_raises(tmp_path):
    path = tmp_path / "i16.png"
    Image.new("I;16", (5, 5)).save(path, format="PNG")
    with pytest.raises(GtFileError):
        read_tracing(str(path))


def test_bad_chunk_json_raises(tmp_path):
    path = tmp_path / "bad_chunk.png"
    ids = np.zeros((5, 5), dtype=np.uint8)
    img = Image.fromarray(ids, mode="P")
    img.putpalette([0, 0, 0] * 256)
    info = PngImagePlugin.PngInfo()
    info.add_text(GT_META_CHUNK, "{not valid json")
    img.save(path, format="PNG", pnginfo=info)
    with pytest.raises(GtFileError):
        read_tracing(str(path))


def test_a_file_pillow_refuses_as_too_large_is_unreadable(tmp_path):
    """Pillow's own error, which is neither OSError nor ValueError, is a ``GtFileError`` with
    Pillow's text (the export lists that text as before)."""
    path = tmp_path / "bomb.png"
    write_header_bomb(path)

    with pytest.raises(GtFileError) as caught:
        read_tracing(str(path))

    assert str(caught.value).startswith("Image size (200000000 pixels) exceeds limit")
    assert isinstance(caught.value.__cause__, Image.DecompressionBombError)


def test_a_broken_png_chunk_is_unreadable(tmp_path):
    path = str(tmp_path / "gt" / "a.png")
    write_tracing(path, Tracing(ids=_ids_with_values((40, 50), [0, 1, 2])), default_registry())
    break_a_chunk(path)

    with pytest.raises(GtFileError, match="broken PNG file"):
        read_tracing(path)


def test_file_without_chunk_reads_empty_meta(tmp_path):
    path = tmp_path / "no_chunk.png"
    ids = _ids_with_values((5, 5), [0, 1])
    Image.fromarray(ids, mode="L").save(path, format="PNG")
    result = read_tracing(str(path))
    assert result is not None
    assert result.file_classes == {}


def test_align_same_project_noop(tmp_path):
    ids = _ids_with_values((10, 10), [0, 1, 2, 3, 255])
    registry = default_registry()
    path = str(tmp_path / "gt" / "a.png")
    write_tracing(path, Tracing(ids=ids), registry)
    tracing = read_tracing(path)

    aligned, warnings, changed = align_to_registry(tracing, registry)

    assert changed is False
    assert warnings == []
    assert np.array_equal(aligned.ids, ids)


def test_align_unknown_key_to_ignore():
    ids = _ids_with_values((4, 4), [1, 2])
    tracing = Tracing(ids=ids, file_classes={1: "root", 2: "weed"})
    registry = default_registry()

    aligned, warnings, changed = align_to_registry(tracing, registry)

    assert changed is True
    assert len(warnings) == 1
    assert "weed" in warnings[0]
    assert set(np.unique(aligned.ids).tolist()) == {1, 255}
    assert np.all(aligned.ids[ids == 2] == 255)
    assert np.all(aligned.ids[ids == 1] == 1)


def test_align_remaps_by_key():
    ids = _ids_with_values((4, 4), [1])
    tracing = Tracing(ids=ids, file_classes={1: "nodule"})
    registry = default_registry()

    aligned, warnings, changed = align_to_registry(tracing, registry)

    assert changed is True
    assert warnings == []
    assert np.all(aligned.ids == 2)


def test_align_id_missing_from_map():
    ids = _ids_with_values((4, 4), [1, 7])
    tracing = Tracing(ids=ids, file_classes={})
    registry = default_registry()

    aligned, warnings, changed = align_to_registry(tracing, registry)

    assert changed is True
    assert len(warnings) == 1
    assert "7" in warnings[0]
    assert np.all(aligned.ids[ids == 7] == 255)
    assert np.all(aligned.ids[ids == 1] == 1)


def test_present_keys():
    ids = _ids_with_values((4, 4), [1, 255])
    tracing = Tracing(ids=ids)
    registry = default_registry()

    assert present_keys(tracing, registry) == {"root"}


def _reference_align(tracing, registry):
    warnings = []
    lut = np.arange(256, dtype=np.uint8)
    for old_id in np.unique(tracing.ids).tolist():
        if old_id == GT_BACKGROUND_ID:
            continue
        key = tracing.file_classes.get(old_id)
        if key is not None:
            try:
                lut[old_id] = registry.by_key(key).id
            except KeyError:
                lut[old_id] = GT_IGNORE_ID
                warnings.append(f"File id {old_id} has unknown key {key!r}; mapped to Ignore")
        elif registry.by_id(old_id) is not None:
            lut[old_id] = old_id
        else:
            lut[old_id] = GT_IGNORE_ID
            warnings.append(
                f"File id {old_id} has no key mapping and is unknown to the registry; "
                "mapped to Ignore"
            )
    new_ids = lut[tracing.ids]
    changed = not np.array_equal(new_ids, tracing.ids)
    return new_ids.astype(np.uint8), warnings, changed


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


@pytest.mark.parametrize("seed", range(12))
def test_align_to_registry_matches_the_original(seed):
    rng = np.random.default_rng(seed)
    registry = _registry()
    tracing = _random_tracing(rng)
    if seed % 3 == 0:
        tracing = Tracing(tracing.ids, file_classes={})  # no id -> key map
    before = tracing.ids.copy()
    expected_ids, expected_warnings, expected_changed = _reference_align(tracing, registry)
    aligned, warnings, changed = align_to_registry(tracing, registry)
    assert np.array_equal(aligned.ids, expected_ids) and aligned.ids.dtype == np.uint8
    assert warnings == expected_warnings and changed == expected_changed
    assert not np.shares_memory(aligned.ids, tracing.ids)  # a new array either way
    assert np.array_equal(tracing.ids, before)  # the input is left alone


def test_align_to_registry_of_an_aligned_tracing_changes_nothing():
    registry = default_registry()
    ids = np.zeros((30, 40), np.uint8)
    ids[5:9, 5:9] = 2
    ids[20:25, 3:30] = 255
    aligned, warnings, changed = align_to_registry(
        Tracing(ids, file_classes={2: "nodule"}), registry
    )
    assert not changed and warnings == [] and np.array_equal(aligned.ids, ids)


@pytest.mark.parametrize("seed", range(8))
def test_present_keys_matches_the_original(seed):
    rng = np.random.default_rng(100 + seed)
    registry = _registry()
    tracing = _random_tracing(rng)
    present = set(np.unique(tracing.ids).tolist())
    expected = {c.key for c in registry.classes if not c.is_ignore and c.id in present}
    assert present_keys(tracing, registry) == expected
    assert present_keys(Tracing(np.zeros((0, 0), np.uint8)), registry) == set()


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


@pytest.mark.parametrize("limit", [0, 50, 100, 1 << 24])
def test_histogram_is_exact_on_both_paths(monkeypatch, limit):
    """``histogram`` equals ``np.bincount`` on every shape (2-D, strided, transposed, 1-D,
    empty), on the calcHist path and on the bincount path (``_EXACT_HIST_MAX``), as int64.
    It is the class labels' counter too (``classes_paint.histogram``)."""
    assert classes_paint.histogram is histogram
    monkeypatch.setattr(store, "_EXACT_HIST_MAX", limit)
    rng = np.random.default_rng(limit)
    random_ids = rng.integers(0, 256, size=(301, 257)).astype(np.uint8)
    random_ids[:50] = 3
    planes = [
        *_planes(rng),
        random_ids,
        random_ids[7:200, 9:100],  # a box of the plane
        random_ids.ravel()[::3],  # a strided 1-D view
    ]
    for ids in planes:
        expected = np.bincount(ids.ravel(), minlength=256)
        counts = histogram(ids)
        assert counts.dtype == np.int64 and counts.shape == (256,)
        assert np.array_equal(counts, expected)


def test_histogram_is_exact_above_the_float32_range():
    """cv2.calcHist counts in float32: 2**24 + 3 would read 2**24 + 4 (np.bincount there)."""
    ids = np.zeros(2**24 + 8, np.uint8)
    ids[:5] = 7
    hist = classes_paint.histogram(ids)
    assert hist[0] == 2**24 + 3 and hist[7] == 5


# --- the ids at the edges -------------------------------------------------------------------
def test_a_class_with_id_254_survives_write_read_align_and_export(tmp_path):
    data = default_registry().to_dict()
    data["classes"].insert(-1, {"id": 254, "key": "edge", "name": "Edge", "color": "#3fa7ff"})
    registry = Registry.from_dict(data)
    ids = np.array([[0, 254, 255, 1]], np.uint8)
    path = str(tmp_path / "gt" / "a.png")

    write_tracing(path, Tracing(ids=ids), registry)
    tracing = read_tracing(path)
    aligned, warnings, changed = align_to_registry(tracing, registry)

    assert tracing.file_classes[254] == "edge" and warnings == [] and not changed
    assert aligned.ids.tolist() == [[0, 254, 255, 1]]
    assert remap_ids(aligned.ids, registry).tolist() == [[0, 4, 255, 1]]  # edge is the 4th


def test_id_0_stays_background_whatever_the_file_map_says():
    ids = np.array([[0, 0, 1]], np.uint8)
    tracing = Tracing(ids=ids, file_classes={0: "nodule", 1: "root"})

    aligned, warnings, changed = align_to_registry(tracing, default_registry())

    assert aligned.ids.tolist() == [[0, 0, 1]] and warnings == [] and not changed


def test_a_3d_array_is_refused_and_nothing_is_written(tmp_path):
    path = tmp_path / "gt" / "a.png"
    with pytest.raises(GtFileError, match="2-D"):
        write_tracing(str(path), Tracing(ids=np.zeros((4, 4, 3), np.uint8)), default_registry())
    assert not path.exists()

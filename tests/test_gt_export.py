"""
Tests for the dataset exporter (``app/gt/export.py``).

Covers ``remap_ids`` (background/active/ignore/retired/unknown mapping),
``assign_splits`` (determinism, tube grouping, non-empty splits, val_fraction
0.0), and ``export_dataset`` (every image with a GT file exported, output layout,
``classes.json``/``manifest.csv`` contents, non-empty ``out_dir`` guard, a
corrupt GT file landing in ``errors`` without aborting the run, and source
files being left untouched).
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

import app
from app.config import GT_IGNORE_ID, GT_META_CHUNK
from app.gt.export import ExportError, ExportOptions, assign_splits, export_dataset
from app.gt.registry import default_registry
from app.gt.store import Tracing, gt_path, write_tracing
from app.gui.export_dataset_dialog import ExportDatasetDialog
from tests.gt_support import STEMS as _STEMS
from tests.gt_support import build_project as _build_project
from tests.qt_teardown import delete_widgets

windows_only = pytest.mark.skipif(os.name != "nt", reason="drive letters are Windows paths")


# --------------------------- remap_ids --------------------------------------


# --------------------------- assign_splits -----------------------------------


def test_assign_splits_deterministic_same_seed():
    a = assign_splits(list(_STEMS), 0.34, True, seed=7)
    b = assign_splits(list(_STEMS), 0.34, True, seed=7)
    assert a == b


def test_assign_splits_different_seed_can_differ():
    results = {seed: assign_splits(list(_STEMS), 0.5, True, seed=seed) for seed in range(8)}
    distinct = {tuple(sorted(r.items())) for r in results.values()}
    assert len(distinct) > 1, "expected at least two seeds to disagree"


def test_assign_splits_group_by_tube_never_splits_a_tube():
    splits = assign_splits(list(_STEMS), 0.5, True, seed=3)
    for tube in ("T1", "T2", "T3"):
        tube_splits = {splits[s] for s in _STEMS if f"_{tube}_" in s}
        assert len(tube_splits) == 1


def test_assign_splits_both_non_empty_with_two_or_more_groups():
    splits = assign_splits(list(_STEMS), 0.5, True, seed=1)
    values = set(splits.values())
    assert "train" in values
    assert "val" in values


def test_assign_splits_zero_val_fraction_is_all_train():
    splits = assign_splits(list(_STEMS), 0.0, True, seed=0)
    assert set(splits.values()) == {"train"}


def test_assign_splits_groups_by_the_tube_token_not_a_project_prefix():
    """``PLOT12_T001`` is tube 1, not 12: two tubes sharing a prefix are two groups."""
    stems = ["PLOT12_T001_L001_2024.07.18_111618", "PLOT12_T002_L001_2024.07.18_111618"]
    splits = assign_splits(stems, 0.5, True, seed=0)
    assert sorted(splits.values()) == ["train", "val"]


# --------------------------- export_dataset ----------------------------------


def _add_old_reviewed_list(path, reviewed):
    """Re-save the GT file at ``path`` as a version before 2026-11-01 wrote it: its metadata
    chunk also holds a ``reviewed`` list (the old "checked" boxes)."""
    with Image.open(path) as image:
        image.load()
        meta = json.loads(image.text[GT_META_CHUNK])
        old = image.copy()
    meta["reviewed"] = sorted(reviewed)
    info = PngImagePlugin.PngInfo()
    info.add_text(GT_META_CHUNK, json.dumps(meta))
    old.save(path, format="PNG", pnginfo=info, transparency=0)


def test_export_includes_an_old_file_with_an_incomplete_reviewed_list(tmp_path):
    """Every image with a GT file is exported: the ``reviewed`` list of an older file
    (here not every class) no longer filters anything."""
    project_dir, images, _write_reg, export_registry = _build_project(tmp_path)
    old_file = gt_path(images[_STEMS[0]])
    _add_old_reviewed_list(old_file, {"root"})
    with Image.open(old_file) as image:
        assert json.loads(image.text[GT_META_CHUNK])["reviewed"] == ["root"]

    result = export_dataset(
        project_dir, images, export_registry, ExportOptions(out_dir=str(tmp_path / "out"))
    )

    assert result.exported == 6 and not result.errors
    assert (tmp_path / "out" / "masks" / f"{_STEMS[0]}.png").exists()


def test_export_output_layout(tmp_path):
    project_dir, images, _write_reg, export_registry = _build_project(
        tmp_path, dead_root_stem=_STEMS[1]
    )
    out_dir = tmp_path / "out"
    options = ExportOptions(out_dir=str(out_dir), val_fraction=0.34, seed=2)

    result = export_dataset(project_dir, images, export_registry, options)

    exported_stems = set(_STEMS)  # every image with a GT file is exported
    assert result.exported == 6

    for stem in exported_stems:
        image_out = out_dir / "images" / f"{stem}.png"
        assert image_out.exists()
        assert image_out.read_bytes() == (tmp_path / "project" / f"{stem}.png").read_bytes()

        mask_out = out_dir / "masks" / f"{stem}.png"
        assert mask_out.exists()
        with Image.open(mask_out) as im:
            assert im.mode == "L"

    with Image.open(out_dir / "masks" / f"{_STEMS[1]}.png") as im:
        remapped_vals = set(np.array(im).tolist()[0])
    assert GT_IGNORE_ID in remapped_vals  # the dead_root (id 4) pixels

    with open(out_dir / "classes.json", encoding="utf-8") as fh:
        classes_json = json.load(fh)
    assert classes_json["ignore_index"] == GT_IGNORE_ID
    ids_in_json = sorted(c["id"] for c in classes_json["classes"])
    assert ids_in_json == [0, 1, 2, 3]

    with open(out_dir / "splits" / "train.txt", encoding="utf-8") as fh:
        train_stems = {line.strip() for line in fh if line.strip()}
    with open(out_dir / "splits" / "val.txt", encoding="utf-8") as fh:
        val_stems = {line.strip() for line in fh if line.strip()}
    assert train_stems | val_stems == exported_stems
    assert train_stems & val_stems == set()

    with open(out_dir / "manifest.csv", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows
    assert list(rows[0].keys()) == [
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
    assert {row["stem"] for row in rows} == exported_stems
    for row in rows:
        assert row["split"] in ("train", "val")


def test_export_manifest_reads_tube_and_position_tokens(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    stem = "CAL1_T014_L011_2024.07.18_111618_001_SMM"  # "CAL1" is not position 1
    image = project / f"{stem}.png"
    Image.new("RGB", (8, 8)).save(image)
    registry = default_registry()
    write_tracing(gt_path(image), Tracing(ids=np.zeros((8, 8), np.uint8)), registry)

    export_dataset(str(project), {stem: str(image)}, registry, ExportOptions(str(tmp_path / "out")))

    with open(tmp_path / "out" / "manifest.csv", encoding="utf-8", newline="") as fh:
        (row,) = list(csv.DictReader(fh))
    assert (row["tube"], row["position"], row["date"]) == ("14", "11", "2024.07.18")


# --------------------------- provenance sidecar --------------------------------


def _strings(value):
    """Every string inside a parsed JSON value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def test_export_records_its_options_counts_and_registry_next_to_the_manifest(tmp_path):
    project_dir, images, _write_reg, export_registry = _build_project(tmp_path)
    out_dir = tmp_path / "out"
    options = ExportOptions(out_dir=str(out_dir), val_fraction=0.34, seed=7)

    result = export_dataset(project_dir, images, export_registry, options)

    sidecar = out_dir / "manifest.csv.provenance.json"
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert record["kind"] == "gt_export"
    assert record["app"]["version"] == app.__version__
    assert record["parameters"] == {
        "val_fraction": 0.34,
        "group_by_tube": True,
        "seed": 7,
    }
    assert record["outputs"] == {
        "exported": 6,
        "train": result.train,
        "val": result.val,
        "errors": 0,
    }
    assert record["inputs"]["images"] == 6
    assert record["inputs"]["registry"] == export_registry.to_dict()  # ids before the remap
    assert record["models"] == {}  # ground truth never comes from a model
    assert not [s for s in _strings(record) if str(tmp_path) in s], "local paths leaked"


def test_export_still_succeeds_when_the_provenance_file_cannot_be_written(tmp_path, monkeypatch):
    project_dir, images, _write_reg, export_registry = _build_project(tmp_path)

    def _disk_full(output_path, record):
        raise OSError("disk full")

    monkeypatch.setattr("app.gt.export.write_sidecar", _disk_full)
    result = export_dataset(
        project_dir, images, export_registry, ExportOptions(out_dir=str(tmp_path / "out"))
    )
    assert result.exported == 6
    assert (tmp_path / "out" / "manifest.csv").exists()


def test_export_non_empty_out_dir_raises(tmp_path):
    project_dir, images, _write_reg, export_registry = _build_project(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "existing.txt").write_text("x")

    options = ExportOptions(out_dir=str(out_dir))
    with pytest.raises(ExportError):
        export_dataset(project_dir, images, export_registry, options)


@pytest.mark.parametrize(
    "sub",
    [
        "gt",
        os.path.join("gt", "export"),
        os.path.join("mask", "export"),
        os.path.join("skeletons", "export"),
        os.path.join("gt", "skeletons", "deep", "export"),
    ],
)
def test_export_refuses_out_dir_inside_project_data_folders(tmp_path, sub):
    project_dir, images, _write_reg, export_registry = _build_project(tmp_path)
    out_dir = os.path.join(project_dir, sub)
    gt_before = sorted(os.listdir(os.path.join(project_dir, "gt")))

    with pytest.raises(ExportError, match="inside"):
        export_dataset(project_dir, images, export_registry, ExportOptions(out_dir=out_dir))
    if sub != "gt":
        assert not os.path.exists(out_dir)
    assert sorted(os.listdir(os.path.join(project_dir, "gt"))) == gt_before


def test_export_allows_sibling_folder_with_similar_name(tmp_path):
    """``gt_export`` next to ``gt/`` is not inside it (no prefix matching)."""
    project_dir, images, _write_reg, export_registry = _build_project(tmp_path)
    out_dir = os.path.join(project_dir, "gt_export")
    result = export_dataset(project_dir, images, export_registry, ExportOptions(out_dir=out_dir))
    assert result.exported == 6


def test_progress_reports_each_percent_once(tmp_path):
    """One call per new percent, from 0 at the first image to 100 at the end, not one per
    image (46k queued signals took about a tenth of a large export)."""
    project_dir, images, _write_reg, export_registry = _build_project(tmp_path)
    for i in range(300):  # images without labels count too
        images[f"plot_T9_L{i:03d}_2024.01.01_120000"] = os.path.join(project_dir, f"u{i}.png")
    values: list[int] = []

    export_dataset(
        project_dir, images, export_registry, ExportOptions(str(tmp_path / "out")), values.append
    )

    n = len(images)
    assert values == sorted({int(100 * i / n) for i in range(n)}) + [100]
    assert values[:2] == [0, 1] and len(values) == 101

    empty: list[int] = []
    export_dataset(
        project_dir, {}, export_registry, ExportOptions(str(tmp_path / "e")), empty.append
    )
    assert empty == [100]


def test_export_missing_out_dir_is_created(tmp_path):
    project_dir, images, _write_reg, export_registry = _build_project(tmp_path)
    out_dir = tmp_path / "nested" / "out"

    options = ExportOptions(out_dir=str(out_dir))
    export_dataset(project_dir, images, export_registry, options)

    assert out_dir.exists()


def _drive(path: str) -> str:
    """``"C:"`` on Windows, ``""`` elsewhere."""
    return os.path.splitdrive(path)[0]


def test_a_drive_relative_folder_is_resolved_once_when_the_export_starts(tmp_path, monkeypatch):
    project_dir, images, _write_registry, registry = _build_project(tmp_path)
    start, later = tmp_path / "start", tmp_path / "later"
    start.mkdir()
    later.mkdir()
    monkeypatch.chdir(start)
    moved: list[str] = []

    def progress(_percent: int) -> None:
        if not moved:  # the working folder changes while the export runs
            moved.append(str(later))
            os.chdir(later)

    options = ExportOptions(out_dir=f"{_drive(str(tmp_path))}out")
    result = export_dataset(project_dir, images, registry, options, progress)

    out = start / "out"
    assert moved and os.path.isabs(result.out_dir) and os.path.samefile(result.out_dir, out)
    assert result.errors == [] and result.exported == len(_STEMS)
    assert len(os.listdir(out / "images")) == len(os.listdir(out / "masks")) == len(_STEMS)
    for name in ("splits/train.txt", "splits/val.txt", "classes.json", "manifest.csv"):
        assert (out / name).is_file()
    assert (out / "manifest.csv.provenance.json").is_file()
    assert not (later / "out").exists()


def test_a_drive_relative_folder_inside_the_project_data_is_refused(tmp_path, monkeypatch):
    project_dir, images, _write_registry, registry = _build_project(tmp_path)
    monkeypatch.chdir(project_dir)
    options = ExportOptions(out_dir=f"{_drive(project_dir)}mask")

    with pytest.raises(ExportError, match="mask/") as caught:
        export_dataset(project_dir, images, registry, options)

    assert str(caught.value).endswith(os.path.join(project_dir, "mask"))  # named in full
    assert not os.path.exists(os.path.join(project_dir, "mask"))


@windows_only
def test_a_folder_on_another_drive_is_named_in_full(tmp_path, monkeypatch):
    """``D:`` with the project and the working folder on ``C:`` is drive D's current
    folder. It is not empty, so nothing is written; the refusal names it in full."""
    project_dir, images, _write_registry, registry = _build_project(tmp_path)
    other = _drive(os.getcwd())
    if not other or other.upper() == _drive(project_dir).upper():
        pytest.skip("the tests do not run from a drive other than the temporary folder's")
    monkeypatch.chdir(tmp_path)
    folder = os.path.abspath(other)
    if not os.path.isdir(folder) or not os.listdir(folder):
        pytest.skip(f"{folder} is not a folder with something in it")

    with pytest.raises(ExportError) as caught:
        export_dataset(project_dir, images, registry, ExportOptions(out_dir=other))

    assert os.path.isabs(folder) and _drive(folder).upper() == other.upper()
    assert str(caught.value) == f"Output directory is not empty: {folder}"


@windows_only
def test_the_dialog_spells_out_a_folder_on_another_drive(app, tmp_path):
    project_dir, images, _write_registry, registry = _build_project(tmp_path)
    other = "E:" if _drive(project_dir).upper() == "D:" else "D:"
    dialog = ExportDatasetDialog(project_dir, images, registry)
    try:
        dialog.output_edit.setText(f"{other}out")

        folder = dialog._output_folder()  # resolved by name only: nothing is written

        assert os.path.isabs(folder) and _drive(folder).upper() == other
        assert os.path.basename(folder) == "out" and folder == os.path.abspath(f"{other}out")
    finally:
        delete_widgets([dialog])


# --------------------------- the edges of the request ----------------------------------------
def test_train_never_empties_at_the_highest_validation_share():
    """At the dialog's highest share (90 %) the last tube group always stays in train: with
    2-5 groups of 1-3 images and any seed, train has a group, val too; one group is all train."""
    for seed in range(300):
        rng = np.random.default_rng(seed)
        groups = int(rng.integers(2, 6))
        stems = [
            f"plot_T{t}_L{i}_2024.01.01_120000"
            for t in range(1, groups + 1)
            for i in range(int(rng.integers(1, 4)))
        ]
        splits = assign_splits(stems, 0.9, True, seed)
        assert sorted(splits) == sorted(stems), f"seed {seed}"
        assert "train" in splits.values() and "val" in splits.values(), f"seed {seed}: {splits}"
    one_tube = [f"plot_T1_L{i}_2024.01.01_120000" for i in range(4)]
    assert set(assign_splits(one_tube, 0.9, True, 0).values()) == {"train"}


def test_an_out_folder_that_is_a_file_is_refused(tmp_path):
    project_dir, images, _write_reg, export_registry = _build_project(tmp_path)
    taken = tmp_path / "dataset"
    taken.write_text("a file, not a folder")

    with pytest.raises(ExportError, match="exists and is not a directory"):
        export_dataset(project_dir, images, export_registry, ExportOptions(out_dir=str(taken)))

    assert taken.read_text() == "a file, not a folder"


def test_an_existing_empty_out_folder_is_used(tmp_path):
    project_dir, images, _write_reg, export_registry = _build_project(tmp_path)
    out_dir = tmp_path / "dataset"
    out_dir.mkdir()

    result = export_dataset(project_dir, images, export_registry, ExportOptions(str(out_dir)))

    assert result.exported == 6 and len(os.listdir(out_dir / "masks")) == 6


def test_labels_holding_only_ignore_are_exported_as_ignore(tmp_path):
    """An image marked all Ignore is exported: its mask is all 255, it counts only Ignore
    pixels and lists no class."""
    project = tmp_path / "project"
    project.mkdir()
    stem = "plot_T1_L1_2024.01.01_120000"
    image = project / f"{stem}.png"
    Image.new("RGB", (8, 6)).save(image)
    registry = default_registry()
    write_tracing(gt_path(image), Tracing(np.full((6, 8), 255, np.uint8)), registry)

    result = export_dataset(
        str(project), {stem: str(image)}, registry, ExportOptions(str(tmp_path / "out"))
    )

    assert (result.exported, result.errors) == (1, [])
    with Image.open(tmp_path / "out" / "masks" / f"{stem}.png") as mask:
        assert (np.array(mask) == 255).all()
    with open(tmp_path / "out" / "manifest.csv", encoding="utf-8", newline="") as fh:
        (row,) = list(csv.DictReader(fh))
    assert (row["px_ignore"], row["px_background"], row["classes_present"]) == ("48", "0", "")

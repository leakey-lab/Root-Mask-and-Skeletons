"""What the dataset export does when a stem or a file fails (``app/gt/export.py``, spec 5.4).

One bad stem is that image's error and the export carries on: unreadable labels (a corrupt or
broken PNG), a missing source image, labels of another size than the image (R7: a pair a
training pipeline would misalign), an image or labels Pillow refuses as too large (the tests
lower ``Image.MAX_IMAGE_PIXELS`` instead of writing 179 Mpx). Nothing of a failed stem is left
in ``images/``, ``masks/``, the splits or the manifest; ``errors.txt`` lists each one on one
line. The text files go through the shared atomic writer (R1): complete or absent, a file held
open elsewhere raises ``FileLockedError``.
"""

from __future__ import annotations

import csv
import errno
import os
import re

import numpy as np
import pytest
from PIL import Image

from app import io_utils
from app.gt.export import (
    ExportOptions,
    export_dataset,
)
from app.gt.store import (
    Tracing,
    gt_path,
    write_tracing,
)
from app.io_utils import FileLockedError
from tests.gt_support import STEMS as _STEMS
from tests.gt_support import break_a_chunk
from tests.gt_support import build_project as _build_project

# 20 x 20 px (the project's images and tracings) pass; 40 x 30 px is over twice the limit.
_LIMIT = 500
_BIG = (40, 30)
_TEXT_FILES = ("splits/train.txt", "splits/val.txt", "classes.json", "manifest.csv", "errors.txt")


def test_export_writes_errors_txt_only_when_there_are_errors(tmp_path):
    project_dir, images, _write_reg, export_registry = _build_project(tmp_path)
    clean_out = tmp_path / "clean"
    export_dataset(project_dir, images, export_registry, ExportOptions(out_dir=str(clean_out)))
    assert not (clean_out / "errors.txt").exists()

    corrupt_stem = _STEMS[2]
    with open(gt_path(images[corrupt_stem]), "wb") as fh:
        fh.write(b"not a png")
    out_dir = tmp_path / "out"
    result = export_dataset(
        project_dir, images, export_registry, ExportOptions(out_dir=str(out_dir))
    )
    assert [stem for stem, _ in result.errors] == [corrupt_stem]
    lines = (out_dir / "errors.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    stem, message = lines[0].split("\t", 1)
    assert stem == corrupt_stem and message == " ".join(result.errors[0][1].split())


def test_export_corrupt_gt_file_lands_in_errors_others_continue(tmp_path):
    project_dir, images, _write_reg, export_registry = _build_project(tmp_path)
    corrupt_stem = _STEMS[0]
    corrupt_path = gt_path(images[corrupt_stem])
    original_contents = {stem: open(gt_path(images[stem]), "rb").read() for stem in _STEMS}
    with open(corrupt_path, "wb") as fh:
        fh.write(b"not a png")
    corrupted_contents = open(corrupt_path, "rb").read()

    options = ExportOptions(out_dir=str(tmp_path / "out"))
    result = export_dataset(project_dir, images, export_registry, options)

    assert len(result.errors) == 1
    assert result.errors[0][0] == corrupt_stem
    assert result.exported == 5

    # Source GT files (including the corrupted one) are left untouched.
    for stem in _STEMS:
        current = open(gt_path(images[stem]), "rb").read()
        if stem == corrupt_stem:
            assert current == corrupted_contents
        else:
            assert current == original_contents[stem]


def test_export_missing_source_image_lands_in_errors_others_continue(tmp_path):
    """A GT-traced image whose source file went missing must not abort the run.

    remap_ids/mask-save only depends on the GT tracing, so the mask for the
    missing-source stem would be written before ``shutil.copy2`` raises;
    that partial mask must not survive, and the stem must not appear in the
    split files or the manifest.
    """
    project_dir, images, _write_reg, export_registry = _build_project(tmp_path)
    missing_stem = _STEMS[0]
    os.remove(images[missing_stem])  # source image gone; its GT file remains

    options = ExportOptions(out_dir=str(tmp_path / "out"))
    result = export_dataset(project_dir, images, export_registry, options)

    assert len(result.errors) == 1
    assert result.errors[0][0] == missing_stem
    assert result.exported == 5
    assert result.train + result.val == 5

    out_dir = tmp_path / "out"
    assert not (out_dir / "images" / f"{missing_stem}.png").exists()
    assert not (out_dir / "masks" / f"{missing_stem}.png").exists()

    with open(out_dir / "splits" / "train.txt", encoding="utf-8") as fh:
        train_stems = {line.strip() for line in fh if line.strip()}
    with open(out_dir / "splits" / "val.txt", encoding="utf-8") as fh:
        val_stems = {line.strip() for line in fh if line.strip()}
    assert missing_stem not in train_stems
    assert missing_stem not in val_stems
    assert len(train_stems | val_stems) == 5

    with open(out_dir / "manifest.csv", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert missing_stem not in {row["stem"] for row in rows}
    assert len(rows) == 5

    # Other images still exported normally.
    other_stem = _STEMS[1]
    assert (out_dir / "images" / f"{other_stem}.png").exists()
    assert (out_dir / "masks" / f"{other_stem}.png").exists()


def test_a_broken_tracing_is_that_images_export_error(tmp_path):
    project_dir, images, _write_registry, registry = _build_project(tmp_path)
    broken = _STEMS[3]
    break_a_chunk(gt_path(images[broken]))
    out_dir = tmp_path / "out"

    result = export_dataset(project_dir, images, registry, ExportOptions(str(out_dir)))

    assert [stem for stem, _ in result.errors] == [broken]
    assert "broken PNG file" in result.errors[0][1]
    assert result.exported == len(_STEMS) - 1
    assert not (out_dir / "masks" / f"{broken}.png").exists()
    assert (out_dir / "errors.txt").read_text(encoding="utf-8").startswith(f"{broken}\t")


def test_a_tracing_of_another_size_is_an_export_error(tmp_path):
    project_dir, images, _write_registry, registry = _build_project(tmp_path)
    resized = _STEMS[1]
    Image.new("RGB", (40, 30), color=(10, 20, 30)).save(images[resized])  # GT is 20 x 20
    out_dir = tmp_path / "out"

    result = export_dataset(project_dir, images, registry, ExportOptions(out_dir=str(out_dir)))

    assert [stem for stem, _ in result.errors] == [resized]
    message = result.errors[0][1]
    assert "20x20" in message and "40x30" in message
    assert result.exported == len(_STEMS) - 1
    assert not (out_dir / "images" / f"{resized}.png").exists()
    assert not (out_dir / "masks" / f"{resized}.png").exists()
    with open(out_dir / "manifest.csv", encoding="utf-8", newline="") as fh:
        assert resized not in {row["stem"] for row in csv.DictReader(fh)}
    assert resized in (out_dir / "errors.txt").read_text(encoding="utf-8")


def test_every_exported_image_matches_its_mask(tmp_path):
    project_dir, images, _write_registry, registry = _build_project(tmp_path)
    out_dir = tmp_path / "out"

    result = export_dataset(project_dir, images, registry, ExportOptions(out_dir=str(out_dir)))

    assert result.errors == [] and result.exported == len(_STEMS)
    for stem in _STEMS:
        with (
            Image.open(out_dir / "images" / f"{stem}.png") as im,
            Image.open(out_dir / "masks" / f"{stem}.png") as mask,
        ):
            assert im.size == mask.size


@pytest.fixture
def project(tmp_path):
    return _build_project(tmp_path)


def _export(project, tmp_path):
    project_dir, images, _write_registry, registry = project
    out_dir = tmp_path / "out"
    options = ExportOptions(out_dir=str(out_dir))
    return out_dir, export_dataset(project_dir, images, registry, options)


def _assert_only_error(out_dir, result, stem, ext=".png"):
    assert [s for s, _ in result.errors] == [stem]
    assert "exceeds limit" in result.errors[0][1]
    assert result.exported == len(_STEMS) - 1
    assert result.train + result.val == len(_STEMS) - 1
    assert not (out_dir / "images" / f"{stem}{ext}").exists()
    assert not (out_dir / "masks" / f"{stem}.png").exists()
    with open(out_dir / "manifest.csv", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert stem not in {row["stem"] for row in rows} and len(rows) == len(_STEMS) - 1
    assert (out_dir / "errors.txt").read_text(encoding="utf-8").startswith(f"{stem}\t")
    assert (out_dir / "manifest.csv.provenance.json").exists()


def test_a_source_image_pillow_refuses_is_that_images_error(project, tmp_path, monkeypatch):
    _project_dir, images, _write_registry, _registry = project
    big = _STEMS[2]
    # Pillow refuses the image; its 20 x 20 tracing (read first) is read as usual.
    Image.new("RGB", _BIG, color=(10, 20, 30)).save(images[big])
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", _LIMIT)

    out_dir, result = _export(project, tmp_path)

    _assert_only_error(out_dir, result, big)


def test_a_tracing_pillow_refuses_is_that_images_error(project, tmp_path, monkeypatch):
    _project_dir, images, write_registry, _registry = project
    big = _STEMS[4]
    width, height = _BIG
    Image.new("RGB", _BIG, color=(10, 20, 30)).save(images[big])
    ids = np.zeros((height, width), np.uint8)
    write_tracing(gt_path(images[big]), Tracing(ids=ids), write_registry)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", _LIMIT)

    out_dir, result = _export(project, tmp_path)

    _assert_only_error(out_dir, result, big)


def test_the_other_images_export_as_without_it(project, tmp_path, monkeypatch):
    """The images that pass are exported exactly as with Pillow's default limit."""
    _project_dir, images, _write_registry, _registry = project
    reference_dir, reference = _export(project, tmp_path / "reference")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", _LIMIT)
    Image.new("RGB", _BIG, color=(10, 20, 30)).save(images[_STEMS[2]])

    out_dir, result = _export(project, tmp_path)

    assert reference.errors == [] and [s for s, _ in result.errors] == [_STEMS[2]]
    for stem in _STEMS:
        if stem == _STEMS[2]:
            continue
        for sub in ("images", "masks"):
            assert (out_dir / sub / f"{stem}.png").read_bytes() == (
                reference_dir / sub / f"{stem}.png"
            ).read_bytes()


def _export_with_an_error(tmp_path, out_name="out"):
    """Export six stems, one of them with a corrupt GT file (so errors.txt exists)."""
    project_dir, images, _write_registry, registry = _build_project(tmp_path)
    with open(gt_path(images[_STEMS[2]]), "wb") as fh:
        fh.write(b"not a png\r\nsecond line")  # the message must stay one line
    out_dir = tmp_path / out_name
    result = export_dataset(project_dir, images, registry, ExportOptions(out_dir=str(out_dir)))
    return out_dir, result, registry


#: ``classes.json`` of the six-stem project (``dead_root`` retired), as ``json.dumps(indent=2)``
#: writes it: background, then the labelled classes 1..N in registry order, Ignore as 255.
EXPECTED_CLASSES_JSON = """{
  "version": 1,
  "ignore_index": 255,
  "classes": [
    {
      "id": 0,
      "key": "background",
      "name": "Background"
    },
    {
      "id": 1,
      "key": "root",
      "name": "Root",
      "color": "#56b4e9"
    },
    {
      "id": 2,
      "key": "nodule",
      "name": "Nodule",
      "color": "#ff7eb6"
    },
    {
      "id": 3,
      "key": "root_tip",
      "name": "Root tip",
      "color": "#ffd166"
    }
  ]
}"""
#: The split of seed 0 at a 20 % share (one tube group, T1, to val), one stem per line.
EXPECTED_TRAIN = [_STEMS[3], _STEMS[4], _STEMS[5]]
EXPECTED_VAL = [_STEMS[0], _STEMS[1]]
MANIFEST_HEADER = (
    "stem,source_image,split,tube,position,date,classes_present,exif_orientation,"
    "px_background,px_root,px_nodule,px_root_tip,px_ignore"
)


def _text_bytes(text: str) -> bytes:
    """``text`` as the export writes it in text mode (``os.linesep`` line ends, as before R1)."""
    return text.replace("\n", os.linesep).encode("utf-8")


def test_the_exported_text_files_hold_the_expected_bytes(tmp_path):
    """The split files, ``classes.json``, ``errors.txt`` and the manifest's header, written out
    by hand (they were compared with a copy of the pre-R1 writers built on the export's own
    helpers, so a wrong class name passed). Line ends: text mode for all but the manifest,
    whose rows end in CRLF (``csv``)."""
    out_dir, result, _registry = _export_with_an_error(tmp_path)

    assert (out_dir / "classes.json").read_bytes() == _text_bytes(EXPECTED_CLASSES_JSON)
    train = "".join(stem + "\n" for stem in EXPECTED_TRAIN)
    val = "".join(stem + "\n" for stem in EXPECTED_VAL)
    assert (out_dir / "splits" / "train.txt").read_bytes() == _text_bytes(train)
    assert (out_dir / "splits" / "val.txt").read_bytes() == _text_bytes(val)
    [(stem, message)] = result.errors
    assert stem == _STEMS[2] and message.startswith("Could not read GT file ")
    line = f"{stem}\t{' '.join(message.split())}\n"  # one line, whatever the message holds
    assert (out_dir / "errors.txt").read_bytes() == _text_bytes(line)
    manifest = (out_dir / "manifest.csv").read_bytes().decode("utf-8")
    rows = manifest.split("\r\n")
    assert rows[0] == MANIFEST_HEADER and rows[-1] == ""
    assert [row.split(",")[0] for row in rows[1:-1]] == [s for s in _STEMS if s != _STEMS[2]]


def test_the_text_files_are_swapped_in_through_the_shared_writer(tmp_path, monkeypatch):
    swapped = []
    real_replace = os.replace

    def spy(src, dst):
        swapped.append((os.path.basename(src), os.path.basename(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(io_utils.os, "replace", spy)
    out_dir, _result, _registry = _export_with_an_error(tmp_path)

    targets = [dst for _src, dst in swapped]
    for rel in _TEXT_FILES:
        assert os.path.basename(rel) in targets, rel
    for src, dst in swapped:  # ".<name>.<8 hex>.tmp": the name the stale-file sweep knows
        assert re.fullmatch(rf"\.{re.escape(dst)}\.[0-9a-f]{{8}}\.tmp", src), src
    leftovers = [n for _root, _dirs, files in os.walk(out_dir) for n in files if n.endswith(".tmp")]
    assert leftovers == []


def test_a_manifest_write_that_fails_part_way_leaves_no_truncated_manifest(tmp_path, monkeypatch):
    real_writerow = csv.DictWriter.writerow
    written = []

    def writerow(self, row):
        if len(written) == 3:  # the header and two images are out; the disk fills up
            raise OSError(errno.ENOSPC, "No space left on device (simulated)")
        written.append(row)
        return real_writerow(self, row)

    monkeypatch.setattr(csv.DictWriter, "writerow", writerow)
    with pytest.raises(OSError, match="No space left"):
        _export_with_an_error(tmp_path)

    out_dir = tmp_path / "out"
    assert not (out_dir / "manifest.csv").exists()  # complete or absent, never truncated
    assert [n for n in os.listdir(out_dir) if n.endswith(".tmp")] == []


@pytest.mark.parametrize("rel", ["splits/train.txt", "classes.json", "manifest.csv"])
def test_a_text_file_held_open_by_another_program_raises_file_locked_error(
    tmp_path, monkeypatch, rel
):
    real_replace = os.replace

    def locked(src, dst):
        if os.path.basename(dst) == os.path.basename(rel):
            exc = PermissionError(errno.EACCES, "The process cannot access the file", dst)
            exc.winerror = 32  # ERROR_SHARING_VIOLATION
            raise exc
        real_replace(src, dst)

    monkeypatch.setattr(io_utils.os, "replace", locked)
    monkeypatch.setattr(io_utils.time, "sleep", lambda s: None)
    with pytest.raises(FileLockedError, match="open in another program"):
        _export_with_an_error(tmp_path)
    assert not (tmp_path / "out" / rel).exists()

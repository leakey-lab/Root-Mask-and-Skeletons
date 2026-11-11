"""Correct reads a GT mask as ``MaskStore.read`` reads a prediction (S01-07).

``mask_files._read_gt`` decodes ``gt/mask/<stem>.png`` through ``app.masks.read_mask_file``:
the exact fast path for plain grey and grey RGBA PNGs (every mask the editor saves), else the
reference decode; a file it cannot decode is "unreadable", one of another size than the photo
is "size" (never resized), both logged. Every file kind against the reference decode:
``tests/test_mask_store_decode.py``.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from app.gui.correct import mask_files
from app.masks import read_mask_file, store, write_mask_file


def test_an_editor_gt_mask_takes_the_fast_path(tmp_path, monkeypatch):
    on = np.zeros((48, 64), bool)
    on[5:20, 3:60] = True
    path = str(tmp_path / "gt" / "mask" / "a.png")
    write_mask_file(path, on)
    calls = []
    real_imdecode, real_reference = cv2.imdecode, store._decode_reference
    monkeypatch.setattr(cv2, "imdecode", lambda *a: calls.append("cv2") or real_imdecode(*a))
    monkeypatch.setattr(
        store, "_decode_reference", lambda d: calls.append("ref") or real_reference(d)
    )

    plane, why = mask_files._read_gt(path, (64, 48))

    assert (why, calls) == ("", [])  # no OpenCV decode: the exact fast path read it
    assert plane.dtype == np.uint8 and np.array_equal(plane, on)
    assert np.array_equal(read_mask_file(path), on)


def test_a_missing_or_folder_gt_mask_is_unreadable_and_logged(tmp_path, caplog):
    folder = tmp_path / "gt" / "mask" / "b.png"
    folder.mkdir(parents=True)
    with caplog.at_level(logging.WARNING, logger="app.gui.correct.mask_files"):
        for path in (str(tmp_path / "gt" / "mask" / "a.png"), str(folder)):
            assert mask_files._read_gt(path, (64, 48)) == (None, "unreadable")
            assert read_mask_file(path) is None
    assert [r.getMessage() for r in caplog.records] == [
        f"Failed to load the GT mask {tmp_path / 'gt' / 'mask' / 'a.png'}",
        f"Failed to load the GT mask {folder}",
    ]


def test_a_gt_mask_of_another_size_is_named_in_the_log(tmp_path, caplog):
    path = str(tmp_path / "a.png")
    write_mask_file(path, np.ones((60, 80), bool))
    with caplog.at_level(logging.WARNING, logger="app.gui.correct.mask_files"):
        assert mask_files._read_gt(path, (200, 150)) == (None, "size")
    assert [r.getMessage() for r in caplog.records] == [
        f"Mask of {path} is 80x60, image is 200x150; ignoring it"
    ]

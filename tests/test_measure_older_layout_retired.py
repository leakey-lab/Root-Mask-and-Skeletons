"""Measure reads only today's folders: the older layout is retired (#38, 2026-10-31).

Measure Area used to read an older ``output/mask`` folder before ``mask/`` (with a warning
that the masks Generate and Correct save to ``mask/`` were left out), and Measure Length
fell back to ``[output/]skeletonizer/test_latest/images`` or a ``test_latest`` folder found
by a walk of the folders nearby. No project uses that layout any more: Measure reads
``mask/`` and ``skeletons/`` (and the GT files) and writes its CSVs there; the older
folders are neither read nor written, and nothing warns about them.
"""

from __future__ import annotations

import cv2
import numpy as np

from tests.measure_support import _handler, _project, _wait_idle


def _warnings(mw) -> list[str]:
    return [c.args[0] for c in mw.feedback.calls if c.name == "notify" and c.args[1] == "warn"]


def test_an_older_mask_folder_next_to_mask_is_ignored(tmp_path, qapp):
    _project(tmp_path, skeletons=False)  # mask/: where Generate and Correct save
    older = tmp_path / "output" / "mask"
    older.mkdir(parents=True)
    cv2.imwrite(str(older / "x_T1_L2_2024.01.02_010200_001.png"), np.zeros((64, 80), np.uint8))
    handler, mw = _handler(tmp_path, qapp)

    handler.calculate_root_area()
    _wait_idle(handler, qapp)

    assert _warnings(mw) == []
    assert (tmp_path / "mask" / "root_areas.csv").is_file()
    assert sorted(p.name for p in older.iterdir()) == ["x_T1_L2_2024.01.02_010200_001.png"]


def test_a_project_with_only_mask_gets_no_warning(tmp_path, qapp):
    _project(tmp_path, skeletons=False)
    handler, mw = _handler(tmp_path, qapp)

    handler.calculate_root_area()
    _wait_idle(handler, qapp)

    assert _warnings(mw) == []


def test_an_older_layout_alone_has_nothing_to_measure(tmp_path, qapp):
    """Skeletons only in ``skeletonizer/test_latest/images`` and masks only in
    ``output/mask``: Measure says there is nothing to measure and writes no CSV."""
    _project(tmp_path, skeletons=False, masks=False)
    plane = np.zeros((64, 80), np.uint8)
    for folder, suffix in (
        (tmp_path / "skeletonizer" / "test_latest" / "images", "_fake.png"),
        (tmp_path / "output" / "mask", ".png"),
    ):
        folder.mkdir(parents=True)
        cv2.imwrite(str(folder / f"x_T1_L2_2024.01.02_010200_001{suffix}"), plane)
    handler, mw = _handler(tmp_path, qapp)

    handler.calculate_root_length()
    handler.calculate_root_area()
    _wait_idle(handler, qapp)

    assert _warnings(mw) == [
        "No skeleton images found. Generate skeletons first.",
        "No mask images found. Generate masks first.",
    ]
    assert "task_started" not in mw.feedback.names()
    assert sorted(p.name for p in tmp_path.rglob("*.csv")) == []

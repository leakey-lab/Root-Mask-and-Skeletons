"""LibraryIndex neighbours and positions: what PgDn / PgUp would open, and "Image i / n".

``neighbours(path, n)`` names the images PgDn / PgUp would open, nearest first (next,
previous, next but one, ...), skipping hidden rows; ``position_of`` gives the 1-based place
among the shown images. The Correct page and the Library display read those neighbours
ahead; the Correct page's status chip shows the position.

Offscreen and file-free.
"""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QTreeWidget

from app.gui.library_index import LibraryIndex, index_names

STEMS = [f"24EFCAM1_T001_L0{d}_2024.07.18_1115{d}_001_SMM" for d in range(11, 17)]


def _index(app) -> tuple[QTreeWidget, LibraryIndex]:
    tree = QTreeWidget()
    index = LibraryIndex(tree, path_of=lambda name: f"D:/p/{name}.jpg")
    index.build(*index_names(STEMS))
    return tree, index


def _path(i: int) -> str:
    return f"D:/p/{STEMS[i]}.jpg"


def test_neighbours_are_next_then_previous_nearest_first(app):
    tree, index = _index(app)
    assert index.neighbours(_path(2)) == [_path(3), _path(1)]
    assert index.neighbours(_path(2), 2) == [_path(3), _path(1), _path(4), _path(0)]
    assert index.neighbours(STEMS[2]) == [_path(3), _path(1)]  # a stem works too
    tree.deleteLater()


def test_neighbours_at_the_ends_and_of_unknown_images(app):
    tree, index = _index(app)
    assert index.neighbours(_path(0)) == [_path(1)]
    assert index.neighbours(_path(5), 2) == [_path(4), _path(3)]
    assert index.neighbours("D:/p/elsewhere.jpg") == []
    assert index.neighbours("") == []
    tree.deleteLater()


def test_neighbours_skip_hidden_rows_like_paging_does(app):
    tree, index = _index(app)
    for i in (1, 3):
        index.leaves()[i].setHidden(True)
    assert index.neighbours(_path(2)) == [_path(4), _path(0)]
    assert index.step(index.leaves()[2], 1) is index.leaves()[4]  # the same rule
    assert index.position_of(_path(1)) is None  # a hidden image has no place
    tree.deleteLater()


def test_neighbours_and_position_follow_the_search(app):
    tree, index = _index(app)
    assert index.apply_filter("_L012_") == 1
    assert index.neighbours(_path(1)) == []
    assert index.position_of(_path(1)) == (1, 1)
    assert index.position_of(_path(2)) is None
    index.apply_filter("")
    assert index.position_of(_path(2)) == (3, 6)
    tree.deleteLater()


def test_position_of_is_one_based_among_the_images(app):
    tree, index = _index(app)
    assert index.position_of(_path(0)) == (1, 6)
    assert index.position_of(STEMS[5]) == (6, 6)
    assert index.position_of("D:/p/elsewhere.jpg") is None
    tree.deleteLater()


def _grid() -> list[str]:
    return [
        f"24EFCAM1_T{t:03d}_L{d:03d}_{date}_1115{d % 60:02d}_001_SMM"
        for t in (1, 2, 3)
        for date in ("2024.07.18", "2024.08.07")
        for d in range(11, 17)
    ]


@pytest.mark.parametrize("seed", range(4))
def test_paging_neighbours_and_position_agree_with_the_shown_rows(app, seed):
    """Under random searches, step / neighbours / position_of answer from the rows on screen:
    the reference is the list of shown image rows itself."""
    rng = np.random.default_rng(seed)
    tree = QTreeWidget()
    index = LibraryIndex(tree, path_of=lambda name: f"D:/p/{name}.jpg")
    index.build(*index_names(_grid()))
    words = ["", "t001", "t002", "_l013_", "_l01", "2024.08", "l011_2024.07", "zzz", "T3", "L1"]
    for query in rng.choice(words, size=6):
        index.apply_filter(str(query))
        shown = index.leaves(visible_only=True)
        for leaf in index.leaves():
            stem = leaf.data(0, Qt.ItemDataRole.UserRole)
            if leaf not in shown:
                assert index.position_of(stem) is None and index.neighbours(stem) == []
                continue
            i = shown.index(leaf)
            after, before = shown[i + 1 : i + 3], shown[max(i - 2, 0) : i][::-1]
            assert index.position_of(stem) == (i + 1, len(shown)), (query, stem)
            assert index.step(leaf, 1) is (after[0] if after else None)
            assert index.step(leaf, -1) is (before[0] if before else None)
            order = [x for pair in zip(after, before, strict=False) for x in pair]
            order += after[len(before) :] + before[len(after) :]
            paths = [f"D:/p/{x.data(0, Qt.ItemDataRole.UserRole)}.jpg" for x in order]
            assert index.neighbours(stem, 2) == paths, (query, stem)
    tree.deleteLater()

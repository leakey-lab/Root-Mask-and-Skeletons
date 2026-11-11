"""The Qt-free vector extraction of the skeleton item (app/imaging/skeleton_vectors.py).

The m-adjacency links between set pixels (a diagonal only where no corner pixel is set; an
isolated pixel is a dot), the links of a tile (each link belongs to the tile of its first
pixel, so the tiles of a mask give every whole-frame link exactly once), the merged
collinear runs of a tile, and ``clip_box``. Also: the module loads no Qt and no cv2, and the
item draws with these very functions. (The extraction tests came from
``tests/test_canvas_skeleton.py``.)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from app.imaging.skeleton_vectors import clip_box, skeleton_links, tile_segments

ROOT = Path(__file__).resolve().parents[1]


def test_the_vector_module_loads_no_qt_and_no_cv2():
    code = (
        "import sys, app.imaging.skeleton_vectors; "
        "print(sorted(m for m in sys.modules if m.split('.')[0] in {'PyQt6', 'cv2', 'torch'}))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"


def test_the_item_draws_with_the_qt_free_functions():
    import app.gui.canvas_layers.skeleton as item_module
    import app.imaging.skeleton_vectors as vectors

    # A from-import: monkeypatching the item module's name reaches the item's extraction.
    assert item_module.tile_segments is vectors.tile_segments
    assert item_module.clip_box is vectors.clip_box
    assert item_module.Box is vectors.Box


@pytest.mark.parametrize(
    ("box", "clipped"),
    [
        (None, (0, 0, 40, 30)),
        ((5, 6, 10, 8), (5, 6, 15, 14)),
        ((-4, -3, 10, 8), (0, 0, 6, 5)),
        ((35, 25, 10, 10), (35, 25, 40, 30)),
        ((50, 40, 10, 10), (50, 40, 40, 30)),  # outside: an empty box (end <= start)
        ((2.9, 3.1, 4.0, 5.0), (2, 3, 6, 8)),  # truncated to int, as the item passes them
    ],
)
def test_clip_box_gives_end_exclusive_edges_inside_the_mask(box, clipped):
    assert clip_box(box, (30, 40)) == clipped
    assert clip_box(box, (30, 40, 4)) == clipped  # extra dimensions are ignored


# ---------------------------------------------------------------- the vector extraction
TILE = 256


def _mask(w: int, h: int, pixels) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    for x, y in pixels:
        m[y, x] = 255
    return m


def _links(mask: np.ndarray, box=None) -> set[tuple[int, int, int, int]]:
    return {tuple(int(v) for v in row) for row in skeleton_links(mask, box)}


def _segments(mask: np.ndarray, box) -> set[tuple[int, int, int, int]]:
    return {tuple(int(v) for v in row) for row in tile_segments(mask, box)}


def _unit_steps(segment) -> list[tuple[int, int, int, int]]:
    """Expand a merged run (x0, y0, x1, y1) back into its unit links (a dot stays a dot)."""
    x0, y0, x1, y1 = (int(v) for v in segment)
    n = max(abs(x1 - x0), abs(y1 - y0))
    if n == 0:
        return [(x0, y0, x0, y0)]
    dx, dy = (x1 - x0) // n, (y1 - y0) // n
    return [(x0 + i * dx, y0 + i * dy, x0 + (i + 1) * dx, y0 + (i + 1) * dy) for i in range(n)]


def _tiles(mask: np.ndarray):
    h, w = mask.shape
    for ty in range(0, h, TILE):
        for tx in range(0, w, TILE):
            yield (tx, ty, TILE, TILE)


def _random_skeleton(seed: int, w: int = 600, h: int = 540, walks: int = 12) -> np.ndarray:
    """8-connected random walks (with crossings, spurs and blobs where walks meet)."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((h, w), np.uint8)
    steps = np.array([(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)])
    for _ in range(walks):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        for _ in range(int(rng.integers(50, 900))):
            mask[y, x] = 255
            dx, dy = steps[rng.integers(0, 8)]
            x, y = int(np.clip(x + dx, 0, w - 1)), int(np.clip(y + dy, 0, h - 1))
    for _ in range(40):  # scattered isolated pixels
        mask[int(rng.integers(0, h)), int(rng.integers(0, w))] = 255
    return mask


def test_l_corner_links_skip_the_corner_diagonal():
    mask = _mask(6, 6, [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3)])
    assert _links(mask) == {(1, 1, 1, 2), (1, 2, 1, 3), (1, 3, 2, 3), (2, 3, 3, 3)}
    assert _segments(mask, (0, 0, 6, 6)) == {(1, 1, 1, 3), (1, 3, 3, 3)}


def test_t_junction_links_join_the_stem_to_the_bar_only():
    mask = _mask(6, 6, [(1, 1), (2, 1), (3, 1), (2, 2), (2, 3)])
    assert _links(mask) == {(1, 1, 2, 1), (2, 1, 3, 1), (2, 1, 2, 2), (2, 2, 2, 3)}
    assert _segments(mask, (0, 0, 6, 6)) == {(1, 1, 3, 1), (2, 1, 2, 3)}


def test_an_isolated_pixel_is_a_dot():
    mask = _mask(6, 6, [(4, 2)])
    assert _links(mask) == {(4, 2, 4, 2)}
    assert _segments(mask, (0, 0, 6, 6)) == {(4, 2, 4, 2)}


def test_diagonal_lines_link_pixel_centres_and_merge_into_one_run():
    diagonal = _mask(6, 6, [(0, 0), (1, 1), (2, 2), (3, 3)])
    assert _links(diagonal) == {(0, 0, 1, 1), (1, 1, 2, 2), (2, 2, 3, 3)}
    assert _segments(diagonal, (0, 0, 6, 6)) == {(0, 0, 3, 3)}

    anti = _mask(6, 6, [(3, 0), (2, 1), (1, 2), (0, 3)])
    assert _links(anti) == {(3, 0, 2, 1), (2, 1, 1, 2), (1, 2, 0, 3)}
    assert _segments(anti, (0, 0, 6, 6)) == {(3, 0, 0, 3)}


def test_a_four_connected_staircase_has_no_corner_triangles():
    mask = _mask(6, 6, [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)])
    expected = {(0, 0, 1, 0), (1, 0, 1, 1), (1, 1, 2, 1), (2, 1, 2, 2), (2, 2, 3, 2)}
    assert _links(mask) == expected
    assert _segments(mask, (0, 0, 6, 6)) == expected  # no collinear chain to merge


def test_a_two_by_two_blob_is_a_square_without_diagonals():
    mask = _mask(4, 4, [(1, 1), (2, 1), (1, 2), (2, 2)])
    assert _links(mask) == {(1, 1, 2, 1), (1, 2, 2, 2), (1, 1, 1, 2), (2, 1, 2, 2)}


def test_pixels_on_the_image_border_link_without_wrapping():
    mask = _mask(5, 4, [(4, 0), (4, 1), (0, 3), (1, 3), (0, 0)])
    assert _links(mask) == {(4, 0, 4, 1), (0, 3, 1, 3), (0, 0, 0, 0)}


def test_links_belong_to_the_tile_of_their_first_pixel():
    mask = np.zeros((300, 300), np.uint8)
    mask[5, 250:261] = 255  # a horizontal line across the x = 256 tile border

    assert _segments(mask, (0, 0, TILE, TILE)) == {(250, 5, 256, 5)}
    assert _segments(mask, (TILE, 0, TILE, TILE)) == {(256, 5, 260, 5)}


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_per_tile_links_equal_the_whole_frame_links_for_random_skeletons(seed):
    mask = _random_skeleton(seed)
    whole = skeleton_links(mask)
    per_tile = [skeleton_links(mask, box) for box in _tiles(mask)]

    assert sum(len(p) for p in per_tile) == len(whole)  # every link extracted exactly once
    assert set().union(*({tuple(r) for r in p.tolist()} for p in per_tile)) == _links(mask)


def test_per_tile_links_equal_the_whole_frame_links_for_lines_crossing_tile_corners():
    mask = np.zeros((520, 530), np.uint8)
    for i in range(-6, 7):
        mask[256 + i, 256 + i] = 255  # diagonal through the corner of four tiles
        mask[255 - i, 256 + i] = 255  # anti-diagonal through the same corner
        mask[255, 250 + i] = 255  # along the last row of the top tiles
        mask[250 + i, 256] = 255  # along the first column of the right tiles
    mask[511:514, 511:514] = 255  # a blob on the next corner
    per_tile = [_links(mask, box) for box in _tiles(mask)]

    assert sum(len(p) for p in per_tile) == len(_links(mask))
    assert set().union(*per_tile) == _links(mask)


@pytest.mark.parametrize("seed", [5, 6])
def test_merged_segments_are_the_tile_links_joined_into_maximal_collinear_runs(seed):
    mask = _random_skeleton(seed)
    for box in _tiles(mask):
        segments = tile_segments(mask, box)
        expanded = [step for seg in segments for step in _unit_steps(seg)]
        assert sorted(expanded) == sorted(_links(mask, box))  # same links, none twice
        runs = [s for s in segments.tolist() if _direction(s) is not None]
        ends = {(tuple(s[2:]), _direction(s)) for s in runs}
        starts = {(tuple(s[:2]), _direction(s)) for s in runs}
        assert not ends & starts, "two collinear runs of one tile should have merged"


def _direction(seg) -> tuple[int, int] | None:
    x0, y0, x1, y1 = seg
    n = max(abs(x1 - x0), abs(y1 - y0))
    return None if n == 0 else ((x1 - x0) // n, (y1 - y0) // n)


def test_extraction_reads_the_mask_without_changing_it():
    mask = _random_skeleton(7)
    before = mask.copy()
    skeleton_links(mask)
    tile_segments(mask, (0, 0, TILE, TILE))
    assert np.array_equal(mask, before)


def test_an_empty_or_outside_box_has_no_links():
    mask = _random_skeleton(8)
    assert len(skeleton_links(np.zeros((40, 40), np.uint8))) == 0
    assert len(tile_segments(mask, (1000, 1000, TILE, TILE))) == 0

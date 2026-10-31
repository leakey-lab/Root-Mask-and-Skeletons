"""Select can pick a closed skeleton ring next to other branches (S03-06).

A ring whose every pixel has two neighbours has no endpoint or junction, so no walk from a
node reaches it. ``vectorize_skeleton`` traced such rings only when the whole skeleton had
no node at all; with any other branch in the image the ring was missing from the topology
and Select could not pick it (only the Eraser removed it). Rings now come after the node
walks, so the polylines before them, and their indices, are as they were.

Topology is display and pick only: never saved or measured.
"""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import Qt

from app.gui.skeleton_graph_model import vectorize_skeleton
from tests.skeleton_edit_support import edits, key, lines, open_layer, pick

pytestmark = pytest.mark.usefixtures("app", "no_prompts")

DIAMOND = [(400, 260), (440, 300), (400, 340), (360, 300), (400, 260)]  # a node-free ring
SMALL = [(150, 360), (170, 380), (150, 400), (130, 380), (150, 360)]
LINE = [(100, 100), (300, 100)]


def _degrees(mask) -> np.ndarray:
    on = (mask > 0).astype(np.int32)
    padded = np.pad(on, 1)
    h, w = on.shape
    total = sum(
        padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if dy or dx
    )
    return np.where(on > 0, total, -1)


def _pixels(polyline) -> set[tuple[int, int]]:
    return {(int(x), int(y)) for x, y in polyline}


def test_the_test_rings_have_no_node():
    for ring in (DIAMOND, SMALL):
        degree = _degrees(lines(ring))
        assert set(np.unique(degree[degree >= 0]).tolist()) == {2}


@pytest.mark.parametrize("eps", [0.0, 1.5])
def test_a_ring_next_to_a_line_is_traced_after_the_lines_polylines(eps):
    line_only = vectorize_skeleton(lines(LINE), simplify_epsilon=eps)
    ring_mask = lines(DIAMOND)

    topo = vectorize_skeleton(lines(LINE, DIAMOND), simplify_epsilon=eps)

    assert topo.polylines[:-1] == line_only.polylines, "the node walks come first, unchanged"
    assert topo.endpoints == line_only.endpoints and topo.junctions == line_only.junctions
    ring = topo.polylines[-1]
    assert ring[0] == ring[-1], "the ring is closed"
    if eps == 0.0:
        ys, xs = np.nonzero(ring_mask)
        assert _pixels(ring) == set(zip(xs.tolist(), ys.tolist(), strict=True))


def test_a_ring_next_to_a_stray_pixel_is_traced():
    mask = lines(DIAMOND)
    mask[50, 50] = 255  # an isolated pixel is a node (degree 0) with no polyline

    topo = vectorize_skeleton(mask, simplify_epsilon=1.5)

    assert len(topo.polylines) == 1 and topo.polylines[0][0] == topo.polylines[0][-1]


def test_two_rings_and_a_line_give_both_rings():
    topo = vectorize_skeleton(lines(LINE, DIAMOND, SMALL), simplify_epsilon=0.0)

    rings = [p for p in topo.polylines if len(p) > 2 and p[0] == p[-1]]
    assert len(rings) == 2
    assert {len(_pixels(r)) for r in rings} == {
        int(np.count_nonzero(lines(DIAMOND))),
        int(np.count_nonzero(lines(SMALL))),
    }


def test_a_ring_alone_is_traced_as_before():
    topo = vectorize_skeleton(lines(DIAMOND), simplify_epsilon=1.5)

    assert len(topo.polylines) == 1 and topo.polylines[0][0] == topo.polylines[0][-1]


def test_select_picks_a_ring_next_to_a_line_and_delete_removes_only_the_ring(tmp_path):
    skeleton = lines(LINE, DIAMOND)
    layer, host = open_layer(tmp_path, skeleton)

    pick(layer, 421, 279)  # on the ring's upper right side

    assert key(layer, Qt.Key.Key_Delete) is True
    assert np.array_equal(layer.model.mask, lines(LINE)), "the ring went, the line stayed"
    assert edits(host) == 1
    layer.cancel_scheduled_render()

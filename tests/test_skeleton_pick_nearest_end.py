"""A picked branch goes on from its end nearest the pointer (owner bug report 2026-11-02).

Select hands the picked branch to the Polyline tool in the order ``vectorize_skeleton``
traced it: from whichever of its two nodes the walk met first, the order of a Python set,
which has nothing to do with where the pointer is. The preview's rubber band and the next
click always went on from the branch's last traced point, so a click beyond one end often
drew a line from the far end instead ("the spline starts from a random node").

Now the point nearest the pointer is where the pick goes on: the nearer end of an open
branch (straight-line distance; at equal distance the end with the lower ``(y, x)``), the
nearest point of a closed one (a ring, or a loop whose ends are the same pixel or two
pixels of a junction cluster). A click adds its point there and the preview's rubber band
runs from it; a closed pick is previewed as Enter draws it, so the pointer does not move
the seam of its smoothed curve. Undoing the click (mid-polyline undo, or a double-click
taking its own point back) undoes the turn too: the pick is as picked again. A new line
keeps its points in the order the user placed them.
"""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, Qt

from app.gui.skeleton_overlay import polyline_path
from app.gui.skeleton_tools import PolylineDraft
from tests.skeleton_edit_support import click, edits, key, lines, open_layer, pick

pytestmark = pytest.mark.usefixtures("app", "no_prompts")

STRAIGHT = [[(100, 240), (539, 240)]]  # traced from (539, 240) to (100, 240)
Y = [[(320, 240), (320, 80)], [(320, 240), (200, 360)], [(320, 240), (440, 360)]]
CURVE = [[(100, 300), (180, 200), (260, 170), (340, 200), (420, 300)]]
L = [[(100, 100), (300, 100), (300, 300)]]  # traced (100, 100), (299, 100), (300, 300)
DIAMOND = [(400, 260), (440, 300), (400, 340), (360, 300), (400, 260)]  # a node-free ring
LINE = [(100, 100), (300, 100)]
# A loop on a stem: thinning leaves a 4 px junction cluster at (297..299, 100..101), and the
# loop is traced from (299, 100) round to (298, 101), two pixels of it: not closed exactly.
BIG_LOOP = [
    (300, 100), (420, 160), (460, 260), (400, 380), (260, 400), (180, 300), (200, 170),
    (300, 100),
]  # fmt: skip

# shape, where Select picks, a click beyond one end, that end (a vertex of the pick).
BEYOND_AN_END = {
    "straight, right end": (STRAIGHT, (300, 242), (600, 300), (539, 240)),
    "straight, left end": (STRAIGHT, (300, 242), (40, 300), (100, 240)),
    "Y arm, tip": (Y, (320, 160), (320, 30), (320, 80)),
    "Y arm, junction": (Y, (320, 160), (380, 250), (320, 240)),
    "curve, left end": (CURVE, (260, 172), (60, 360), (100, 300)),
    "curve, right end": (CURVE, (260, 172), (460, 360), (420, 300)),
    "L, first end": (L, (200, 102), (40, 100), (100, 100)),
    "L, last end": (L, (200, 102), (300, 360), (300, 300)),
}


def _centre(p) -> tuple[float, float]:
    return (p[0] + 0.5, p[1] + 0.5)


def _path_points(layer) -> list[tuple[float, float]]:
    path = layer.decorations.polyline_item.path()
    return [(path.elementAt(i).x, path.elementAt(i).y) for i in range(path.elementCount())]


@pytest.mark.parametrize("name", sorted(BEYOND_AN_END))
def test_a_click_beyond_an_end_of_a_pick_goes_on_from_that_end(tmp_path, name):
    shape, at, (x, y), end = BEYOND_AN_END[name]
    layer, _host = open_layer(tmp_path, lines(*shape))
    pick(layer, *at)
    branch = list(layer.draft.edit_original)
    assert end in (branch[0], branch[-1]), "the test end is not an end of the pick"

    click(layer, x, y)

    points = layer.draft.xy()
    assert points[-2:] == [end, (x, y)], f"the line went on from {points[-2]}, not {end}"
    assert points[:-1] in (branch, branch[::-1]), "the picked branch kept its shape"
    layer.cancel_scheduled_render()


@pytest.mark.parametrize("name", sorted(BEYOND_AN_END))
def test_the_preview_runs_from_the_end_nearest_the_pointer(tmp_path, name):
    shape, at, (x, y), end = BEYOND_AN_END[name]
    layer, _host = open_layer(tmp_path, lines(*shape))
    pick(layer, *at)
    points = list(layer.draft.edit_original)

    layer.move(QPoint(x, y), False)  # hover

    path = _path_points(layer)
    assert path[-2:] == [_centre(end), _centre((x, y))], "the rubber band is not from the end"
    assert len(path) == len(points) + 1
    assert layer.draft.xy() == points, "a hover changes no point of the pick"
    layer.cancel_scheduled_render()


@pytest.mark.parametrize("side", ["right", "left"])
def test_the_saved_line_joins_the_near_end_with_no_chord_across(tmp_path, side):
    # The owner's case end to end: pick a branch, click beyond one end, Enter. The saved
    # skeleton has the new segment at that end, and nothing along the chord that a line
    # from the far end to the click would draw.
    shape, at, (x, y), end = BEYOND_AN_END[f"straight, {side} end"]
    skeleton = lines(*shape)
    layer, host = open_layer(tmp_path, skeleton)
    pick(layer, *at)

    click(layer, x, y)
    key(layer, Qt.Key.Key_Return)

    mask = layer.model.mask
    assert edits(host) == 1 and layer.draft.points == []
    mx, my = (end[0] + x) // 2, (end[1] + y) // 2  # the middle of the new segment
    assert mask[my - 3 : my + 4, mx - 3 : mx + 4].any(), "the segment at the end is missing"
    far = (100, 240) if end == (539, 240) else (539, 240)
    cx, cy = (far[0] + x) // 2, (far[1] + y) // 2  # the middle of the chord from the far end
    assert not mask[cy - 6 : cy + 7, cx - 6 : cx + 7].any(), "a chord from the far end"
    assert np.array_equal(mask[235:246, 110:530] > 0, skeleton[235:246, 110:530] > 0)
    layer.cancel_scheduled_render()


def test_a_click_near_the_middle_of_a_long_branch_goes_on_from_the_nearer_end(tmp_path):
    # The rule for a click anywhere: the end at the shorter straight-line distance.
    layer, _host = open_layer(tmp_path, lines(*STRAIGHT))
    pick(layer, 300, 242)
    assert layer.draft.xy()[-1] == (100, 240), "the test needs the far end traced last"

    click(layer, 330, 270)  # 211 px from (539, 240), 232 px from (100, 240)

    assert layer.draft.xy()[-2:] == [(539, 240), (330, 270)]
    layer.cancel_scheduled_render()


def test_a_ring_goes_on_from_its_point_nearest_the_click(tmp_path):
    layer, _host = open_layer(tmp_path, lines(LINE, DIAMOND))
    pick(layer, 440, 300)
    ring = list(layer.draft.edit_original)
    assert ring[0] == ring[-1] and ring[0] != (440, 300), "the test ring opens elsewhere"

    click(layer, 480, 300)

    points = layer.draft.xy()
    assert points[-2:] == [(440, 300), (480, 300)]
    assert points[0] == (440, 300), "the ring is still closed, at its new opening"
    assert sorted(set(points[:-1])) == sorted(set(ring)), "the ring kept its points"
    layer.cancel_scheduled_render()


def test_with_smooth_on_the_curve_goes_on_from_the_end_nearest_the_click(tmp_path):
    layer, _host = open_layer(tmp_path, lines(*L))
    pick(layer, 200, 102)
    key(layer, Qt.Key.Key_S)

    click(layer, 40, 100)

    assert layer.draft.xy()[-2:] == [(100, 100), (40, 100)]
    assert layer.draft.bends()
    layer.cancel_scheduled_render()


def test_a_loop_open_at_a_junction_cluster_goes_on_from_its_point_nearest_the_click(tmp_path):
    skeleton = lines(LINE, BIG_LOOP)
    layer, host = open_layer(tmp_path, skeleton)
    pick(layer, 460, 262)
    loop = list(layer.draft.edit_original)
    (x0, y0), (x1, y1) = loop[0], loop[-1]
    assert loop[0] != loop[-1] and max(abs(x0 - x1), abs(y0 - y1)) <= 1, "not a cluster loop"
    assert (460, 260) in loop

    click(layer, 520, 262)

    points = layer.draft.xy()
    assert points[-2:] == [(460, 260), (520, 262)], f"went on from {points[-2]}"
    assert points[0] == (460, 260), "the loop is closed at its new opening"
    assert set(points[:-1]) == set(loop), "the loop kept its points"
    key(layer, Qt.Key.Key_Return)
    mask = layer.model.mask
    assert edits(host) == 1
    assert mask[257:266, 486:495].any(), "the segment from the loop is missing"
    # The chord from the loop's traced end (299, 100) to the click passes (409, 181), well
    # inside the loop: nothing there.
    assert not mask[176:187, 404:415].any(), "a chord across the loop"
    layer.cancel_scheduled_render()


def test_undo_in_the_gesture_takes_back_the_click_that_turned_the_pick(tmp_path):
    layer, _host = open_layer(tmp_path, lines(*STRAIGHT))
    pick(layer, 300, 242)
    branch = list(layer.draft.edit_original)
    click(layer, 600, 300)
    assert layer.draft.xy()[0] == branch[-1], "the click turned the pick"

    assert layer.undo_in_gesture() is True

    assert layer.draft.xy() == branch, "the branch as picked, in its picked order"
    assert layer.draft.edit_original == branch and not layer.is_dirty()
    layer.cancel_scheduled_render()


def test_extending_both_ends_then_undoing_twice_gives_the_pick_back_as_picked(tmp_path):
    layer, _host = open_layer(tmp_path, lines(*STRAIGHT))
    pick(layer, 300, 242)
    picked = list(layer.draft.points)
    original = layer.draft.edit_original
    click(layer, 600, 300)  # beyond the right end
    click(layer, 40, 300)  # beyond the left end: the line turns again
    assert layer.draft.xy() == [(600, 300), (539, 240), (100, 240), (40, 300)]

    assert layer.undo_in_gesture() is True
    assert layer.draft.xy() == [(100, 240), (539, 240), (600, 300)]
    assert layer.undo_in_gesture() is True

    assert len(layer.draft.points) == len(picked)
    assert all(a is b for a, b in zip(layer.draft.points, picked, strict=True)), "as picked"
    assert layer.draft.edit_original == original and not layer.is_dirty()
    assert layer.draft.turns == []
    layer.cancel_scheduled_render()


def test_a_ring_clicked_then_undone_smooths_exactly_as_the_plain_pick(tmp_path):
    masks = []
    for name, with_click in (("plain", False), ("clicked", True)):
        layer, host = open_layer(tmp_path / name, lines(LINE, DIAMOND))
        pick(layer, 440, 300)
        if with_click:
            click(layer, 480, 300)
            layer.undo_in_gesture()
        key(layer, Qt.Key.Key_S)
        key(layer, Qt.Key.Key_Return)
        assert edits(host) == 1
        masks.append(layer.model.mask.copy())
        layer.cancel_scheduled_render()
    assert np.array_equal(masks[0], masks[1]), "the undone click moved the ring's seam"


def test_an_undo_after_a_drag_takes_back_only_the_last_point(tmp_path):
    # A point moved since the click: the draft is not the one the click made, so the undo
    # drops the click alone and keeps the moved point (no restore of an older draft).
    layer, _host = open_layer(tmp_path, lines(*STRAIGHT))
    pick(layer, 300, 242)
    click(layer, 600, 300)
    layer.press(QPoint(539, 240), Qt.KeyboardModifier.NoModifier)  # the right end's handle
    layer.move(QPoint(539, 262), True)
    layer.release(QPoint(539, 262))
    assert layer.draft.xy() == [(100, 240), (539, 262), (600, 300)]

    assert layer.undo_in_gesture() is True
    assert layer.draft.xy() == [(100, 240), (539, 262)]
    assert layer.undo_in_gesture() is True
    assert layer.draft.xy() == [(100, 240)]
    layer.cancel_scheduled_render()


def test_an_undo_after_dragging_the_clicked_point_gives_the_pick_back_as_picked(tmp_path):
    # The clicked point goes with the undo anyway, so dragging it keeps no turned order:
    # the pick is back as picked, its points and original included.
    layer, _host = open_layer(tmp_path, lines(*STRAIGHT))
    pick(layer, 300, 242)
    picked, original = list(layer.draft.points), layer.draft.edit_original
    click(layer, 600, 300)
    layer.press(QPoint(600, 300), Qt.KeyboardModifier.NoModifier)  # the clicked point's handle
    layer.move(QPoint(620, 320), True)
    layer.release(QPoint(620, 320))
    assert layer.draft.xy() == [(100, 240), (539, 240), (620, 320)]

    assert layer.undo_in_gesture() is True
    assert len(layer.draft.points) == len(picked)
    assert all(a is b for a, b in zip(layer.draft.points, picked, strict=True))
    assert layer.draft.edit_original == original
    assert layer.draft.turns == []
    assert layer.draft.plain_pick()
    layer.cancel_scheduled_render()


def test_a_closed_pick_is_previewed_as_enter_draws_it_wherever_the_pointer_is(tmp_path):
    # With Smooth on, opening the ring where the pointer is would move the seam of the
    # curve with the pointer. The ring is shown as picked (as Enter draws it) and the rubber
    # band runs apart from its point nearest the pointer, where a click goes on from.
    layer, _host = open_layer(tmp_path, lines(LINE, DIAMOND))
    pick(layer, 440, 300)
    key(layer, Qt.Key.Key_S)
    ring = layer.draft.xy()
    drawn = polyline_path([_centre(p) for p in ring], True)
    as_enter = [(drawn.elementAt(i).x, drawn.elementAt(i).y) for i in range(drawn.elementCount())]

    layer.move(QPoint(480, 300), False)  # hover

    path = _path_points(layer)
    assert path[:-2] == as_enter, "the ring is not shown as Enter draws it"
    assert path[-2:] == [_centre((440, 300)), _centre((480, 300))]
    assert layer.draft.xy() == ring
    layer.cancel_scheduled_render()


@pytest.mark.parametrize("which", [0, -1])
def test_dragging_an_end_handle_draws_no_line_from_the_other_end(tmp_path, which):
    layer, _host = open_layer(tmp_path, lines(*STRAIGHT))
    pick(layer, 300, 242)
    handle = layer.draft.points[which]

    layer.press(QPoint(handle), Qt.KeyboardModifier.NoModifier)
    layer.move(QPoint(handle.x(), handle.y() + 20), True)

    assert len(_path_points(layer)) == len(layer.draft.points), "an extra rubber band"
    layer.release(QPoint(handle.x(), handle.y() + 20))
    layer.cancel_scheduled_render()


def test_a_double_click_off_a_plain_pick_near_its_first_end_leaves_it_as_picked(tmp_path):
    # The double-click's first press turned the pick round to add its point; taking the
    # point back (S03-02) restores the pick as it was, its order included.
    layer, host = open_layer(tmp_path, lines(*STRAIGHT))
    pick(layer, 300, 242)
    picked = layer.draft.xy()
    assert picked[0] == (539, 240)

    layer.press(QPoint(600, 300), Qt.KeyboardModifier.NoModifier)
    layer.release(QPoint(600, 300))
    layer.double_click(QPoint(600, 300))
    layer.release(QPoint(600, 300))

    assert layer.draft.xy() == picked and layer.draft.edit_original == picked
    assert not layer.is_dirty() and edits(host) == 0
    layer.cancel_scheduled_render()


def test_a_new_line_goes_on_from_its_last_point_wherever_the_pointer_is(tmp_path):
    layer, _host = open_layer(tmp_path, lines(*STRAIGHT))
    layer.set_tool("polyline")
    click(layer, 100, 400)
    click(layer, 500, 400)

    click(layer, 90, 440)  # nearer the first point: still the line's next point

    assert layer.draft.xy() == [(100, 400), (500, 400), (90, 440)]
    layer.cancel_scheduled_render()


# ------------------------------------------------------------------- the rule itself
def _picked(*points) -> PolylineDraft:
    draft = PolylineDraft(points=[QPoint(*p) for p in points])
    draft.edit_original = list(points)
    return draft


@pytest.mark.parametrize(
    ("points", "pointer", "last"),
    [
        ([(300, 100), (300, 300)], (300, 330), (300, 300)),  # nearer the last end
        ([(300, 100), (300, 300)], (300, 60), (300, 100)),  # nearer the first end
        ([(300, 300), (300, 100)], (400, 200), (300, 100)),  # a tie: the lower y
        ([(300, 100), (300, 300)], (400, 200), (300, 100)),
        ([(300, 200), (100, 200)], (200, 300), (100, 200)),  # a tie on one row: the lower x
        ([(100, 200), (300, 200)], (200, 300), (100, 200)),
    ],
)
def test_facing_puts_the_end_nearest_the_pointer_last(points, pointer, last):
    draft = _picked(*points)

    faced = [(p.x(), p.y()) for p in draft.facing(QPoint(*pointer))]

    assert faced[-1] == last and sorted(faced) == sorted(points)
    assert draft.xy() == points, "facing only reads the draft"


def test_facing_opens_a_closed_pick_at_its_point_nearest_the_pointer():
    draft = _picked((10, 0), (20, 10), (10, 20), (0, 10), (10, 0))

    faced = [(p.x(), p.y()) for p in draft.facing(QPoint(30, 12))]

    assert faced == [(20, 10), (10, 20), (0, 10), (10, 0), (20, 10)]


def test_face_turns_a_plain_pick_and_what_it_compares_with_together():
    draft = _picked((300, 100), (300, 300))

    draft.face(QPoint(300, 60))

    assert draft.xy() == [(300, 300), (300, 100)]
    assert draft.edit_original == [(300, 300), (300, 100)] and draft.plain_pick()


def test_face_leaves_a_changed_picks_original_alone():
    draft = _picked((300, 100), (300, 300))
    draft.points.append(QPoint(350, 350))  # an added point: the pick is changed

    draft.face(QPoint(300, 60))

    assert draft.xy() == [(350, 350), (300, 300), (300, 100)]
    assert draft.edit_original == [(300, 100), (300, 300)] and draft.pending()


def test_a_new_line_never_faces_the_pointer():
    draft = PolylineDraft(points=[QPoint(300, 100), QPoint(300, 300)])

    draft.face(QPoint(300, 60))

    assert draft.xy() == [(300, 100), (300, 300)]
    assert draft.facing(QPoint(300, 60)) == draft.points

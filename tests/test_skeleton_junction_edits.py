"""Branch Delete and branch edits at junctions erase only the picked branch (S03-01).

A branch picked with Select is erased by its own pixels: the pixels of its walk from node
to node without the junction pixels at its ends. It used to be a 9 px band along the
branch, whose round cap at a junction also cut the first pixels of every other branch
there, so one Delete split a Y into two pieces and lowered the GT root length.

Cases: a Y, an X of two diagonals, a + cross (a junction of five pixels), branches that run
to the image edge, short spurs, an edited branch, a click right on a junction, and a ring
next to a line (no junction at all). Each
check: the other branches keep every pixel outside a 3 px box around the junction, and
the skeleton stays as many pieces as it was.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import Qt

from app.gui.skeleton_graph_model import SkeletonCorrectionModel
from tests.real_data import real_data_dir
from tests.skeleton_edit_support import (
    components,
    drag,
    edits,
    key,
    lines,
    open_layer,
    pick,
    tips,
)

pytestmark = pytest.mark.usefixtures("app", "no_prompts")

BOX = 3  # px around the nominal junction where thinning may move the junction pixels

# name: (junction, arms); each arm runs from the junction to its tip.
JUNCTIONS = {
    "Y": (
        (320, 240),
        [[(320, 240), (320, 80)], [(320, 240), (200, 360)], [(320, 240), (440, 360)]],
    ),
    "X": (
        (320, 240),
        [
            [(320, 240), (220, 140)],
            [(320, 240), (420, 140)],
            [(320, 240), (220, 340)],
            [(320, 240), (420, 340)],
        ],
    ),
    "plus": (
        (320, 240),
        [
            [(320, 240), (480, 240)],
            [(320, 240), (160, 240)],
            [(320, 240), (320, 80)],
            [(320, 240), (320, 400)],
        ],
    ),
    "Y at the top edge": (
        (320, 60),
        [[(320, 60), (320, 0)], [(320, 60), (200, 180)], [(320, 60), (440, 180)]],
    ),
    "Y at the left edge": (
        (300, 240),
        [[(300, 240), (0, 240)], [(300, 240), (420, 120)], [(300, 240), (420, 360)]],
    ),
    "X at the bottom right corner": (
        (590, 430),
        [
            [(590, 430), (639, 479)],
            [(590, 430), (540, 380)],
            [(590, 430), (639, 381)],
            [(590, 430), (541, 479)],
        ],
    ),
}
CASES = [(name, arm) for name, (_, arms) in JUNCTIONS.items() for arm in range(len(arms))]


def _box(junction) -> np.ndarray:
    x, y = junction
    box = np.zeros((480, 640), bool)
    box[max(0, y - BOX) : y + BOX + 1, max(0, x - BOX) : x + BOX + 1] = True
    return box


def _arm_pixels(skeleton, arm, junction) -> np.ndarray:
    """The pixels of ``skeleton`` on ``arm`` (drawn alone), outside the junction box."""
    return (lines(arm) > 0) & (skeleton > 0) & ~_box(junction)


def _midpoint(arm) -> tuple[int, int]:
    (x0, y0), (x1, y1) = arm
    return ((x0 + x1) // 2, (y0 + y1) // 2)


def _check_only_the_arm_went(before, after, arms, junction, gone: int) -> None:
    box = _box(junction)
    removed = (before > 0) & (after == 0)
    assert not ((after > 0) & (before == 0)).any(), "a delete adds no pixel"
    own = _arm_pixels(before, arms[gone], junction)
    assert own.sum() > 20
    assert not (own & (after > 0)).any(), "the picked branch is gone"
    assert not (removed & ~own & ~box).any(), "nothing outside the branch and its junction"
    for i, arm in enumerate(arms):
        if i != gone:
            others = _arm_pixels(before, arm, junction)
            assert (after[others] > 0).all(), f"arm {i} lost pixels next to the junction"
    assert components(after) == components(before) == 1, "the other branches stay one piece"


@pytest.mark.parametrize(("name", "arm"), CASES, ids=[f"{n}-arm{a}" for n, a in CASES])
def test_delete_erases_only_the_picked_branch_at_a_junction(tmp_path, name, arm):
    junction, arms = JUNCTIONS[name]
    skeleton = lines(*arms)
    layer, host = open_layer(tmp_path, skeleton)
    pick(layer, *_midpoint(arms[arm]))
    branch = np.zeros(skeleton.shape, bool)
    own = layer.draft.edit_pixels
    branch[[y for _, y in own], [x for x, _ in own]] = True

    assert key(layer, Qt.Key.Key_Delete) is True

    _check_only_the_arm_went(skeleton, layer.model.mask, arms, junction, arm)
    removed = (skeleton > 0) & (layer.model.mask == 0)
    # Strict form: besides the branch's own pixels only junction pixels may go.
    assert not (removed & ~branch & (_degree(skeleton) < 3)).any(), "another branch lost a pixel"
    assert edits(host) == 1
    assert layer.undo() is True
    assert np.array_equal(layer.model.mask, skeleton)
    layer.cancel_scheduled_render()


def test_a_click_on_the_junction_picks_one_branch_and_delete_removes_only_it(tmp_path):
    junction, arms = JUNCTIONS["plus"]  # a junction of five pixels
    skeleton = lines(*arms)
    layer, _ = open_layer(tmp_path, skeleton)
    pick(layer, *junction)

    key(layer, Qt.Key.Key_Delete)

    after = layer.model.mask
    gone = [i for i, a in enumerate(arms) if not (after[_arm_pixels(skeleton, a, junction)]).any()]
    assert len(gone) == 1, "one arm went, not an edge between two junction pixels"
    _check_only_the_arm_went(skeleton, after, arms, junction, gone[0])
    layer.cancel_scheduled_render()


@pytest.mark.parametrize("length", [2, 3, 6])
def test_deleting_a_short_spur_leaves_the_root_whole_and_no_stub(tmp_path, length):
    root = [(100, 240), (539, 240)]
    spur = [(300, 240), (300, 240 + length)]
    skeleton = lines(root, spur)
    assert tips(skeleton) == 3
    layer, _ = open_layer(tmp_path, skeleton)
    pick(layer, 300, 240 + length + 3)  # just below the spur's tip

    key(layer, Qt.Key.Key_Delete)

    after = layer.model.mask
    assert (after[240, 100:540] > 0).all(), "the root keeps every pixel"
    assert not after[241:, :].any(), "the spur is gone"
    assert components(after) == 1 and tips(after) == 2  # no stub of the spur is left


def test_an_edited_branch_keeps_the_other_branches_whole(tmp_path):
    junction, arms = JUNCTIONS["Y"]
    skeleton = lines(*arms)
    layer, host = open_layer(tmp_path, skeleton)
    pick(layer, 320, 160)
    tip = next(p for p in layer.draft.points if (p.x(), p.y()) == (320, 80))
    drag(layer, (tip.x(), tip.y()), (360, 80))  # move the branch's tip 40 px right

    assert key(layer, Qt.Key.Key_Return) is True

    after = layer.model.mask
    for arm in arms[1:]:
        assert (after[_arm_pixels(skeleton, arm, junction)] > 0).all()
    assert not after[90:200, 320].any(), "the old branch is gone"
    assert after[78:83, 357:364].any(), "the new branch reaches the moved tip"
    assert components(after) == 1, "the new branch joins the junction"
    assert edits(host) == 1
    layer.cancel_scheduled_render()


EDIT_CASES = [(n, a) for n in ("Y", "X", "plus") for a in range(len(JUNCTIONS[n][1]))]


@pytest.mark.parametrize(("name", "arm"), EDIT_CASES, ids=[f"{n}-arm{a}" for n, a in EDIT_CASES])
def test_a_moved_tip_and_enter_keep_the_other_branches_whole(tmp_path, name, arm):
    junction, arms = JUNCTIONS[name]
    skeleton = lines(*arms)
    layer, host = open_layer(tmp_path, skeleton)
    pick(layer, *_midpoint(arms[arm]))
    (jx, jy), (tx, ty) = arms[arm]
    tip = min(layer.draft.points, key=lambda p: abs(p.x() - tx) + abs(p.y() - ty))
    # Move the tip 20 px across the arm, away from the other arms' directions.
    dx, dy = (ty - jy, jx - tx)
    norm = max(abs(dx), abs(dy))
    drag(layer, (tip.x(), tip.y()), (tip.x() + 20 * dx // norm, tip.y() + 20 * dy // norm))

    assert key(layer, Qt.Key.Key_Return) is True

    after = layer.model.mask
    for i, other in enumerate(arms):
        if i != arm:
            kept = _arm_pixels(skeleton, other, junction)
            assert (after[kept] > 0).all(), f"arm {i} lost pixels next to the junction"
    assert components(after) == 1, "the new branch joins the junction"
    assert edits(host) == 1
    layer.cancel_scheduled_render()


def test_the_picked_branch_without_a_junction_is_erased_whole(tmp_path):
    skeleton = lines([(100, 240), (539, 240)], [(100, 100), (539, 100)])
    layer, _ = open_layer(tmp_path, skeleton)
    pick(layer, 300, 242)

    key(layer, Qt.Key.Key_Delete)

    after = layer.model.mask
    assert not after[235:246, :].any()
    assert np.array_equal(after[:200], skeleton[:200]), "the other line is untouched"
    layer.cancel_scheduled_render()


DIAMOND = [(400, 260), (440, 300), (400, 340), (360, 300), (400, 260)]  # a node-free ring
LINE = [(100, 100), (300, 100)]


def test_select_picks_a_ring_next_to_a_line_and_delete_removes_only_the_ring(tmp_path):
    # A ring has no node: Select could not pick it while another branch was in the image
    # (S03-06); its own pixels are the whole ring.
    skeleton = lines(LINE, DIAMOND)
    layer, host = open_layer(tmp_path, skeleton)

    pick(layer, 421, 279)  # on the ring's upper right side

    assert key(layer, Qt.Key.Key_Delete) is True
    assert np.array_equal(layer.model.mask, lines(LINE)), "the ring went, the line stayed"
    assert edits(host) == 1
    layer.cancel_scheduled_render()


# ------------------------------------------------------- real skeletons (the owner's data)
REAL_DATA = real_data_dir()
# The densest generated skeletons among the first 6000 files of the 2024 project.
REAL_SAMPLE = (
    "24EFCAM1_T019_L036_2024.07.18_093022_001_SMM_fake.png",
    "24EFCAM1_T057_L032_2024.08.07_092811_002_SMM_fake.png",
    "24EFCAM1_T026_L036_2024.07.18_114325_001_SMM_fake.png",
)


def _degree(mask: np.ndarray) -> np.ndarray:
    on = (mask > 0).astype(np.float32)
    kernel = np.ones((3, 3), np.float32)
    kernel[1, 1] = 0
    return cv2.filter2D(on, -1, kernel, borderType=cv2.BORDER_CONSTANT).astype(np.int32)


def _junction_branches(topo, deg) -> list[tuple[list[tuple[int, int]], bool]]:
    """``(own pixels, from a tip)`` of the branches with a junction at one end or both."""
    found = []
    for poly, own in zip(topo.polylines, topo.own, strict=True):
        ends = sorted(int(deg[y, x]) for x, y in (poly[0], poly[-1]))
        if own and ends[1] >= 3:
            found.append((own, ends[0] == 1))
    return found


@pytest.mark.skipif(REAL_DATA is None, reason="real 2024 EF dataset not present")
@pytest.mark.parametrize("name", REAL_SAMPLE)
def test_deleting_branches_of_real_skeletons_keeps_every_other_branch(name):
    """Every branch at a junction of a generated skeleton, deleted as the tool does (its own
    pixels, then the thinning) and undone: only its pixels and junction pixels go, and after
    a branch from a tip the skeleton stays as many pieces as it was (a branch between two
    junctions may hold two pieces together)."""
    path = Path(REAL_DATA) / "skeletons" / name
    if not path.exists():
        pytest.skip(f"{name} not in the real data")
    gray = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)
    model = SkeletonCorrectionModel()
    model.load_from_raster(gray, target_size=(640, 480))  # as Correct opens it
    before = model.mask.copy()
    deg, pieces = _degree(before), components(before)
    branches = _junction_branches(model.topology(simplify_epsilon=1.5), deg)
    assert sum(tip for _, tip in branches) >= 10

    for own, from_tip in branches:
        model.push_undo()
        model.erase_pixels(own)
        model.skeletonize()
        after = model.mask
        branch = np.zeros(before.shape, bool)
        branch[[y for _, y in own], [x for x, _ in own]] = True
        removed = (before > 0) & (after == 0)
        assert not ((after > 0) & (before == 0)).any()
        assert not (removed & ~branch & (deg < 3)).any(), "a pixel of another branch went"
        if from_tip:
            assert components(after) == pieces
        assert model.undo() is True
        assert np.array_equal(model.mask, before)

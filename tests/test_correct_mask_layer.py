"""MaskLayer: the mask as an edit layer of the Correct workspace (C2, C13).

The layer composes the Qt-free ``MaskDocument`` (P4a), the ``PaintCanvas`` painting engine
(P2) and the ``RasterLayerItem`` render item (P1a), with Brush / Eraser / Fill tools and a
brush ring. The oracle for every pixel is a direct ``PaintCanvas`` replay with today's
lookup table, canvas-to-plane and fill-source rules written out here by hand
(``mask_tracing_interface.py:702-712``). A recording ``FakeHost`` stands in for the
workspace, so the signal rules of C2 are checked call by call.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QPoint, QRect, QRectF, Qt
from PyQt6.QtGui import (
    QImage,
    QKeyEvent,
    QPainter,
)

from app.gui.canvas_layers.raster import RasterLayerItem
from app.gui.correct import mask_tools
from app.gui.correct.layer import (
    LayerStatus,
    LayerTool,
    Notice,
    SaveOutcome,
    ToolSpec,
    missing_members,
)
from app.gui.correct.mask_layer import MaskLayer
from app.gui.paint_canvas import pixmap_to_argb_array
from app.gui.widgets import tokens
from app.masks import write_mask_file
from tests.correct_fakes import FakeHost
from tests.correct_mask_layer_support import (
    CLEARED,
    FILL_MISS,
    MASK_COLOUR,
    NO_MODS,
    NOT_SHOWN_SIZE_TEXT,
    PREDICTION,
    STEM,
    H,
    Replay,
    W,
    boxes_plane,
    canvas_shows_the_plane,
    click,
    count,
    gt_file,
    image_path,
    mask_file,
    open_layer,
    plane_of,
    read_rgba,
    stroke,
)
from tests.correct_mask_layer_support import scene as scene  # fixture

pytestmark = pytest.mark.usefixtures("no_prompts")


# --------------------------------------------------------------------------- contract
def test_the_layer_and_its_tools_implement_the_protocols(scene, tmp_path):
    layer, _ = open_layer(scene, tmp_path)

    assert missing_members(layer) == []
    assert missing_members(layer, mask_tools.MaskToolHost) == []
    for cls in mask_tools.TOOL_CLASSES:
        assert missing_members(cls(layer), LayerTool) == [], cls.__name__


def test_identity_and_tools_are_the_contracts(scene):
    layer = MaskLayer(FakeHost(), scene)

    assert (layer.layer_id, layer.noun, layer.color) == ("mask", "mask", tokens.MASK)
    assert layer.tools == (
        ToolSpec("brush", "brush", "Brush", "B", "hold B + scroll to resize", True),
        ToolSpec("eraser", "erase", "Eraser", "E", "[ and ] resize", True),
        ToolSpec(
            "fill", "fill", "Fill a closed outline", "F", "click inside a closed outline", False
        ),
    )
    assert tuple(cls.spec for cls in mask_tools.TOOL_CLASSES) == layer.tools
    assert layer.current_tool == "brush"
    assert layer.size() == 8
    assert layer.opacity() == pytest.approx(0.60)


def test_the_item_is_a_raster_layer_at_z_one_in_the_scene(scene):
    layer = MaskLayer(FakeHost(), scene)

    assert isinstance(layer.item, RasterLayerItem)
    assert layer.item.zValue() == 1.0
    assert layer.item.scene() is scene
    assert layer.item.opacity() == pytest.approx(0.60)


def test_an_unknown_tool_is_refused(scene):
    layer = MaskLayer(FakeHost(), scene)

    with pytest.raises(ValueError):
        layer.set_tool("polyline")
    assert layer.current_tool == "brush"


# --------------------------------------------------------------------------- pixels
def test_the_paint_rules_are_todays():
    # _plane_lut: 0 transparent, every other value the opaque mask colour (B, G, R, A)
    lut = mask_tools.mask_lut()
    assert lut.shape == (256, 4) and lut.dtype == np.uint8
    assert not lut[0].any()
    assert (lut[1:] == (MASK_COLOUR.blue(), MASK_COLOUR.green(), MASK_COLOUR.red(), 255)).all()
    # _argb_to_plane: any alpha is 1, colour without alpha is 0; strided input too
    argb = np.zeros((3, 256, 4), np.uint8)
    argb[0, :, 3] = np.arange(256)
    argb[1, :, :3] = 200
    argb[2] = argb[0, ::-1]
    expected = np.zeros((3, 256), np.uint8)
    expected[0, 1:] = 1
    expected[2, :-1] = 1
    assert np.array_equal(mask_tools.argb_to_plane(argb, (0, 0, 256, 3)), expected)
    assert mask_tools.argb_to_plane(argb, (0, 0, 256, 3)).dtype == np.uint8
    assert np.array_equal(mask_tools.argb_to_plane(argb[:, ::2], (0, 0, 128, 3)), expected[:, ::2])
    # _fill_source: the mask as a 0/255 outline image, filled with 1
    plane = np.array([[0, 1, 2], [0, 0, 1]], np.uint8)
    outline, value = mask_tools.fill_source(plane)
    assert value == 1
    assert np.array_equal(outline, [[0, 255, 255], [0, 0, 255]])


def test_brush_eraser_and_fill_through_the_layer_equal_a_direct_canvas_replay(scene, tmp_path):
    start = boxes_plane()
    layer, host = open_layer(scene, tmp_path, start)
    oracle = Replay(start)
    square = [(20, 60), (50, 60), (50, 100), (20, 100), (20, 60)]
    steps = [
        ("brush", 6, "stroke", [(20, 20), (60, 30), (100, 70)]),
        ("brush", 3, "stroke", square),
        ("fill", None, "click", (35, 80)),  # inside the closed square
        ("eraser", 5, "stroke", [(10, 10), (150, 110)]),
        ("fill", None, "click", (150, 5)),  # outside every closed outline
        ("brush", 1, "click", (140, 20)),
        ("eraser", 9, "click", (90, 60)),
    ]

    for tool, size, gesture, where in steps:
        layer.set_tool(tool)
        if size is not None:
            layer.set_size(size)
        if gesture == "stroke":
            stroke(layer, where)
            oracle.stroke(where, size, erase=tool == "eraser")
        elif tool == "fill":
            click(layer, where)
            oracle.fill(where)
        else:
            click(layer, where)
            oracle.stroke([where], size, erase=tool == "eraser")

        assert np.array_equal(plane_of(layer), oracle.plane), (tool, gesture, where)
        assert canvas_shows_the_plane(layer), (tool, gesture, where)
    assert np.array_equal(
        pixmap_to_argb_array(layer.canvas.image), pixmap_to_argb_array(oracle.canvas.image)
    )
    assert len(layer.document.history.undo_stack) == 6  # the missed fill recorded nothing


def test_a_fill_outside_every_outline_says_so_and_records_nothing(scene, tmp_path):
    layer, host = open_layer(scene, tmp_path, boxes_plane())
    layer.set_tool("fill")
    host.calls.clear()

    click(layer, (150, 110))

    assert host.calls == [("notify", FILL_MISS, "info")]
    assert MaskLayer.FILL_MISS_NOTICE == FILL_MISS
    assert not layer.can_undo()
    assert np.array_equal(plane_of(layer), boxes_plane())


def test_a_fill_inside_a_closed_outline_is_one_action(scene, tmp_path):
    ring = np.zeros((H, W), np.uint8)
    ring[20:60, 30:80] = 1
    ring[22:58, 32:78] = 0
    layer, host = open_layer(scene, tmp_path, ring)
    layer.set_tool("fill")

    click(layer, (50, 40))

    expected = ring.copy()
    expected[20:60, 30:80] = 1
    assert np.array_equal(plane_of(layer), expected)
    assert count(host, "layer_edited") == 1
    assert count(host, "notify") == 0


def _shifted(plane: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """``plane`` moved by ``(dx, dy)`` pixels; what leaves the frame is gone, 0 comes in."""
    out = np.zeros_like(plane)
    h, w = plane.shape
    src = plane[max(0, -dy) : h - max(0, dy), max(0, -dx) : w - max(0, dx)]
    out[max(0, dy) : max(0, dy) + src.shape[0], max(0, dx) : max(0, dx) + src.shape[1]] = src
    return out


@pytest.mark.parametrize(
    "edge",
    [(0, 0), (W - 1, H - 1), (40, 0), (0, 50)],
    ids=["top-left", "bottom-right", "top", "left"],
)
def test_a_dab_at_the_image_edge_is_the_in_image_part_of_the_same_dab(scene, tmp_path, edge):
    """Oracle: the same dab painted in the interior, moved to the edge and clipped by the frame.

    Odd sizes only: Qt's aliased ellipse of an even diameter places its tie pixels by the
    position on the canvas, edge or not (a 2 px dab is 1 px at x = 80 and 2 px at x = 159),
    so moving it is not the identity there; those pixels are pinned as they are by the goldens.
    """
    layer, _ = open_layer(scene, tmp_path)
    centre = (W // 2, H // 2)
    for size in (1, 3, 9):
        layer.set_size(size)
        click(layer, centre)
        interior = plane_of(layer).copy()
        assert layer.undo()
        click(layer, edge)

        expected = _shifted(interior, edge[0] - centre[0], edge[1] - centre[1])
        assert np.array_equal(plane_of(layer), expected), (size, edge)
        assert canvas_shows_the_plane(layer), (size, edge)
        assert layer.undo() and not plane_of(layer).any(), (size, edge)  # one action, all of it


def test_a_fill_of_an_outline_on_the_image_edge(scene, tmp_path):
    """A closed outline whose left side is the frame's first column fills, border included;
    one whose open side is the frame is not closed: nothing is filled, the user is told."""
    closed = np.zeros((H, W), np.uint8)
    closed[10:40, 0:30] = 1
    closed[12:38, 2:28] = 0
    layer, host = open_layer(scene, tmp_path, closed)
    layer.set_tool("fill")

    click(layer, (15, 25))

    expected = closed.copy()
    expected[10:40, 0:30] = 1
    assert np.array_equal(plane_of(layer), expected)

    open_at_edge = np.zeros((H, W), np.uint8)
    open_at_edge[10:40, 0:30] = 1
    open_at_edge[12:38, 0:28] = 0
    other, host = open_layer(scene, tmp_path / "open", open_at_edge)
    other.set_tool("fill")
    host.calls.clear()

    click(other, (10, 25))

    assert np.array_equal(plane_of(other), open_at_edge)
    assert host.calls == [("notify", FILL_MISS, "info")] and not other.can_undo()


def test_clear_is_one_undoable_action_with_todays_notice(scene, tmp_path):
    layer, host = open_layer(scene, tmp_path, boxes_plane())

    assert layer.clear() is True

    assert not plane_of(layer).any()
    assert ("notify", CLEARED, "info") in host.calls
    assert MaskLayer.CLEARED_NOTICE == CLEARED
    assert count(host, "layer_edited") == 1
    assert canvas_shows_the_plane(layer)
    assert layer.undo() is True
    assert np.array_equal(plane_of(layer), boxes_plane())
    assert canvas_shows_the_plane(layer)


def test_clear_without_an_open_mask_does_nothing(scene):
    host = FakeHost()
    layer = MaskLayer(host, scene)

    assert layer.clear() is False
    assert host.calls == []


def test_a_double_click_paints_one_dab_and_one_history_entry(scene, tmp_path):
    layer, _ = open_layer(scene, tmp_path)
    layer.set_size(9)
    p = QPoint(70, 50)

    layer.press(p, NO_MODS)
    layer.release(p)
    layer.double_click(p)  # Qt sends the second press of a double-click as this
    layer.release(p)

    oracle = Replay(np.zeros((H, W), np.uint8))
    oracle.stroke([(70, 50)], 9)
    assert np.array_equal(plane_of(layer), oracle.plane)
    assert len(layer.document.history.undo_stack) == 1


def test_a_press_during_a_stroke_commits_that_stroke_first(scene, tmp_path):
    # A lost release (e.g. taken by a modal box): the next press starts a new action.
    layer, host = open_layer(scene, tmp_path)
    layer.set_size(4)

    layer.press(QPoint(10, 10), NO_MODS)
    layer.move(QPoint(40, 10), True)
    stroke(layer, [(10, 60), (40, 60)])

    oracle = Replay(np.zeros((H, W), np.uint8))
    oracle.stroke([(10, 10), (40, 10)], 4)
    oracle.stroke([(10, 60), (40, 60)], 4)
    assert np.array_equal(plane_of(layer), oracle.plane)
    assert count(host, "layer_edited") == 2
    assert len(layer.document.history.undo_stack) == 2


def test_the_item_hears_each_segment_rect_and_the_stroke_never_copies_the_canvas(
    scene, tmp_path, monkeypatch
):
    rects: list[QRect] = []
    original = RasterLayerItem.canvas_changed

    def spy(item, rect):
        rects.append(QRect(rect))
        original(item, rect)

    monkeypatch.setattr(RasterLayerItem, "canvas_changed", spy)
    layer, _ = open_layer(scene, tmp_path)
    layer.set_size(6)
    serial = layer.canvas.image.cacheKey() >> 32
    target = QImage(W, H, QImage.Format.Format_ARGB32_Premultiplied)

    layer.press(QPoint(10, 10), NO_MODS)
    layer.move(QPoint(20, 10), True)
    painter = QPainter(target)  # the scene paints between segments, as a view would
    scene.render(painter, QRectF(0, 0, W, H), QRectF(0, 0, W, H))
    painter.end()
    layer.move(QPoint(20, 30), True)
    layer.release(QPoint(20, 30))

    # today's dirty rects: the segment's box padded by r + 2 = 5, aligned outwards
    assert rects == [QRect(5, 5, 11, 11), QRect(5, 5, 21, 11), QRect(15, 5, 11, 31)]
    assert layer.canvas.image.cacheKey() >> 32 == serial


# --------------------------------------------------------------------------- signals
def test_layer_edited_fires_once_per_stroke_fill_and_clear_and_never_otherwise(scene, tmp_path):
    ring = np.zeros((H, W), np.uint8)
    ring[20:60, 30:80] = 1
    ring[22:58, 32:78] = 0
    layer, host = open_layer(scene, tmp_path, ring)

    stroke(layer, [(100, 90), (140, 100)])
    assert count(host, "layer_edited") == 1
    layer.set_tool("fill")
    click(layer, (50, 40))
    assert count(host, "layer_edited") == 2
    layer.clear()
    assert count(host, "layer_edited") == 3
    assert host.calls.count(("layer_edited", "mask")) == 3

    layer.undo()
    layer.redo()
    layer.save()
    layer.revert()
    layer.open(image_path(tmp_path), (W, H), None)
    layer.close()
    assert count(host, "layer_edited") == 3


def test_layer_changed_fires_when_a_stroke_ends_never_per_segment(scene, tmp_path):
    layer, host = open_layer(scene, tmp_path)

    layer.press(QPoint(10, 10), NO_MODS)
    for x in range(12, 60, 4):
        layer.move(QPoint(x, 20), True)
    assert count(host, "layer_changed") == 0

    layer.release(QPoint(60, 20))
    assert count(host, "layer_changed") >= 1
    assert host.calls.index(("layer_edited", "mask")) < host.calls.index(("layer_changed", "mask"))


@pytest.mark.parametrize(
    "action",
    ["undo", "redo", "save", "revert", "set_tool", "open", "close", "step_size"],
)
def test_layer_changed_fires_after_each_ui_visible_change(scene, tmp_path, action):
    layer, host = open_layer(scene, tmp_path)
    stroke(layer, [(10, 10), (60, 40)])
    if action == "redo":
        layer.undo()
    host.calls.clear()

    {
        "undo": lambda: layer.undo(),
        "redo": lambda: layer.redo(),
        "save": lambda: layer.save(),
        "revert": lambda: layer.revert(),
        "set_tool": lambda: layer.set_tool("eraser"),
        "open": lambda: layer.open(image_path(tmp_path), (W, H), None),
        "close": lambda: layer.close(),
        "step_size": lambda: layer.step_size(1),
    }[action]()

    assert count(host, "layer_changed") >= 1
    assert count(host, "layer_edited") == 0


def test_no_change_fires_nothing(scene, tmp_path):
    layer, host = open_layer(scene, tmp_path)

    layer.set_tool("brush")  # already the tool
    layer.set_size(layer.size())
    layer.drop_redo()  # nothing to drop
    layer.finish_gesture()  # no gesture
    layer.hover(QPoint(5, 5))
    layer.move(QPoint(6, 6), False)

    assert host.calls == []


def test_drop_redo_forgets_the_redo_branch_only(scene, tmp_path):
    layer, host = open_layer(scene, tmp_path)
    stroke(layer, [(10, 10), (40, 10)])
    stroke(layer, [(10, 50), (40, 50)])
    layer.undo()
    host.calls.clear()

    layer.drop_redo()

    assert (layer.can_undo(), layer.can_redo()) == (True, False)
    assert host.calls == [("layer_changed", "mask")]
    assert layer.redo() is False


# --------------------------------------------------------------------------- history
def test_undo_and_redo_step_one_entry_and_report_it(scene, tmp_path):
    layer, _ = open_layer(scene, tmp_path)
    assert (layer.undo(), layer.redo()) == (False, False)
    stroke(layer, [(10, 10), (60, 40)])
    painted = plane_of(layer).copy()

    assert layer.can_undo() is True
    assert layer.undo() is True
    assert not plane_of(layer).any()
    assert canvas_shows_the_plane(layer)
    assert layer.undo() is False  # empty: the journal may drop its entry
    assert layer.redo() is True
    assert np.array_equal(plane_of(layer), painted)
    assert canvas_shows_the_plane(layer)
    assert layer.redo() is False


def test_redo_mid_stroke_commits_the_stroke_which_ends_the_redo_branch(scene, tmp_path):
    layer, host = open_layer(scene, tmp_path)
    stroke(layer, [(10, 10), (40, 10)])
    layer.undo()
    assert layer.can_redo()
    host.calls.clear()
    layer.press(QPoint(10, 50), NO_MODS)
    layer.move(QPoint(40, 50), True)

    assert layer.redo() is False  # nothing left to redo: a new action ended the branch

    assert layer.gesture_active() is False
    assert count(host, "layer_edited") == 1
    assert (layer.can_undo(), layer.can_redo()) == (True, False)


def test_undo_back_to_the_saved_mask_reads_clean(scene, tmp_path):
    layer, _ = open_layer(scene, tmp_path, boxes_plane())
    assert layer.is_dirty() is False
    assert layer.status() == PREDICTION

    stroke(layer, [(10, 100), (60, 110)])
    assert layer.is_dirty() is True
    assert layer.status() == LayerStatus("dirty")

    layer.undo()
    assert layer.is_dirty() is False
    assert layer.status() == PREDICTION


def test_key_consumes_nothing_and_there_is_no_draft_to_pop(scene, tmp_path):
    layer, _ = open_layer(scene, tmp_path)
    enter = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, NO_MODS)

    assert layer.key(enter) is False
    assert layer.undo_in_gesture() is False


# --------------------------------------------------------------------------- documents
def test_open_hands_back_the_documents_notices(scene, tmp_path):
    write_mask_file(gt_file(tmp_path), np.ones((H + 1, W), bool))  # a GT mask of another size
    layer = MaskLayer(FakeHost(), scene)

    notices = layer.open(image_path(tmp_path), (W, H), None)

    assert notices == [Notice(NOT_SHOWN_SIZE_TEXT, "warn")]
    assert layer.is_open() and not plane_of(layer).any()


def test_unloaded_label_names_the_gt_file_that_was_not_shown(scene, tmp_path):
    path = gt_file(tmp_path)
    write_mask_file(path, np.ones((H // 2, W // 2), bool))  # a GT mask of another size
    layer = MaskLayer(FakeHost(), scene)
    layer.open(image_path(tmp_path), (W, H), None)
    assert layer.unloaded_label() == f"gt/mask/{STEM}.png"  # a save would replace it

    stroke(layer, [(20, 20), (40, 40)])

    assert layer.unloaded_label() == f"gt/mask/{STEM}.png"
    assert layer.status() == LayerStatus("dirty")
    os.remove(path)
    assert layer.unloaded_label() == ""  # gone since the open: nothing left to replace


def test_an_unshown_prediction_is_never_named_for_replacing(scene, tmp_path):
    os.makedirs(os.path.dirname(mask_file(tmp_path)))
    with open(mask_file(tmp_path), "wb") as fh:
        fh.write(b"not a png")  # an unreadable prediction
    layer = MaskLayer(FakeHost(), scene)
    layer.open(image_path(tmp_path), (W, H), None)

    stroke(layer, [(20, 20), (40, 40)])

    assert layer.unloaded_label() == ""  # a save writes gt/mask, never the prediction


def test_a_save_with_nothing_painted_keeps_the_unshown_file(scene, tmp_path):
    path = gt_file(tmp_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_mask_file(path, np.ones((H // 2, W // 2), bool))
    with open(path, "rb") as fh:
        before = fh.read()
    layer = MaskLayer(FakeHost(), scene)
    layer.open(image_path(tmp_path), (W, H), None)

    outcome = layer.save()

    assert outcome == SaveOutcome(
        "mask", ok=True, message="Nothing to save: the saved mask was kept", kind="info"
    )
    with open(path, "rb") as fh:
        assert fh.read() == before
    assert layer.status() == LayerStatus("error", "Mask not shown", NOT_SHOWN_SIZE_TEXT)


def test_a_layer_with_its_mask_shown_names_no_unloaded_file(scene, tmp_path):
    layer, _ = open_layer(scene, tmp_path, boxes_plane())

    stroke(layer, [(20, 20), (40, 40)])

    assert layer.unloaded_label() == ""


def test_open_shows_the_saved_mask(scene, tmp_path):
    layer, _ = open_layer(scene, tmp_path, boxes_plane())

    assert np.array_equal(plane_of(layer), boxes_plane())
    assert canvas_shows_the_plane(layer)
    assert layer.item.boundingRect() == QRectF(0, 0, W, H)


def test_close_forgets_the_document_and_empties_the_item(scene, tmp_path):
    layer, _ = open_layer(scene, tmp_path, boxes_plane())

    layer.close()

    assert layer.is_open() is False
    assert layer.canvas.image is None
    assert layer.item.boundingRect().isEmpty()


def test_revert_goes_back_to_the_saved_mask(scene, tmp_path):
    layer, _ = open_layer(scene, tmp_path, boxes_plane())
    stroke(layer, [(10, 100), (60, 110)])

    assert layer.revert() == []

    assert np.array_equal(plane_of(layer), boxes_plane())
    assert canvas_shows_the_plane(layer)
    assert (layer.is_dirty(), layer.can_undo()) == (False, False)


def test_save_writes_the_mask_and_reports_its_path(scene, tmp_path):
    layer, _ = open_layer(scene, tmp_path)
    stroke(layer, [(10, 10), (60, 40)])

    outcome = layer.save()

    assert outcome == SaveOutcome("mask", ok=True, path=outcome.path)
    assert os.path.normcase(os.path.normpath(outcome.path)) == os.path.normcase(
        os.path.normpath(gt_file(tmp_path))
    )
    assert np.array_equal(read_rgba(gt_file(tmp_path))[:, :, 3] == 255, plane_of(layer) > 0)
    assert layer.is_dirty() is False
    assert not os.path.exists(mask_file(tmp_path))  # no prediction is ever written


def test_saving_an_emptied_mask_writes_an_empty_gt_mask_and_says_so(scene, tmp_path):  # O9
    layer, _ = open_layer(scene, tmp_path, boxes_plane())
    layer.clear()

    outcome = layer.save()

    name = f"{STEM}.jpg"
    assert outcome == SaveOutcome(
        "mask",
        ok=True,
        path=gt_file(tmp_path),
        message=f"Empty mask saved as GT for {name}: root area 0",
        kind="info",
    )
    assert not read_rgba(gt_file(tmp_path)).any() and os.path.exists(mask_file(tmp_path))


def test_saving_the_prediction_again_removes_the_gt_file_and_says_so(scene, tmp_path):
    write_mask_file(gt_file(tmp_path), np.zeros((H, W), bool))
    layer, host = open_layer(scene, tmp_path, boxes_plane())
    assert layer.load_prediction() is True
    assert np.array_equal(plane_of(layer), boxes_plane()) and layer.can_undo()
    assert host.calls.count(("layer_edited", "mask")) == 1

    outcome = layer.save()

    message = f"Mask equals the prediction: removed gt/mask/{STEM}.png"
    assert outcome == SaveOutcome("mask", ok=True, cleared=True, message=message, kind="info")
    assert not os.path.exists(gt_file(tmp_path)) and layer.status() == PREDICTION


def test_saving_an_empty_mask_without_a_file_writes_and_says_nothing(scene, tmp_path):
    layer, _ = open_layer(scene, tmp_path)

    assert layer.save() == SaveOutcome("mask", ok=True)
    assert not os.path.exists(gt_file(tmp_path))  # O2: no edits, no GT file
    assert not os.path.exists(mask_file(tmp_path))


def _raise(message: str):
    def raiser(*_args, **_kwargs):
        raise OSError(message)

    return raiser


def test_a_failed_write_is_a_danger_outcome_with_todays_text(scene, tmp_path, monkeypatch):
    layer, _ = open_layer(scene, tmp_path)
    stroke(layer, [(10, 10), (60, 40)])
    monkeypatch.setattr(os, "replace", _raise("disk full"))

    outcome = layer.save()

    target = os.path.normpath(gt_file(tmp_path))
    assert outcome == SaveOutcome(
        "mask",
        ok=False,
        path=None,  # nothing was written
        message=f"Could not save mask to {target}: disk full",
        kind="danger",
    )
    assert layer.status() == LayerStatus("error", "Not saved", "disk full")
    assert layer.is_dirty() is True


def test_a_failed_removal_is_a_danger_outcome_with_todays_text(scene, tmp_path, monkeypatch):
    write_mask_file(gt_file(tmp_path), np.zeros((H, W), bool))
    layer, _ = open_layer(scene, tmp_path, boxes_plane())
    layer.load_prediction()
    monkeypatch.setattr(os, "remove", _raise("in use"))

    outcome = layer.save()

    message = f"Could not remove gt/mask/{STEM}.png: in use"
    assert outcome == SaveOutcome("mask", ok=False, path=None, message=message, kind="danger")
    assert layer.is_dirty() is True


def test_a_one_pixel_click_paints_its_pixel_as_one_undo_step(scene, tmp_path):
    layer, _ = open_layer(scene, tmp_path)
    layer.set_size(1)
    click(layer, (70, 50))
    assert plane_of(layer).sum() == 1
    assert plane_of(layer)[50, 70] == 1
    assert len(layer.document.history.undo_stack) == 1
    assert layer.document.is_dirty()
    layer.undo()
    assert plane_of(layer).sum() == 0

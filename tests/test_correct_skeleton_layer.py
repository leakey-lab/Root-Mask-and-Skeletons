"""SkeletonLayer: the skeleton of the Correct workspace as an edit layer (plan P5b, C14).

The layer-level port of ``test_skeleton_editor_lifecycle.py`` and
``test_skeleton_editor_tools.py`` (both go with the old editor at the swap), plus the
Correct rules: one ``layer_edited`` per action and no ``layer_changed`` per drag frame,
undo / redo steps that drop a polyline draft, the mid-gesture rule, the reference-layer
view (dimmed, markers and previews hidden), the z-order under the decorations and the
item update through ``take_changed_box``. Offscreen, with a recording ``FakeHost``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, QRectF, Qt

from app.gui import skeleton_tools
from app.gui.canvas_layers.skeleton import SkeletonLayerItem
from app.gui.correct.layer import (
    LayerStatus,
    Notice,
    SaveOutcome,
    ToolSpec,
    missing_members,
)
from app.gui.correct.skeleton_layer import SkeletonLayer
from app.gui.key_text import platform_keys
from app.gui.widgets import tokens
from tests.correct_skeleton_layer_support import (
    EMPTY_HINT,
    GAP,
    LINE,
    NO_MOD,
    H,
    W,
    _click,
    _commit,
    _count,
    _drag,
    _key,
    _open,
    _write_raster,
)
from tests.correct_skeleton_layer_support import gap_layer as gap_layer  # fixture
from tests.correct_skeleton_layer_support import host as host  # fixture
from tests.correct_skeleton_layer_support import layer as layer  # fixture
from tests.correct_skeleton_layer_support import project as project  # fixture
from tests.correct_skeleton_layer_support import scene as scene  # fixture
from tests.correct_skeleton_layer_support import tool_calls as tool_calls  # fixture

pytestmark = pytest.mark.usefixtures("no_prompts")


# ---------------------------------------------------------------------------- contract
def test_the_layer_implements_edit_layer_and_the_tool_host(layer):
    assert missing_members(layer) == []
    assert missing_members(layer, skeleton_tools.ToolHost) == []
    assert (layer.layer_id, layer.noun, layer.color) == ("skeleton", "skeleton", tokens.SKEL)


def test_the_rail_tools_are_the_skeleton_tools_and_only_the_eraser_is_sized(layer):
    assert SkeletonLayer.tools == (
        ToolSpec("select", "cursor", "Select", "V", "pick a branch, Delete removes it", False),
        ToolSpec("eraser", "erase", "Eraser", "E", "[ and ] resize", True),
        ToolSpec("polyline", "polyline", "Polyline", "P", "Enter finish · Esc cancel", False),
        ToolSpec("connect", "link", "Connect endpoints", "C", "drag from an endpoint", False),
    )
    for spec in SkeletonLayer.tools:
        layer.set_tool(spec.tool_id)
        assert layer.current_tool == spec.tool_id
        assert (layer.size_spec() is not None) == spec.sized


def test_the_eraser_size_spec_is_the_diameter_range(layer):
    layer.set_tool("eraser")
    spec = layer.size_spec()
    assert (spec.minimum, spec.maximum) == (1, 80)


def test_the_item_sits_on_the_scene_below_every_decoration(layer, scene):
    assert isinstance(layer.item, SkeletonLayerItem)
    assert layer.item.scene() is scene
    assert layer.item.zValue() == 1.5
    deco = layer.decorations
    decorations = (
        deco.polyline_item,
        deco.handles_item,
        deco.endpoints_item,
        deco.highlight_item,
        deco.connect_item,
    )
    assert all(item.zValue() > layer.item.zValue() for item in decorations)
    assert layer.eraser_ring.inner.zValue() > max(item.zValue() for item in decorations)


# ---------------------------------------------------------------------- document glue
def test_open_shows_the_prediction_clean(host, scene, project):
    root, image = project
    _write_raster(root, "skeletons/a_fake.png", LINE)
    layer = SkeletonLayer(host, scene)

    assert layer.open(image, (W, H), root) == []

    assert layer.is_open() and not layer.is_dirty()
    assert layer.document.source == "prediction"
    assert layer.document.model.mask[240, 100:540].all()
    assert layer.item.boundingRect() == QRectF(0, 0, W, H)
    assert layer.status() == LayerStatus("none", "Prediction", "Not corrected yet")
    assert host.calls == [("layer_changed", "skeleton")]  # once, even for a new layer


def test_the_resolver_reaches_the_document(host, scene, project):
    root, image = project
    first = _write_raster(root, "out/a_fake.png", [(50, 150, 30)])
    layer = SkeletonLayer(host, scene, resolver=lambda _image: first)
    layer.open(image, (W, H), root)
    assert layer.document.model.mask[30, 50:150].all()

    second = _write_raster(root, "out2/a_fake.png", [(50, 150, 60)])
    layer.set_resolver(lambda _image: second)
    layer.revert()

    assert layer.document.model.mask[60, 50:150].all()
    assert not layer.document.model.mask[30].any()


def test_an_unreadable_skeleton_opens_empty_with_a_danger_notice(host, scene, project):
    root, image = project
    bad = os.path.join(root, "skeletons", "a_fake.png")
    os.makedirs(os.path.dirname(bad))
    with open(bad, "wb") as fh:
        fh.write(b"not an image")
    layer = SkeletonLayer(host, scene)

    assert layer.open(image, (W, H), root) == [
        Notice("Could not read skeleton a_fake.png", "danger")
    ]
    assert not layer.document.model.mask.any()


def test_hints_are_todays_including_the_empty_skeleton_hint(host, scene, project):
    layer = SkeletonLayer(host, scene)
    assert layer.tool_hint() == ("Select an image to correct its skeleton.", "")

    root, image = project
    layer.open(image, (W, H), root)
    assert layer.tool_hint() == EMPTY_HINT
    assert layer.status() == LayerStatus("none", "No skeleton")

    layer.set_tool("polyline")
    assert layer.tool_hint()[0] == (
        platform_keys("Polyline · click to place points · Ctrl+click a segment adds a bend")
    )


def test_the_hint_follows_the_current_tool(gap_layer):
    gap_layer.set_tool("eraser")
    assert gap_layer.tool_hint()[0] == "Eraser · drag over the skeleton to erase it"

    gap_layer.set_tool("connect")
    assert gap_layer.tool_hint()[0] == "Connect · drag from an endpoint (orange dot)"
    gap_layer.press(QPoint(199, 100), NO_MOD)
    assert gap_layer.tool_hint()[0] == "Connect · release on the point to join"


def test_close_empties_the_layer(layer):
    layer.set_tool("polyline")
    _click(layer, 50, 50)

    layer.close()

    assert not layer.is_open() and not layer.is_dirty()
    assert layer.item.boundingRect().isEmpty()
    assert layer.file_stamps() == ()
    assert not layer.can_undo()
    assert layer.draft.points == []
    assert layer.decorations.endpoints_item.path().isEmpty()


def test_revert_rereads_the_start_file_and_clears_the_history(layer):
    start = layer.document.model.mask.copy()
    _commit(layer, (50, 50), (200, 100))
    assert layer.is_dirty()

    assert layer.revert() == []

    assert np.array_equal(layer.document.model.mask, start)
    assert not layer.is_dirty() and not layer.can_undo()


def test_save_writes_the_gt_skeleton_and_reports_its_path(layer, host, project):
    root, _image = project
    _commit(layer, (50, 50), (200, 100))
    host.calls.clear()

    outcome = layer.save()

    expected = os.path.join(root, "gt", "skeletons", "a.png")
    assert outcome.ok and os.path.isfile(expected)
    assert os.path.normcase(os.path.abspath(outcome.path)) == os.path.normcase(expected)
    assert outcome == SaveOutcome("skeleton", True, path=outcome.path)
    assert not layer.is_dirty()
    assert layer.status().state == "clean"
    assert ("layer_changed", "skeleton") in host.calls


def test_a_failed_save_names_the_file_keeps_the_old_one_and_stays_dirty(
    layer, project, monkeypatch
):
    root, _image = project
    old = _write_raster(root, "gt/skeletons/a.png", [(10, 20, 5)])
    before = Path(old).read_bytes()
    _commit(layer, (50, 50), (200, 100))

    def fail(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", fail)
    outcome = layer.save()
    monkeypatch.undo()

    assert outcome == SaveOutcome(
        "skeleton", False, message="Could not save gt/skeletons/a.png: disk full", kind="danger"
    )
    assert Path(old).read_bytes() == before
    assert not [f for f in os.listdir(os.path.dirname(old)) if f.endswith(".tmp")]
    assert layer.is_dirty()
    assert layer.status().state == "error"


def test_load_file_replaces_the_skeleton_and_forgets_its_history(layer, project):
    root, _image = project
    other = _write_raster(root, "elsewhere/x.png", [(30, 90, 400)])
    _commit(layer, (50, 50), (200, 100))

    assert layer.load_file(other) == []

    mask = layer.document.model.mask
    assert mask[400, 30:90].all() and not mask[240].any()
    assert not layer.can_undo() and layer.is_dirty()  # unsaved until saved (E1)
    assert layer.document.source == "loaded" and layer.document.last_load_ok is True


@pytest.mark.parametrize("step", ["close", "revert", "reopen", "load"])
def test_a_polyline_draft_dropped_by_a_reload_is_announced_once(layer, host, project, step):
    root, image = project
    other = _write_raster(root, "elsewhere/x.png", [(30, 90, 400)])
    actions = {
        "close": layer.close,
        "revert": layer.revert,
        "reopen": lambda: layer.open(image, (W, H), root),
        "load": lambda: layer.load_file(other),
    }
    layer.set_tool("polyline")
    _click(layer, 60, 300)
    host.calls.clear()

    actions[step]()
    actions[step]()  # nothing left to drop: no second notice

    assert host.calls.count(("notify", SkeletonLayer.DRAFT_DISCARDED_NOTICE, "info")) == 1
    assert layer.draft.points == []


def test_an_unreadable_file_to_load_keeps_the_history_and_the_draft(layer, project):
    root, _image = project
    bad = os.path.join(root, "broken.png")
    Path(bad).write_bytes(b"not an image")
    _commit(layer, (50, 50), (200, 100))
    _click(layer, 60, 300)
    _click(layer, 160, 300)

    layer.load_file(bad)

    assert layer.can_undo() and len(layer.draft.points) == 2
    assert layer.current_tool == "polyline"


def test_dialog_start_is_the_generated_file(layer, host, scene, project):
    root, _image = project
    generated = os.path.join(root, "skeletons", "a_fake.png")
    assert os.path.normcase(os.path.abspath(layer.dialog_start())) == os.path.normcase(generated)
    assert SkeletonLayer(host, scene).dialog_start() == ""


def test_a_closed_layer_ignores_input(host, scene):
    layer = SkeletonLayer(host, scene)
    _click(layer, 10, 10)
    layer.move(QPoint(20, 20), True)
    layer.double_click(QPoint(20, 20))

    assert layer.key(_key(Qt.Key.Key_Delete)) is False
    assert layer.clear() is False
    assert not layer.can_undo() and layer.undo() is False
    assert _count(host, "layer_edited") == 0


# ------------------------------------------------------------------------ host signals
@pytest.mark.parametrize("action", ["polyline", "eraser", "connect", "delete", "clear"])
def test_each_action_is_reported_once_as_layer_edited(host, scene, project, action):
    layer = _open(host, scene, project, [*LINE, *GAP])
    if action == "polyline":
        _commit(layer, (50, 300), (200, 360))
    elif action == "eraser":
        layer.set_tool("eraser")
        _drag(layer, [(120, 240), (160, 240), (200, 240)])
    elif action == "connect":
        layer.set_tool("connect")
        _drag(layer, [(199, 100), (260, 100)])
    elif action == "delete":
        layer.set_tool("select")
        _click(layer, 300, 242)
        layer.key(_key(Qt.Key.Key_Delete))
    else:
        layer.clear()

    edits = [call for call in host.calls if call[0] == "layer_edited"]
    assert edits == [("layer_edited", "skeleton")]
    layer.cancel_scheduled_render()


def test_history_steps_and_document_calls_are_not_edits(layer, host, project):
    root, image = project
    _commit(layer, (50, 50), (200, 100))
    host.calls.clear()

    layer.undo()
    layer.redo()
    layer.save()
    layer.revert()
    layer.open(image, (W, H), root)
    layer.close()

    assert _count(host, "layer_edited") == 0


@pytest.mark.parametrize(
    "call",
    [
        "set_tool",
        "add_point",
        "commit",
        "undo",
        "redo",
        "save",
        "save_again",
        "revert",
        "close",
        "drop_redo",
    ],
)
def test_ui_changes_notify_the_host_once_per_call(layer, host, call):
    _commit(layer, (50, 50), (200, 100))
    _commit(layer, (50, 400), (200, 420))
    if call in ("redo", "drop_redo"):
        layer.undo()
    if call == "commit":
        _click(layer, 300, 300)
        _click(layer, 400, 300)
    if call == "save_again":  # the source stays "corrected": only the dirty state changes
        layer.save()
        _commit(layer, (300, 300), (400, 320))
    host.calls.clear()

    {
        "set_tool": lambda: layer.set_tool("eraser"),
        "add_point": lambda: _click(layer, 300, 300),
        "commit": layer.finish_polyline,
        "undo": layer.undo,
        "redo": layer.redo,
        "save": layer.save,
        "save_again": layer.save,
        "revert": layer.revert,
        "close": layer.close,
        "drop_redo": layer.drop_redo,
    }[call]()

    assert _count(host, "layer_changed") == 1


def test_escape_during_a_connect_drag_refreshes_the_hint(gap_layer, host):
    gap_layer.set_tool("connect")
    gap_layer.press(QPoint(199, 100), NO_MOD)
    assert gap_layer.tool_hint()[0] == "Connect · release on the point to join"
    host.calls.clear()

    gap_layer.cancel_transient()

    assert gap_layer.tool_hint()[0] == "Connect · drag from an endpoint (orange dot)"
    assert _count(host, "layer_changed") == 1  # the chip's hint is stale otherwise


def test_a_hover_that_changes_nothing_the_ui_shows_is_silent(layer, host):
    layer.set_tool("polyline")
    _click(layer, 50, 50)
    host.calls.clear()

    for x in range(60, 200, 10):
        layer.move(QPoint(x, 80), False)

    assert host.calls == []


# ------------------------------------------------------------------------- undo / redo
def test_undo_and_redo_step_one_snapshot_and_report_it(layer):
    model = layer.document.model
    start = model.mask.copy()
    _commit(layer, (50, 50), (200, 100))
    first = model.mask.copy()
    _commit(layer, (50, 400), (200, 420))
    second = model.mask.copy()

    assert layer.undo() is True and np.array_equal(model.mask, first)
    assert layer.undo() is True and np.array_equal(model.mask, start)
    assert layer.undo() is False and np.array_equal(model.mask, start)
    assert layer.redo() is True and np.array_equal(model.mask, first)
    assert layer.redo() is True and np.array_equal(model.mask, second)
    assert layer.redo() is False


def test_undo_drops_the_whole_draft_instead_of_popping_a_point(layer, host):
    _commit(layer, (50, 50), (200, 100))
    for x, y in ((300, 300), (350, 320), (400, 300)):
        _click(layer, x, y)
    host.calls.clear()

    assert layer.undo() is True

    assert layer.draft.points == []
    assert not [call for call in host.calls if call[0] == "notify"]  # it was on screen


@pytest.mark.parametrize("step", ["undo", "redo"])
def test_a_history_step_that_drops_a_hidden_draft_says_so(layer, host, step):
    _commit(layer, (50, 50), (200, 100))
    if step == "redo":
        layer.undo()
    _click(layer, 300, 300)
    _click(layer, 400, 300)
    layer.set_active(False)
    host.calls.clear()

    assert getattr(layer, step)() is True

    assert layer.draft.points == []
    assert ("notify", "Polyline draft discarded", "info") in host.calls


def test_an_empty_stack_keeps_the_draft(layer, host):
    layer.set_tool("polyline")
    _click(layer, 300, 300)
    _click(layer, 400, 300)
    layer.set_active(False)
    host.calls.clear()

    assert layer.undo() is False
    assert layer.redo() is False

    assert len(layer.draft.points) == 2
    assert host.calls == []


def test_drop_redo_forgets_only_the_redo_branch(layer):
    _commit(layer, (50, 50), (200, 100))
    _commit(layer, (50, 400), (200, 420))
    layer.undo()
    before = layer.document.model.mask.copy()
    assert layer.can_undo() and layer.can_redo()

    layer.drop_redo()

    assert not layer.can_redo() and layer.can_undo()
    assert np.array_equal(layer.document.model.mask, before)


def test_an_undo_mid_drag_finishes_the_drag_first_then_undoes_it(gap_layer):
    start = gap_layer.document.model.mask.copy()
    gap_layer.set_tool("connect")
    gap_layer.press(QPoint(199, 100), NO_MOD)
    gap_layer.move(QPoint(260, 100), True)

    assert gap_layer.undo() is True  # the join was recorded, then stepped back

    assert np.array_equal(gap_layer.document.model.mask, start)
    assert not gap_layer.gesture_active()
    assert gap_layer.redo() is True
    assert gap_layer.document.model.mask[100, 200:260].all()

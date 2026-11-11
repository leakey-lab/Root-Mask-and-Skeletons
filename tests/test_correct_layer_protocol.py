"""The layer contract of the Correct workspace (C2, ``app/gui/correct/layer.py``).

``missing_members`` finds what an object lacks of a protocol, so layer tests can assert
``missing_members(layer) == []``: in declaration order, attributes and methods, a declared
method that is not callable counts as missing. The test fakes implement the protocols, and
equal ``SizeSpec``s are one dict key (the dock keeps one slider per distinct spec).
"""

from __future__ import annotations

from types import SimpleNamespace

from app.gui.correct.layer import LayerHost, LayerTool, SizeSpec, missing_members

# C2's EditLayer, in declaration order: five attributes, then the methods.
_EDIT_LAYER_MEMBERS = [
    "layer_id",
    "noun",
    "color",
    "tools",
    "current_tool",
    "open",
    "close",
    "is_open",
    "file_stamps",
    "is_dirty",
    "save",
    "revert",
    "status",
    "clear",
    "can_undo",
    "can_redo",
    "undo",
    "redo",
    "drop_redo",
    "undo_in_gesture",
    "gesture_active",
    "finish_gesture",
    "cancel_transient",
    "set_tool",
    "tool_hint",
    "size_spec",
    "size",
    "set_size",
    "step_size",
    "press",
    "move",
    "release",
    "double_click",
    "key",
    "hover",
    "set_active",
    "is_active",
    "set_visible",
    "is_visible",
    "opacity",
    "set_opacity",
]


def _noop(*_args, **_kwargs) -> None:
    return None


def _complete_layer() -> SimpleNamespace:
    return SimpleNamespace(**{name: _noop for name in _EDIT_LAYER_MEMBERS})


def test_an_empty_object_misses_every_edit_layer_member_in_declaration_order():
    assert missing_members(object()) == _EDIT_LAYER_MEMBERS


def test_a_complete_layer_misses_nothing():
    assert missing_members(_complete_layer()) == []


def test_only_the_absent_members_are_reported():
    layer = _complete_layer()
    del layer.press
    del layer.layer_id
    layer.tools = ()  # a declared attribute may hold any value

    assert missing_members(layer) == ["layer_id", "press"]


def test_a_declared_method_that_is_not_callable_counts_as_missing():
    layer = _complete_layer()
    layer.undo = True

    assert missing_members(layer) == ["undo"]


def test_other_protocols_are_checked_by_their_own_declarations():
    host = SimpleNamespace(notify=_noop, layer_edited=_noop, layer_changed=_noop)
    assert missing_members(host, LayerHost) == ["is_panning"]

    assert missing_members(object(), LayerTool) == [
        "tool_id",
        "press",
        "move",
        "release",
        "double_click",
        "key",
        "deactivate",
        "hint",
    ]


def test_the_test_fakes_implement_the_protocols():
    from tests.correct_fakes import FakeHost, FakeLayer

    assert missing_members(FakeLayer("mask")) == []
    assert missing_members(FakeLayer("skeleton")) == []
    assert missing_members(FakeHost(), LayerHost) == []


def test_equal_size_specs_are_one_dict_key():
    # The Correct dock keeps one size slider per distinct SizeSpec (C20).
    brush = SizeSpec("brush", "Brush size", 1, 100)
    eraser = SizeSpec("brush", "Brush size", 1, 100)
    skeleton_eraser = SizeSpec("erase", "Eraser size", 1, 80)

    sliders = {brush: "slider A", eraser: "slider A", skeleton_eraser: "slider B"}

    assert len(sliders) == 2

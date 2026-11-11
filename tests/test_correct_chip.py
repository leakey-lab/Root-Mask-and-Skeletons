"""What the Correct page's status chip shows (``app/gui/correct/chip.py``, ``chip_content``;
Qt-free): the file name, the worst save state of the layers (error > unsaved > saved >
none), the failed layer's label and reason, a layer's own lock (``LayerStatus.local``)
counted only while that layer is edited, and the hint "<Layer> · <tool hint>" in the edited
layer's colour. The chip on the page: ``test_correct_editor_controls.py``.
"""

from __future__ import annotations

from app.design import tokens
from app.gui.correct.chip import chip_content
from app.gui.correct.layer import CLASSES, MASK, SKELETON, LayerStatus
from tests.correct_fakes import FakeLayer

IMAGE = "/data/images/a.png"


def state_of(statuses, layer=None) -> tuple:
    """``(state, label, tooltip)`` the chip shows for ``statuses``, the mask edited."""
    return chip_content(IMAGE, statuses, layer or FakeLayer(MASK), None)[1]


def test_the_worst_state_of_the_layers_wins():
    clean, dirty, none = LayerStatus("clean"), LayerStatus("dirty"), LayerStatus("none")

    assert state_of({MASK: none, SKELETON: none})[0] == "none"
    assert state_of({MASK: clean, SKELETON: none})[0] == "clean"
    assert state_of({MASK: clean, SKELETON: dirty})[0] == "dirty"


def test_a_failed_save_outranks_unsaved_edits_and_names_its_reason():
    statuses = {
        MASK: LayerStatus("dirty"),
        SKELETON: LayerStatus("error", "Not saved", "Could not save gt/skeletons/a.png: disk full"),
    }

    assert state_of(statuses) == (
        "error",
        "Not saved",
        "Could not save gt/skeletons/a.png: disk full",
    )


def test_a_layers_own_lock_counts_only_while_that_layer_is_edited():
    locked = LayerStatus("error", "Read-only", "gt/classes.json is invalid", local=True)
    statuses = {MASK: LayerStatus("clean"), CLASSES: locked}

    assert state_of(statuses, FakeLayer(MASK)) == ("clean", None, "")
    assert state_of(statuses, FakeLayer(CLASSES)) == (
        "error",
        "Read-only",
        "gt/classes.json is invalid",
    )


def test_the_hint_names_the_edited_layer_and_its_tool_in_its_colour():
    name, _state, hint, position = chip_content(
        IMAGE, {SKELETON: LayerStatus("clean")}, FakeLayer(SKELETON), (12, 40)
    )

    assert name == "a.png" and position == (12, 40)
    assert hint == ("Skeleton · Select", "pick a branch, Delete removes it", tokens.SKEL)


def test_with_no_image_the_chip_says_so_and_how_to_open_one():
    name, state, hint, _position = chip_content(
        "", {MASK: LayerStatus("dirty")}, FakeLayer(MASK), None
    )

    assert name == "No image" and state == ("none", None, "")
    assert hint == ("Select an image in the Library", "", tokens.MASK)

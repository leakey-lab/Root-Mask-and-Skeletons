"""The pixel goldens of the pre-merge editors, replayed through the whole Correct page
(plan P7b Step 3, contract C18).

``tests/data/correct_goldens.json`` holds what the pre-merge mask and skeleton editors did for
two synthetic projects: the plane or raster after every op of each script and the saved
file. Here the scripts drive one ``CorrectEditor`` the way a user does: the rail's tool
buttons, the dock's size sliders, the polyline prompt, Clear, Save, Undo / Redo and the
canvas input (the router, or real mouse events through the view). Saved pixels must stay
identical to the goldens (the skeleton entries were re-recorded once for Q1: bilinear
open, corrections saved at image resolution; see CHANGELOG). The mask is saved as GT
(``gt/mask/<stem>.png``, WG-T2); Clear and Save on the model mask writes an empty GT file
(O9), and a save with nothing unsaved writes nothing (O2):

- ``ops.mask`` then ``ops.skeleton`` in one page, each op and each save matching its golden
  (the skeleton's save never rewrites the clean mask);
- ``ops.mask_clear`` and ``ops.skeleton_clear`` on fresh projects;
- the ops of both scripts other than undo, redo and save, interleaved with a layer switch
  between every op, leave each layer exactly as its ops alone do and save the same files;
- the mask strokes as real mouse events through the canvas at zoom 1.0 and 3.7.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("ROOT_VIEWER_ENABLE_OPENGL_VIEWPORT", "0")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.gui.correct.layer import MASK, SKELETON  # noqa: E402
from app.io_utils import imread_unicode  # noqa: E402
from tests.correct_goldens_page_support import (  # noqa: E402
    GOLDENS,
    OPS,
    apply,
    assert_mask_file,
    assert_skeleton_file,
    files,
    mouse_stroke,
    replay,
    skeleton_start,
    state_of,
)
from tests.correct_goldens_page_support import app as app  # noqa: E402 - fixture
from tests.correct_goldens_page_support import guards as guards  # noqa: E402 - fixture
from tests.correct_goldens_page_support import make_editor as make_editor  # noqa: E402 - fixture
from tests.correct_goldens_support import (  # noqa: E402
    SYNTH_CASES,
    make_synth_project,
    sha,
)

HISTORY = {"undo", "redo", "save"}


# ============================================================================= the goldens
@pytest.mark.parametrize("case", sorted(SYNTH_CASES))
def test_the_mask_then_the_skeleton_script_in_one_page_match_their_goldens(
    make_editor, tmp_path, case
):
    golden = GOLDENS["cases"][case]
    folder, image = make_synth_project(str(tmp_path), case)
    mask_file, correction = files(folder)
    editor = make_editor(image, folder)
    assert skeleton_start(editor) == golden["skeleton"]["start"]

    saved = replay(editor, MASK, "mask", golden["mask"]["per_op_plane_sha256"])
    assert saved is golden["mask"]["save_returned"]
    assert_mask_file(mask_file, golden["mask"]["saved"])
    assert not os.path.exists(correction)
    mask_stamp = os.stat(mask_file).st_mtime_ns

    assert skeleton_start(editor) == golden["skeleton"]["start"]  # untouched by the mask ops
    saved = replay(editor, SKELETON, "skeleton", golden["skeleton"]["per_op_raster_sha256"])
    assert saved is golden["skeleton"]["save_returned"]
    assert_skeleton_file(correction, golden["skeleton"]["saved"])
    assert os.stat(mask_file).st_mtime_ns == mask_stamp  # the clean mask was not rewritten
    assert_mask_file(mask_file, golden["mask"]["saved"])


@pytest.mark.parametrize("script", ["mask_clear", "skeleton_clear"])
@pytest.mark.parametrize("case", sorted(SYNTH_CASES))
def test_the_clear_scripts_on_a_fresh_project_match_their_goldens(
    make_editor, tmp_path, case, script
):
    golden = GOLDENS["cases"][case][script]
    folder, image = make_synth_project(str(tmp_path), case)
    mask_file, correction = files(folder)
    editor = make_editor(image, folder)
    layer_id = MASK if script == "mask_clear" else SKELETON
    key = "per_op_plane_sha256" if layer_id == MASK else "per_op_raster_sha256"
    if layer_id == SKELETON:
        assert skeleton_start(editor) == golden["start"]

    assert replay(editor, layer_id, script, golden[key]) is golden["save_returned"]

    if layer_id == MASK:
        assert_mask_file(mask_file, golden["saved"])
        assert not os.path.exists(correction)
    else:
        assert_skeleton_file(correction, golden["saved"])


def run_alone(
    make_editor, root: str, case: str, layer_id: str, ops: list
) -> tuple[list, np.ndarray]:
    """``ops`` of one layer alone in a fresh page: the per-op shas and that layer's saved file."""
    folder, image = make_synth_project(root, case)
    editor = make_editor(image, folder)
    editor.set_active_layer(layer_id)
    shas = []
    for op in ops:
        apply(editor, op)
        shas.append(sha(state_of(editor, layer_id)))
    save_each_layer(editor)
    path = files(folder)[0 if layer_id == MASK else 1]
    return shas, imread_unicode(path, cv2.IMREAD_UNCHANGED)


def save_each_layer(editor) -> None:
    """Ctrl+S once: it saves every unsaved layer, mask first (O2: a clean layer, such as a
    blank skeleton cleared again, is not written, and with none unsaved nothing is)."""
    unsaved = editor.has_unsaved_changes()
    assert editor.save() is unsaved


@pytest.mark.parametrize("case", sorted(SYNTH_CASES))
def test_interleaved_ops_with_a_switch_between_every_op_do_what_each_script_does_alone(
    make_editor, tmp_path, case
):
    ops = {
        layer_id: [op for op in OPS[script] if op[0] not in HISTORY]
        for layer_id, script in ((MASK, "mask"), (SKELETON, "skeleton"))
    }
    alone = {
        layer_id: run_alone(
            make_editor, str(tmp_path / f"alone_{layer_id}"), case, layer_id, ops[layer_id]
        )
        for layer_id in (MASK, SKELETON)
    }
    order = []
    for index in range(max(len(ops[MASK]), len(ops[SKELETON]))):
        order += [(layer_id, index) for layer_id in (MASK, SKELETON) if index < len(ops[layer_id])]
    folder, image = make_synth_project(str(tmp_path / "interleaved"), case)
    editor = make_editor(image, folder)
    switches = []
    editor.active_layer_changed.connect(switches.append)

    previous = None
    for layer_id, index in order:
        if layer_id == previous:  # the other script ran out: switch away and back
            editor.swap_layer()
        editor.set_active_layer(layer_id)
        apply(editor, ops[layer_id][index])
        where = (layer_id, index, ops[layer_id][index])
        assert sha(state_of(editor, layer_id)) == alone[layer_id][0][index], where
        previous = layer_id
    save_each_layer(editor)

    assert len(switches) >= len(order) - 1  # a layer switch between every two ops
    mask_file, correction = files(folder)
    assert np.array_equal(imread_unicode(mask_file, cv2.IMREAD_UNCHANGED), alone[MASK][1])
    assert np.array_equal(imread_unicode(correction, cv2.IMREAD_UNCHANGED), alone[SKELETON][1])


@pytest.mark.parametrize("zoom", [1.0, 3.7])
def test_real_mouse_strokes_through_the_canvas_match_the_mask_golden(make_editor, tmp_path, zoom):
    case = "synth_model_mask_and_skeleton"
    golden = GOLDENS["cases"][case]["mask"]
    folder, image = make_synth_project(str(tmp_path), case)
    editor = make_editor(image, folder)
    editor.graphics_view.set_zoom(zoom)

    saved = replay(editor, MASK, "mask", golden["per_op_plane_sha256"], stroke=mouse_stroke)

    assert saved is golden["save_returned"]
    assert_mask_file(files(folder)[0], golden["saved"])

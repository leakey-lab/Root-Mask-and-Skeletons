"""What the canvas shows never changes what is saved or measured.

Zoom, the pixel grid, Enhance (Contrast Stretching here; Deblock in
test_enhance_display_only.py) and a swap of the viewport are display only. With them on,
the Correct goldens replay saves the golden mask and skeleton files, exactly as
test_correct_goldens.py checks them (``tests/data/correct_goldens.json``, read only).

The skeleton layer item keeps the skeleton raster by reference and only reads it. That
array is the GT skeleton the page saves and Measure takes root length from. A read-only
array (numpy refuses every write) is the oracle, through every tier, box updates and the
idle vector warm-up.
"""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import QPointF
from PyQt6.QtWidgets import QGraphicsView

from app.gui import canvas_gl
from app.gui.canvas_layers.skeleton import SkeletonLayerItem
from app.gui.correct.layer import MASK, SKELETON
from tests.canvas_skeleton_support import (
    _busy_mask,
    _drop_scenes,  # noqa: F401 - the module's autouse fixture
    _edit,
    _in_scene,
    _render,
    _unbudgeted,
)
from tests.correct_goldens_page_support import (
    GOLDENS,
    assert_mask_file,
    assert_skeleton_file,
    files,
    replay,
    router_stroke,
    skeleton_start,
)
from tests.correct_goldens_support import make_synth_project
from tests.qt_teardown import delete_widgets
from tests.qt_wait import wait_until

CASE = "synth_model_mask_and_skeleton"


# ------------------------------------------------------------------ saved files, at the page
@pytest.mark.usefixtures("correct_guards")
@pytest.mark.parametrize("zoom", [0.3, 8.0], ids=["coverage-tiers", "exact-pixels-and-grid"])
def test_the_goldens_save_the_same_files_whatever_the_canvas_shows(
    make_editor, tmp_path, monkeypatch, zoom
):
    golden = GOLDENS["cases"][CASE]
    folder, image = make_synth_project(str(tmp_path), CASE)
    editor = make_editor(image, folder)
    view = editor.graphics_view
    raw = editor.photo_item.pixmap().toImage()
    controls = editor.norm_controls
    controls.method_combo.setCurrentText("Contrast Stretching")
    controls.apply_button.click()
    assert editor.photo_item.pixmap().toImage() != raw  # Enhance really changed the photo
    monkeypatch.setattr(canvas_gl, "opengl_viewports_allowed", lambda: True)
    view.set_opengl_viewport_enabled(True)  # a GL viewport, then the raster one again
    assert view.viewport().inherits("QOpenGLWidget")
    view.set_opengl_viewport_enabled(False)
    view.set_pixel_grid(True)
    view.set_zoom(zoom)
    view.centerOn(QPointF(64.0, 48.0))

    def stroke_then_paint(page, points) -> None:
        router_stroke(page, points)
        page.graphics_view.viewport().repaint()  # every layer drawn at this zoom

    mask_file, skeleton_file = files(folder)
    assert skeleton_start(editor) == golden["skeleton"]["start"]
    saved = replay(
        editor, MASK, "mask", golden["mask"]["per_op_plane_sha256"], stroke=stroke_then_paint
    )
    assert saved is golden["mask"]["save_returned"]
    assert_mask_file(mask_file, golden["mask"]["saved"])
    saved = replay(
        editor,
        SKELETON,
        "skeleton",
        golden["skeleton"]["per_op_raster_sha256"],
        stroke=stroke_then_paint,
    )
    assert saved is golden["skeleton"]["save_returned"]
    assert_skeleton_file(skeleton_file, golden["skeleton"]["saved"])


# ------------------------------------------------------------------ the skeleton item only reads
def _read_only(mask: np.ndarray) -> np.ndarray:
    mask.flags.writeable = False
    return mask


@pytest.mark.usefixtures("slot_errors")
@pytest.mark.parametrize("vectors", [True, False], ids=["correct", "library"])
def test_the_skeleton_item_never_writes_the_skeleton_it_shows(app, vectors):
    mask = _read_only(_busy_mask())
    before = mask.copy()
    item = _unbudgeted(_in_scene(SkeletonLayerItem(halo=vectors, vectors=vectors)))
    item.set_mask(mask)
    for zoom in (0.3, 1.0, 1.5, 4.0):  # LOD, pixel tiles, smooth and vector tiers
        _render(item, zoom, (400, 300), origin=(280.0, 180.0))

    for box in ((300, 200, 32, 32), (0, 0, 8, 8)):  # an edit: a new read-only array each time
        edited = mask.copy()
        _edit(edited, box)
        mask = _read_only(edited)
        before = mask.copy()
        item.set_mask(mask, changed=box)
        for zoom in (0.3, 1.0, 4.0):
            _render(item, zoom, (400, 300), origin=(280.0, 180.0))

    view = QGraphicsView(item.scene())  # the idle warm-up extracts only while a view shows it
    try:
        view.show()
        if vectors:
            assert wait_until(lambda: len(item.vector_tiles()) == 4 * 3)
    finally:
        delete_widgets([view])
    assert np.array_equal(mask, before)

"""A stem that ends in an image extension is never stripped a second time (final review D1).

Image ``X.png.jpg`` has the stem ``X.png`` (only the last image extension goes). Every
lookup of its GT and its prediction keeps that stem: ``gt/skeletons/X.png.png``,
``gt/mask/X.png.png``, ``gt/X.png.png``, ``skeletons/X.png_fake.png``. Before the fix the
"stem or path" methods of ``ProjectLayout`` stripped ``X.png`` again to ``X``: the skeleton
layer opened and saved ``X.png``'s GT skeleton, Remove GT… deleted ``X.png``'s three GT files,
Measure recorded the saved GT as "image gone", and the Library marks and the metrics bar
looked under ``X``. Driven through the real Correct page, the Measure handler, the Library's
``ImageManager`` with the window's report functions, and the metrics bar's
``MeasurementIndex``.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from app.config import AREA_CSV_HEADERS, LENGTH_CSV_HEADERS
from app.data_processing.measurement_index import MeasurementIndex
from app.gt import Tracing, default_registry, write_tracing
from app.gui.correct.layer import SKELETON
from app.gui.image_manager import ImageManager
from app.gui.window import editors
from app.masks import write_mask_file
from app.project_layout import ProjectLayout
from tests.correct_support import (
    H,
    W,
    box_plane,
    draw_line,
    write_image,
    write_mask,
    write_skeleton,
)
from tests.measure_support import run_area, run_length

pytestmark = pytest.mark.usefixtures("correct_guards")

LENGTH, AREA = LENGTH_CSV_HEADERS[5], AREA_CSV_HEADERS[5]


class Confirm:
    """Remove GT…'s question, answered Yes; the names it was asked about."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, _parent, name: str, detail: str) -> bool:
        self.calls.append((name, detail))
        return True


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def gt_of_x(layout: ProjectLayout, x_png: str) -> dict[str, str]:
    """The three GT files of image ``X.png``, written from its path; returns kind -> path."""
    on = np.zeros((H, W), bool)
    on[10:40, 10:60] = True
    paths = {
        "mask": layout.gt_mask_path(x_png),
        "skeleton": layout.gt_skeleton_path(x_png),
        "labels": layout.gt_path(x_png),
    }
    write_mask_file(paths["mask"], on, durable=False)
    write_skeleton(paths["skeleton"], row=30)
    write_tracing(paths["labels"], Tracing(np.ones((H, W), np.uint8)), default_registry())
    return paths


def fake_window(manager: ImageManager, open_image: str = "") -> SimpleNamespace:
    """What the window's report functions read: the Library, its rows, the metrics bar."""
    return SimpleNamespace(
        image_manager=manager,
        library_index=SimpleNamespace(refresh_status=lambda stem: None),
        metrics=SimpleNamespace(refresh_if_current=lambda stem: None),
        correct_editor=SimpleNamespace(current_image_path=open_image),
    )


def library(folder: str) -> ImageManager:
    """The Library of ``folder`` as a finished scan leaves it (its GT marks from disk)."""
    manager = ImageManager()
    manager.original_folder = folder
    layout = ProjectLayout(folder)
    manager.on_loading_finished(layout.scan_images(), {}, layout.scan_masks(), False)
    return manager


@pytest.fixture
def saved(make_editor, tmp_path) -> SimpleNamespace:
    """Images ``X.png`` and ``X.png.jpg`` with their predictions and X.png's three GT files;
    the skeleton layer of X.png.jpg opened, a line drawn, saved. What the layer showed before
    the edit is in ``opened``, what the save reported in ``saved``."""
    folder = str(tmp_path / "project")
    layout = ProjectLayout(folder)
    x_png = write_image(os.path.join(folder, "X.png"))
    x_jpg = write_image(os.path.join(folder, "X.png.jpg"), value=60)  # a converted copy
    scanned = sorted(layout.scan_images())
    for stem, row in (("X", 110), ("X.png", 60)):  # what Generate writes for each
        write_skeleton(os.path.join(layout.skeletons_dir, f"{stem}_fake.png"), row=row)
        write_mask(os.path.join(layout.mask_dir, f"{stem}.png"), box_plane())
    x_gt = gt_of_x(layout, x_png)
    before = {kind: read_bytes(path) for kind, path in x_gt.items()}
    editor = make_editor(x_jpg, folder)
    skeleton = editor.layers[SKELETON]
    opened = SimpleNamespace(
        source=skeleton.document.source,
        row_60=skeleton.document.model.mask[60].any(),
        row_110=skeleton.document.model.mask[110].any(),
    )
    draw_line(editor)
    reported: list[str] = []
    editor.skeleton_saved.connect(reported.append)
    ok = editor.save()
    return SimpleNamespace(
        folder=folder,
        layout=layout,
        x_png=x_png,
        x_jpg=x_jpg,
        scanned=scanned,
        x_gt=x_gt,
        before=before,
        own_gt=os.path.join(layout.gt_skeletons_dir, "X.png.png"),
        editor=editor,
        skeleton=skeleton,
        opened=opened,
        saved=reported,
        ok=ok,
    )


def test_the_skeleton_layer_of_x_png_jpg_opens_and_saves_its_own_files(saved):
    assert saved.scanned == ["X", "X.png"]
    assert saved.opened.source == "prediction"
    assert saved.opened.row_60 and not saved.opened.row_110
    assert saved.ok
    assert [os.path.normcase(p) for p in saved.saved] == [os.path.normcase(saved.own_gt)]
    assert os.path.isfile(saved.own_gt)
    assert {kind: read_bytes(path) for kind, path in saved.x_gt.items()} == saved.before


def test_the_library_marks_each_image_by_its_own_gt(saved):
    own_gt, x_gt = saved.own_gt, saved.x_gt
    manager = library(saved.folder)
    assert manager.gt_skeleton_stems == {"X", "X.png"}
    manager.gt_skeleton_stems.discard("X.png")  # as before the save
    editors.on_skeleton_saved(fake_window(manager), saved.saved[0])
    assert manager.gt_skeleton_stems == {"X", "X.png"}
    editors.sync_open_correction(fake_window(manager, saved.x_jpg))
    assert manager.gt_skeleton_stems == {"X", "X.png"}
    assert os.path.normcase(manager.gt_skeleton_path_for("X.png")) == os.path.normcase(own_gt)
    assert os.path.normcase(manager.gt_skeleton_path_for("X")) == os.path.normcase(x_gt["skeleton"])
    manager.set_gt("X.png", "mask", True)
    assert manager.gt_mask_stems == {"X", "X.png"}
    manager.set_gt("X.png", "mask", False)
    assert manager.gt_mask_stems == {"X"}


def test_measure_gives_each_image_a_row_from_its_own_gt(saved, app):
    lengths = run_length(saved.layout, app)
    assert sorted(lengths) == ["X", "X.png"]
    assert [lengths[s]["Source"] for s in ("X", "X.png")] == ["gt", "gt"]
    assert lengths["X"][LENGTH] != lengths["X.png"][LENGTH]
    areas = run_area(saved.layout, app)
    assert [areas[s]["Source"] for s in ("X", "X.png")] == ["gt", "prediction"]


def test_the_metrics_bar_finds_each_images_own_row_and_file(saved, app):
    lengths = run_length(saved.layout, app)
    areas = run_area(saved.layout, app)

    index = MeasurementIndex(saved.folder)
    for stem in ("X", "X.png"):
        assert index.lookup(stem).length_mm == float(lengths[stem][LENGTH])
        assert index.lookup(stem).area_mm2 == float(areas[stem][AREA])
    assert index.measured_from("X.png") == {
        "length": ("gt/skeletons/X.png.png", "gt"),
        "area": ("mask/X.png.png", "prediction"),
    }
    assert index.measured_from("X") == {
        "length": ("gt/skeletons/X.png", "gt"),
        "area": ("gt/mask/X.png", "gt"),
    }
    assert index.outdated("X.png") == {"length": False, "area": False}


def test_remove_gt_on_x_png_jpg_removes_only_its_own_file(saved):
    editor, skeleton, manager = saved.editor, saved.skeleton, library(saved.folder)
    confirm = Confirm()
    editor.actions.confirm_remove = confirm
    editor.remove_gt()
    assert confirm.calls and not os.path.exists(saved.own_gt)
    assert {kind: read_bytes(path) for kind, path in saved.x_gt.items()} == saved.before
    editors.on_gt_removed(fake_window(manager), saved.x_jpg)
    assert manager.gt_skeleton_stems == {"X"}
    assert manager.gt_mask_stems == manager.gt_label_stems == {"X"}

    # X.png still opens its own GT skeleton.
    editor.load_image(saved.x_png, saved.folder)
    assert skeleton.document.source == "gt" and skeleton.document.model.mask[30].any()


def test_x_png_jpg_alone_is_measured_from_the_gt_it_saved(make_editor, app, tmp_path):
    folder = str(tmp_path / "project")
    layout = ProjectLayout(folder)
    x_jpg = write_image(os.path.join(folder, "X.png.jpg"))
    write_skeleton(os.path.join(layout.skeletons_dir, "X.png_fake.png"), row=60)
    own_gt = os.path.join(layout.gt_skeletons_dir, "X.png.png")

    editor = make_editor(x_jpg, folder)
    skeleton = editor.layers[SKELETON]
    assert skeleton.document.source == "prediction"  # Generate's X.png_fake.png
    draw_line(editor)
    assert editor.save() and os.path.isfile(own_gt)
    assert sorted(os.listdir(layout.gt_skeletons_dir)) == ["X.png.png"]

    lengths = run_length(layout, app)
    assert [(name, row["Source"]) for name, row in lengths.items()] == [("X.png", "gt")]
    assert MeasurementIndex(folder).lookup("X.png").length_source == "gt"
    manager = library(folder)
    assert manager.gt_skeleton_stems == {"X.png"}

    confirm = Confirm()
    editor.actions.confirm_remove = confirm
    editor.remove_gt()
    assert confirm.calls == [("X.png.jpg", confirm.calls[0][1])]
    assert "gt/skeletons/X.png.png" in confirm.calls[0][1]
    assert not os.path.exists(own_gt)
    assert skeleton.document.source == "prediction"

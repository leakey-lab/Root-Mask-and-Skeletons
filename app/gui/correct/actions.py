"""The Correct page's file actions and reports: :class:`PageActions` (WG-T5, Design 3.4 and
3.6).

What the page does to files beyond a layer's own Save: open another skeleton file (Load
skeleton file…), put a layer's prediction back (Load prediction), edit the project's classes
(Classes…, the class editor: ``gt/classes.json``), delete the open image's GT (Remove GT…,
the topbar button it owns: on while an image is open and nothing is unsaved) and save its
class labels with nothing labelled (Mark empty, its other topbar button: "checked, none of
the classes", a negative example for training; on while the class labels are untouched and
no labels file is saved or kept for the image). Each ends a drag in progress first, as every
transition of the page does. It also turns the session's report of each layer save into the
page's signal: ``mask_saved`` / ``mask_cleared`` / ``skeleton_saved`` /
``skeleton_cleared`` / ``gt_saved``, on every save path.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QWidget

from app import gt
from app.dialogs import run_modal
from app.gt.files import GtRemoveError, gt_files, remove_gt
from app.gui.correct.classes_document import EMPTY_CLOSED, REGISTRY_LABEL
from app.gui.correct.layer import CLASSES, MASK, SKELETON, SaveOutcome, project_label
from app.gui.widgets import keep_canvas_focus
from app.gui.widgets.buttons import ghost_button
from app.project_layout import ProjectLayout

if TYPE_CHECKING:  # pragma: no cover
    from PyQt6.QtWidgets import QPushButton

    from app.gui.correct.editor import CorrectEditor

__all__ = ["EMPTY_NO_IMAGE_TIP", "EMPTY_TIP", "REMOVE_DETAIL", "PageActions", "ask_remove_gt"]

SKELETON_FILES = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
REMOVE_DETAIL = (
    "Deletes gt/mask/{stem}.png, gt/skeletons/{stem}.png and gt/{stem}.png where they exist. "
    "The predictions in mask/ and skeletons/ are kept. This cannot be undone."
)
REMOVE_TIP = "Remove the GT of this image: its saved mask, skeleton and class labels"
REMOVE_UNSAVED_TIP = "Save or discard your edits first"
REMOVE_NO_IMAGE_TIP = "Open an image to remove its GT"
EMPTY_TIP = (
    "Mark this image empty: save its class labels with nothing labelled, as it shows none of "
    "the classes (an empty example for training)"
)
EMPTY_NO_IMAGE_TIP = EMPTY_CLOSED


def ask_remove_gt(parent: QWidget | None, name: str, detail: str) -> bool:
    """The Remove GT box: True when the user chose "Remove GT" (Cancel is the default and
    Esc)."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("Remove GT")
    box.setText(f"Remove the GT of {name}?")
    box.setInformativeText(detail)
    remove = box.addButton("Remove GT", QMessageBox.ButtonRole.DestructiveRole)
    cancel = box.addButton(QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(cancel)
    box.setEscapeButton(cancel)
    return run_modal(box, lambda answered: answered.clickedButton() is remove)


class PageActions(QObject):
    """The page's file actions, its Remove GT… and Mark empty buttons (``remove_button``,
    ``empty_button``) and its reports. ``confirm_remove(parent, name, detail) -> bool`` asks
    before Remove GT deletes anything (:func:`ask_remove_gt`; a test hands its own)."""

    def __init__(self, editor: CorrectEditor) -> None:
        super().__init__(editor)
        self._editor = editor
        self.confirm_remove: Callable[[QWidget | None, str, str], bool] = ask_remove_gt
        self.remove_button = ghost_button("Remove GT…", "trash", compact=True)
        self.remove_button.setEnabled(False)
        self.remove_button.setToolTip(REMOVE_NO_IMAGE_TIP)
        # No icon: with one the topbar outgrows the 900 px window budget (the Correct page is
        # its widest action page; tests/test_window_width_budget.py).
        self.empty_button = ghost_button("Mark empty", compact=True)
        self.empty_button.setEnabled(False)
        self.empty_button.setToolTip(EMPTY_NO_IMAGE_TIP)
        keep_canvas_focus(self.remove_button, self.empty_button)
        self.remove_button.clicked.connect(self.remove_gt)
        self.empty_button.clicked.connect(self.save_empty_labels)

    # ------------------------------------------------------------------ actions
    def load_skeleton_file(self) -> None:
        """Open another skeleton file; an unsaved skeleton is asked about first."""
        editor = self._editor
        editor.session.finish_gestures()
        skeleton = editor.layers[SKELETON]
        if not skeleton.is_open():
            editor.notify("Open an image first", "warn")
            return
        start = skeleton.dialog_start()  # the generated skeleton, else skeletons/
        title = "Select Skeleton File"
        path, _ = QFileDialog.getOpenFileName(editor, title, start, SKELETON_FILES)
        if not path:
            return
        # Read before the prompt, so a file that cannot be read costs no edits (C6-03);
        # load_file reads it again after the prompt, as a Save there can rewrite it.
        unreadable = skeleton.document.check_file(path)
        for notice in unreadable:
            editor.notify(notice.message, notice.kind)
        if unreadable or not editor.session.release([SKELETON]):
            return
        for notice in skeleton.load_file(path):
            editor.notify(notice.message, notice.kind)
        if skeleton.document.last_load_ok:  # unreadable: nothing changed, the timeline stays
            editor.session.forget(SKELETON)  # its own history is gone too

    def load_prediction(self) -> None:
        """Load prediction: the edited layer's prediction as one undoable action (if any)."""
        editor = self._editor
        editor.session.finish_gestures()
        getattr(editor.layers[editor.active_layer_id], "load_prediction", lambda: None)()

    def edit_classes(self) -> None:
        """Classes…: the class editor on the project's classes. OK saves them to
        gt/classes.json and recolours the class labels (their ids and history stay); the
        page takes that write as its own (no change on disk). Refused while the file is
        invalid (the class labels' one notice says why)."""
        editor = self._editor
        session, layer = editor.session, editor.layers[CLASSES]
        session.finish_gestures()
        if not layer.is_open():
            editor.notify("Open an image first; classes are saved with its folder.", "info")
            return
        if layer.locked():
            return
        from app.gui.class_editor_dialog import ClassEditorDialog  # its first use loads it

        dialog = ClassEditorDialog(layer.registry(), editor)  # deleted once answered
        registry = run_modal(dialog, lambda answered: answered.result_registry())
        editor.graphics_view.setFocus()
        if registry is None:
            return
        unchanged = not session.files_changed()  # a change on disk before stays one
        try:
            gt.save_registry(ProjectLayout.for_image(session.image_path).root, registry)
        except OSError as exc:
            editor.notify(f"Could not save {REGISTRY_LABEL}: {exc}", "danger")
            return
        layer.apply_registry(registry)
        if unchanged:
            session.take_stamps(CLASSES)
        editor.notify("Classes updated", "success")

    def remove_gt(self) -> None:
        """Remove GT…: once the user confirms, delete the open image's GT files (never a
        prediction, never gt/classes.json) and read its layers again from what is left. A
        file that could not be deleted is named; the image is read again either way."""
        editor = self._editor
        session = editor.session
        session.finish_gestures()
        path = session.image_path
        if not path or session.is_dirty():
            return  # the button is off then
        layout, stem = ProjectLayout.for_image(path), ProjectLayout.stem(path)
        name = os.path.basename(path)
        found = gt_files(layout, path)  # the path: a stem is never stripped twice (D1)
        if not any((found.mask, found.skeleton, found.labels)):
            editor.notify(f"No GT to remove for {name}", "info")
            return
        if not self.confirm_remove(editor, name, REMOVE_DETAIL.format(stem=stem)):
            return
        failed: dict[str, OSError] = {}
        try:
            removed = remove_gt(layout, path)
        except GtRemoveError as exc:
            removed, failed = exc.removed, exc.failed
        editor.paging.forget(path)  # what was read of it holds the GT
        session.reload_documents()  # the predictions, or empty
        editor.gt_removed.emit(path)
        if not failed:
            editor.notify(f"Removed GT · {name}", "success")
            return
        root = layout.root
        done = ", ".join(project_label(file, root) for file in removed) or "nothing"
        kept = ", ".join(
            f"{project_label(file, root)} ({exc.strerror or exc})" for file, exc in failed.items()
        )
        editor.notify(f"Remove GT · {name}: removed {done}; could not remove {kept}", "danger")

    def save_empty_labels(self) -> None:
        """Mark empty: the open image's class labels saved with nothing labelled (all
        background) through the session's save, which reports, toasts and takes the stamps as
        for any save; only the class labels are saved. Saved labels are never replaced: the
        button is off while a labels file is saved or kept for the image (Remove GT… deletes
        it again)."""
        self._editor.session.finish_gestures()
        if self._empty_refusal():
            return  # the button is off then
        self._editor.session.save_layer(CLASSES)

    def save_for_paging(self) -> bool:
        """PgDn / PgUp: True when the image may be left. The unsaved layers are saved without
        a prompt, unless one must be asked about (``save_needs_prompt``: a lossy class remap
        is never written silently): then the one prompt, whose Cancel stays."""
        session = self._editor.session
        dirty = [session.layers[layer_id] for layer_id in session.dirty_layers()]
        if any(getattr(layer, "save_needs_prompt", bool)() for layer in dirty):
            return session.release()
        return session.save_dirty().ok  # a failed save stays (its toast says why)

    def sync(self) -> None:
        """The Remove GT… button: on while an image is open and nothing is unsaved. Mark empty:
        on while an empty save would replace no file and drop no painted label (its tooltip
        says why not)."""
        session = self._editor.session
        if not session.image_path:
            enabled, tip = False, REMOVE_NO_IMAGE_TIP
        elif session.is_dirty():
            enabled, tip = False, REMOVE_UNSAVED_TIP
        else:
            enabled, tip = True, REMOVE_TIP
        _show(self.remove_button, enabled, tip)
        refusal = self._empty_refusal()
        _show(self.empty_button, not refusal, refusal or EMPTY_TIP)

    def _empty_refusal(self) -> str:
        """Why Mark empty is off ("" while it is on)."""
        if not self._editor.session.image_path:
            return EMPTY_NO_IMAGE_TIP
        return self._editor.layers[CLASSES].document.empty_save_refusal()

    # ------------------------------------------------------------------ reports
    def report(self, outcome: SaveOutcome) -> None:
        """One layer save attempt -> the page's signal for MainWindow (a failed save reports
        nothing); what was read ahead of the image goes (its files changed)."""
        editor = self._editor
        path = editor.session.image_path
        editor.paging.forget(path)
        if not outcome.ok:
            return
        layer_id, written = outcome.layer_id, outcome.path
        if layer_id == MASK and (written or outcome.cleared):
            (editor.mask_saved if written else editor.mask_cleared).emit(path)
        elif layer_id == SKELETON and outcome.cleared:
            editor.skeleton_cleared.emit(path)
        elif layer_id == SKELETON and written:
            editor.skeleton_saved.emit(written)
        elif layer_id == CLASSES and written:
            editor.gt_saved.emit(path)


def _show(button: QPushButton, enabled: bool, tip: str) -> None:
    """Set ``button``'s state and tooltip where they changed."""
    if button.isEnabled() != enabled:
        button.setEnabled(enabled)
    if button.toolTip() != tip:
        button.setToolTip(tip)

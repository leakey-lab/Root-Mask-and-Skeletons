"""Helpers of the CorrectControls and LayerPanel tests: the Correct page as P7b composes
it from its parts (``Page``, fixture ``make_page``), the event settle, a signal recorder
and colour readers of real renders. Moved here from ``test_correct_controls.py``, whose
LayerPanel tests now live in ``test_correct_layer_panel.py``.
"""

from __future__ import annotations

from collections import Counter

import pytest
from PyQt6.QtCore import QPoint, QRectF
from PyQt6.QtGui import QColor, QImage
from PyQt6.QtWidgets import (
    QGraphicsScene,
    QVBoxLayout,
    QWidget,
)

from app.enhancement import EnhanceDefaultsStore
from app.gui.correct.controls import CorrectControls
from app.gui.correct.layer import MASK, SKELETON
from app.gui.correct.layer_panel import LayerPanel
from app.gui.correct.mask_layer import MaskLayer
from app.gui.correct.skeleton_layer import SkeletonLayer
from app.gui.editor_canvas import EditorCanvasView
from app.gui.editor_chrome import EditorChrome
from app.gui.image_normalization_interface import NormalizationControls
from app.gui.widgets.overlay_geometry import LayoutMetrics
from tests.correct_fakes import FakeHost, FakeLayer
from tests.qt_teardown import delete_widgets

M = LayoutMetrics()


class Page(QWidget):
    """The Correct page as P7b composes it: canvas, chrome, both layers, controls, panel."""

    def __init__(self, store: EnhanceDefaultsStore, *, fake: bool):
        super().__init__()
        self.view = EditorCanvasView()
        self.scene = QGraphicsScene(self.view)
        self.view.setScene(self.scene)
        self.view.set_content_rect(QRectF(0, 0, 640, 480))
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self.view)
        self.norm = NormalizationControls(store=store)
        self.chrome = EditorChrome(self, self.view, self.norm)
        self.host = FakeHost()
        if fake:
            self.layers = {MASK: FakeLayer(MASK), SKELETON: FakeLayer(SKELETON)}
        else:
            self.layers = {
                MASK: MaskLayer(self.host, self.scene),
                SKELETON: SkeletonLayer(self.host, self.scene),
            }
        self.controls = CorrectControls(self, self.chrome, self.layers)
        self.panel = LayerPanel(list(self.layers.values()), self)
        self.chrome.register(side_panel=self.panel)


@pytest.fixture
def make_page(app, tmp_path):
    made = []

    def make(size=(1000, 700), *, show=True, fake=False):
        store = EnhanceDefaultsStore(
            user_path=str(tmp_path / "user.json"), factory_path=str(tmp_path / "factory.json")
        )
        page = Page(store, fake=fake)
        page.resize(*size)
        if show:
            page.show()
            settle(app, page)
        made.append(page)
        return page

    yield make
    # Delete the pages while their wrappers are intact (page -> chrome -> OverlayLayout -> page
    # is a cycle the GC could otherwise empty while Qt still delivers teardown events).
    delete_widgets(made, prepare=_quiet_skeleton)


def _quiet_skeleton(page) -> None:
    """Stop the skeleton layer's scheduled render before its page goes."""
    skeleton = page.layers[SKELETON]
    if isinstance(skeleton, SkeletonLayer):
        skeleton.cancel_scheduled_render()


# ------------------------------------------------------------------------------ helpers
def settle(app, page=None) -> None:
    for _ in range(3):
        app.processEvents()
    if page is not None:
        page.chrome.relayout()
        app.processEvents()


def record(signal) -> list[tuple]:
    seen: list[tuple] = []
    signal.connect(lambda *args: seen.append(args))
    return seen


def opaque_colours(image: QImage) -> set[str]:
    image = image.convertToFormat(QImage.Format.Format_ARGB32)
    return {
        QColor(image.pixel(x, y)).name()
        for y in range(image.height())
        for x in range(image.width())
        if QColor.fromRgba(image.pixel(x, y)).alpha() == 255
    }


def text_colours(image: QImage, min_alpha: int = 128, tolerance: int = 2) -> set[str]:
    """The colours the text is drawn in. Antialiased glyphs of a small font (DejaVu on Linux)
    have no fully opaque pixel, so every pixel at least ``min_alpha`` covered counts. The
    grab is premultiplied; un-premultiplying a partly covered pixel is off by a level or two
    (the 2 levels ``drawn_in`` allows), so colours within ``tolerance`` levels of a more
    frequent one count as it."""
    image = image.convertToFormat(QImage.Format.Format_ARGB32)
    seen = Counter(
        QColor(image.pixel(x, y)).name()
        for y in range(image.height())
        for x in range(image.width())
        if QColor.fromRgba(image.pixel(x, y)).alpha() >= min_alpha
    )
    found: list[QColor] = []
    for name, _count in seen.most_common():
        colour = QColor(name)
        if not any(
            max(
                abs(colour.red() - c.red()),
                abs(colour.green() - c.green()),
                abs(colour.blue() - c.blue()),
            )
            <= tolerance
            for c in found
        ):
            found.append(colour)
    return {colour.name() for colour in found}


def pixel(image: QImage, point: QPoint) -> str:
    return QColor(image.pixel(point)).name()


def right(widget) -> int:
    return widget.geometry().x() + widget.geometry().width()

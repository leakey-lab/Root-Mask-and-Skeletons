"""The Library's status badge: three marks (mask, skeleton, class labels), its words and tooltip.

Colour says the kind (mask green, skeleton orange, class labels purple); the shape says the
state: a ring is "none", a filled circle a model prediction, a filled diamond GT (for the
class labels: "traced"). The delegate draws it from the injected status providers at paint
time, in the same pixmap whatever the row's selection (Qt's Selected mode tinted it, W10), and
a selected row's badge and label do not move.

Offscreen and file-free.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import pytest
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QColor, QIcon, QImage, QPalette
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QStyle, QStyleOptionViewItem, QTreeWidget

from app.gui.library_index import (
    LibraryIndex,
    _badge_pixmap,
    build_hierarchy,
    status_text,
)
from app.gui.widgets import tokens
from tests.qt_teardown import flush_deleted

STEM = "24EFCAM1_T001_L011_2024.07.18_111530_001_SMM"


def _mark_pixels(pixmap, mark: int):
    """The pixels of mark ``mark`` (0 mask, 1 skeleton, 2 labels): its 10 px column, which
    starts 1 px right of the pixmap column grid so that no diamond tip leaves it."""
    image = pixmap.toImage()
    dpr = pixmap.devicePixelRatio()
    left, right = round((10 * mark + 1) * dpr), round((10 * mark + 11) * dpr)
    for x in range(left, min(right, image.width())):
        for y in range(image.height()):
            yield image.pixelColor(x, y)


def opaque(pixmap, mark: int) -> int:
    """How many pixels of the mark are more than half covered."""
    return sum(1 for c in _mark_pixels(pixmap, mark) if c.alpha() > 128)


def dominant_colour(pixmap, mark: int) -> str:
    """The most common fully covered colour of the mark, as ``#rrggbb``."""
    counts = Counter(c.name() for c in _mark_pixels(pixmap, mark) if c.alpha() == 255)
    return counts.most_common(1)[0][0]


def test_the_badge_has_three_marks_and_the_shape_carries_the_state(app):
    pix = _badge_pixmap("gt", "prediction", True)
    assert pix.size() / pix.devicePixelRatio() == QSize(32, 12)
    ring = opaque(_badge_pixmap("none", "none", False), 0)
    circle = opaque(_badge_pixmap("prediction", "none", False), 0)
    diamond = opaque(_badge_pixmap("gt", "none", False), 0)
    assert circle > diamond > ring > 0  # 63 px² circle, ~50 px² diamond, a thin ring (dpr 2)
    assert dominant_colour(pix, 0) == tokens.MASK
    assert dominant_colour(pix, 1) == tokens.SKEL
    assert dominant_colour(pix, 2) == tokens.ACCENT
    # the labels diamond is the same shape as the other diamonds
    assert opaque(_badge_pixmap("gt", "gt", True), 2) == opaque(
        _badge_pixmap("none", "none", True), 2
    )
    assert opaque(_badge_pixmap("none", "none", True), 2) == diamond


def test_each_mark_follows_its_own_state_only(app):
    base = _badge_pixmap("none", "none", False)
    mask_only = _badge_pixmap("gt", "none", False)
    assert opaque(mask_only, 0) != opaque(base, 0)
    assert opaque(mask_only, 1) == opaque(base, 1) and opaque(mask_only, 2) == opaque(base, 2)
    skeleton_only = _badge_pixmap("none", "prediction", False)
    assert opaque(skeleton_only, 1) != opaque(base, 1)
    assert opaque(skeleton_only, 0) == opaque(base, 0) and opaque(skeleton_only, 2) == opaque(
        base, 2
    )
    labels_only = _badge_pixmap("none", "none", True)
    assert opaque(labels_only, 2) != opaque(base, 2)
    assert opaque(labels_only, 0) == opaque(base, 0) and opaque(labels_only, 1) == opaque(base, 1)
    assert dominant_colour(base, 0) == tokens.TEXT_FAINT


def test_status_text_words():
    assert status_text("gt", "prediction", False) == "Mask: GT · Skeleton: prediction · No classes"
    assert status_text("none", "none", True) == "No mask · No skeleton · Classes: traced"
    assert status_text("prediction", "gt", False) == "Mask: prediction · Skeleton: GT · No classes"


def _providers(masks=None, skeletons=None, labels=()):
    masks, skeletons, labels = dict(masks or {}), dict(skeletons or {}), set(labels)
    return (
        dict(
            path_of=lambda n: f"D:/p/{n}.jpg",
            mask_state=lambda n: masks.get(n, "none"),
            skeleton_state=lambda n: skeletons.get(n, "none"),
            has_labels=lambda n: n in labels,
        ),
        masks,
        skeletons,
        labels,
    )


@pytest.fixture
def index_with_providers(app):
    providers, masks, skeletons, labels = _providers(
        masks={STEM: "gt"}, skeletons={STEM: "prediction"}
    )
    tree = QTreeWidget()
    index = LibraryIndex(tree, **providers)
    index.build(build_hierarchy([STEM]))
    index.states = (masks, skeletons, labels)
    yield tree, index
    tree.deleteLater()
    del tree, index
    flush_deleted()


def test_status_of_is_a_state_triple(index_with_providers):
    _tree, index = index_with_providers
    assert index.status_of(STEM) == ("gt", "prediction", False)
    assert index.status_of("other") == ("none", "none", False)


def test_the_tooltip_names_the_state_of_each_kind(index_with_providers):
    _tree, index = index_with_providers
    assert index.tooltip_for(STEM) == (
        f"{STEM}\nMask: GT · Skeleton: prediction · No classes\nD:/p/{STEM}.jpg"
    )


def test_the_delegate_caches_one_icon_per_state_triple(index_with_providers):
    """A repaint of a row whose state did not change draws a cached icon, not a new one."""
    tree, _index = index_with_providers
    delegate = tree.itemDelegate()
    first = delegate.status_icon("gt", "prediction", False).cacheKey()
    other = delegate.status_icon("none", "none", True).cacheKey()
    assert delegate.status_icon("gt", "prediction", False).cacheKey() == first
    assert delegate.status_icon("none", "none", True).cacheKey() == other != first


@pytest.mark.parametrize(
    "state",
    [
        ("none", "none", False),
        ("prediction", "none", False),
        ("none", "gt", False),
        ("prediction", "gt", False),
        ("gt", "prediction", True),
        ("gt", "gt", True),
    ],
)
def test_the_badge_is_not_tinted_when_selected(index_with_providers, state):
    tree, _index = index_with_providers
    icon = tree.itemDelegate().status_icon(*state)
    size = QSize(32, 12)
    normal = icon.pixmap(size, QIcon.Mode.Normal).toImage()
    for mode in (QIcon.Mode.Selected, QIcon.Mode.Active):
        assert icon.pixmap(size, mode).toImage() == normal, mode


def test_the_delegate_draws_the_state_the_providers_report(index_with_providers):
    tree, index = index_with_providers
    masks, _skeletons, labels = index.states
    delegate = tree.itemDelegate()
    model_index = tree.indexFromItem(index.item_for(STEM))

    def icon_key():
        option = QStyleOptionViewItem()
        delegate.initStyleOption(option, model_index)
        assert not option.icon.isNull() and option.decorationSize == QSize(32, 12)
        return option.icon.cacheKey()

    before = icon_key()
    masks[STEM] = "prediction"
    index.refresh_status(STEM)
    assert icon_key() != before
    changed = icon_key()
    labels.add(STEM)
    assert icon_key() != changed


def test_a_group_row_keeps_its_own_icon_and_gets_no_badge(index_with_providers):
    tree, _index = index_with_providers
    delegate = tree.itemDelegate()
    rows = [tree.topLevelItem(0)]
    while rows[-1].childCount() and rows[-1].child(0).childCount():
        rows.append(rows[-1].child(0))  # field, tube, date
    assert len(rows) == 3
    for row in rows:
        option = QStyleOptionViewItem()
        delegate.initStyleOption(option, tree.indexFromItem(row))
        assert option.icon.cacheKey() == row.icon(0).cacheKey(), row.text(0)
        assert option.decorationSize != QSize(32, 12), row.text(0)


def _name(tube, depth, date="2024.07.18", hhmmss="111530", field="24EFCAM1"):
    return f"{field}_T{tube:03d}_L{depth:03d}_{date}_{hhmmss}_001_SMM"


def synth_names(n_tubes=3, dates=("2024.07.18", "2024.08.07"), depths=(11, 12, 13, 14, 15)):
    names = []
    for t in range(1, n_tubes + 1):
        for d in dates:
            for k, depth in enumerate(depths):
                names.append(_name(t, depth, d, f"{9 + k // 60:02d}{k % 60:02d}{(7 * k) % 60:02d}"))
    return names


def _built(app, names=None, **kw):
    tree = QTreeWidget()
    index = LibraryIndex(tree, **kw)
    index.build(build_hierarchy(names or synth_names()))
    return tree, index


def test_status_icon_renders_under_the_app_stylesheet(app):
    """ForegroundRole is ignored under the app QSS (UI-09); the status badge is not."""
    from PyQt6.QtGui import QColor

    from app.gui.widgets import tokens

    qss_path = os.path.join(
        os.path.dirname(__file__), "..", "resources", "themes", "dark_theme.qss"
    )
    with open(qss_path, encoding="utf-8") as fh:
        qss = fh.read()
    name = _name(1, 11)
    tree, index = _built(app, [name], mask_state=lambda n: "prediction")
    tree.setStyleSheet(qss)
    tree.resize(320, 200)
    tree.expandAll()
    tree.show()
    QApplication.processEvents()
    img = tree.viewport().grab().toImage()
    rect = tree.visualItemRect(index.item_for(name))
    target = QColor(tokens.MASK)
    hits = 0
    for x in range(rect.left(), min(rect.right(), rect.left() + 120)):
        for y in range(rect.top(), rect.bottom()):
            c = img.pixelColor(x, y)
            if (
                abs(c.red() - target.red()) < 40
                and abs(c.green() - target.green()) < 40
                and abs(c.blue() - target.blue()) < 40
            ):
                hits += 1
    tree.close()
    assert hits > 5


APP_QSS = Path(__file__).resolve().parents[1] / "resources" / "themes" / "dark_theme.qss"


STEMS = [f"24EFCAM1_T001_L0{d}_2024.07.18_1115{d}_001_SMM" for d in (11, 12, 13)]


@pytest.fixture
def styled_app(app):
    """The application under the app stylesheet and highlight colour, for one test only (the
    badge tests above run without the stylesheet)."""
    a = app
    previous_qss, previous_palette = a.styleSheet(), a.palette()
    palette = QPalette(previous_palette)
    palette.setColor(QPalette.ColorRole.Highlight, QColor(tokens.SEL))  # as main.py does
    a.setPalette(palette)
    a.setStyleSheet(APP_QSS.read_text(encoding="utf-8"))
    yield a
    a.setStyleSheet(previous_qss)
    a.setPalette(previous_palette)


@pytest.fixture
def library(styled_app):
    """A library tree styled like the app's (its own stylesheet over the app's)."""
    from app.gui.library_index import LibraryIndex, index_names
    from app.gui.ui_panels import library_tree_qss

    tree = QTreeWidget()
    tree.setStyleSheet(library_tree_qss())
    index = LibraryIndex(
        tree,
        path_of=lambda n: n,
        mask_state=lambda n: "prediction",
        skeleton_state=lambda n: "prediction",
    )
    index.build(*index_names(STEMS))
    tree.expandAll()
    tree.resize(260, 300)
    tree.show()
    QTest.qWaitForWindowExposed(tree)
    yield tree, index
    tree.close()
    tree.deleteLater()
    del tree, index
    flush_deleted()


def _rects(tree, index, item, selected: bool) -> tuple[int, int]:
    """x of the badge and of the text of ``item``, painted selected or not."""
    model_index = tree.indexFromItem(item)
    option = QStyleOptionViewItem()
    option.rect = tree.visualRect(model_index)
    tree.itemDelegate().initStyleOption(option, model_index)
    if selected:
        option.state |= QStyle.StateFlag.State_Selected
    else:
        option.state &= ~QStyle.StateFlag.State_Selected
    style = tree.style()
    decoration = style.subElementRect(QStyle.SubElement.SE_ItemViewItemDecoration, option, tree)
    text = style.subElementRect(QStyle.SubElement.SE_ItemViewItemText, option, tree)
    return decoration.x(), text.x()


def test_selecting_a_row_does_not_move_its_badge_or_label(library):
    tree, index = library
    leaf = index.leaves()[1]
    assert _rects(tree, index, leaf, True) == _rects(tree, index, leaf, False)


def test_the_selected_row_keeps_its_accent_bar(library):
    tree, index = library
    leaf = index.leaves()[1]
    tree.setCurrentItem(leaf)
    QApplication.processEvents()
    rect = tree.visualItemRect(leaf)
    image = tree.viewport().grab(rect).toImage().convertToFormat(QImage.Format.Format_ARGB32)
    y = image.height() // 2
    left = [QColor(image.pixel(x, y)) for x in range(0, 6)]
    bar = QColor(tokens.SEL)
    assert any(
        abs(c.red() - bar.red()) + abs(c.green() - bar.green()) + abs(c.blue() - bar.blue()) < 40
        for c in left
    ), [c.name() for c in left]

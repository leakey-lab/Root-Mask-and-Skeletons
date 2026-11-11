"""The zoom cluster against a real canvas view (app/gui/widgets ZoomCluster,
app/gui/canvas_view.py): its buttons step along the view's zoom ladder, and its readout's
menu lists Fit then the presets, checks the current zoom and sets it.
"""

import pytest
from PyQt6.QtCore import QPoint, QRectF, Qt
from PyQt6.QtGui import QColor, QPixmap
from PyQt6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QMenu, QWidget

from app.gui import canvas_gl
from app.gui.widgets import ZoomCluster
from tests.canvas_view_support import View as _View
from tests.qt_teardown import flush_deleted

IMAGE_W, IMAGE_H = 800, 600

pytestmark = pytest.mark.usefixtures("no_prompts")


@pytest.fixture
def view(app, monkeypatch):
    """A shown 400x300 raster view over an 800x600 image at 100 %."""
    monkeypatch.setenv(canvas_gl.ENV_VAR, "0")
    view = _View()
    scene = QGraphicsScene(view)
    pm = QPixmap(IMAGE_W, IMAGE_H)
    pm.fill(QColor(90, 90, 90))
    scene.addItem(QGraphicsPixmapItem(pm))
    view.setScene(scene)
    view.set_content_rect(QRectF(0, 0, IMAGE_W, IMAGE_H))
    view.resize(400, 300)
    view.show()
    app.processEvents()
    view.set_zoom(1.0)
    yield view
    view.close()
    view.deleteLater()
    del view, scene
    flush_deleted()


# ------------------------------------------------------------------------ zoom cluster
@pytest.fixture
def cluster(view):
    host = QWidget()
    zoom = ZoomCluster(host)
    zoom.bind(view)
    yield zoom
    host.deleteLater()
    del host, zoom
    flush_deleted()


def _menu_entries(menu: QMenu) -> list[str]:
    return [action.text() for action in menu.actions() if not action.isSeparator()]


def test_zoom_cluster_buttons_step_along_the_ladder(view, cluster):
    view.set_zoom(1.63)
    cluster.zoom_in_button.click()
    assert view.zoom_factor == pytest.approx(2.0)
    cluster.zoom_in_button.click()
    assert view.zoom_factor == pytest.approx(3.0)
    cluster.zoom_out_button.click()
    cluster.zoom_out_button.click()
    assert view.zoom_factor == pytest.approx(1.5)
    assert cluster.zoom_pct_label.text() == "150%"


def test_zoom_readout_menu_lists_fit_then_the_presets(view, cluster):
    menu = cluster.preset_menu()
    assert _menu_entries(menu) == ["Fit", "50%", "100%", "200%", "400%", "800%", "1600%"]
    presets = [a for a in menu.actions() if not a.isSeparator() and a.text() != "Fit"]
    assert all(action.isCheckable() for action in presets)


@pytest.mark.parametrize(("zoom", "checked"), [(2.0, ["200%"]), (1.0, ["100%"]), (1.63, [])])
def test_zoom_readout_menu_checks_the_current_zoom(view, cluster, zoom, checked):
    view.set_zoom(zoom)
    menu = cluster.preset_menu()
    assert [a.text() for a in menu.actions() if a.isChecked()] == checked


@pytest.mark.parametrize(
    ("entry", "zoom"),
    [("50%", 0.5), ("100%", 1.0), ("200%", 2.0), ("400%", 4.0), ("800%", 8.0), ("1600%", 16.0)],
)
def test_zoom_readout_presets_set_that_zoom(view, cluster, entry, zoom):
    view.set_zoom(1.63)
    menu = cluster.preset_menu()
    next(a for a in menu.actions() if a.text() == entry).trigger()
    assert view.zoom_factor == pytest.approx(zoom)


def test_zoom_readout_menu_fit_fits_the_view(view, cluster):
    view.fit_to_view()
    fitted = view.zoom_factor
    view.set_zoom(8.0)
    menu = cluster.preset_menu()
    next(a for a in menu.actions() if a.text() == "Fit").trigger()
    assert view.zoom_factor == pytest.approx(fitted)


def test_zoom_readout_menu_fit_uses_the_editor_fit(view):
    host = QWidget()
    zoom = ZoomCluster(host)
    fits = []
    zoom.bind(view, fit=lambda: fits.append("fit"))
    next(a for a in zoom.preset_menu().actions() if a.text() == "Fit").trigger()
    assert fits == ["fit"]
    host.deleteLater()


def test_right_click_on_the_readout_opens_the_presets(view, cluster, monkeypatch):
    shown = []
    monkeypatch.setattr(QMenu, "exec", lambda menu, *args: shown.append(_menu_entries(menu)))
    cluster.zoom_pct_label.customContextMenuRequested.emit(QPoint(5, 5))
    assert shown == [["Fit", "50%", "100%", "200%", "400%", "800%", "1600%"]]
    assert cluster.zoom_pct_label.contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu

"""tests/window_support.py: ``open_folder`` waits until the load it started has ended.

Waiting for the image count and the tree's rows is not enough: the Library keeps the
previous project until the scan delivers, so a folder opened after one with as many images
passed at once. The other helpers are straight-line and have no test of their own.
"""

import time

import pytest

from app.gui import image_manager
from tests.qt_teardown import delete_widgets
from tests.qt_wait import wait_until
from tests.window_support import build_window, open_folder, quiet_window, write_photos

pytestmark = pytest.mark.usefixtures("window_guards")

FIRST = [f"24EFCAM1_T001_L0{d}_2024.07.18_1115{d}_001_SMM" for d in (11, 12, 13)]
SECOND = [f"24EFCAM1_T002_L0{d}_2024.07.18_1116{d}_001_SMM" for d in (31, 32, 33)]


@pytest.fixture
def make_window(app, tmp_path):
    made: list = []

    def make(**kwargs):
        made.append(build_window(tmp_path / f"s{len(made)}.ini", **kwargs))
        return made[-1]

    yield make
    delete_widgets(made, prepare=quiet_window)


@pytest.fixture
def slow_scan(monkeypatch):
    """Every folder scan starts ``delay[0]`` seconds late, as on a busy machine."""
    delay = [0.3]
    real = image_manager.ImageLoaderThread.work

    def work(self):
        time.sleep(delay[0])
        real(self)

    monkeypatch.setattr(image_manager.ImageLoaderThread, "work", work)
    return delay


def _stems(window) -> list[str]:
    return sorted(window.image_manager.images)


def test_open_folder_waits_for_the_load_to_end(make_window, tmp_path):
    window = make_window()
    open_folder(window, write_photos(tmp_path / "first", FIRST), len(FIRST))

    assert not window._loading_project
    assert _stems(window) == FIRST
    assert len(window._image_leaves()) == len(FIRST)


def test_a_second_folder_of_as_many_images_is_waited_for_too(make_window, tmp_path, slow_scan):
    window = make_window()
    open_folder(window, write_photos(tmp_path / "first", FIRST), len(FIRST))

    # The Library keeps the first folder until the late scan delivers the second.
    open_folder(window, write_photos(tmp_path / "second", SECOND), len(SECOND))

    assert not window._loading_project
    assert _stems(window) == SECOND


def test_open_folder_fails_when_the_load_does_not_end_in_time(make_window, tmp_path, slow_scan):
    window = make_window()
    slow_scan[0] = 1.0
    with pytest.raises(AssertionError, match="did not end with 3 images in 0.2 s"):
        open_folder(window, write_photos(tmp_path / "first", FIRST), len(FIRST), timeout=0.2)
    assert wait_until(lambda: not window._loading_project)  # no late scan in a later test

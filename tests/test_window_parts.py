"""The library navigator and the project loader (``app/gui/window/navigator.py``,
``project_loader.py``) against a stand-in window (review CQ-07).

Each part works on the window it is given, so a SimpleNamespace stands in for it and each
branch runs without a MainWindow. The real-window tests of these units are the Library's.

(Parked here by A10 in wave 1; the lead decides the final home at integration.)
"""

from __future__ import annotations

from types import SimpleNamespace

from app.gui.panels import Panel
from app.gui.window import navigator, project_loader


class Item:
    """A library row holding ``name`` (None: a group row), under the group ``parent``."""

    def __init__(self, name: str | None, parent=None, *, expanded=True, opened=None) -> None:
        self.name = name
        self._parent = parent
        self.expanded = expanded
        self.opened = [] if opened is None else opened  # names ``setExpanded`` reached

    def data(self, _column, _role):
        return self.name

    def parent(self):
        return self._parent

    def isExpanded(self) -> bool:  # noqa: N802 - Qt API
        return self.expanded

    def setExpanded(self, expanded: bool) -> None:  # noqa: N802 - Qt API
        self.expanded = expanded
        self.opened.append(self.name)


def test_the_debounce_starts_for_image_rows_only():
    started = []
    window = SimpleNamespace(_tree_load_timer=SimpleNamespace(start=lambda: started.append(1)))
    navigator.on_tree_current_changed(window, None)
    navigator.on_tree_current_changed(window, Item(None))  # a group row
    assert started == []
    navigator.on_tree_current_changed(window, Item("a"), Item("b"))
    assert started == [1]


def test_the_debounce_opens_a_new_image_and_leaves_the_keyboard_on_the_tree():
    opened, focused = [], []
    item = Item("b")
    window = SimpleNamespace(
        file_list=SimpleNamespace(
            currentItem=lambda: item, hasFocus=lambda: True, setFocus=lambda: focused.append(1)
        ),
        _shown_leaf_name="a",
        on_image_selected=opened.append,
    )
    navigator.load_current_tree_item(window)
    assert opened == [item] and focused == [1]
    window._shown_leaf_name = "b"  # already shown: nothing to load
    navigator.load_current_tree_item(window)
    assert opened == [item]


def paging_window(
    opened: list, step_to, *, editor=None, current=None, shown=None
) -> SimpleNamespace:
    """``current``: the tree's current row (default an image row); ``shown``: the path
    on the display."""
    selected, scrolled = [], []
    row = Item("tree") if current is None else current
    window = SimpleNamespace(
        right_panel=SimpleNamespace(currentIndex=lambda: Panel.CORRECT),
        file_list=SimpleNamespace(
            currentItem=lambda: row,
            setCurrentItem=selected.append,
            scrollToItem=scrolled.append,
        ),
        library_index=SimpleNamespace(step=lambda current, delta, visible_only: step_to(current)),
        display_controller=SimpleNamespace(current_image=shown),
        on_image_selected=opened.append,
        selected=selected,
        scrolled=scrolled,
    )
    window._editors = lambda: {} if editor is None else {Panel.CORRECT: editor}
    window._leaf_for_image = lambda path: Item(f"leaf of {path}") if path else None
    return window


def test_paging_steps_from_the_image_open_in_the_editor():
    opened, froms = [], []
    target = Item("next")
    editor = SimpleNamespace(current_image_path="open.jpg")
    window = paging_window(
        opened, lambda current: froms.append(current.name) or target, editor=editor
    )
    navigator.navigate_image(window, 1)
    assert froms == ["leaf of open.jpg"] and window.selected == [target] and opened == [target]


def test_paging_stops_after_the_last_image():
    opened = []
    window = paging_window(opened, lambda _current: None)
    navigator.navigate_image(window, 1)
    assert window.selected == [] and opened == []


def test_paging_from_a_group_row_steps_from_the_image_on_the_display():
    # X6-01 (owner: A): after a click on a tube, date or field row, PgDn / PgUp go on
    # from the image on screen, as Correct does (they used to start from the group row).
    opened, froms = [], []
    target = Item("next")
    window = paging_window(
        opened,
        lambda current: froms.append(current.name) or target,
        current=Item(None),
        shown="shown.jpg",
    )
    navigator.navigate_image(window, 1)
    assert froms == ["leaf of shown.jpg"] and opened == [target]


def test_paging_from_a_group_row_with_nothing_shown_steps_from_the_row():
    opened, froms = [], []
    group = Item(None)
    window = paging_window(
        opened, lambda current: froms.append(current) or Item("first"), current=group
    )
    navigator.navigate_image(window, 1)
    assert froms == [group]  # LibraryIndex.step gives the first image from a group row


def test_paging_opens_every_closed_group_above_the_opened_row():
    # H1-01: Qt's scrollToItem opens no group during the tree's expand animation, so a
    # fast PgDn across collapsed groups left the row hidden; the navigator opens them.
    opened, expanded = [], []
    field = Item("field", opened=expanded)
    tube = Item("tube", field, expanded=False, opened=expanded)
    target = Item("next", Item("date", tube, expanded=False, opened=expanded))
    window = paging_window(opened, lambda _current: target)
    navigator.navigate_image(window, 1)
    assert expanded == ["date", "tube"]  # innermost first; the open field is left alone
    assert window.selected == [target] and window.scrolled == [target]
    assert opened == [target]


def test_an_editor_that_kept_its_image_points_the_library_back(tmp_path):
    reselected = []
    window = SimpleNamespace(_select_tree_image=reselected.append)
    kept = str(tmp_path / "kept.jpg")
    editor = SimpleNamespace(current_image_path=kept)
    assert navigator.kept_other_image(window, editor, str(tmp_path / "asked.jpg")) is True
    assert reselected == [kept]
    assert navigator.kept_other_image(window, editor, kept) is False


def loader_window(calls: list, *, keeps_edits=False) -> SimpleNamespace:
    editor = SimpleNamespace(maybe_save_changes=lambda: not keeps_edits)
    window = SimpleNamespace(
        notify=lambda message, kind="success", timeout=3200, key=None: calls.append(
            (message, kind, key)
        ),
        _refresh_recents=lambda: calls.append("recents"),
        app_stack=SimpleNamespace(setCurrentIndex=lambda i: calls.append(("page", i))),
        image_manager=SimpleNamespace(load_images=lambda d: calls.append("scan"), images={}),
    )
    window._editors = lambda: {Panel.CORRECT: editor}
    return window


def test_opening_a_missing_folder_says_so_and_changes_nothing(tmp_path):
    calls = []
    missing = str(tmp_path / "gone")
    project_loader.open_path(loader_window(calls), missing)
    assert calls == [(f"Folder not found: {missing}", "danger", "project-load"), "recents"]


def test_an_editor_that_keeps_its_edits_cancels_the_open(tmp_path):
    calls = []
    window = loader_window(calls, keeps_edits=True)
    project_loader.open_path(window, str(tmp_path))
    assert calls == [] and not hasattr(window, "_last_loaded_dir")


def test_a_folder_without_images_keeps_what_was_loaded(tmp_path):
    calls, done = [], []
    window = loader_window(calls)
    window._loading_project = True
    window.empty_state = SimpleNamespace(set_loading=lambda name: calls.append(("empty", name)))
    window.task_progress = SimpleNamespace(finish=done.append, fail=lambda m: calls.append(m))
    window._end_failed_load = lambda m, k: project_loader.end_failed_load(window, m, k)
    project_loader.on_loading_empty(window, str(tmp_path / "trial"))
    message = "No .png/.jpg/.jpeg images in trial — nothing loaded"
    assert window._loading_project is False and done == [message]
    # Nothing was loaded before: back to the Welcome screen.
    assert calls == [("empty", None), (message, "warn", "project-load"), ("page", 0)]

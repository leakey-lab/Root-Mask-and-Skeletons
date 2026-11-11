"""Guard: the right-panel page contract after the Correct swap (spec 3.1, plan C19).

``MainWindow.right_panel`` is a ``QStackedWidget`` whose pages are named by
``app.gui.panels.Panel``. One Correct stage replaced the mask and skeleton editors and the
pages were renumbered on purpose, with no aliases (spec D18); the Trace GT page went too
(WG-T11): ``DISPLAY = 0``, ``CORRECT = 1``, ``VISUALIZATION = 2``. Two kinds of stale caller
would survive that silently, so both are rejected anywhere under ``app/``:

* an int literal page index: ``right_panel.setCurrentIndex(<int>)``,
  ``right_panel.widget(<int>)``, or ``right_panel.currentIndex()`` compared with ``==``,
  ``!=``, ``in`` or ``not in`` to an int literal. Name pages with ``Panel`` members.
* a page key ``Panel.coerce`` rejects: ``switch_right_panel`` logs and ignores it, so
  ``switch_right_panel("mask_tracing")`` (a removed editor's key) or a typo such as
  ``"dispaly"`` would do nothing, without an error. Every string literal passed to
  ``switch_right_panel(...)`` or ``Panel.coerce(...)`` must be a key ``Panel.coerce``
  accepts.

The scan is syntactic (AST): it cannot follow an index or a key through a variable, so it
pins the call sites as written. The self-tests prove each rule can fail. The ``Panel`` enum
itself (values, keys, ``coerce``) is tested at the end.
"""

from __future__ import annotations

import ast
import textwrap
from collections.abc import Callable
from pathlib import Path

import pytest

from app.gui.panels import Panel

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"

#: ``right_panel`` methods whose first argument is a page index.
PAGE_ARGUMENT = frozenset({"setCurrentIndex", "widget"})
#: ``right_panel`` methods that return a page index.
PAGE_RESULT = frozenset({"currentIndex"})
#: Comparisons that test a page index for a value.
MATCHING = (ast.Eq, ast.NotEq, ast.In, ast.NotIn)


def _is_right_panel(node: ast.AST) -> bool:
    """``right_panel`` as a name or as the last attribute (``self.right_panel``)."""
    return (isinstance(node, ast.Name) and node.id == "right_panel") or (
        isinstance(node, ast.Attribute) and node.attr == "right_panel"
    )


def _right_panel_call(node: ast.AST, methods: frozenset[str]) -> bool:
    """``<right_panel>.<method>(...)`` for one of ``methods``."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in methods
        and _is_right_panel(node.func.value)
    )


def _is_int_literal(node: ast.AST) -> bool:
    """An int constant (``True`` / ``False`` are ints to Python, not page indices)."""
    return isinstance(node, ast.Constant) and type(node.value) is int


def _names_an_int(node: ast.AST) -> bool:
    """An int literal, or a tuple / list / set display holding one."""
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return any(_is_int_literal(item) for item in node.elts)
    return _is_int_literal(node)


def _compares_index_to_int(node: ast.Compare) -> bool:
    """A ``==`` / ``!=`` / ``in`` / ``not in`` link of ``currentIndex()`` and an int literal."""
    operands = [node.left, *node.comparators]
    return any(
        isinstance(op, MATCHING)
        and (
            (_right_panel_call(left, PAGE_RESULT) and _names_an_int(right))
            or (_right_panel_call(right, PAGE_RESULT) and _names_an_int(left))
        )
        for op, left, right in zip(node.ops, operands[:-1], operands[1:], strict=True)
    )


def int_page_indices(source: str) -> list[int]:
    """Line numbers where an int literal names a right-panel page in ``source``."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if _right_panel_call(node, PAGE_ARGUMENT):
            arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
            if any(_is_int_literal(argument) for argument in arguments):
                found.append(node.lineno)
        elif isinstance(node, ast.Compare) and _compares_index_to_int(node):
            found.append(node.lineno)
    return sorted(found)


def right_panel_page_calls(source: str) -> int:
    """How many ``right_panel`` page calls (index argument or result) ``source`` makes."""
    methods = PAGE_ARGUMENT | PAGE_RESULT
    return sum(_right_panel_call(node, methods) for node in ast.walk(ast.parse(source)))


def _takes_a_page_key(node: ast.AST) -> bool:
    """``<x>.switch_right_panel(...)`` or ``Panel.coerce(...)``."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return False
    if node.func.attr == "switch_right_panel":
        return True
    return (
        node.func.attr == "coerce"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "Panel"
    )


def literal_page_keys(source: str) -> list[tuple[int, str]]:
    """``(line, key)`` of every string literal passed as the page of a key-taking call."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if _takes_a_page_key(node):
            for argument in [*node.args[:1], *(k.value for k in node.keywords)]:
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    found.append((node.lineno, argument.value))
    return sorted(found)


def _accepted(key: str) -> bool:
    try:
        Panel.coerce(key)
    except ValueError:
        return False
    return True


def unknown_page_keys(source: str) -> list[int]:
    """Line numbers where a literal page key is one ``Panel.coerce`` rejects."""
    return [line for line, key in literal_page_keys(source) if not _accepted(key)]


def _app_sources() -> dict[str, str]:
    """Every module under ``app/`` by repo-relative path."""
    return {
        path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(APP.rglob("*.py"))
    }


def _offenders(sources: dict[str, str], scan: Callable[[str], list[int]]) -> dict[str, list[str]]:
    """``{path: ["<line>: <code>", ...]}`` for every module where ``scan`` finds lines."""
    found = {}
    for path, source in sources.items():
        if lines := scan(source):
            code = source.splitlines()
            found[path] = [f"{line}: {code[line - 1].strip()}" for line in lines]
    return found


def test_the_index_scan_flags_int_literals_and_passes_panel_members() -> None:
    source = textwrap.dedent(
        """\
        self.right_panel.setCurrentIndex(1)
        mw.right_panel.widget(3)
        right_panel.setCurrentIndex(index=2)
        if self.right_panel.currentIndex() == 4:
            pass
        if 1 != mw.right_panel.currentIndex():
            pass
        if self.right_panel.currentIndex() in (0, Panel.CORRECT):
            pass
        self.right_panel.setCurrentIndex(Panel.CORRECT)
        self.right_panel.widget(Panel.VISUALIZATION)
        self.right_panel.setCurrentIndex(target)
        if self.right_panel.currentIndex() == Panel.DISPLAY:
            pass
        if self.right_panel.currentIndex() < 0:
            pass
        self.right_panel.addWidget(page, 1)
        self.stack.setCurrentIndex(1)
        self.right_panel.setCurrentIndex(True)
        """
    )
    assert int_page_indices(source) == [1, 2, 3, 4, 6, 8]
    assert right_panel_page_calls(source) == 12


def test_the_key_scan_flags_keys_panel_rejects_and_passes_the_others() -> None:
    source = textwrap.dedent(
        '''\
        """Docstring: "mask_tracing" and "gt_tracing" are gone."""
        self.switch_right_panel("mask_tracing")
        mw.switch_right_panel("gt_tracing")
        self._mw.switch_right_panel("dispaly")
        Panel.coerce("SKELETON_CORRECTION")
        window.switch_right_panel(panel="nope")
        mw.switch_right_panel("display")
        Panel.coerce("CORRECT")
        mw.switch_right_panel(Panel.VISUALIZATION)
        mw.switch_right_panel(target)
        other.coerce("mask_tracing")
        message = "switch_right_panel('mask_tracing')"
        '''
    )
    assert unknown_page_keys(source) == [2, 3, 4, 5, 6]
    assert len(literal_page_keys(source)) == 7


def test_no_int_literal_names_a_right_panel_page() -> None:
    sources = _app_sources()
    seen = sum(right_panel_page_calls(source) for source in sources.values())
    assert seen > 0, "the scan found no right_panel page call: it would pass vacuously"
    offenders = _offenders(sources, int_page_indices)
    assert offenders == {}, "name right_panel pages with app.gui.panels.Panel members"


def test_every_literal_page_key_in_app_is_one_panel_accepts() -> None:
    sources = _app_sources()
    seen = sum(len(literal_page_keys(source)) for source in sources.values())
    assert seen > 0, "the scan found no literal page key: it would pass vacuously"
    offenders = _offenders(sources, unknown_page_keys)
    assert offenders == {}, (
        "switch_right_panel ignores a key Panel.coerce rejects: use a Panel member or its key"
    )


# ------------------------------------------------------------- the Panel enum itself
# (from test_foundation_runtime_panels_feedback.py at b9f7d19, A3 hand-over)


def test_panel_values_pin_the_right_panel_contract():
    assert [int(p) for p in Panel] == [0, 1, 2]
    assert Panel.DISPLAY == 0
    assert Panel.CORRECT == 1
    assert Panel.VISUALIZATION == 2


@pytest.mark.parametrize(
    ("key", "panel"),
    [
        ("display", Panel.DISPLAY),
        ("correct", Panel.CORRECT),
        ("visualization", Panel.VISUALIZATION),
    ],
)
def test_panel_key_round_trips_switch_keys(key, panel):
    assert panel.key == key
    assert Panel.coerce(key) is panel
    assert Panel.coerce(key.upper()) is panel


def test_panel_coerce_accepts_panels_and_indices():
    assert Panel.coerce(Panel.VISUALIZATION) is Panel.VISUALIZATION
    assert Panel.coerce(1) is Panel.CORRECT


@pytest.mark.parametrize(
    "bad",
    # 3 and 4 were the removed editors' pages (no aliases); a key Panel does not know
    [3, 4, "nope", "", 7, -1, None, 1.5],
)
def test_panel_coerce_rejects_unknown_values(bad):
    with pytest.raises(ValueError):
        Panel.coerce(bad)

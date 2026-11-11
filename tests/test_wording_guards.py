"""Wording guard: no text of the app uses a retired term (``CLAUDE.md`` "Vocabulary",
KNOWN_ISSUES G4).

Model output is a *prediction*, a human-saved result is *GT*, and Correct is one editor
with three layers. The terms below are the ones a person writing a new tooltip or message
still reaches for; none may occur in a string constant under ``app/`` (docstrings may name
them). The app is parsed once for every term (``tests/source_scan_support.py``).

The `_fake.png` single source is tested in ``test_single_sources.py``; the button and
stage texts in ``test_window_layout.py``.
"""

from __future__ import annotations

import ast
import functools
import re

import pytest

from tests.source_scan_support import app_files, parse, rel, string_constants

#: term id -> pattern of a retired term
RETIRED = {
    "hand-traced": r"(?i)\bhand-traced\b",
    "hand-corrected": r"(?i)\bhand-corrected\b",
    "GT tracing": r"(?i)\bGT tracing\b",
}


def uses_of(pattern: str, strings) -> list[str]:
    """``"<where>:<line>"`` of each ``(where, line, text)`` whose text matches ``pattern``."""
    regex = re.compile(pattern)
    return [f"{where}:{line}" for where, line, text in strings if regex.search(text)]


@functools.cache
def _app_strings() -> tuple[tuple[str, int, str], ...]:
    """Every non-docstring string constant of ``app/`` as ``(file, line, text)``."""
    return tuple(
        (rel(path), line, text)
        for path in app_files()
        for line, text in string_constants(parse(path))
    )


def test_the_term_scan_finds_a_term_in_code_but_not_in_a_docstring():
    tree = ast.parse(
        '"""The GT tracing editor is gone."""\n'
        'TIP = "Save the hand-traced mask"\n'
        'OTHER = f"{n} GT Tracing images"\n'
        'FINE = "Save the GT mask"\n'
    )
    strings = [("x.py", line, text) for line, text in string_constants(tree)]
    assert uses_of(RETIRED["hand-traced"], strings) == ["x.py:2"]
    assert uses_of(RETIRED["GT tracing"], strings) == ["x.py:3"]
    assert len(_app_strings()) > 1000, "the scan sees almost no app text: it would pass vacuously"


@pytest.mark.parametrize("term", sorted(RETIRED))
def test_no_app_text_uses_a_retired_term(term):
    assert uses_of(RETIRED[term], _app_strings()) == []

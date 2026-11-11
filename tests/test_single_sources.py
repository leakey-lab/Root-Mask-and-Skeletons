"""Facts that live in one place in ``app/`` (``CLAUDE.md``: single sources of truth).

- The skeleton suffix ``_fake.png`` is written only in ``app/project_layout.py``: a second
  literal drifts when the suffix rule changes, and Measure Length then misses skeletons.
- The file-stamp helpers are defined only in ``app/io_utils.py``: two ``file_stamp``s that
  drift apart make the read-ahead caches (used only while the stamp matches) serve a stale
  image.
"""

from __future__ import annotations

import ast

from tests.source_scan_support import APP, ROOT, app_files, parse, rel, string_constants

STAMP_HELPERS = frozenset({"_stamp", "_file_stamp", "file_stamp", "_same_file", "same_file"})


def test_no_fake_png_literal_outside_project_layout():
    layout_file = APP / "project_layout.py"
    found = [
        f"{rel(path)}:{line}"
        for path in app_files()
        if path != layout_file
        for line, value in string_constants(parse(path))
        if "_fake.png" in value
    ]
    assert found == []


def test_stamp_helpers_are_defined_once():
    home = ROOT / "app" / "io_utils.py"
    defined = {
        rel(path): sorted(
            node.name
            for node in parse(path).body
            if isinstance(node, ast.FunctionDef) and node.name in STAMP_HELPERS
        )
        for path in app_files()
        if path != home
    }
    assert {name: found for name, found in defined.items() if found} == {}
    tree = parse(home)
    assert {n.name for n in tree.body if isinstance(n, ast.FunctionDef)} >= {
        "file_stamp",
        "same_file",
    }

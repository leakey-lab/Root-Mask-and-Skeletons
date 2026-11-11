"""Helpers of the source scans (the repository's rules read from the code, not run).

- ``ROOT``: the repository root; ``app_files()``: every ``app/*.py`` in a stable order.
- ``parse(path)``: the module's AST, parsed once per process (the scans used to parse every
  app file again, each one of them).
- ``imported_names(path)``: every module a file imports, relative imports resolved against
  its package, and ``from X import y`` also counted as ``X.y`` (``y`` may be a submodule).
- ``string_constants(tree)``: ``(line, text)`` of every str constant that is not a docstring
  (f-string parts included).

Qt-free and app-free: importing it loads only the standard library.
"""

from __future__ import annotations

import ast
import functools
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"

#: Nodes whose first statement, when a string, is their docstring.
DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def app_files() -> list[Path]:
    """Every module under ``app/``, sorted."""
    return sorted(APP.rglob("*.py"))


def rel(path: Path) -> str:
    """``path`` relative to the repository root, with forward slashes."""
    return Path(path).resolve().relative_to(ROOT).as_posix()


@functools.cache
def _parse_cached(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def parse(path) -> ast.Module:
    """The AST of ``path``, parsed once per process."""
    return _parse_cached(Path(path).resolve())


def names_in(tree: ast.Module, package: tuple[str, ...]) -> set[str]:
    """Every module ``tree`` imports. ``package`` is the dotted package of the file (for
    ``app/handlers/x.py``: ``("app", "handlers")``), against which relative imports are
    resolved; ``from X import y`` adds ``X`` and ``X.y``."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            parts = list(package[: len(package) - node.level + 1]) if node.level else []
            base = ".".join(parts + [node.module] if node.module else parts)
            found.add(base)
            found.update(f"{base}.{alias.name}" for alias in node.names)
    return found


def package_of(path: Path) -> tuple[str, ...]:
    """The dotted package a file belongs to: ``app/handlers/x.py`` -> ``("app", "handlers")``."""
    return tuple(Path(rel(path)).parts[:-1])


def imported_names(path) -> set[str]:
    """Every module the file at ``path`` imports (see :func:`names_in`)."""
    return names_in(parse(path), package_of(Path(path)))


def docstring_ids(tree: ast.Module) -> set[int]:
    """``id()`` of every docstring constant of ``tree``."""
    return {
        id(first.value)
        for node in ast.walk(tree)
        if isinstance(node, DOCSTRING_OWNERS)
        and node.body
        and isinstance(first := node.body[0], ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    }


def string_constants(tree: ast.Module) -> Iterator[tuple[int, str]]:
    """``(line, text)`` of every str constant of ``tree`` that is not a docstring."""
    docstrings = docstring_ids(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                yield node.lineno, node.value

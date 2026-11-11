"""Guard: the public names of ``app.gui`` resolve.

``app.gui`` re-exports its public classes lazily (PEP 562, ``app/gui/_lazy.py``), so an
export whose module was deleted or renamed breaks only when something touches it. Every
name in ``app.gui.__all__`` resolves to the object defined under that name (no aliases).
"""

from __future__ import annotations

from types import ModuleType

import pytest

import app.gui as gui


def _own_name(value: object) -> str | None:
    """The name ``value`` was defined under (for a module, the last dotted part)."""
    name = getattr(value, "__name__", None)
    if isinstance(value, ModuleType) and name:
        return name.rpartition(".")[2]
    return name


@pytest.mark.parametrize("name", sorted(gui.__all__))
def test_every_public_name_resolves_to_the_object_of_that_name(name: str) -> None:
    assert _own_name(getattr(gui, name)) == name

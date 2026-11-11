"""The FAIR reference docs stay in step with the code they describe (FR-07, FR-14).

* ``docs/MODELS.md`` lists every checkpoint registered in ``app.models`` with its
  exact path and SHA-256, so users can verify downloaded weights.
* ``docs/DATA_DICTIONARY.md`` documents every pinned metric CSV column verbatim.
* ``README.md`` and ``docs/DATA_DICTIONARY.md`` name the GT files where
  ``app.project_layout.ProjectLayout`` puts them (owner decision O1).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.config import AREA_CSV_HEADERS, LENGTH_CSV_HEADERS
from app.models import MODELS
from app.project_layout import ProjectLayout

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def undocumented_models(model_card: str) -> list[str]:
    """Registered checkpoints whose path or digest the model card does not state."""
    return [
        key
        for key, spec in MODELS.items()
        if f"`{spec.relpath}`" not in model_card or f"`{spec.sha256}`" not in model_card
    ]


def undocumented_columns(dictionary: str) -> list[str]:
    """Pinned CSV columns the data dictionary does not name (as `code`)."""
    headers = dict.fromkeys(LENGTH_CSV_HEADERS + AREA_CSV_HEADERS)
    return [h for h in headers if f"| `{h}` |" not in dictionary]


def test_model_card_states_every_registered_checkpoint_and_digest():
    assert undocumented_models(_read("docs/MODELS.md")) == []


def test_data_dictionary_documents_every_metric_csv_column():
    assert undocumented_columns(_read("docs/DATA_DICTIONARY.md")) == []


def test_model_card_states_the_weights_licence():
    assert "CC-BY-4.0" in (ROOT / "docs" / "MODELS.md").read_text(encoding="utf-8")


def gt_paths() -> list[str]:
    """The GT file of an image ``<stem>`` (mask, skeleton, class labels), relative to the
    project folder, as ``ProjectLayout`` builds them (the one place that makes the paths)."""
    layout = ProjectLayout("project")
    paths = (
        layout.gt_mask_path_for_stem("<stem>"),
        layout.gt_skeleton_path_for_stem("<stem>"),
        layout.gt_path("<stem>"),
    )
    return [os.path.relpath(path, layout.root).replace(os.sep, "/") for path in paths]


@pytest.mark.parametrize("doc", ["README.md", "docs/DATA_DICTIONARY.md"])
def test_the_docs_name_the_gt_paths_the_project_layout_writes(doc):
    text = _read(doc)
    assert [path for path in gt_paths() if f"`{path}`" not in text] == []

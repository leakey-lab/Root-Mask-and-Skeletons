"""Citation and release metadata stay in step with the code (FAIR4RS F1.2, F4, R1.1; FR-10, FR-01).

``CITATION.cff`` (GitHub "Cite this repository", Zenodo), ``codemeta.json`` and
``pyproject.toml`` describe the same software; ``CHANGELOG.md`` records every
release. Each copy of a fact is checked against its single source so a version
bump, a new author or a license decision cannot leave one file behind.

The licence decision (O32): GPL-3.0-or-later code, CC-BY-4.0 weights, one SPDX id
(``app.LICENSE_SPDX``) in every metadata file, the FSF GPL text in ``LICENSE``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import tomllib
from pathlib import Path

import pytest

import app
from app.models import MODELS

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
# FSF gpl-3.0.txt (674 lines, 35,149 bytes), fetched 2026-10-25.
GPL3_SHA256 = "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986"


@pytest.fixture(scope="module")
def cff() -> dict:
    return yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def codemeta() -> dict:
    return json.loads((ROOT / "codemeta.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def project() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def test_citation_cff_has_the_fields_github_needs_to_cite_it(cff):
    assert cff["cff-version"] == "1.2.0"
    assert cff["type"] == "software"
    assert cff["message"].strip() and cff["title"].strip()
    assert cff["authors"], "at least one author"
    for author in cff["authors"]:
        assert author.get("family-names") or author.get("name"), author


def test_the_three_authors_are_in_the_agreed_order(cff):
    assert [a["family-names"] for a in cff["authors"]] == ["Parth", "Varela", "Leakey"]
    assert cff["title"] == "SPROUTS"


def test_readme_cites_the_current_version_and_every_author(cff):
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    section = text.split("## How to cite", 1)[1].split("\n## ", 1)[0]
    match = re.search(r"version (\d+\.\d+\.\d+)", section)
    assert match and match.group(1) == app.__version__
    for author in cff["authors"]:
        assert author["family-names"] in section


def test_orcids_have_the_registry_form_when_present(cff):
    for author in cff["authors"]:
        if "orcid" in author:
            assert re.fullmatch(r"https://orcid\.org/\d{4}-\d{4}-\d{4}-\d{3}[\dX]", author["orcid"])


def test_version_is_the_package_version_everywhere(cff, codemeta):
    assert cff["version"] == codemeta["version"] == app.__version__


def test_repository_and_name_agree_across_metadata_files(cff, codemeta, project):
    repository = project["urls"]["Repository"]
    assert cff["repository-code"] == codemeta["codeRepository"] == repository
    assert codemeta["name"] == cff["title"]
    assert codemeta["issueTracker"] == project["urls"]["Issues"]


def test_authors_and_keywords_agree_across_metadata_files(cff, codemeta, project):
    cff_names = [f"{a['given-names']} {a['family-names']}" for a in cff["authors"]]
    codemeta_names = [f"{a['givenName']} {a['familyName']}" for a in codemeta["author"]]
    assert cff_names == codemeta_names == [a["name"] for a in project["authors"]]
    assert cff["keywords"] == codemeta["keywords"] == project["keywords"]


def test_changelog_top_release_is_the_package_version():
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    sections = re.findall(r"^## \[([^\]]+)\]", text, flags=re.MULTILINE)
    assert sections[:2] == ["Unreleased", app.__version__]


def test_the_pix2pix_bsd_notice_is_reproduced():
    text = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    for needle in (
        "Copyright (c) 2017, Jun-Yan Zhu and Taesung Park",
        "Copyright (c) 2016, Phillip Isola and Jun-Yan Zhu",
        "Copyright (c) 2015, Facebook, Inc.",
        "Redistributions of source code must retain the above copyright notice",
    ):
        assert needle in text
    # The derived module names its origin and points at the notice.
    source = (ROOT / "app" / "inference" / "skeleton_inference.py").read_text(encoding="utf-8")
    docstring = ast.get_docstring(ast.parse(source)) or ""
    assert "THIRD_PARTY_NOTICES.md" in docstring and "pix2pix" in docstring


# ------------------------------------------------------------------ the licence (O32)


def test_license_is_gpl3_text_under_the_projects_copyright_line():
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")  # universal newlines
    assert "Copyright (c) 2026 the SPROUTS authors" in text
    marker = "                    GNU GENERAL PUBLIC LICENSE\n                       Version 3, 29 June 2007"
    body = text[text.index(marker) :]
    assert len(body.splitlines()) == 674
    assert hashlib.sha256(body.encode()).hexdigest() == GPL3_SHA256


def test_spdx_constant_is_the_one_in_every_metadata_file():
    assert app.LICENSE_SPDX == "GPL-3.0-or-later"
    assert app.COPYRIGHT_HOLDER == "the SPROUTS authors"
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    cff = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    codemeta = json.loads((ROOT / "codemeta.json").read_text(encoding="utf-8"))
    assert project["license"] == app.LICENSE_SPDX
    assert cff["license"] == app.LICENSE_SPDX
    assert codemeta["license"].rsplit("/", 1)[-1] == app.LICENSE_SPDX


def test_every_registered_checkpoint_is_cc_by_4():
    assert MODELS
    assert {spec.license for spec in MODELS.values()} == {"CC-BY-4.0"}

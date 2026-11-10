"""``app.gt.files.remove_labels``: Unmark's removal deletes one class-labels file only.

The Correct page's Unmark takes Mark empty back: it deletes the image's empty class labels
``gt/<stem>.png`` and nothing else, never ``gt/classes.json``, the GT mask or skeleton
(``gt/mask/``, ``gt/skeletons/``) or a prediction (``mask/``, ``skeletons/``).
"""

from __future__ import annotations

import os

import pytest

from app.gt.files import remove_labels


def touch(path) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"x")
    return str(path)


@pytest.fixture
def project(tmp_path):
    """Every file an image may have: its labels, the registry, GT and predictions."""
    folder = tmp_path / "project"
    files = {
        "labels": touch(folder / "gt" / "a.png"),
        "registry": touch(folder / "gt" / "classes.json"),
        "gt_mask": touch(folder / "gt" / "mask" / "a.png"),
        "gt_skeleton": touch(folder / "gt" / "skeletons" / "a.png"),
        "mask": touch(folder / "mask" / "a.png"),
        "skeleton": touch(folder / "skeletons" / "a_fake.png"),
        "photo": touch(folder / "a.png"),
    }
    return folder, files


def every_file(folder) -> set[str]:
    return {os.path.join(root, name) for root, _, names in os.walk(folder) for name in names}


def test_it_deletes_the_labels_file_and_nothing_else(project):
    folder, files = project
    before = every_file(folder)

    assert remove_labels(files["labels"]) is True

    assert before - every_file(folder) == {files["labels"]}
    assert every_file(folder) <= before


def test_a_file_already_gone_is_no_error(project):
    _, files = project
    os.remove(files["labels"])

    assert remove_labels(files["labels"]) is False


@pytest.mark.parametrize(
    "kind", ["registry", "gt_mask", "gt_skeleton", "mask", "skeleton", "photo"]
)
def test_it_refuses_every_other_file(project, kind):
    folder, files = project
    before = every_file(folder)

    with pytest.raises(ValueError):
        remove_labels(files[kind])

    assert every_file(folder) == before


def test_a_gt_folder_named_in_another_case_is_still_gt(tmp_path):
    path = touch(tmp_path / "GT" / "a.png")

    assert remove_labels(path) is True
    assert not os.path.exists(path)


def test_a_file_that_cannot_be_deleted_raises_its_os_error(project, monkeypatch):
    _, files = project

    def locked(path):
        raise PermissionError(13, "Permission denied", path)

    monkeypatch.setattr("app.gt.files.os.remove", locked)

    with pytest.raises(PermissionError):
        remove_labels(files["labels"])
    assert os.path.exists(files["labels"])

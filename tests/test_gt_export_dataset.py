"""The exported dataset, file by file, against the test's own reading of the labels.

``export_dataset`` turns the class labels into a training set. Seeded random projects (labelled,
empty, Ignore-only, from another project's registry with an unknown class and other ids, and
unlabelled images; PNG and JPEG photos; three tubes) are exported with a random validation share
and seed, then every output is checked against what the test computes from the files it wrote,
sharing no code with ``app/gt/export.py``: its own reading of each labels PNG (Pillow, the
``GT_META_CHUNK`` text), its own id table built from the registry it wrote (by key, else by id,
else Ignore; then background 0, the labelled classes 1..N in registry order, everything else
255), its own ``np.bincount``. Checked: ``masks/``, ``images/`` (bytes), ``manifest.csv`` (rows,
counts, ``source_image``, the name tokens, the split), ``splits/``, ``classes.json``, the
provenance counts, and that no file of the project changed. A failure names the seed.
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np
import pytest
from PIL import Image

from app.config import GT_META_CHUNK
from app.gt import Registry, Tracing, default_registry, gt_path, save_registry, write_tracing
from app.gt.export import ExportOptions, export_dataset

SEEDS = range(12)
H, W = 18, 24
KINDS = ("random", "empty", "ignore_only", "foreign", "unlabelled")


def tree(folder: str) -> dict[str, bytes]:
    files = {}
    for dirpath, _dirs, names in os.walk(folder):
        for name in names:
            path = os.path.join(dirpath, name)
            with open(path, "rb") as fh:
                files[os.path.relpath(path, folder)] = fh.read()
    return files


def project_registry():
    """The defaults with a class added (hypha, id 4) and one retired (dead, id 5)."""
    return (
        default_registry()
        .add(key="hypha", name="Hypha", color="#3fa7ff")
        .add(key="dead", name="Dead root", color="#7f5af0")
        .retire("dead")
    )


def foreign_registry():
    """Another project's registry: root and nodule swapped, a class "weed" unknown here (4)."""
    data = default_registry().to_dict()
    data["classes"][0]["id"], data["classes"][1]["id"] = 2, 1
    data["classes"].insert(-1, {"id": 4, "key": "weed", "name": "Weed", "color": "#3a86ff"})
    return Registry.from_dict(data)


def make_project(root: str, rng) -> dict:
    """Write a random project; ``{"images": stem -> path, "kinds": stem -> kind}``."""
    os.makedirs(root)
    save_registry(root, project_registry())
    images, kinds = {}, {}
    n = int(rng.integers(5, 10))
    for i in range(n):
        tube = int(rng.integers(1, 4))
        stem = f"plot_T{tube}_L{i + 1}_2024.0{int(rng.integers(1, 9))}.01_120000"
        ext = str(rng.choice([".png", ".jpg"]))
        path = os.path.join(root, stem + ext)
        Image.fromarray(rng.integers(0, 256, (H, W, 3), dtype=np.uint8)).save(path)
        kind = KINDS[i % len(KINDS)] if i < len(KINDS) else str(rng.choice(KINDS))
        images[stem], kinds[stem] = path, kind
        if kind == "unlabelled":
            continue
        ids = np.zeros((H, W), np.uint8)
        registry = project_registry()
        if kind == "random":
            ids = rng.choice(np.array([0, 1, 2, 3, 4, 5, 9, 255], np.uint8), size=(H, W))
        elif kind == "ignore_only":
            ids[:] = 255
        elif kind == "foreign":
            ids = rng.choice(np.array([0, 1, 2, 4, 255], np.uint8), size=(H, W))
            registry = foreign_registry()
        write_tracing(gt_path(path), Tracing(ids.astype(np.uint8)), registry)
    return {"images": images, "kinds": kinds}


# ------------------------------------------------------------------ the test's own reading
def read_labels(path: str) -> tuple[np.ndarray, dict[int, str]]:
    """The labels PNG's pixel ids and its own id -> key map, read with Pillow alone."""
    with Image.open(path) as image:
        ids = np.array(image, dtype=np.uint8)
        meta = json.loads(image.text[GT_META_CHUNK])
    return ids, {int(k): v for k, v in meta["classes"].items()}


def expected_export_ids(ids: np.ndarray, file_map: dict[int, str]) -> np.ndarray:
    """The export ids of a labels file under :func:`project_registry`, by the rules."""
    registry = project_registry().to_dict()["classes"]
    id_of_key = {c["key"]: c["id"] for c in registry}
    known_ids = {c["id"] for c in registry}
    labelled = [c for c in registry if not c["retired"] and c["key"] != "ignore"]
    export_of = {c["id"]: n for n, c in enumerate(labelled, start=1)}
    table = np.full(256, 255, np.uint8)
    table[0] = 0
    for file_id in range(1, 256):
        key = file_map.get(file_id)
        if key is not None:
            project_id = id_of_key.get(key)  # an unknown key: Ignore
        else:
            project_id = file_id if file_id in known_ids else None
        table[file_id] = export_of.get(project_id, 255)
    return table[ids]


def expected_classes_json() -> dict:
    registry = project_registry().to_dict()["classes"]
    labelled = [c for c in registry if not c["retired"] and c["key"] != "ignore"]
    classes = [{"id": 0, "key": "background", "name": "Background"}]
    classes += [
        {"id": n, "key": c["key"], "name": c["name"], "color": c["color"]}
        for n, c in enumerate(labelled, start=1)
    ]
    return {"version": 1, "ignore_index": 255, "classes": classes}


def read_lines(path) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return [line for line in fh.read().splitlines() if line]


@pytest.mark.parametrize("seed", SEEDS)
def test_the_exported_dataset_matches_an_independent_reading_of_the_labels(tmp_path, seed):
    rng = np.random.default_rng(seed)
    root = str(tmp_path / "project")
    made = make_project(root, rng)
    images, kinds = made["images"], made["kinds"]
    labelled = sorted(s for s, k in kinds.items() if k != "unlabelled")
    out = tmp_path / "dataset"
    share = float(rng.choice([0.0, 0.2, 0.5, 0.9]))
    options = ExportOptions(str(out), val_fraction=share, seed=int(rng.integers(0, 1000)))
    before = tree(root)

    result = export_dataset(root, images, project_registry(), options)

    where = f"seed {seed}"
    assert tree(root) == before, where  # nothing of the project changed
    assert (result.exported, result.errors) == (len(labelled), []), where
    assert sorted(os.listdir(out / "masks")) == sorted(f"{s}.png" for s in labelled), where
    labelled_classes = expected_classes_json()["classes"][1:]
    with open(out / "manifest.csv", encoding="utf-8", newline="") as fh:
        rows = {row["stem"]: row for row in csv.DictReader(fh)}
    assert sorted(rows) == labelled, where
    for stem in labelled:
        here = f"{where}, {stem} ({kinds[stem]})"
        ids, file_map = read_labels(gt_path(images[stem]))
        expected = expected_export_ids(ids, file_map)
        with Image.open(out / "masks" / f"{stem}.png") as mask:
            assert mask.mode == "L", here
            assert np.array_equal(np.array(mask), expected), here
        ext = os.path.splitext(images[stem])[1]
        with open(images[stem], "rb") as a, open(out / "images" / f"{stem}{ext}", "rb") as b:
            assert a.read() == b.read(), here
        row = rows[stem]
        counts = np.bincount(expected.ravel(), minlength=256)
        assert row["source_image"] == images[stem], here
        tube, position = stem.split("_")[1][1:], stem.split("_")[2][1:]
        assert (row["tube"], row["position"], row["date"]) == (tube, position, stem.split("_")[3])
        assert int(row["px_background"]) == counts[0] and int(row["px_ignore"]) == counts[255]
        for c in labelled_classes:
            assert int(row[f"px_{c['key']}"]) == counts[c["id"]], here
        present = [c["key"] for c in labelled_classes if counts[c["id"]]]
        assert row["classes_present"] == ";".join(present), here
        if kinds[stem] in ("empty", "ignore_only"):
            assert row["classes_present"] == "", here
    train, val = read_lines(out / "splits" / "train.txt"), read_lines(out / "splits" / "val.txt")
    assert sorted(train + val) == labelled and not set(train) & set(val), where
    assert train, where  # train never empties
    assert all(rows[s]["split"] == "train" for s in train), where
    assert all(rows[s]["split"] == "val" for s in val), where
    for tube in {rows[s]["tube"] for s in labelled}:  # a tube never spans both splits
        assert len({rows[s]["split"] for s in labelled if rows[s]["tube"] == tube}) == 1, where
    with open(out / "classes.json", encoding="utf-8") as fh:
        assert json.load(fh) == expected_classes_json(), where
    with open(out / "manifest.csv.provenance.json", encoding="utf-8") as fh:
        record = json.load(fh)
    assert record["outputs"] == {
        "exported": len(labelled),
        "train": len(train),
        "val": len(val),
        "errors": 0,
    }, where

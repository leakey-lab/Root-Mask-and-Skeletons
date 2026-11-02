"""Class-label goldens of the Trace GT editor (Qt-free at import).

``tests/data/gt_goldens.json`` holds what ``GtTracingInterface`` did, before the Trace GT
stage was merged into the Correct editor (a "Classes" layer), for the scripted sessions in
:data:`GT_OPS`. The merged editor must reproduce the same class-id pixels, unsaved flag and
saved file. ``cases.*`` is the contract; ``ops`` records the scripts.

Re-recorded 2026-11-01 (owner decision: the class panel's box marks the active class; there
is no "checked" / ``reviewed`` state and Save writes no ``reviewed`` list): every class-id
plane, count, palette and file id map stayed the same; the ``reviewed`` entries went, the
saved files' bytes changed with their chunk, and a ``("box", key)`` op (a click on a class's
box, which picks the class) no longer makes the labels unsaved.

Cases (each on its own 640x480 project, :func:`make_gt_project`):

- ``two_classes_crossing``: two crossing strokes of two classes, the eraser through the
  crossing, a root outline filled (Fill reads the active class only), Clear then Undo, a
  click on ``root_tip``'s box, Save;
- ``fill_active_class_only``: Fill reads the outlines of the active class only. A closed
  ``nodule`` outline, then a ``root`` fill inside it and one outside every outline (both
  change nothing), an eraser stroke across the outline (a gap), a ``nodule`` fill inside the
  gapped outline (nothing), a closed ``root`` outline over the nodule outline's left half, a
  ``root`` fill inside both (fills the root outline's whole inside, the nodule outline's left
  half included; its right half stays) and one outside (nothing), Save;
- ``dirty_and_history``: the unsaved flag through Save, box clicks after it (a class's box,
  the same box again, a painted class's box: none is an edit), an edit undone back to the
  saved state, Undo past the saved state and Redo back (Redo after Undo, twice in a row), a
  box click over an edit, a second Save, the ticked box clicked twice after it, Clear then
  Undo back to the saved state;
- ``ignore_interplay``: the Ignore class (id 255) active: a stroke, the eraser across it, a
  ``root`` outline over it and a ``root`` fill inside (overwrites Ignore), an Ignore outline
  with an Ignore fill, a ``root`` stroke across that, Save;
- ``empty_save``: Save on an untouched image (an empty tracing is written);
- ``lossy_remap_on_load``: a ``gt/<stem>.png`` whose chunk maps id 2 to an unknown key, so
  the load remaps it to Ignore (255) with one warning and the tracing is unsaved; Save. The
  file is written as before 2026-11-01, with a ``reviewed`` list (read past).

Op semantics (positions are image pixels):

- ``("open",)``: open the image (done first when a script does not start with it);
- ``("class", key)``: make ``key`` the active class;
- ``("size", n)``: set the brush size;
- ``("tool", t)``: ``brush``, ``eraser`` or ``fill``;
- ``("stroke", pts)``: press at the first point, a dragging move per further point,
  release at the last;
- ``("click", p)``: press and release at ``p`` (with the Fill tool: fill around ``p``);
- ``("clear",)`` / ``("undo",)`` / ``("redo",)``: Clear the tracing / undo one action / redo
  one;
- ``("box", key)``: click the class's box (it picks the class, like a click on its row;
  the box was a "checked" tick until 2026-11-01);
- ``("save",)``: save the tracing.

An editor is driven through a *driver* (:func:`replay`), so the legacy editor and the
merged one run the very same script. A driver has: ``open(image_path)``, ``set_class(key)``,
``set_size(n)``, ``set_tool(name)``, ``stroke(points)``, ``click(point)``, ``clear()``,
``undo()``, ``redo()``, ``box(key)``, ``save() -> bool`` and the readers ``ids()`` (the
class-id plane, 2-D uint8), ``dirty()`` and ``warnings()`` (the number of remap warnings of
the load).

- ``dirty()`` is True while Save would write something new: after an edit, and from the open
  on when the load remapped ids. An undo back to the saved state makes it False, and an undo
  past the saved state makes it True again.

Per case the record holds the state after every op (``per_op_ids_sha256``,
``per_op_dirty``) and the end state. Notice texts are not gated: the
port renames them by design. ``dirty_before_save``, ``dirty_after_save``, ``warnings`` and
``save_returned`` are ``None`` for a script without a save op.

Import it as ``from tests.gt_goldens_support import ...``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import struct
import zlib

import numpy as np
from PIL import Image

from app.config import GT_META_CHUNK
from app.gt import (
    Tracing,
    default_registry,
    gt_path,
    read_tracing,
    save_registry,
    write_tracing,
)
from app.io_utils import imwrite_unicode
from tests.correct_goldens_support import file_sha, sha

__all__ = [
    "GOLDENS",
    "GT_CASES",
    "GT_OPS",
    "GT_STEM",
    "describe_file",
    "describe_ids",
    "load_goldens",
    "make_gt_project",
    "replay",
    "write_goldens",
    "write_legacy_tracing",
]

#: The committed golden data (``json.dump(..., indent=1, sort_keys=True)``).
GOLDENS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gt_goldens.json")

#: Stem of the synthetic image (a valid SPROUTS name).
GT_STEM = "SYNTH_T002_L001_2024.01.01_000000_001_SMM"

GT_CASES = (
    "two_classes_crossing",
    "empty_save",
    "lossy_remap_on_load",
    "fill_active_class_only",
    "dirty_and_history",
    "ignore_interplay",
)

#: The scripted sessions, as data (JSON turns tuples into lists; ``ops`` in the goldens).
GT_OPS: dict[str, list[tuple]] = {
    "two_classes_crossing": [
        ("class", "root"),
        ("size", 8),
        ("stroke", [(100, 100), (150, 150), (200, 200), (250, 250), (300, 300)]),
        ("class", "nodule"),
        ("size", 25),
        ("stroke", [(300, 100), (250, 150), (200, 200), (150, 250), (100, 300)]),
        ("tool", "eraser"),
        ("size", 10),
        ("stroke", [(200, 180), (200, 200), (200, 220)]),
        ("tool", "brush"),
        ("class", "root"),
        ("size", 3),
        ("stroke", [(150, 350), (250, 350), (250, 420), (150, 420), (150, 350)]),
        ("tool", "fill"),
        ("click", (200, 385)),
        ("clear",),
        ("undo",),
        ("box", "root_tip"),
        ("save",),
    ],
    "empty_save": [("save",)],
    "lossy_remap_on_load": [("open",), ("save",)],
    "dirty_and_history": [
        ("class", "root"),
        ("size", 8),
        ("stroke", [(100, 100), (200, 150), (300, 100)]),
        ("save",),
        ("box", "nodule"),
        ("box", "nodule"),
        ("box", "root"),
        ("class", "nodule"),
        ("size", 12),
        ("stroke", [(100, 300), (300, 300)]),
        ("undo",),
        ("redo",),
        ("undo",),
        ("undo",),
        ("redo",),
        ("redo",),
        ("box", "root_tip"),
        ("save",),
        ("box", "root_tip"),
        ("box", "root_tip"),
        ("clear",),
        ("undo",),
    ],
    "ignore_interplay": [
        ("class", "ignore"),
        ("size", 20),
        ("stroke", [(100, 200), (300, 200)]),
        ("tool", "eraser"),
        ("size", 8),
        ("stroke", [(200, 170), (200, 230)]),
        ("tool", "brush"),
        ("class", "root"),
        ("size", 3),
        ("stroke", [(150, 150), (250, 150), (250, 250), (150, 250), (150, 150)]),
        ("tool", "fill"),
        ("click", (160, 160)),
        ("class", "ignore"),
        ("tool", "brush"),
        ("size", 3),
        ("stroke", [(350, 150), (450, 150), (450, 250), (350, 250), (350, 150)]),
        ("tool", "fill"),
        ("click", (400, 200)),
        ("class", "root"),
        ("tool", "brush"),
        ("size", 12),
        ("stroke", [(340, 200), (460, 200)]),
        ("save",),
    ],
    "fill_active_class_only": [
        ("class", "nodule"),
        ("size", 3),
        ("stroke", [(300, 100), (400, 100), (400, 200), (300, 200), (300, 100)]),
        ("class", "root"),
        ("tool", "fill"),
        ("click", (350, 150)),
        ("click", (50, 50)),
        ("tool", "eraser"),
        ("size", 10),
        ("stroke", [(280, 150), (320, 150)]),
        ("tool", "fill"),
        ("class", "nodule"),
        ("click", (350, 150)),
        ("tool", "brush"),
        ("class", "root"),
        ("size", 3),
        ("stroke", [(250, 50), (350, 50), (350, 250), (250, 250), (250, 50)]),
        ("tool", "fill"),
        ("click", (330, 150)),
        ("click", (50, 400)),
        ("save",),
    ],
}

_LOSSY_KEY = "unknown_x"  # what the prepared file calls id 2


def load_goldens() -> dict:
    """The committed goldens: ``{"meta": ..., "ops": ..., "cases": ...}`` (a fresh copy)."""
    with open(GOLDENS, encoding="utf-8") as fh:
        return json.load(fh)


def write_goldens(data: dict) -> None:
    """Write the goldens file (the capture run, ``SPROUTS_WRITE_GT_GOLDENS=1``)."""
    os.makedirs(os.path.dirname(GOLDENS), exist_ok=True)
    with open(GOLDENS, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
        fh.write("\n")


def _photo() -> np.ndarray:
    """The 640x480 BGR gradient photo (no RNG)."""
    yy, xx = np.mgrid[0:480, 0:640]
    return np.dstack([(xx * 255 // 639), (yy * 255 // 479), ((xx + yy) * 255 // 1118)]).astype(
        np.uint8
    )


def make_gt_project(root: str, case: str) -> tuple[str, str]:
    """A fresh project folder ``<root>/<case>`` for ``case``; returns ``(image path, folder)``.

    It holds the 640x480 photo ``<GT_STEM>.png``; for ``lossy_remap_on_load`` also the
    prepared ``gt/<stem>.png`` (id 1 and id 2 areas; id 2 named ``"unknown_x"`` in the file's
    chunk, ``root`` in its old ``reviewed`` list) and ``gt/classes.json`` (the default
    registry). An existing
    folder of that name is removed first. An unknown case raises ``KeyError``.
    """
    if case not in GT_OPS:
        raise KeyError(case)
    folder = os.path.join(root, case)
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    image_path = os.path.join(folder, f"{GT_STEM}.png")
    imwrite_unicode(image_path, _photo(), durable=False)
    if case == "lossy_remap_on_load":
        registry = default_registry()
        save_registry(folder, registry)
        file_registry = dataclasses.replace(
            registry,
            classes=tuple(
                dataclasses.replace(c, key=_LOSSY_KEY) if c.id == 2 else c for c in registry.classes
            ),
        )
        ids = np.zeros((480, 640), np.uint8)
        ids[100:200, 100:300] = 1
        ids[300:400, 350:500] = 2
        write_legacy_tracing(gt_path(image_path), Tracing(ids), file_registry, {"root"})
    return image_path, folder


def write_legacy_tracing(path: str, tracing: Tracing, registry, reviewed) -> None:
    """``write_tracing`` as SPROUTS wrote it before 2026-11-01: the same file, its chunk
    with a ``reviewed`` list (the old "checked" classes) after the id map.

    The PNG of ``write_tracing`` gets its ``GT_META_CHUNK`` text chunk rewritten in place
    (with its CRC); every other chunk keeps its bytes."""
    write_tracing(path, tracing, registry)
    with open(path, "rb") as fh:
        data = fh.read()
    key = GT_META_CHUNK.encode("latin-1") + b"\0"
    out, pos = bytearray(data[:8]), 8
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        kind, body = data[pos + 4 : pos + 8], data[pos + 8 : pos + 8 + length]
        if kind == b"tEXt" and body.startswith(key):
            meta = json.loads(body[len(key) :])
            meta["reviewed"] = sorted(reviewed)
            body = key + json.dumps(meta).encode("latin-1")
        out += struct.pack(">I", len(body)) + kind + body
        out += struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
        pos += 12 + length
    with open(path, "wb") as fh:
        fh.write(bytes(out))


def describe_ids(ids: np.ndarray) -> dict:
    """``ids_sha256`` (dtype, shape, bytes) and ``counts`` (``{"<id>": pixels}``, ids with
    pixels only)."""
    plane = np.asarray(ids)
    values, counts = np.unique(plane, return_counts=True)
    return {
        "ids_sha256": sha(plane),
        "counts": {str(int(v)): int(n) for v, n in zip(values, counts, strict=True)},
    }


def describe_file(path: str) -> dict:
    """What a golden records for a saved ``gt/<stem>.png`` (``cases[...]["file"]``).

    The contract is every entry but ``file_sha256_informational`` (encoder bytes may change
    with Pillow or zlib while the pixels stay).
    """
    if not os.path.exists(path):
        return {"exists": False}
    with Image.open(path) as image:
        image.load()
        mode, size = image.mode, list(image.size)
        palette = list(image.getpalette() or [])[:768]
        palette += [0] * (768 - len(palette))  # Pillow may trim the PLTE chunk: 256 entries
        transparency = image.info.get("transparency")
    tracing = read_tracing(path)
    return {
        "exists": True,
        "mode": mode,
        "size": size,
        "palette_sha256": sha(np.array(palette, np.uint8)),
        "transparency": transparency,
        "ids_sha256": sha(tracing.ids),
        "classes": {str(i): key for i, key in sorted(tracing.file_classes.items())},
        "file_sha256_informational": file_sha(path),
    }


def replay(driver, image_path: str, case: str) -> dict:
    """Run ``GT_OPS[case]`` through ``driver``; returns the case record of the goldens.

    ``dirty_before_save`` and ``warnings`` are read just before the save op,
    ``dirty_after_save`` just after it. The file is described after the last op.
    """
    ops = GT_OPS[case]
    if ops[0][0] != "open":
        driver.open(image_path)
    per_op: list[str] = []
    per_op_dirty: list[bool] = []
    dirty_before_save: bool | None = None
    dirty_after_save: bool | None = None
    warnings: int | None = None
    save_returned: bool | None = None
    for op in ops:
        kind = op[0]
        if kind == "open":
            driver.open(image_path)
        elif kind == "class":
            driver.set_class(op[1])
        elif kind == "size":
            driver.set_size(op[1])
        elif kind == "tool":
            driver.set_tool(op[1])
        elif kind == "stroke":
            driver.stroke(op[1])
        elif kind == "click":
            driver.click(op[1])
        elif kind == "clear":
            driver.clear()
        elif kind == "undo":
            driver.undo()
        elif kind == "redo":
            driver.redo()
        elif kind == "box":
            driver.box(op[1])
        elif kind == "save":
            dirty_before_save = bool(driver.dirty())
            warnings = int(driver.warnings())
            save_returned = bool(driver.save())
            dirty_after_save = bool(driver.dirty())
        else:
            raise ValueError(f"unknown golden op {op!r}")
        per_op.append(sha(driver.ids()))
        per_op_dirty.append(bool(driver.dirty()))
    record = {
        "per_op_ids_sha256": per_op,
        "per_op_dirty": per_op_dirty,
        **describe_ids(driver.ids()),
        "dirty_before_save": dirty_before_save,
        "dirty_after_save": dirty_after_save,
        "save_returned": save_returned,
        "warnings": warnings,
        "file": describe_file(gt_path(image_path)),
    }
    return record

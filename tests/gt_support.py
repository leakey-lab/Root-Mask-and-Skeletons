"""Shared helpers of the class-label tests (``app/gt``): a small labelled project and broken PNGs.

Qt-free; no asserts. Import as ``from tests.gt_support import ...``.

- :func:`build_project`: six 20 x 20 photos of three tubes, each with class labels, under a
  registry with an added class that the export registry retires (:data:`STEMS`);
- :func:`write_header_bomb` / :func:`break_a_chunk`: PNGs Pillow refuses (too large, broken
  chunk), for the unreadable-labels paths.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np
from PIL import Image

from app.gt.registry import default_registry
from app.gt.store import Tracing, gt_path, write_tracing

__all__ = ["STEMS", "break_a_chunk", "build_project", "make_ids", "write_header_bomb"]

#: Six stems: three tubes (T1-T3), two positions each (L1/L2).
STEMS = [
    "plot_T1_L1_2024.01.01_120000",
    "plot_T1_L2_2024.01.01_120000",
    "plot_T2_L1_2024.01.01_120000",
    "plot_T2_L2_2024.01.01_120000",
    "plot_T3_L1_2024.01.01_120000",
    "plot_T3_L2_2024.01.01_120000",
]


def make_ids(shape, values):
    """A deterministic (h, w) uint8 array whose pixels cycle through ``values``."""
    h, w = shape
    flat = np.array([values[i % len(values)] for i in range(h * w)], dtype=np.uint8)
    return flat.reshape(h, w)


def build_project(tmp_path, *, dead_root_stem=None):
    """Six 20x20 RGB images + GT files (registry has an added-then-retired class).

    Returns (project_dir, images, write_registry, export_registry).
    ``write_registry`` still has ``dead_root`` active (used to write the GT
    file that paints it); ``export_registry`` has it retired, and is what
    callers pass to ``export_dataset``.
    """
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    write_registry = default_registry().add(key="dead_root", name="Dead root", color="#111111")
    export_registry = write_registry.retire("dead_root")

    images: dict[str, str] = {}

    for stem in STEMS:
        image_path = project_dir / f"{stem}.png"
        Image.new("RGB", (20, 20), color=(10, 20, 30)).save(image_path, format="PNG")
        images[stem] = str(image_path)

        if stem == dead_root_stem:
            ids = make_ids((20, 20), [0, 1, 2, 4])
        else:
            ids = make_ids((20, 20), [0, 1, 2, 3, 255])

        write_tracing(gt_path(image_path), Tracing(ids=ids), write_registry)

    return str(project_dir), images, write_registry, export_registry


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(kind: bytes, data: bytes) -> bytes:
    """One PNG chunk: length, type, data and CRC."""
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def write_header_bomb(path) -> None:
    """A palette PNG whose header says 20000 x 10000 px, over twice Pillow's default limit:
    Pillow refuses it when it opens the file, whatever follows the header."""
    header = struct.pack(">IIBBBBB", 20000, 10000, 8, 3, 0, 0, 0)
    with open(path, "wb") as fh:
        fh.write(_PNG_SIGNATURE + _chunk(b"IHDR", header) + _chunk(b"PLTE", bytes(3)))
        fh.write(_chunk(b"IDAT", zlib.compress(b"\0")) + _chunk(b"IEND", b""))


def break_a_chunk(path) -> None:
    """Split the pixel data of the PNG at ``path`` in two chunks and corrupt the second one's
    type: Pillow opens the file, then raises ``SyntaxError`` ("broken PNG file") on load."""
    data, pos, out = open(path, "rb").read(), len(_PNG_SIGNATURE), _PNG_SIGNATURE
    while pos < len(data):
        (size,) = struct.unpack(">I", data[pos : pos + 4])
        kind, body = data[pos + 4 : pos + 8], data[pos + 8 : pos + 8 + size]
        pos += 12 + size
        if kind != b"IDAT":
            out += _chunk(kind, body)
            continue
        half = len(body) // 2
        second = _chunk(b"IDAT", body[half:])
        out += _chunk(b"IDAT", body[:half]) + second[:4] + b"I\x00AT" + second[8:]
    with open(path, "wb") as fh:
        fh.write(out)

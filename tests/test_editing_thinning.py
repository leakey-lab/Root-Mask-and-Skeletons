"""The skeleton thinning helpers of ``app.editing.thinning`` (Qt-free, W1-T3, O2).

``thin_raster`` (a skeleton file as Correct opens it: upscaled bilinearly before the
threshold, inverted when the background is white, thinned; a file at image size reopens
unchanged) and the lazy scikit-image import live here; ``app.gui.skeleton_graph_model``
imports them back under the same names. ``is_generated_as_opened`` says whether a skeleton is
the generated one as Correct opens it: a skeleton Save writes a correction only when it is
not. ``app.editing.thinning`` loading no PyQt6 is pinned by ``tests/test_correct_layering.py``.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import cv2
import numpy as np
import pytest

import app.gui.skeleton_graph_model as sgm
from app.editing import thinning
from app.editing.thinning import _skeletonize_uint8, is_generated_as_opened, thin_raster
from app.gui.skeleton_graph_model import SkeletonCorrectionModel

ROOT = REPO_ROOT = Path(__file__).resolve().parents[1]
PHOTO = (640, 480)  # (w, h)


def _line(row: int) -> np.ndarray:
    """A generated skeleton raster at SKELETON_OUTPUT_SIZE (341 x 256): one line on ``row``."""
    gray = np.zeros((256, 341), np.uint8)
    gray[row, 30:300] = 255
    return gray


def _generated(tmp_path) -> tuple[str, np.ndarray]:
    """The generated skeleton file of the tests (one line on row 100) and its raster."""
    gray = _line(100)
    path = str(tmp_path / "a_fake.png")
    cv2.imwrite(path, gray)
    return path, gray


def test_skeleton_graph_model_reexports_the_same_functions():
    assert sgm.thin_raster is thinning.thin_raster
    assert sgm.prewarm_skeletonize is thinning.prewarm_skeletonize
    assert sgm._skeletonize_uint8 is thinning._skeletonize_uint8
    assert sgm._ensure_uint8_binary is thinning._ensure_uint8_binary


def test_is_generated_as_opened(tmp_path):
    path, gray = _generated(tmp_path)
    opened = thin_raster(gray, PHOTO)
    other = thin_raster(_line(40), PHOTO)
    empty = np.zeros((PHOTO[1], PHOTO[0]), np.uint8)

    assert opened.any()
    assert is_generated_as_opened(opened, path, PHOTO)
    assert not is_generated_as_opened(other, path, PHOTO)
    assert is_generated_as_opened(empty, None, PHOTO)
    assert not is_generated_as_opened(empty, path, PHOTO)  # emptied: a real correction
    assert is_generated_as_opened(empty, str(tmp_path / "missing.png"), PHOTO)
    assert not is_generated_as_opened(opened, str(tmp_path / "missing.png"), PHOTO)


def test_an_unreadable_generated_file_counts_as_empty(tmp_path):
    path = tmp_path / "broken_fake.png"
    path.write_bytes(b"not a png")
    empty = np.zeros((PHOTO[1], PHOTO[0]), np.uint8)

    assert is_generated_as_opened(empty, str(path), PHOTO)
    assert not is_generated_as_opened(empty + 255, str(path), PHOTO)


def test_a_raster_with_a_white_background_opens_as_its_dark_lines():
    # A skeleton file of another tool can be black lines on white: Correct opens its lines,
    # not the skeleton of its background.
    raster = np.zeros((60, 80), np.uint8)
    cv2.line(raster, (5, 30), (75, 30), 255, 1)
    cv2.line(raster, (40, 5), (40, 55), 255, 1)

    white_lines = thin_raster(raster, (80, 60))
    dark_lines = thin_raster(255 - raster, (80, 60))

    assert int(np.count_nonzero(white_lines)) > 100
    assert np.array_equal(dark_lines, white_lines)


def test_the_values_of_the_mask_do_not_matter_only_where_it_is_set(tmp_path):
    path, gray = _generated(tmp_path)
    opened = thin_raster(gray, PHOTO)

    assert is_generated_as_opened((opened > 0).astype(np.uint8), path, PHOTO)


def _reference_skeletonize_uint8(mask: np.ndarray) -> np.ndarray:
    """``_skeletonize_uint8`` as it was before the lean form (2026-10-26): the 0/255 copy of
    ``_ensure_uint8_binary``, then two passes back to 0/255."""
    mask = thinning._ensure_uint8_binary(mask)
    skel_bool = thinning._skeletonize_fn()(mask > 0)
    return skel_bool.astype(np.uint8) * 255


def _layouts(plane: np.ndarray):
    yield plane
    yield np.asfortranarray(plane)
    yield plane[::2, 1::3]
    yield plane.T
    yield plane[::-1, ::-1]


def test_the_skeletonize_wrapper_gives_the_reference_bytes_for_any_input():
    rng = np.random.default_rng(20261026)
    planes = []
    for dtype in (np.uint8, np.int8, np.uint16, np.int16, np.int32, np.int64):
        values = rng.integers(-300, 600, (37, 53)) * (rng.random((37, 53)) < 0.4)
        planes.append(values.astype(dtype))
    for dtype in (np.float32, np.float64):
        values = rng.normal(0.0, 200.0, (37, 53)) * (rng.random((37, 53)) < 0.4)
        values[0, :4] = (0.5, -0.5, 255.9, 256.0)
        planes.append(values.astype(dtype))
    planes.append(rng.random((37, 53)) < 0.3)  # bool
    blob = np.zeros((60, 80), np.uint8)
    cv2.circle(blob, (40, 30), 20, 255, -1)  # a thick blob: thinning removes pixels
    cv2.line(blob, (0, 59), (79, 0), 7, 3)  # values other than 255
    planes.append(blob)
    compared = 0
    with np.errstate(invalid="ignore"):
        for plane in planes:
            for mask in _layouts(plane):
                want = _reference_skeletonize_uint8(mask)
                got = thinning._skeletonize_uint8(mask)
                assert got.dtype == want.dtype == np.uint8 and got.shape == want.shape
                assert np.array_equal(got, want), (plane.dtype, mask.strides)
                assert got.flags.c_contiguous and got.flags.writeable and got.flags.owndata
                compared += 1
    assert compared == len(planes) * 5


def test_the_skeletonize_wrapper_refuses_what_is_not_a_plane():
    for bad in (np.zeros(5, np.uint8), np.zeros((4, 5, 3), np.uint8), np.uint8(1), np.zeros(())):
        for fn in (thinning._skeletonize_uint8, _reference_skeletonize_uint8):
            try:
                fn(np.asarray(bad))
            except ValueError as exc:
                assert str(exc) == "Expected a 2D binary mask."
            else:
                raise AssertionError(f"{fn.__name__} took a {np.asarray(bad).shape} array")


# ------------------------------------------------------------- the bilinear upscale
#: (source (w, h), target (w, h)): a prediction to photo sizes, downscales, the exact 2x
#: shrink (OpenCV serves it from INTER_AREA), odd and 1-px sizes
_RESIZE_SIZES = [
    ((341, 256), (640, 480)),
    ((341, 256), (2048, 1536)),
    ((640, 480), (341, 256)),
    ((682, 512), (341, 256)),
    ((100, 80), (37, 53)),
    ((13, 7), (5, 3)),
    ((5, 3), (13, 7)),
    ((1, 1), (9, 9)),
    ((300, 1), (77, 5)),
    ((640, 480), (639, 481)),
]
#: sha256 of resize_linear_u8 over _resize_cases(): recorded on x86, where it equals
#: cv2.resize, and the same under QEMU aarch64 (2026-10-29, A7)
_RESIZE_DIGEST = "dda0a814447e3128a745293399ad95c06bc5ab66cb377afbe5b9a09168effd51"


def _resize_cases():
    rng = np.random.default_rng(29)
    for (sw, sh), size in _RESIZE_SIZES:
        yield rng.integers(0, 256, (sh, sw), dtype=np.uint8), size


def _resize_goes_to_an_arm_hal() -> bool:
    """True for OpenCV builds with the carotene or KleidiCV HAL (the arm64 wheels), whose
    cv2.resize rounds unlike OpenCV's own code."""
    hal = [line for line in cv2.getBuildInformation().splitlines() if "Custom HAL:" in line]
    return any(name in line for line in hal for name in ("carotene", "KleidiCV"))


def test_the_upscale_gives_the_same_pixels_on_every_platform():
    """thin_raster's resize is integer numpy: the digest recorded on x86 holds on arm64 too,
    where cv2.resize gave other pixels (macOS CI run 40970595195)."""
    digest = hashlib.sha256()
    for img, size in _resize_cases():
        out = thinning.resize_linear_u8(img, size)
        assert out.dtype == np.uint8 and out.shape == (size[1], size[0])
        digest.update(out.tobytes())
    assert digest.hexdigest() == _RESIZE_DIGEST


@pytest.mark.skipif(
    _resize_goes_to_an_arm_hal(), reason="cv2.resize goes to an arm64 HAL here: why the port exists"
)
def test_the_upscale_equals_opencvs_bilinear_resize():
    rng = np.random.default_rng(123)
    cases = list(_resize_cases())
    for i in range(80):
        sw, sh = int(rng.integers(1, 400)), int(rng.integers(1, 400))
        if i % 4 == 0:  # the exact 2x shrink
            sw, sh = 2 * max(1, sw // 2), 2 * max(1, sh // 2)
            size = (sw // 2, sh // 2)
        else:
            size = (int(rng.integers(1, 900)), int(rng.integers(1, 900)))
        cases.append((rng.integers(0, 256, (sh, sw), dtype=np.uint8), size))
    for img, size in cases:
        want = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        assert np.array_equal(thinning.resize_linear_u8(img, size), want), (img.shape, size)


# --------------------------------------------------------------------------------------
# Lazy scikit-image (C4): the import waits for the first skeletonize or the prewarm
# --------------------------------------------------------------------------------------


def _run_python(code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_scikit_image_loads_on_the_first_skeletonize_not_at_import():
    result = _run_python(
        """
        import sys
        import app.gui.skeleton_graph_model as sgm
        print("import", "skimage" in sys.modules)
        model = sgm.SkeletonCorrectionModel()
        model.set_empty((40, 30))
        model.draw_polyline([(5, 5), (30, 20)], thickness=3)
        print("edit", "skimage" in sys.modules)
        model.skeletonize()
        print("skeletonize", "skimage.morphology" in sys.modules)
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["import", "False", "edit", "False", "skeletonize", "True"]


def test_prewarm_loads_scikit_image_once_from_a_worker_thread():
    result = _run_python(
        """
        import builtins
        import sys
        import threading
        import app.gui.skeleton_graph_model as sgm
        worker = threading.Thread(target=sgm.prewarm_skeletonize)
        worker.start()
        worker.join(120)
        print("prewarmed", "skimage.morphology" in sys.modules)
        imported = []
        real_import = builtins.__import__
        def spy(name, *args, **kwargs):
            imported.append(name)
            return real_import(name, *args, **kwargs)
        builtins.__import__ = spy
        sgm.prewarm_skeletonize()
        builtins.__import__ = real_import
        print("again", sum(name.startswith("skimage") for name in imported))
        model = sgm.SkeletonCorrectionModel()
        model.set_empty((40, 30))
        model.draw_polyline([(5, 5), (30, 20)], thickness=3)
        model.skeletonize()
        print("thin", int(model.mask.max()), int((model.mask > 0).sum()) < 40)
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["prewarmed", "True", "again", "0", "thin", "255", "True"]


# ----------------------------------------------- opening a small generated file
def test_a_small_file_opens_upscaled_smoothly_as_one_line():
    """The generated skeleton is small (SKELETON_OUTPUT_SIZE, 341 x 256) with ~2 px lines.
    Opening it at image size (640 x 480) upscales the grey file bilinearly before
    thresholding, so the line keeps its connectivity and its measured length: within 1 %
    (bilinear +0.23 %; a nearest-neighbour staircase measures +2.19 %). The lengths are
    ``root_length_mm`` values, each file with its own per-axis pixel pitch (W3-T8 review I1).
    """
    from app.config import SKELETON_OUTPUT_SIZE
    from app.inference.metrics import root_length_mm

    w, h = SKELETON_OUTPUT_SIZE
    small = np.zeros((h, w), np.uint8)
    cv2.line(small, (20, 20), (320, 236), 255, 2)
    cv2.ellipse(small, (170, 128), (100, 70), 15, 0, 270, 255, 2)
    model = SkeletonCorrectionModel()
    model.load_from_raster(small, target_size=(640, 480))
    mask = model.mask
    assert mask.shape == (480, 640)
    source = _skeletonize_uint8(small)
    n_src, _ = cv2.connectedComponents(source, connectivity=8)
    n_new, _ = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    assert n_new == n_src
    assert abs(root_length_mm(mask) / root_length_mm(source) - 1.0) < 0.01


def test_a_file_at_image_size_opens_unchanged():
    """A correction saved at image resolution reopens pixel for pixel (no resize)."""
    model = SkeletonCorrectionModel()
    raster = np.zeros((60, 80), np.uint8)
    cv2.line(raster, (5, 5), (70, 50), 255, 1)
    cv2.line(raster, (5, 50), (40, 8), 255, 1)
    model.load_from_raster(raster, target_size=(80, 60))
    first = model.mask.copy()
    model.load_from_raster(first, target_size=(80, 60))
    assert np.array_equal(model.mask, first)

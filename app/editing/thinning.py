"""Skeleton thinning as Correct opens a skeleton file (Qt-free: numpy and cv2).

- :func:`thin_raster`: a grayscale skeleton raster resized to the image and thinned to 1 px,
  the skeleton Correct shows for a prediction or a GT skeleton.
- :func:`is_generated_as_opened`: whether a skeleton is the prediction as Correct opens
  it. Saving is not acceptance (O2): a skeleton Save writes a GT skeleton only when this is
  False.
- The lazy scikit-image import (:func:`prewarm_skeletonize`) and the 0/255 helpers the
  skeleton model shares (``app.gui.skeleton_graph_model`` imports them back).

scikit-image (about 0.2 s to import, most of it scipy) is imported on the first
skeletonize, or ahead of time by :func:`prewarm_skeletonize`, which ``MainWindow`` runs in
the thread pool once the window shows; importing this module stays cheap. No Qt, no torch:
``app.inference`` may import it.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from app.io_utils import imread_unicode

__all__ = ["is_generated_as_opened", "prewarm_skeletonize", "resize_linear_u8", "thin_raster"]

# scikit-image's skeletonize, imported on first use (see prewarm_skeletonize).
_skimage_skeletonize: Callable[[np.ndarray], np.ndarray] | None = None


def _skeletonize_fn() -> Callable[[np.ndarray], np.ndarray]:
    """scikit-image's ``skeletonize``, imported (once) on first use."""
    global _skimage_skeletonize
    if _skimage_skeletonize is None:
        from skimage.morphology import skeletonize

        _skimage_skeletonize = skeletonize
    return _skimage_skeletonize


def prewarm_skeletonize() -> None:
    """Import scikit-image's skeletonize now, so the first skeleton edit does not wait.

    Idempotent: later calls return at once. Safe from a worker thread (Python's import
    lock makes a skeletonize on another thread wait for the import to finish).
    """
    _skeletonize_fn()


def _ensure_uint8_binary(mask: np.ndarray) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.ndim != 2:
        raise ValueError("Expected a 2D binary mask.")
    # Normalize to 0/255. uint8 scalars keep np.where from building an int64 temporary.
    return np.where(mask > 0, np.uint8(255), np.uint8(0))


def _skeletonize_uint8(mask: np.ndarray) -> np.ndarray:
    """Skeletonize a uint8 0/255 foreground mask to 1px (uint8 0/255).

    Foreground is what :func:`_ensure_uint8_binary` keeps (non-zero after a uint8 cast),
    tested without building its 0/255 copy; the result is a new C-contiguous array.
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.ndim != 2:
        raise ValueError("Expected a 2D binary mask.")
    skel = np.asarray(_skeletonize_fn()(mask > 0), dtype=bool)
    return skel.view(np.uint8) * np.uint8(255)


_COEF_SCALE = 2048  # OpenCV's INTER_RESIZE_COEF_SCALE: 11-bit fixed-point weights


def _linear_taps(src: int, dst: int) -> tuple[np.ndarray, np.ndarray]:
    """Per output index along one axis: the source index and the float32 fraction,
    computed in the order and precision of OpenCV's ``resize`` (half-pixel centres)."""
    scale = 1.0 / (dst / src)
    f = ((np.arange(dst, dtype=np.float64) + 0.5) * scale - 0.5).astype(np.float32)
    s = np.floor(f).astype(np.intp)
    return s, f - s.astype(np.float32)


def _weights(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The two fixed-point weights of fraction ``f`` (rounded half to even, as cvRound)."""
    w0 = np.rint((np.float32(1.0) - f) * np.float32(_COEF_SCALE)).astype(np.int32)
    w1 = np.rint(f * np.float32(_COEF_SCALE)).astype(np.int32)
    return w0, w1


def resize_linear_u8(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """``cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)`` of a 2D uint8 image, pixel
    for pixel as OpenCV computes it on x86, but on every CPU.

    On arm64 (macOS, Linux aarch64) OpenCV's wheels route this resize through their
    KleidiCV / carotene HAL, which rounds differently (up to 1.2 % of the pixels of an
    upscaled prediction, by 1-3 levels): Correct then opened a prediction as a different
    skeleton than on Windows or Linux x86 and saved other pixels. This is OpenCV's own
    integer algorithm (11-bit weights, ``(b0 * (r0 >> 4) >> 16) + (b1 * (r1 >> 4) >> 16)``
    rounded by 2 bits) in numpy, with its exact 2x shrink (a 2x2 box mean) too.
    """
    h, w = img.shape
    dw, dh = size
    src = img.astype(np.int32)
    if (w, h) == (2 * dw, 2 * dh):  # OpenCV takes INTER_AREA's fast path here
        quad = src[0::2, 0::2] + src[0::2, 1::2] + src[1::2, 0::2] + src[1::2, 1::2]
        return ((quad + 2) >> 2).astype(np.uint8)
    sx, fx = _linear_taps(w, dw)
    outside = (sx < 0) | (sx >= w - 1)  # columns: clamped, and the fraction dropped
    fx[outside] = 0
    sx = np.clip(sx, 0, w - 1)
    a0, a1 = _weights(fx)
    rows = src[:, sx] * a0 + src[:, np.minimum(sx + 1, w - 1)] * a1
    rows >>= 4  # the vertical pass reads each row >> 4 (at most 32640: int32 stays exact)
    sy, fy = _linear_taps(h, dh)  # rows: the fraction kept, the two rows clamped
    b0, b1 = _weights(fy)
    out = rows[np.clip(sy, 0, h - 1)]
    out *= b0[:, None]
    out >>= 16
    bottom = rows[np.clip(sy + 1, 0, h - 1)]
    bottom *= b1[:, None]
    bottom >>= 16
    out += bottom
    out += 2
    out >>= 2
    return out.astype(np.uint8)


def thin_raster(
    raster_gray: np.ndarray, target_size: tuple[int, int], threshold: str = "otsu"
) -> np.ndarray:
    """A grayscale skeleton raster at ``target_size``, 1 px thin (uint8 0/255); any thread.

    A raster of another size (the generated skeleton is ``SKELETON_OUTPUT_SIZE``) is
    upscaled bilinearly *before* thresholding, so thin lines come out as smooth,
    connected 1 px lines of the right length instead of a nearest-neighbour
    staircase. A raster already at ``target_size`` (a GT skeleton saved at image
    resolution) is only thresholded and thinned, so it reopens unchanged. The upscale is
    :func:`resize_linear_u8`, so every platform opens a skeleton as the same pixels.
    """
    if raster_gray.ndim != 2:
        raise ValueError("Expected a grayscale image.")

    img = raster_gray
    w_t, h_t = target_size
    if (img.shape[1], img.shape[0]) != (w_t, h_t):
        img = resize_linear_u8(img, (w_t, h_t))
    if threshold == "otsu":
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

    # Ensure skeleton is foreground (white) by inverting if background dominates.
    if float(np.mean(binary)) > 127.0:
        binary = 255 - binary
    return _skeletonize_uint8(binary)


def is_generated_as_opened(
    mask: np.ndarray, generated_path: str | None, photo_size: tuple[int, int]
) -> bool:
    """True when ``mask`` sets exactly the pixels of the generated skeleton as Correct opens
    it: the file at ``generated_path`` read in grayscale and :func:`thin_raster`-ed to
    ``photo_size`` ``(w, h)``.

    No path, a missing file or an unreadable one opens as an empty skeleton, so an empty
    ``mask`` is then the generated one; an emptied skeleton over a non-empty generated one
    is not (real GT). Only where ``mask`` is set counts, not its values.
    """
    gray = imread_unicode(generated_path, cv2.IMREAD_GRAYSCALE) if generated_path else None
    if gray is None:
        return not mask.any()
    opened = thin_raster(gray, photo_size)
    return opened.shape == mask.shape and np.array_equal(opened > 0, mask > 0)

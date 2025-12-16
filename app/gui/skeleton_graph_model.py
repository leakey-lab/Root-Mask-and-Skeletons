"""
Skeleton correction model utilities.

This module intentionally lives in the *correction stream* and does NOT affect the
existing skeleton generation / overlay / measurement logic.

The editor keeps an internal binary skeleton mask (uint8 0/255) and provides:
- Raster import (threshold + optional inversion + resize) and skeletonization
- Polyline drawing + erasing on the raster mask
- Vectorization helpers (endpoints/junctions + polyline tracing) to support
  connect/snap UX
- Saving corrected skeletons as `<base>_skeleton.png`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize

Point = Tuple[int, int]  # (x, y)


@dataclass(frozen=True)
class SkeletonTopology:
    """Derived topology information from a 1px skeleton mask."""

    endpoints: List[Point]
    junctions: List[Point]
    polylines: List[List[Point]]


def _neighbors8() -> List[Tuple[int, int]]:
    return [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]


def _ensure_uint8_binary(mask: np.ndarray) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.ndim != 2:
        raise ValueError("Expected a 2D binary mask.")
    # Normalize to 0/255
    out = np.where(mask > 0, 255, 0).astype(np.uint8)
    return out


def _skeletonize_uint8(mask: np.ndarray) -> np.ndarray:
    """Skeletonize a uint8 0/255 foreground mask to 1px (uint8 0/255)."""
    mask = _ensure_uint8_binary(mask)
    skel_bool = skeletonize(mask > 0)
    return (skel_bool.astype(np.uint8) * 255)


def _compute_degree_map(skel_bool: np.ndarray) -> np.ndarray:
    """Degree (# of 8-neighbors) for each skeleton pixel."""
    h, w = skel_bool.shape
    padded = np.pad(skel_bool, 1, constant_values=False)
    deg = np.zeros((h, w), dtype=np.uint8)
    for dy, dx in _neighbors8():
        deg += padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
    return deg


def _rdp_simplify(points: Sequence[Point], epsilon: float) -> List[Point]:
    """Ramer–Douglas–Peucker polyline simplification."""
    if len(points) < 3:
        return list(points)

    pts = np.array(points, dtype=np.float32)
    start = pts[0]
    end = pts[-1]

    # Distance from point to line segment (start-end)
    seg = end - start
    seg_len_sq = float(seg[0] ** 2 + seg[1] ** 2)
    if seg_len_sq == 0:
        dists = np.linalg.norm(pts - start, axis=1)
    else:
        t = np.clip(((pts - start) @ seg) / seg_len_sq, 0.0, 1.0)
        proj = start + np.outer(t, seg)
        dists = np.linalg.norm(pts - proj, axis=1)

    idx = int(np.argmax(dists))
    max_dist = float(dists[idx])
    if max_dist <= epsilon:
        return [points[0], points[-1]]

    left = _rdp_simplify(points[: idx + 1], epsilon)
    right = _rdp_simplify(points[idx:], epsilon)
    return left[:-1] + right


def vectorize_skeleton(mask_1px: np.ndarray, *, simplify_epsilon: float = 1.5) -> SkeletonTopology:
    """
    Vectorize a *1px* skeleton mask into polylines and derive endpoints/junctions.

    This is a best-effort topology extraction suitable for UI affordances (endpoint
    picking + connect). It is not intended to be a perfect graph extractor.
    """
    mask_1px = _ensure_uint8_binary(mask_1px)
    skel_bool = mask_1px > 0
    h, w = skel_bool.shape

    deg = _compute_degree_map(skel_bool)
    node_mask = skel_bool & (deg != 2)

    ys, xs = np.where(node_mask)
    nodes = {(int(x), int(y)) for x, y in zip(xs, ys)}

    endpoints = [(x, y) for (x, y) in nodes if int(deg[y, x]) == 1]
    junctions = [(x, y) for (x, y) in nodes if int(deg[y, x]) >= 3]

    # Edge visitation (undirected) on pixel graph
    visited_edges = set()

    def iter_neighbors(p: Point) -> Iterable[Point]:
        x, y = p
        for dy, dx in _neighbors8():
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and skel_bool[ny, nx]:
                yield (nx, ny)

    def mark_edge(a: Point, b: Point) -> None:
        visited_edges.add((a, b))
        visited_edges.add((b, a))

    polylines: List[List[Point]] = []

    # Trace from each node along each unvisited neighbor direction
    for n in list(nodes):
        for nb in iter_neighbors(n):
            if (n, nb) in visited_edges:
                continue
            path = [n, nb]
            mark_edge(n, nb)
            prev = n
            cur = nb
            while True:
                if cur in nodes and cur != n:
                    break
                nbs = [p for p in iter_neighbors(cur) if p != prev]
                if not nbs:
                    break
                nxt = nbs[0]
                path.append(nxt)
                mark_edge(cur, nxt)
                prev, cur = cur, nxt

            if simplify_epsilon > 0:
                path = _rdp_simplify(path, simplify_epsilon)
            polylines.append(path)

    # Handle loops with no nodes (all degree==2)
    if not nodes:
        ys2, xs2 = np.where(skel_bool)
        remaining = {(int(x), int(y)) for x, y in zip(xs2, ys2)}
        while remaining:
            start = next(iter(remaining))
            # pick an arbitrary neighbor to start walking
            nbs = list(iter_neighbors(start))
            if not nbs:
                remaining.remove(start)
                continue
            prev = start
            cur = nbs[0]
            path = [start, cur]
            remaining.discard(start)
            remaining.discard(cur)
            while True:
                nbs2 = [p for p in iter_neighbors(cur) if p != prev]
                if not nbs2:
                    break
                nxt = nbs2[0]
                if nxt == start:
                    path.append(nxt)
                    break
                path.append(nxt)
                remaining.discard(nxt)
                prev, cur = cur, nxt
            if simplify_epsilon > 0:
                path = _rdp_simplify(path, simplify_epsilon)
            polylines.append(path)

    return SkeletonTopology(endpoints=endpoints, junctions=junctions, polylines=polylines)


class SkeletonCorrectionModel:
    """
    Holds the editable skeleton mask and basic undo/redo.

    Representation:
    - `mask` is uint8 0/255 foreground mask at *editor resolution* (typically image size).
    """

    def __init__(self):
        self.mask: Optional[np.ndarray] = None
        self._undo_stack: List[np.ndarray] = []
        self._redo_stack: List[np.ndarray] = []
        self.max_stack_size = 25

    # -------------------- undo/redo --------------------
    def push_undo(self) -> None:
        if self.mask is None:
            return
        # Deep copy to ensure we capture the exact state
        state = self.mask.copy()
        self._undo_stack.append(state)
        if len(self._undo_stack) > self.max_stack_size:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self) -> bool:
        if self.mask is None or not self._undo_stack:
            return False
        # Save current state to redo stack
        current = self.mask.copy()
        self._redo_stack.append(current)
        if len(self._redo_stack) > self.max_stack_size:
            self._redo_stack.pop(0)
        # Restore previous state
        self.mask = self._undo_stack.pop().copy()
        return True

    def redo(self) -> bool:
        if self.mask is None or not self._redo_stack:
            return False
        # Save current state to undo stack
        current = self.mask.copy()
        self._undo_stack.append(current)
        if len(self._undo_stack) > self.max_stack_size:
            self._undo_stack.pop(0)
        # Restore next state
        self.mask = self._redo_stack.pop().copy()
        return True

    def clear_history(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()

    # -------------------- load / transform --------------------
    def set_empty(self, size: Tuple[int, int]) -> None:
        w, h = size
        self.mask = np.zeros((h, w), dtype=np.uint8)
        self.clear_history()

    def load_from_raster(
        self,
        raster_gray: np.ndarray,
        *,
        target_size: Tuple[int, int],
        threshold: str = "otsu",
    ) -> None:
        """Load skeleton from a grayscale raster image into `mask` at target_size."""
        if raster_gray.ndim != 2:
            raise ValueError("Expected a grayscale image.")

        img = raster_gray
        if threshold == "otsu":
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

        # Ensure skeleton is foreground (white) by inverting if background dominates.
        if float(np.mean(binary)) > 127.0:
            binary = 255 - binary

        w_t, h_t = target_size
        if (binary.shape[1], binary.shape[0]) != (w_t, h_t):
            binary = cv2.resize(binary, (w_t, h_t), interpolation=cv2.INTER_NEAREST)

        self.mask = _skeletonize_uint8(binary)
        self.clear_history()

    # -------------------- edits --------------------
    def erase_circle(self, center: Point, radius: int) -> None:
        if self.mask is None:
            return
        x, y = center
        cv2.circle(self.mask, (int(x), int(y)), int(radius), 0, thickness=-1)

    def draw_polyline(self, points: Sequence[Point], thickness: int = 3) -> None:
        if self.mask is None or len(points) < 2:
            return
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(self.mask, [pts], isClosed=False, color=255, thickness=int(thickness))

    def skeletonize(self) -> None:
        if self.mask is None:
            return
        self.mask = _skeletonize_uint8(self.mask)

    def topology(self, *, simplify_epsilon: float = 1.5) -> SkeletonTopology:
        if self.mask is None:
            return SkeletonTopology(endpoints=[], junctions=[], polylines=[])
        # Ensure topology extraction sees a 1px skeleton
        return vectorize_skeleton(self.mask, simplify_epsilon=simplify_epsilon)

    # -------------------- save --------------------
    def render_to_size(self, size: Tuple[int, int]) -> np.ndarray:
        """
        Render current mask to a different size (uint8 0/255) using nearest neighbor.
        """
        if self.mask is None:
            w, h = size
            return np.zeros((h, w), dtype=np.uint8)
        w, h = size
        out = cv2.resize(self.mask, (w, h), interpolation=cv2.INTER_NEAREST)
        out = _skeletonize_uint8(out)
        return out



"""Editing tools of the skeleton editor (AR-12 step 2 and 4).

Each tool is a small State object: the editor forwards pointer and key input to the
active tool (``editor.tools[editor.current_tool]``) instead of branching on the tool
name in every handler. Tools act on the editor through the narrow :class:`ToolHost`
protocol, and keep their own transient state; the polyline being drawn or edited is a
:class:`PolylineDraft` shared with the editor (the Select tool fills it, the Polyline
tool edits and commits it).

Positions are image pixels (``QPoint``), already clamped by the view. Grab and snap
distances (``*_HIT_PX``) are screen px, turned into image px at the canvas zoom by
:func:`reach`, so they match the markers at every zoom.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Protocol

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QKeyEvent

from app.editing.geometry import (
    nearest_polyline_index,
    nearest_segment_index,
    nearest_vertex_index,
    sample_catmull_rom,
)
from app.gui.correct.layer import ToolSpec
from app.gui.key_text import platform_keys

if TYPE_CHECKING:
    from app.gui.skeleton_graph_model import SkeletonCorrectionModel
    from app.gui.skeleton_overlay import SkeletonDecorations

SELECT = "select"
ERASER = "eraser"
POLYLINE = "polyline"
CONNECT = "connect"

# Single-letter tool keys (QuPath / napari style), shown in the rail tooltips.
TOOL_KEYS: dict[str, str] = {SELECT: "V", ERASER: "E", POLYLINE: "P", CONNECT: "C"}
SMOOTH_KEY = "S"

#: The tools as the Correct page's rail shows them (``SkeletonLayer.tools``); only the
#: eraser has a size.
TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(SELECT, "cursor", "Select", "V", "pick a branch, Delete removes it", False),
    ToolSpec(ERASER, "erase", "Eraser", "E", "[ and ] resize", True),
    ToolSpec(POLYLINE, "polyline", "Polyline", "P", "Enter finish · Esc cancel", False),
    ToolSpec(CONNECT, "link", "Connect endpoints", "C", "drag from an endpoint", False),
)
#: The chip's ``(hint, tooltip)`` with no image open, and with an empty skeleton.
CLOSED_HINT = ("Select an image to correct its skeleton.", "")
EMPTY_HINT = (
    "No skeleton yet · Generate, Load one, or draw with Polyline (P)",
    "Generate skeletons first, open a file with Load (dock), or draw the roots with the "
    "Polyline tool.",
)

HANDLE_HIT_PX = 8  # grab a polyline point (screen px, see reach)
BEND_HIT_PX = 8  # Ctrl+click a segment to add a bend
MOVE_HIT_PX = 10  # Shift+drag a segment to move the line
PICK_HIT_PX = 12  # Select: pick a skeleton branch
ENDPOINT_HIT_PX = 10  # Connect: grab an endpoint
REACH_MIN_PX = 1.0  # image px: the pointer is floored to a pixel
REACH_MAX_SCALE = 4.0  # image px: at most four times the distance at 100 %
SMOOTH_STEP_PX = 0.75


def _xy(p: QPoint) -> tuple[int, int]:
    return (p.x(), p.y())


def _closes(points: list[QPoint]) -> bool:
    """Whether a line of 3+ points closes on itself: its ends at most a pixel apart (a ring,
    or a loop whose ends are two pixels of the junction cluster it hangs from)."""
    if len(points) < 3:
        return False
    a, b = points[0], points[-1]
    return max(abs(a.x() - b.x()), abs(a.y() - b.y())) <= 1


def reach(host: ToolHost, screen_px: float) -> float:
    """Image px that ``screen_px`` screen px cover at the canvas zoom (S4).

    A grab or snap distance then stays the same on screen, as the markers do, clamped to
    REACH_MIN_PX .. REACH_MAX_SCALE x ``screen_px`` image px. At 100 %, and without a view,
    it is ``screen_px`` itself: the tools' image px distances of before.
    """
    scale = host.decorations.view_scale()
    return min(max(screen_px / scale, REACH_MIN_PX), screen_px * REACH_MAX_SCALE)


@dataclass(frozen=True)
class DraftState:
    """What the polyline prompt shows: the Polyline tool, its draft points and Smooth."""

    in_polyline: bool
    points: int
    smooth: bool


@dataclass
class PolylineDraft:
    """The polyline being drawn, or a picked branch being edited.

    ``edit_original`` holds the picked branch's points as picked (what :meth:`pending`
    compares), ``edit_pixels`` its own pixels: erased when the edit is committed or the
    branch deleted, so the other branches at its junctions keep theirs (S03-01).
    ``smooth`` mirrors the Smooth toggle and survives :meth:`reset`. ``turns`` keeps, per
    point a click added to a pick, the draft right after the click and its points and
    original before it (:meth:`add`), so :meth:`take_back` undoes the turn with the point.
    """

    points: list[QPoint] = field(default_factory=list)
    drag_index: int | None = None
    translating: bool = False
    anchor: QPoint | None = None
    orig_points: list[QPoint] = field(default_factory=list)
    smooth: bool = False
    edit_original: list[tuple[int, int]] | None = None
    edit_pixels: list[tuple[int, int]] | None = None
    smooth_on_pick: bool = False  # Smooth was already on when the branch was picked
    turns: list[tuple[list[tuple[int, int]], list[QPoint], list[tuple[int, int]] | None]] = field(
        default_factory=list
    )
    SMOOTH_DROPPED_NOTICE: ClassVar[str] = (
        "Smoothed branch not applied · pick it and press Enter to apply"
    )

    def reset(self) -> None:
        """Forget the line and any drag (keeps the Smooth setting)."""
        self.points = []
        self.edit_original = self.edit_pixels = None
        self.smooth_on_pick = False
        self.turns = []
        self.end_drag()

    def end_drag(self) -> None:
        self.drag_index = None
        self.translating = False
        self.anchor = None
        self.orig_points = []

    def xy(self) -> list[tuple[int, int]]:
        return [_xy(p) for p in self.points]

    def pending(self) -> bool:
        """Two or more points that a commit would draw: a new line, or a picked branch
        whose points moved (a branch left as it was changes nothing). Unsaved work (S5)."""
        if len(self.points) < 2:
            return False
        original = self.edit_original
        return original is None or len(original) != len(self.points) or self.xy() != original

    def plain_pick(self) -> bool:
        """A branch picked with Select and left as it was: dropping it loses nothing."""
        return self.edit_original is not None and not self.pending()

    def holds_work(self) -> bool:
        """Points that dropping the draft loses (a new line's, or a picked branch's that
        moved): a dropped draft is announced only then."""
        return bool(self.points) and not self.plain_pick()

    def bends(self) -> bool:
        """Smooth is on and the draft has the 3+ points it needs to bend into a curve."""
        return self.smooth and len(self.points) >= 3

    def unapplied_smooth(self) -> bool:
        """A plain pick whose preview was smoothed by a Smooth switched on during the pick."""
        return self.plain_pick() and self.smooth and not self.smooth_on_pick

    def drop_notice(self, discarded: str) -> str | None:
        """What dropping the draft tells: ``discarded`` for lost work, the smoothed-pick
        notice for an unapplied Smooth, else None."""
        if self.holds_work():
            return discarded
        return self.SMOOTH_DROPPED_NOTICE if self.unapplied_smooth() else None

    def raster_points(self, bounds: tuple[int, int]) -> list[tuple[int, int]]:
        """Pixels to draw: the straight polyline, or the sampled curve when smooth."""
        if self.bends():
            return sample_catmull_rom(self.xy(), step_px=SMOOTH_STEP_PX, bounds=bounds)
        return self.xy()

    def facing(self, pos: QPoint | None) -> list[QPoint]:
        """The points in the order a click at ``pos`` goes on from: its point last.

        A picked branch comes in the order the topology traced it, which says nothing of
        the pointer, so it goes on from its point nearest ``pos`` (owner bug report
        2026-11-02): an open branch from its nearer end (at equal distance the end with
        the lower ``(y, x)``), a closed one (a ring, a loop: :func:`_closes`) opened at its
        nearest point. A new line keeps the order its user placed. Only reads the draft.
        """
        pts = self.points
        if pos is None or self.edit_original is None or len(pts) < 2:
            return pts
        px, py = pos.x(), pos.y()

        def rank(p: QPoint) -> tuple[int, int, int]:
            return ((p.x() - px) ** 2 + (p.y() - py) ** 2, p.y(), p.x())

        if not _closes(pts):
            return pts[::-1] if rank(pts[0]) < rank(pts[-1]) else pts
        ring = pts[:-1] if pts[0] == pts[-1] else pts  # a loop's ends: two cluster pixels
        i = min(range(len(ring)), key=lambda j: rank(ring[j]))
        if ring is pts and i in (0, len(pts) - 1):  # such an end is nearest: as a branch
            return pts[::-1] if i == 0 else pts
        return pts if i == 0 else ring[i:] + ring[:i] + [QPoint(ring[i])]

    def face(self, pos: QPoint) -> None:
        """Reorder the points as :meth:`facing` gives them; a pick left as it was turns
        what :meth:`pending` compares with too, so it still reads as unchanged."""
        points = self.facing(pos)
        if points is not self.points:
            if self.plain_pick():
                self.edit_original = [_xy(p) for p in points]
            self.points = points

    def add(self, point: QPoint) -> None:
        """Add a clicked point: after a new line's last point, at a pick's point nearest the
        click (:meth:`face`), not at its traced end; the turn is kept for :meth:`take_back`."""
        if self.edit_original is None:
            self.points.append(point)
            return
        before = (list(self.points), self.edit_original)
        self.face(point)
        self.points.append(point)
        self.turns.append((self.xy(), *before))

    def take_back(self) -> None:
        """Drop the last point (a mid-polyline undo, a double-click's own point). A click on
        a pick goes with its turn: the draft is as before the click, its order and original
        included (also when the clicked point was dragged: it goes anyway), unless another
        point moved or one was added since; then only the last point goes."""
        turn = self.turns.pop() if self.turns else None
        made = turn[0] if turn is not None else None
        if made is not None and len(self.points) == len(made) and self.xy()[:-1] == made[:-1]:
            self.points, self.edit_original = list(turn[1]), turn[2]
        elif self.points:
            self.points.pop()

    def preview(self, cursor: QPoint | None) -> tuple[list[QPoint], QPoint | None, QPoint | None]:
        """``(points, cursor, start)`` for ``SkeletonDecorations.show_polyline``: the line as
        a click at ``cursor`` goes on from it (:meth:`facing`), its rubber band from its last
        point. A closed pick is shown as Enter draws it, its rubber band apart from its point
        nearest the cursor (``start``), so the pointer does not move its smoothed seam."""
        points = self.points
        if not points or cursor is None:
            return points, None, None
        if self.edit_original is not None and _closes(points):
            return points, cursor, self.facing(cursor)[-1]
        return self.facing(cursor), cursor, None


class ToolHost(Protocol):
    """What a tool may use of the editor."""

    model: SkeletonCorrectionModel
    draft: PolylineDraft
    decorations: SkeletonDecorations
    draw_thickness: int

    def clamp_to_image(self, pt: QPoint) -> QPoint: ...
    def image_size(self) -> tuple[int, int]: ...
    def eraser_footprint_radius(self) -> int: ...
    def render_skeleton(self, *, full: bool = True) -> None: ...
    def schedule_render(self) -> None: ...
    def cancel_scheduled_render(self) -> None: ...
    def refresh_draft(self, cursor: QPoint | None = None) -> None: ...
    def refresh_hint(self) -> None: ...
    def switch_tool(self, tool_id: str, *, keep_draft: bool = False) -> None: ...
    def show_eraser_ring(self, pos: QPoint | None) -> None: ...
    def notify(self, message: str, kind: str = "info") -> None: ...


class SkeletonTool:
    """Base tool: ignores every input. Subclasses override what they use."""

    tool_id = ""

    def __init__(self, host: ToolHost) -> None:
        self.host = host

    def press(self, pos: QPoint, modifiers: Qt.KeyboardModifier) -> None:
        pass

    def move(self, pos: QPoint) -> None:
        pass

    def release(self, pos: QPoint) -> None:
        pass

    def double_click(self, pos: QPoint) -> None:
        pass

    def key(self, event: QKeyEvent) -> bool:
        return False

    def deactivate(self) -> None:
        """Drop transient state when another tool is chosen."""

    def hint(self) -> tuple[str, str]:
        """``(short hint, tooltip)`` for the status chip."""
        return ("Ready to edit skeleton.", "")


class SelectTool(SkeletonTool):
    """Pick a skeleton branch; it opens in the Polyline tool for editing or Delete."""

    tool_id = SELECT

    def press(self, pos, modifiers):
        host = self.host
        topo = host.model.topology(simplify_epsilon=1.5)
        # Only branches with pixels of their own: an edge between two junction pixels has
        # none to erase. The others stay at their index.
        lines = [line if own else [] for line, own in zip(topo.polylines, topo.own, strict=True)]
        i = nearest_polyline_index(lines, _xy(pos), reach(host, PICK_HIT_PX))
        if i is None:
            host.refresh_hint()
            return
        branch = [(int(x), int(y)) for x, y in topo.polylines[i]]
        host.draft.reset()
        host.draft.edit_original = branch
        host.draft.edit_pixels = list(topo.own[i])
        host.draft.smooth_on_pick = host.draft.smooth
        host.draft.points = [QPoint(x, y) for x, y in branch]
        host.switch_tool(POLYLINE, keep_draft=True)
        host.refresh_draft()

    def hint(self):
        return (
            "Select · click a skeleton line to edit it",
            f"Click near a skeleton line to pick it: drag its points, press {SMOOTH_KEY} to "
            "smooth it and Enter to commit, or press Delete to remove the whole branch.",
        )


class EraserTool(SkeletonTool):
    """Drag to erase; each move erases the whole segment from the previous point (PE-02)."""

    tool_id = ERASER

    def __init__(self, host):
        super().__init__(host)
        self._last: tuple[int, int] | None = None

    @property
    def active(self) -> bool:
        return self._last is not None

    def press(self, pos, modifiers):
        host = self.host
        host.model.push_undo()
        self._last = _xy(pos)
        host.model.erase_segment(self._last, self._last, host.eraser_footprint_radius())
        host.decorations.set_endpoints_visible(False)
        host.render_skeleton(full=False)

    def move(self, pos):
        host = self.host
        host.show_eraser_ring(pos)
        if self._last is None:
            return
        pt = _xy(pos)
        host.model.erase_segment(self._last, pt, host.eraser_footprint_radius())
        self._last = pt
        host.schedule_render()

    def release(self, pos):
        if self._last is None:
            return
        self._last = None
        host = self.host
        host.cancel_scheduled_render()
        host.model.skeletonize()  # clean to 1 px for stable endpoints
        host.render_skeleton(full=True)

    def deactivate(self):
        if self._last is not None:
            self.release(QPoint(*self._last))
        self.host.show_eraser_ring(None)

    def hint(self):
        return (
            "Eraser · drag over the skeleton to erase it",
            "Click and drag to erase skeleton pixels. [ and ] change the eraser size.",
        )


class PolylineTool(SkeletonTool):
    """Click points, drag them, Ctrl+click to bend, Shift+drag to move; Enter commits."""

    tool_id = POLYLINE

    def __init__(self, host):
        super().__init__(host)
        # The point this press added to a picked branch (the very object), else None: the
        # double-click whose first press it was takes it back (S03-02).
        self._pick_click: QPoint | None = None

    def press(self, pos, modifiers):
        host, draft = self.host, self.host.draft
        self._pick_click = None
        pts = draft.xy()
        hit = nearest_vertex_index(pts, _xy(pos), reach(host, HANDLE_HIT_PX))
        if hit is not None:
            draft.end_drag()
            draft.drag_index = hit
            return
        if modifiers & Qt.KeyboardModifier.ControlModifier and len(pts) >= 2:
            seg = nearest_segment_index(pts, _xy(pos), reach(host, BEND_HIT_PX))
            if seg is not None:
                draft.points.insert(seg + 1, QPoint(pos))
                host.refresh_draft(pos)
                return
        if modifiers & Qt.KeyboardModifier.ShiftModifier and len(pts) >= 2:
            if nearest_segment_index(pts, _xy(pos), reach(host, MOVE_HIT_PX)) is not None:
                draft.end_drag()
                draft.translating = True
                draft.anchor = QPoint(pos)
                draft.orig_points = [QPoint(p) for p in draft.points]
                return
        point = QPoint(pos)
        draft.add(point)  # a pick goes on from its point nearest the click
        if draft.edit_original is not None:
            self._pick_click = point
        host.refresh_draft(pos)

    def move(self, pos):
        host, draft = self.host, self.host.draft
        if not draft.points:
            return
        if draft.drag_index is not None:
            draft.points[draft.drag_index] = QPoint(pos)
        elif draft.translating and draft.anchor is not None:
            dx, dy = pos.x() - draft.anchor.x(), pos.y() - draft.anchor.y()
            draft.points = [
                host.clamp_to_image(QPoint(p.x() + dx, p.y() + dy)) for p in draft.orig_points
            ]
        host.refresh_draft(pos)

    def release(self, pos):
        draft = self.host.draft
        draft.end_drag()
        if draft.points:
            self.host.refresh_draft(pos)

    def double_click(self, pos):
        draft = self.host.draft
        if not draft.points:
            return
        if draft.edit_original is not None:
            # The double-click's first press added the click as a last point. On a plain
            # pick, or on the branch itself, that is a chord to the click: take it back
            # (S03-02). Off the branch of a changed pick it is the line's new end: keep it,
            # and add where the double-click landed when it drifted, as a new line does.
            # Commit only if the branch changed; a plain pick stays open.
            click, self._pick_click = self._pick_click, None
            if click is not None and draft.points[-1] is click:
                if self._chord(click):  # as before the press, in its order too
                    draft.take_back()
                    self.host.refresh_draft()
                elif pos != click:
                    draft.points.append(QPoint(pos))
            if draft.pending():
                self.commit()
            return
        if pos != draft.points[-1]:
            draft.points.append(QPoint(pos))
        self.commit()

    def _chord(self, click: QPoint) -> bool:
        """Whether ``click``, the picked branch's last point, only draws a chord: the rest
        is still the branch as picked, or the click lies on the rest."""
        draft = self.host.draft
        rest = draft.xy()[:-1]
        if rest == draft.edit_original:
            return True
        return nearest_segment_index(rest, _xy(click), reach(self.host, PICK_HIT_PX)) is not None

    def key(self, event):
        key = event.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and len(self.host.draft.points) >= 2:
            self.commit()
            return True
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self.host.draft.edit_pixels:
            self.delete_branch()
            return True
        return False

    def commit(self) -> None:
        """Draw the draft into the skeleton (replacing the picked branch) as one undo step.

        A picked branch left as it was is dropped instead, with no pixel changed and no
        undo step (S03-02), unless Smooth bends it (the documented pick, S, Enter).
        """
        host, draft = self.host, self.host.draft
        if len(draft.points) < 2 or (draft.plain_pick() and not draft.bends()):
            draft.reset()
            host.refresh_draft()
            return
        pts = draft.raster_points(host.image_size())
        host.model.push_undo()
        if draft.edit_pixels:  # the picked branch's own pixels, not a band (S03-01)
            host.model.erase_pixels(draft.edit_pixels)
        host.model.draw_polyline(pts, thickness=host.draw_thickness)
        host.model.skeletonize()
        draft.reset()
        host.refresh_draft()
        host.render_skeleton(full=True)

    def delete_branch(self) -> None:
        """Remove the picked branch (UXE-12): one undo step. Only its own pixels go; the
        junction pixels at its ends stay with the other branches there (S03-01)."""
        host, draft = self.host, self.host.draft
        pixels = draft.edit_pixels
        if not pixels:
            return
        host.model.push_undo()
        host.model.erase_pixels(pixels)
        host.model.skeletonize()
        draft.reset()
        host.refresh_draft()
        host.render_skeleton(full=True)
        host.notify(platform_keys("Branch deleted · Ctrl+Z to undo"), "info")

    def deactivate(self):
        self._pick_click = None
        self.host.draft.reset()

    def hint(self):
        draft = self.host.draft
        n = len(draft.points)
        help_text = platform_keys(
            "Click to place points. Drag points to adjust. Ctrl+click a segment to add a "
            "bend. Shift+drag a segment to move the whole line. Enter or double-click "
            f"finishes the line, Esc cancels it, {SMOOTH_KEY} smooths it into a curve "
            "(3+ points)."
        )
        if draft.edit_original:
            # Enter on a branch left as it was only drops the pick (S03-02).
            enter = "Enter commits · " if draft.pending() or draft.bends() else ""
            text = f"Editing branch · {SMOOTH_KEY} smooths · {enter}Delete removes it · Esc cancels"
        elif n == 0:
            text = platform_keys(
                "Polyline · click to place points · Ctrl+click a segment adds a bend"
            )
        elif n == 1:
            text = "Polyline · 1 point · click to add more"
        else:
            text = f"Polyline · {n} points · drag points to adjust"
        if draft.smooth and n < 3:
            text += " · Smooth needs 3+ points"
        return (text, help_text)


class ConnectTool(SkeletonTool):
    """Drag from an endpoint to join it to another point with a straight line."""

    tool_id = CONNECT

    def __init__(self, host):
        super().__init__(host)
        self.first: QPoint | None = None

    def _nearest_endpoint(self, pos: QPoint) -> QPoint | None:
        eps = self.host.model.endpoints()
        i = nearest_vertex_index(eps, _xy(pos), reach(self.host, ENDPOINT_HIT_PX))
        return None if i is None else QPoint(*eps[i])

    def press(self, pos, modifiers):
        ep = self._nearest_endpoint(pos)
        if ep is None:
            return
        self.first = ep
        self.host.decorations.highlight_endpoint(ep)
        self.host.decorations.show_connect(ep, pos)
        self.host.refresh_hint()

    def move(self, pos):
        if self.first is not None:
            self.host.decorations.show_connect(self.first, pos)

    def release(self, pos):
        if self.first is None:
            return
        first, self.first = self.first, None
        host = self.host
        host.decorations.clear_connect()
        host.decorations.highlight_endpoint(None)
        host.model.push_undo()
        host.model.draw_polyline([_xy(first), _xy(pos)], thickness=host.draw_thickness)
        host.model.skeletonize()
        host.render_skeleton(full=True)
        host.refresh_hint()

    def key(self, event):
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self.first is not None:
            self.deactivate()
            self.host.refresh_hint()
            return True
        return False

    def deactivate(self):
        self.first = None
        self.host.decorations.clear_connect()
        self.host.decorations.highlight_endpoint(None)

    def hint(self):
        if self.first is None:
            return ("Connect · drag from an endpoint (orange dot)", "")
        return ("Connect · release on the point to join", "")


def build_tools(host: ToolHost) -> dict[str, SkeletonTool]:
    """One instance of every tool, keyed by tool id."""
    return {cls.tool_id: cls(host) for cls in (SelectTool, EraserTool, PolylineTool, ConnectTool)}

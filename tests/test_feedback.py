"""The user-feedback port (``app/feedback.py``) and its Qt adapter (``app/gui/ui_feedback.py``).

Handlers report long-running work through ``UiFeedback`` instead of reaching into the
window's widgets. The two Qt-free doubles must satisfy the protocol (handler tests rely on
them): ``NullFeedback`` takes every call, ``RecordingFeedback`` records them and tracks the
task state. ``QtUiFeedback`` hands a toast its action button, and still serves a notifier
that takes no action.
"""

from __future__ import annotations

from app.feedback import (
    FeedbackCall,
    NoticeAction,
    NullFeedback,
    RecordingFeedback,
    UiFeedback,
)


# --- the doubles ---------------------------------------------------------------------
def test_doubles_satisfy_the_protocol():
    assert isinstance(NullFeedback(), UiFeedback)
    assert isinstance(RecordingFeedback(), UiFeedback)


def test_null_feedback_accepts_every_call():
    fb = NullFeedback()
    fb.task_started("Generating masks", total=10, cancelable=True, on_cancel=lambda: None)
    fb.task_progress(3)
    fb.task_progress(4, 10)
    fb.task_finished("done")
    fb.task_failed("boom")
    fb.status("hello", timeout_ms=10)
    fb.notify("saved", kind="info")


def test_recording_feedback_records_calls_in_order():
    fb = RecordingFeedback()
    fb.task_started("Generating masks", total=4)
    fb.task_progress(1)
    fb.task_finished("Masks ready")
    assert fb.names() == ["task_started", "task_progress", "task_finished"]
    assert fb.calls[0] == FeedbackCall(
        "task_started",
        ("Generating masks",),
        {"total": 4, "cancelable": False, "on_cancel": None},
    )


def test_recording_feedback_tracks_task_state():
    fb = RecordingFeedback()
    assert fb.busy is False
    fb.task_started("Measuring", total=4)
    assert (fb.busy, fb.label, fb.percent) == (True, "Measuring", 0)
    fb.task_progress(1)
    assert fb.percent == 25  # counts against the total given to task_started
    fb.task_progress(50, 100)  # an explicit total overrides it
    assert fb.percent == 50
    fb.task_failed("root_lengths.csv is open in another program")
    assert fb.busy is False
    assert fb.last_message == "root_lengths.csv is open in another program"


def test_recording_feedback_percent_mode_without_total():
    fb = RecordingFeedback()
    fb.task_started("Loading")
    fb.task_progress(37)
    assert fb.percent == 37
    fb.task_finished()
    assert (fb.busy, fb.percent) == (False, 100)


def test_recording_feedback_cancel_invokes_latest_on_cancel():
    fb = RecordingFeedback()
    hits = []
    fb.task_started("Generating skeletons", cancelable=True, on_cancel=lambda: hits.append(1))
    fb.cancel()
    assert hits == [1]


def test_recording_feedback_cancel_without_handler_is_a_no_op():
    fb = RecordingFeedback()
    fb.task_started("Loading")
    fb.cancel()  # must not raise


def test_the_recording_double_keeps_the_action():
    from app.feedback import RecordingFeedback

    feedback = RecordingFeedback()
    action = NoticeAction("Open folder", lambda: None)
    feedback.notify("Saved", "success", action=action)
    feedback.notify("Loaded")
    assert feedback.calls[0].kwargs == {"action": action}
    assert feedback.calls[1].kwargs == {}


# --- the Qt adapter ------------------------------------------------------------------
def test_the_feedback_port_hands_a_notice_its_action(app):
    from app.gui.task_progress import TaskProgressWidget
    from app.gui.ui_feedback import QtUiFeedback

    seen = []
    feedback = QtUiFeedback(
        TaskProgressWidget(),
        notifier=lambda message, kind, action=None: seen.append((message, kind, action)),
    )
    action = NoticeAction("Open folder", lambda: None)
    feedback.notify("Skeletons saved to the skeletons folder", "success", action=action)
    assert seen == [("Skeletons saved to the skeletons folder", "success", action)]


def test_a_notifier_without_actions_still_gets_plain_notices(app):
    from app.gui.task_progress import TaskProgressWidget
    from app.gui.ui_feedback import QtUiFeedback

    seen = []
    feedback = QtUiFeedback(
        TaskProgressWidget(), notifier=lambda message, kind: seen.append((message, kind))
    )
    feedback.notify("Loaded 40 images")
    assert seen == [("Loaded 40 images", "success")]

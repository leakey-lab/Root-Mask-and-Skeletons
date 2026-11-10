"""``CorrectSession.reopen_layer``: one layer read again from disk, the others left alone.

The page's Unmark reads the class labels again after it deleted their file (or found it
changed on disk); the mask and skeleton keep their unsaved edits and timeline entries.
"""

from __future__ import annotations

from tests.test_correct_session import BASE, IMAGE, SIZE, _Rig


def test_reopen_layer_reopens_that_layer_only_and_keeps_the_others_edits():
    rig = _Rig().opened()
    rig.mask.act()
    rig.skeleton.act()
    rig.clear_log()

    rig.session.reopen_layer("skeleton")

    assert rig.skeleton.calls == [("open", IMAGE, SIZE, BASE)]
    assert rig.mask.calls == []
    assert [entry for entry in rig.session.journal.undo_entries] == ["mask"]
    assert rig.of("history") == [()]  # the page is refreshed once, at the end

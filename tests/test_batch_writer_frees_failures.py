"""C6-01 side issue: a failed save must not keep its image in memory until the run ends.

``AsyncWriter`` keeps every failed save's exception for the run's report. The exception's
traceback held the save's frames, and with them the image array and its encoded bytes
(about 0.3-0.8 MiB per failed image), so a run in which every save failed grew by that
much per image. The writer now clears those frames once the failure is logged; the
exception, its message and its traceback's line numbers stay.
"""

from __future__ import annotations

import errno
import logging
import traceback
import weakref

from app.inference.batch_pipeline import AsyncWriter


class _Image:
    """Stands in for an image array: something a frame holds that a weak reference can watch."""


def _failing_save(refs: list):
    def save(key):
        image = _Image()  # a local of the failed save's frame
        refs.append(weakref.ref(image))
        raise OSError(errno.EACCES, "Permission denied", f"out/{key}.png")

    return save


def test_failed_saves_do_not_keep_their_images(caplog):
    refs: list = []
    with caplog.at_level(logging.WARNING, "app.inference.batch_pipeline"):
        with AsyncWriter(max_workers=2, max_pending=4) as writer:
            for i in range(6):
                writer.submit(f"img{i}", _failing_save(refs), f"img{i}")
        errors = writer.errors()

    assert len(errors) == 6 and len(refs) == 6
    assert all(ref() is None for ref in refs)  # every image freed while the run still holds
    for key, exc in errors:  # ... the errors, which still say what failed and where
        assert str(exc) == f"[Errno {errno.EACCES}] Permission denied: 'out/{key}.png'"
        assert exc.__traceback__ is not None
        assert "save" in "".join(traceback.format_exception(exc))
    # The first failure is still logged with its traceback.
    assert any(record.exc_info for record in caplog.records)


def test_a_save_that_succeeds_is_untouched():
    done = []
    with AsyncWriter(max_workers=1) as writer:
        writer.submit("a", done.append, "a")
    assert done == ["a"] and writer.errors() == []

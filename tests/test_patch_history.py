"""Patch storage of the raster editors' undo history (app/gui/patch_history.py).

A large 0/1 region (every mask edit) or 0/255 region (every skeleton edit: the skeleton
model's history stores these patches too) is stored as its packed bits, deflated: 8x less
data to compress on commit and to inflate on undo / redo, and at most one bit per pixel
however noisy the region is. An all-0 region (a Clear's after) stores nothing. Other regions
(class ids, a 0/255 region with any third value) are deflated byte for byte. Every form gives
back exactly the region it stored (values, shape and dtype).
"""

import numpy as np
import pytest

from app.gui.patch_history import COMPRESS_MIN_BYTES, Patch, PatchHistory

OVERHEAD = 128 + 2 * 64  # the entry overhead plus a deflate header / trailer per side


@pytest.mark.parametrize("shape", [(400, 400), (131, 257), (1, COMPRESS_MIN_BYTES + 3)])
def test_a_large_binary_region_is_stored_in_at_most_one_bit_per_pixel(shape):
    rng = np.random.default_rng(1)
    before = (rng.random(shape) < 0.5).astype(np.uint8)  # noise: deflate alone cannot shrink it
    after = 1 - before
    assert before.nbytes >= COMPRESS_MIN_BYTES

    patch = Patch.capture(2, 3, before, after)

    assert patch.nbytes <= 2 * -(-before.size // 8) + OVERHEAD
    for got, want in ((patch.before(), before), (patch.after(), after)):
        assert got.dtype == np.uint8 and got.shape == shape
        assert np.array_equal(got, want)


def test_a_large_region_of_class_ids_round_trips_exactly():
    rng = np.random.default_rng(2)
    before = rng.integers(0, 6, (300, 211), dtype=np.uint8)
    before[::7, ::5] = 255  # Ignore pixels
    after = np.where(before == 2, 1, before).astype(np.uint8)

    patch = Patch.capture(0, 0, before, after)

    assert np.array_equal(patch.before(), before) and np.array_equal(patch.after(), after)
    assert patch.before().dtype == np.uint8


def test_a_binary_region_of_another_dtype_round_trips_exactly():
    before = np.zeros((200, 300), np.uint16)
    before[50:150, 20:280] = 1
    patch = Patch.capture(0, 0, before, np.zeros_like(before))
    assert patch.before().dtype == np.uint16 and np.array_equal(patch.before(), before)
    assert not patch.after().any()


def test_undo_and_redo_of_a_whole_plane_mask_edit_restore_every_pixel():
    rng = np.random.default_rng(3)
    plane = (rng.random((1536, 2048)) < 0.4).astype(np.uint8)
    original = plane.copy()
    history = PatchHistory()
    history.reset(plane)
    after = original.copy()
    after[100:1400, 30:2000] ^= 1
    history.commit((0, 0, 2048, 1536), after)
    edited = history.plane.copy()

    assert history.undo() is not None and np.array_equal(history.plane, original)
    assert history.redo() is not None and np.array_equal(history.plane, edited)


@pytest.mark.parametrize("shape", [(400, 400), (131, 257), (1, COMPRESS_MIN_BYTES + 3)])
def test_a_large_skeleton_region_of_0_and_255_is_stored_in_at_most_one_bit_per_pixel(shape):
    rng = np.random.default_rng(4)
    before = np.where(rng.random(shape) < 0.5, 255, 0).astype(np.uint8)  # noise
    after = 255 - before

    patch = Patch.capture(5, 1, before, after)

    assert patch.nbytes <= 2 * -(-before.size // 8) + OVERHEAD
    for got, want in ((patch.before(), before), (patch.after(), after)):
        assert got.dtype == np.uint8 and got.shape == shape
        assert np.array_equal(got, want)


@pytest.mark.parametrize("third", [1, 2, 128, 254])
def test_a_region_of_0_and_255_with_a_third_value_round_trips_exactly(third):
    """The exactness check: one pixel of another value keeps the region byte for byte."""
    rng = np.random.default_rng(third)
    before = np.where(rng.random((300, 211)) < 0.05, 255, 0).astype(np.uint8)
    before[123, 45] = third
    after = before.copy()
    after[7:9, :] = 255

    patch = Patch.capture(0, 0, before, after, compress_min=1024)

    assert np.array_equal(patch.before(), before) and np.array_equal(patch.after(), after)
    assert patch.before().dtype == np.uint8


def test_an_all_zero_region_stores_nothing_and_gives_zeros_back():
    before = np.zeros((480, 640), np.uint8)
    before[100:300:3, 50:600] = 255
    cleared = np.zeros_like(before)

    patch = Patch.capture(0, 0, before, cleared, compress_min=1024)
    alone = Patch.capture(0, 0, cleared, cleared, compress_min=1024)

    assert alone.nbytes == 128  # only the entry overhead: no stored side
    assert patch.nbytes < 128 + 64 + -(-before.size // 8)
    got = patch.after()
    assert got.dtype == np.uint8 and got.shape == (480, 640) and not got.any()
    assert np.array_equal(patch.before(), before)


def test_undo_and_redo_of_a_whole_plane_skeleton_edit_restore_every_pixel():
    rng = np.random.default_rng(6)
    plane = np.where(rng.random((1536, 2048)) < 0.02, 255, 0).astype(np.uint8)
    original = plane.copy()
    history = PatchHistory()
    history.reset(plane)
    stroke = np.where(rng.random((1536, 2048)) < 0.02, 255, 0).astype(np.uint8)
    history.commit((0, 0, 2048, 1536), stroke)
    edited = history.plane.copy()
    history.commit((0, 0, 2048, 1536), np.zeros((1536, 2048), np.uint8))  # a Clear

    assert history.undo() is not None and np.array_equal(history.plane, edited)
    assert history.undo() is not None and np.array_equal(history.plane, original)
    assert history.redo() is not None and np.array_equal(history.plane, edited)
    assert history.redo() is not None and not history.plane.any()
    assert history.plane.dtype == np.uint8


@pytest.mark.parametrize(
    ("rect", "shape"),
    [((5, 5, 10, 10), (10, 10)), ((-1, 0, 4, 4), (4, 4)), ((0, 0, 4, 4), (3, 4))],
    ids=["past-the-edge", "negative-origin", "region-of-another-shape"],
)
def test_a_commit_that_does_not_fit_the_plane_is_refused_and_changes_nothing(rect, shape):
    """(The refusal itself is also in test_patch_history_undo.py.) Nothing of the region is
    written and no step is recorded: a mis-sized edit cannot corrupt the plane."""
    history = PatchHistory()
    plane = np.zeros((10, 10), np.uint8)
    history.reset(plane)
    history.commit((0, 0, 2, 2), np.ones((2, 2), np.uint8))
    before = plane.copy()

    with pytest.raises(ValueError):
        history.commit(rect, np.ones(shape, np.uint8))

    assert np.array_equal(history.plane, before) and history.plane is plane
    assert len(history.undo_stack) == 1 and not history.can_redo


def test_a_commit_without_a_plane_is_refused():
    history = PatchHistory()
    with pytest.raises(ValueError, match="no plane"):
        history.commit((0, 0, 1, 1), np.ones((1, 1), np.uint8))
    assert len(history.undo_stack) == 0

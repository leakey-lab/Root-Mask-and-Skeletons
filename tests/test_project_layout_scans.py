"""``app.project_layout`` scans and single-stem lookups: one file per stem without case (O7).

Promises: the ranked scan lists one file per ``stem_key``, the lowest ``(extension rank, file
name)`` winning, PNG and JPEG only (O6), and is the scan it was before R6 (the pre-R6 scan is
kept below as the reference); ``scan_image_folder`` names the hidden and unsupported files and
maps names only when a stem was shared; ``key_to_stems`` renames a scan's keys to the
Library's stems, equal to the plain remap of f392a85 (the reference) and building no table when
every key is a stem; single-stem lookups find a case variant, the exact name first.
"""

from __future__ import annotations

import os
import random

import numpy as np
import pytest
from PIL import Image

from app import project_layout
from app.config import IMAGE_EXTENSIONS
from app.gui import image_manager as im
from app.inference.mask_origin import scan_existing_masks
from app.io_utils import path_key
from app.naming import image_stem, stem_key
from app.project_layout import (
    ImageScan,
    ProjectLayout,
    few_keys,
    key_to_stems,
)

pytestmark = pytest.mark.usefixtures("no_temp_sweep")

# ------------------------------------------------------------------ the ranked scan (R6)


def _pre_r6_scan(directory, key_of):
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                key = key_of(entry.name)
                if key is not None and entry.is_file():
                    yield key, entry.path
    except FileNotFoundError, NotADirectoryError:
        return


def _pre_r6_scan_ranked(directory, extensions):
    """``project_layout._scan_ranked`` before R6, verbatim."""
    rank = {ext.lower(): i for i, ext in enumerate(extensions)}

    def ranked_stem(name):
        stem, ext = os.path.splitext(name)
        r = rank.get(ext.lower())
        return None if r is None else (stem, r)

    best = {}
    for (stem, r), path in _pre_r6_scan(directory, ranked_stem):
        if stem not in best or r < best[stem][0]:
            best[stem] = (r, path)
    return {stem: path for stem, (_, path) in best.items()}


NAMES = [
    ".png", "..png", "...jpg", ".a.png", "..a.jpg", "a.png", "a.JPG", "a.jpeg", "b.PnG",
    "c.jpg.png", "d.", "e", ".f", "g.tif", "h.png.tmp", ".x.png.1a2b3c4d.tmp", "i.jpeg",
    "i.jpg", "j .png", "k..png", "Données_根.png", "24EFCAM1_T001_L045_2024.07.18_111618_001_SMM.jpg",
    "24EFCAM1_T001_L045_2024.07.18_111618_001_SMM.PNG", "root_areas.csv", "_sprouts_run.json",
]  # fmt: skip


@pytest.fixture
def folder(tmp_path):
    for name in NAMES:
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "a_folder.png").mkdir()  # a folder is never a file of the project
    return str(tmp_path)


@pytest.mark.parametrize(
    "extensions",
    [IMAGE_EXTENSIONS, (".png",), (".jpg", ".png"), (".png", ".jpg", ".jpeg", ".tif", ".tiff")],
)
def test_the_ranked_scan_is_unchanged(folder, extensions):
    assert project_layout._scan_ranked(folder, extensions) == _pre_r6_scan_ranked(
        folder, extensions
    )


def test_the_layout_scans_are_unchanged(folder):
    layout = ProjectLayout(folder)
    assert layout.scan_images() == _pre_r6_scan_ranked(layout.root, IMAGE_EXTENSIONS)
    assert layout.scan_masks(folder) == _pre_r6_scan_ranked(folder, (".png", ".jpg", ".jpeg"))


def test_a_missing_folder_or_a_file_scans_empty(tmp_path):
    assert project_layout._scan_ranked(str(tmp_path / "missing"), IMAGE_EXTENSIONS) == {}
    (tmp_path / "file.png").write_bytes(b"")
    assert project_layout._scan_ranked(str(tmp_path / "file.png"), IMAGE_EXTENSIONS) == {}


# ---------------------------------------- key_to_stems, scan_image_folder (S16-01, S06-08)


def _key_to_stems_before(found, stems):
    """``key_to_stems`` of f392a85, before its short-cut, verbatim: the reference the
    optimised remap must equal (the remap tests below and the stem-string tests at the end)."""
    by_key = {stem_key(stem): stem for stem in stems}
    return {by_key.get(stem_key(key), key): value for key, value in found.items()}


def _scan_image_folder_today(layout: ProjectLayout) -> ImageScan:
    """``ProjectLayout.scan_image_folder`` before the short-cut (f392a85), verbatim."""
    skips = project_layout._Skips()
    images = project_layout._scan_ranked(layout.root, IMAGE_EXTENSIONS, skips)
    winners = {stem_key(stem): os.path.basename(path) for stem, path in images.items()}
    return ImageScan(
        images=images,
        hidden=tuple(sorted((name, winners[key]) for key, name in skips.losers)),
        unsupported=tuple(sorted(skips.unsupported)),
    )


def test_keys_that_are_stems_are_copied_in_their_order():
    found = {"c": "x", "A": "y"}

    out = key_to_stems(found, {"B": 1, "A": 2, "c": 3})

    assert list(out.items()) == [("c", "x"), ("A", "y")]
    assert out is not found  # a new dict: the Library edits its masks in place


def test_a_case_variant_key_among_exact_ones_is_mapped():
    found = {"b": 1, "a": 2, "orphan": 3}

    assert list(key_to_stems(found, ["A", "b"]).items()) == [("b", 1), ("A", 2), ("orphan", 3)]
    assert list(key_to_stems(found, {"A", "b"}).items()) == [("b", 1), ("A", 2), ("orphan", 3)]


def test_an_empty_scan_gives_a_new_empty_dict():
    first = key_to_stems({}, {"a": 1})

    assert first == {} and key_to_stems({}, {"a": 1}) is not first


def test_keys_named_as_their_images_build_no_stem_key_table(monkeypatch):
    stems = {f"T{i:03d}_L{i % 7}": i for i in range(1000)}
    found = {stem: f"mask/{stem}.png" for stem in list(stems)[::7]}
    calls: list[str] = []
    real = project_layout.stem_key
    monkeypatch.setattr(project_layout, "stem_key", lambda s: calls.append(s) or real(s))

    assert key_to_stems(found, stems) == found
    assert key_to_stems(dict.fromkeys(found), set(stems)) == dict.fromkeys(found)
    assert key_to_stems({}, stems) == {}

    assert calls == []


_ALPHABET = ("a", "A", "b", "B", "ß", "ẞ", "Σ", "σ", "ς", "k", "K", "K", ".png", "_", "1")


_SHAPES = (dict.fromkeys, set, frozenset, list, tuple, iter)


@pytest.mark.parametrize("seed", range(4))
def test_the_remap_equals_todays_for_any_stems_shape(seed):
    """Random stems (one per stem key, as a scan lists them) and keys: exact stems, other
    cases, orphans; the result, its order too, is today's."""
    rng = random.Random(seed)
    for _ in range(1500):
        pool = ["".join(rng.choices(_ALPHABET, k=rng.randint(1, 3))) for _ in range(8)]
        by_key: dict[str, str] = {}
        for name in pool[: rng.randint(0, 8)]:
            by_key.setdefault(stem_key(name), name)
        stems = list(by_key.values())
        found = {}
        for _ in range(rng.randint(0, 6)):
            key = rng.choice([*stems, *pool, "orphan"])
            found[rng.choice((key, key.swapcase(), key.upper(), key.lower()))] = rng.random()
        shape = rng.choice(_SHAPES)

        got = key_to_stems(found, shape(stems))

        assert list(got.items()) == list(_key_to_stems_before(found, stems).items())


def test_a_folder_without_shared_stems_maps_no_file_name(tmp_path, monkeypatch):
    for name in ("a.png", "b.jpg", "c.JPEG", "t.tif"):
        (tmp_path / name).write_bytes(b"")
    layout = ProjectLayout(str(tmp_path))
    calls: list[str] = []
    real = os.path.basename

    with monkeypatch.context() as patched:
        patched.setattr(os.path, "basename", lambda path: calls.append(path) or real(path))
        scan = layout.scan_image_folder()

    assert sorted(scan.images) == ["a", "b", "c"]
    assert (scan.hidden, scan.unsupported) == ((), ("t.tif",))
    assert calls == []


def test_the_hidden_files_name_the_file_listed_instead(tmp_path):
    for name in ("a.png", "a.jpg", "a.jpeg", "b.JPG", "b.jpeg", "c.png", "d.TIF", "e.bmp"):
        (tmp_path / name).write_bytes(b"")
    layout = ProjectLayout(str(tmp_path))

    scan = layout.scan_image_folder()

    assert scan.hidden == (("a.jpeg", "a.png"), ("a.jpg", "a.png"), ("b.jpeg", "b.JPG"))
    assert scan == _scan_image_folder_today(layout)
    assert scan.unsupported == ("d.TIF", "e.bmp")


# ------------------------------------------------------------------ case variants on disk (O7)


def _images(folder, spec):
    """``spec``: ``{file name: grey level}``; every image 64x48 RGB."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, level in spec.items():
        Image.new("RGB", (64, 48), (level, level, level)).save(folder / name)
    return str(folder)


def _case_sensitive(folder) -> bool:
    (folder / "Probe").write_bytes(b"")
    try:
        return not (folder / "probe").exists()
    finally:
        (folder / "Probe").unlink()


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("L", (8, 6)).save(path, "PNG")
    return path


def test_equal_extensions_pick_the_name_that_sorts_first(tmp_path):
    if not _case_sensitive(tmp_path):
        pytest.skip("needs a case-sensitive file system (runs on the ubuntu CI)")
    src = _images(tmp_path / "imgs", {"a.png": 0, "A.png": 255})
    scan = ProjectLayout(src).scan_image_folder()
    assert scan.images == {"A": os.path.join(src, "A.png")}
    assert scan.hidden == (("a.png", "A.png"),)


def test_a_trailing_separator_still_finds_a_case_insensitive_folder(tmp_path):
    if _case_sensitive(tmp_path):
        pytest.skip("needs a case-insensitive file system (Windows, macOS)")
    from app import project_layout

    folder = tmp_path / "Mask"
    folder.mkdir()
    assert project_layout._ignores_case(str(folder) + os.sep)


def test_single_stem_lookups_find_a_file_whose_name_differs_in_case(tmp_path):
    if not _case_sensitive(tmp_path):
        pytest.skip("needs a case-sensitive file system (runs on the ubuntu CI)")
    from app.gui.image_manager import ImageManager

    layout = ProjectLayout(_images(tmp_path / "p", {"A.png": 90}))
    mask = _touch(os.path.join(layout.mask_dir, "a.PNG"))
    skeleton = _touch(os.path.join(layout.skeletons_dir, "a_fake.png"))
    gt_skeleton = _touch(os.path.join(layout.gt_skeletons_dir, "a.png"))

    assert layout.find_mask("A") == mask
    assert layout.find_skeleton("A") == skeleton
    assert layout.find_gt_skeleton("A") == gt_skeleton
    assert layout.find_mask("b") is None and layout.find_skeleton("b") is None

    manager = ImageManager()
    manager.original_folder = layout.root
    manager.on_loading_finished(layout.scan_images(), {}, {}, False)
    assert manager.has_gt_skeleton("A") and manager.gt_skeleton_path_for("A") == gt_skeleton
    manager.processed_base_path = layout.skeletons_dir
    manager.current_view_mode = "Overlay"
    assert manager.get_fake_image_path("A") == skeleton


def test_the_exact_case_file_wins_a_single_stem_lookup(tmp_path):
    if not _case_sensitive(tmp_path):
        pytest.skip("needs a case-sensitive file system (runs on the ubuntu CI)")
    layout = ProjectLayout(_images(tmp_path / "p", {"A.png": 90}))
    _touch(os.path.join(layout.mask_dir, "a.png"))
    exact = _touch(os.path.join(layout.mask_dir, "A.jpg"))
    _touch(os.path.join(layout.skeletons_dir, "a_fake.png"))
    exact_skeleton = _touch(os.path.join(layout.skeletons_dir, "A_fake.png"))

    assert layout.find_mask("A") == exact
    assert layout.find_skeleton("A") == exact_skeleton
    # Without an exact-case file, the case variants rank as the scan does.
    os.remove(exact)
    _touch(os.path.join(layout.mask_dir, "a.jpg"))
    assert layout.find_mask("A") == os.path.join(layout.mask_dir, "a.png")


def test_a_miss_on_a_case_insensitive_disk_lists_no_folder(tmp_path, monkeypatch):
    if _case_sensitive(tmp_path):
        pytest.skip("needs a case-insensitive file system (Windows, macOS)")
    from app import project_layout

    layout = ProjectLayout(_images(tmp_path / "p", {"A.png": 90}))
    _touch(os.path.join(layout.mask_dir, "other.png"))
    os.makedirs(layout.skeletons_dir)

    def no_listing(*_args):
        raise AssertionError("a lookup listed a folder")

    monkeypatch.setattr(project_layout.os, "scandir", no_listing)
    assert layout.find_mask("A") is None and layout.find_skeleton("A") is None
    _touch(os.path.join(layout.mask_dir, "a.png"))
    assert path_key(layout.find_mask("A")) == path_key(os.path.join(layout.mask_dir, "a.png"))


# ---------------------------------------- the Library's own stem strings (received from A5)


STEMS = [
    f"24EFCAM1_T{tube:03d}_L{depth:03d}_2024.06.04_100001_001_SMM"
    for tube in (1, 2)
    for depth in range(1, 13)
]


def _ids(strings) -> set[int]:
    return {id(s) for s in strings}


def _fresh(text: str) -> str:
    """An equal string that is another object, as a second scan makes it."""
    return "".join(list(text))


@pytest.mark.parametrize("shape", [dict.fromkeys, set])
def test_key_to_stems_gives_a_large_scan_the_strings_of_stems_in_its_own_order(shape):
    stems = shape(map(_fresh, STEMS))
    found = {_fresh(stem): f"mask/{stem}.png" for stem in reversed(STEMS[2:])}

    out = key_to_stems(found, stems)

    assert list(out.items()) == list(found.items())
    assert _ids(out) <= _ids(stems)


@pytest.mark.parametrize("n, few", [(2, True), (3, False)])  # 24 images: an eighth is 3
def test_both_remaps_keep_their_own_strings_below_the_same_size(n, few):
    images = dict.fromkeys(map(_fresh, STEMS))
    found = [_fresh(stem) for stem in STEMS[:n]]

    remapped = key_to_stems(dict.fromkeys(found), images)
    stems = im._stems_of(set(found), images)

    assert few_keys(found, images) is few
    for out in (remapped, stems):
        assert set(out) == set(found)
        assert _ids(out) <= (_ids(found) if few else _ids(images))


class _MaskScan:
    """The one layout method :func:`scan_existing_masks` calls."""

    def __init__(self, masks: dict) -> None:
        self._masks = masks

    def scan_masks(self) -> dict:
        return dict(self._masks)


_STEM_ALPHABET = ("a", "A", "b", "ß", "ẞ", "Σ", "σ", "ς", "k", "K", "K", "_", "1")


@pytest.mark.parametrize("seed", range(3))
def test_the_answers_are_those_of_before_for_few_and_many_outputs(seed):
    """Random Libraries (one stem per stem key, as a scan lists them) of up to 40 stems and
    scans of a few or of most of them: exact stems, other cases, orphans."""
    rng = random.Random(seed)
    for _ in range(600):
        pool = ["".join(rng.choices(_STEM_ALPHABET, k=rng.randint(1, 4))) for _ in range(40)]
        by_key: dict[str, str] = {}
        for name in pool[: rng.randint(0, 40)]:
            by_key.setdefault(stem_key(name), name)
        stems = list(by_key.values())
        exact = rng.random() < 0.5  # every output named as its image (nearly always)
        found = {}
        for _ in range(rng.choice((rng.randint(0, 3), rng.randint(0, len(stems) + 3)))):
            if stems and (exact or rng.random() < 0.7):
                key = rng.choice(stems)
            else:
                key = rng.choice([*pool, "orphan"])
            if not exact:
                key = rng.choice((key, key.swapcase(), key.upper(), key.lower()))
            found[key] = rng.random()
        images = dict.fromkeys(stems)
        before = _key_to_stems_before(found, stems)

        assert list(key_to_stems(found, images).items()) == list(before.items())
        for kind in (set, frozenset):
            assert im._stems_of(kind(found), images) == set(
                _key_to_stems_before(dict.fromkeys(kind(found)), stems)
            )
        wanted = set(stems)
        assert scan_existing_masks(_MaskScan(found), stems).stems == frozenset(
            s for s in _key_to_stems_before(found, wanted) if s in wanted
        )


# ------------------------------------------------------------------ added: finders and the scan agree


def _case_sensitive_disk(monkeypatch) -> None:
    """Make this disk behave like Linux's for the layout: case is kept (``_ignores_case``
    False) and an exact-name test matches the exact name only."""
    real_isfile = os.path.isfile

    def exact_isfile(path) -> bool:
        folder, name = os.path.split(os.fspath(path))
        return real_isfile(path) and name in os.listdir(folder or ".")

    monkeypatch.setattr(project_layout, "_ignores_case", lambda _directory: False)
    monkeypatch.setattr(project_layout, "_LISTED_KEYS", {})
    monkeypatch.setattr(project_layout.os.path, "isfile", exact_isfile)


def test_the_gt_finders_find_a_gt_file_whose_extension_differs_in_case(tmp_path, monkeypatch):
    """A GT file renamed by hand, or copied from another computer, as ``a.PNG`` for the image
    ``A.png``: on a case-sensitive disk (Linux) the scans list it as that image's GT, so the
    finders must find it too, or Correct opens the prediction and its save writes a second GT
    file beside the user's. The stand-in disk below is how this runs on Windows and macOS."""
    _case_sensitive_disk(monkeypatch)
    layout = ProjectLayout(_images(tmp_path / "p", {"A.png": 90}))
    mask = _touch(os.path.join(layout.gt_mask_dir, "a.PNG"))
    skeleton = _touch(os.path.join(layout.gt_skeletons_dir, "a.PNG"))
    labels = _touch(os.path.join(layout.gt_dir, "a.PNG"))
    image = os.path.join(layout.root, "A.png")

    assert layout.scan_gt_masks() == {"a": mask}
    assert layout.find_gt_mask(image) == mask
    assert layout.find_gt_skeleton(image) == skeleton
    assert layout.find_gt_skeleton_for_stem("A") == skeleton
    assert layout.find_gt_labels(image) == labels


def test_image_stem_names_every_listed_file_as_the_scan_does(tmp_path):
    """Seeded random names of dots, extensions in any case, spaces and double extensions:
    for every file the image scan lists, ``image_stem`` of its path is the scan's stem, so
    an output named from the photo's path is found under the Library's stem."""
    rng = np.random.default_rng(7)
    parts = ["a", "A", ".", "..", "png", "PNG", "jpg", "JPEG", "jpeg", "tif", " ", "_", "x.png"]
    made = 0
    for _ in range(1500):
        name = "".join(rng.choice(parts, size=rng.integers(1, 6)))
        try:
            (tmp_path / name).write_bytes(b"")
            made += 1
        except OSError:
            continue  # a name this file system refuses (Windows: trailing dot or space)
    scanned = ProjectLayout(str(tmp_path)).scan_images()

    assert made > 500 and len(scanned) > 50
    for stem, path in scanned.items():
        assert image_stem(path) == stem, path
        assert ProjectLayout.stem(os.path.basename(path)) == stem, path


class _Listing:
    """A folder listing in a chosen order, as ``os.scandir`` hands it out."""

    def __init__(self, entries) -> None:
        self.entries = entries

    def __enter__(self):
        return iter(self.entries)

    def __exit__(self, *exc) -> bool:
        return False


@pytest.mark.parametrize("winner_first", [True, False])
def test_the_hidden_files_are_named_whichever_order_the_folder_lists(
    tmp_path, monkeypatch, winner_first
):
    """NTFS lists names alphabetically, so there the winning ``a.png`` always comes last;
    ext4 and APFS list in any order. A file that loses to one listed before it is named too."""
    for name in ("a.png", "a.jpg", "b.jpeg", "b.JPG"):
        (tmp_path / name).write_bytes(b"")
    real_scandir = os.scandir

    def ordered(path):
        with real_scandir(path) as entries:
            listed = sorted(entries, key=lambda e: e.name.lower(), reverse=winner_first)
        return _Listing(listed)

    monkeypatch.setattr(project_layout.os, "scandir", ordered)
    scan = ProjectLayout(str(tmp_path)).scan_image_folder()

    assert sorted(scan.images) == ["a", "b"]
    assert scan.hidden == (("a.jpg", "a.png"), ("b.jpeg", "b.JPG"))

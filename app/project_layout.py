"""
Project folder layout: the one place that turns an image folder into paths.

A SPROUTS project is a folder of images. Every stage writes next to them::

    <root>/<stem>.jpg                               source image (IMAGE_EXTENSIONS)
    <root>/mask/<stem>.png                          mask, generated or traced
    <root>/mask/root_areas.csv                      Measure (area)
    <root>/skeletons/<stem>_fake.png                generated skeleton
    <root>/skeletons/root_lengths.csv               Measure (length)
    <root>/gt/<stem>.png                            GT class labels
    <root>/gt/mask/<stem>.png                       GT mask (a saved edit)
    <root>/gt/skeletons/<stem>.png                  GT skeleton (a saved edit)

The older layout (``output/mask``, ``[output/]skeletonizer/test_latest/images``) is not
read since 2026-10-31 (#38): masks, skeletons and CSVs there are not found.

Qt-free and side-effect free: nothing is created on disk, and each ``scan_*``
method is a single ``os.scandir`` of one folder. The case-variant lookups remember, in
memory only, which stem keys a case-sensitive folder held while it is unchanged.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sized
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field

from app.config import (
    GT_CLASSES_FILENAME,
    GT_DIRNAME,
    IMAGE_EXTENSIONS,
    MASK_DIRNAME,
    ROOT_AREAS_CSV,
    ROOT_LENGTHS_CSV,
    SKELETONS_DIRNAME,
)
from app.naming import LEGACY_IMAGE_EXTENSIONS, image_stem, stem_key

__all__ = [
    "MASK_EXTENSIONS",
    "SKELETON_SUFFIX",
    "GtStems",
    "ImageScan",
    "ProjectLayout",
    "few_keys",
    "find_by_stem",
    "group_skeletons",
    "key_to_stems",
    "project_layout_of",
    "skeleton_stem",
    "skip_gt_variants",
]

#: File-name suffix of a generated skeleton (skeleton_inference output).
SKELETON_SUFFIX = "_fake.png"
#: Mask file types the app reads, in lookup priority: the image formats, as masks use
#: the same scan (one mask per stem without case). Masks are written as .png.
MASK_EXTENSIONS = IMAGE_EXTENSIONS

# Mask lookup order of the mask editor (find_mask_path); the case variants
# matter on case-sensitive file systems.
_MASK_LOOKUP = tuple(v for ext in MASK_EXTENSIONS for v in (ext, ext.upper()))

# Before R8 the GT store and registry wrote "<path>.tmp" next to gt/<stem>.png and
# gt/classes.json; a killed save left that file in gt/ for good.
_LEGACY_GT_TEMP_SUFFIX = ".tmp"


def _scan[K](directory: str, key_of: Callable[[str], K | None]) -> Iterator[tuple[K, str]]:
    """Yield ``(key, path)`` for files in ``directory`` whose name ``key_of`` accepts.

    One ``os.scandir``; a missing folder yields nothing.
    """
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                key = key_of(entry.name)
                if key is not None and entry.is_file():
                    yield key, entry.path
    except FileNotFoundError, NotADirectoryError:
        return


def skeleton_stem(name: str) -> str | None:
    """The stem of a generated skeleton's file name, None for any other name: Measure
    Length's name rule (the suffix in any case of its ASCII letters, ``B_fake.PNG`` too, #3;
    hidden names count; ``_fake.png`` alone has the stem "")."""
    if name.endswith(SKELETON_SUFFIX):
        return name[: -len(SKELETON_SUFFIX)]
    tail = name[-len(SKELETON_SUFFIX) :]
    # isascii: str.lower() maps U+212A KELVIN SIGN to "k", which no disk folds to a "k"
    if tail.isascii() and tail.lower() == SKELETON_SUFFIX:
        return name[: -len(SKELETON_SUFFIX)]
    return None


def _skeleton_rank(path: str) -> tuple[bool, str]:
    """Order of the files of one skeleton stem: the exact suffix first, then the name that
    sorts first."""
    name = os.path.basename(path)
    return not name.endswith(SKELETON_SUFFIX), name


def group_skeletons(
    files: Iterable[tuple[str, str]],
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """``(stem -> path, shadowed)`` of skeleton ``files`` (``(stem, path)`` pairs, stems by
    :func:`skeleton_stem`).

    Of several files with one stem (``a_fake.png`` and ``a_fake.PNG`` on a case-sensitive
    disk) the one with the exact suffix is listed, else the name that sorts first (#3). A
    suffix in another case is listed only for a stem key without an exact-suffix file
    (``A_fake.png`` keeps image ``a``'s row from ``a_FAKE.png``), so it adds a row and never
    changes one (beside a GT skeleton, :func:`skip_gt_variants` leaves it out first). The
    others are ``shadowed``, for :func:`app.inference.input_split.split_inputs`, so none of
    them vanishes unrecorded.
    """
    listed: dict[str, str] = {}
    shadowed: list[tuple[str, str]] = []
    for stem, path in files:
        held = listed.get(stem)
        if held is None:
            listed[stem] = path
        elif _skeleton_rank(path) < _skeleton_rank(held):
            listed[stem] = path
            shadowed.append((stem, held))
        else:
            shadowed.append((stem, path))
    variants = [stem for stem, path in listed.items() if not path.endswith(SKELETON_SUFFIX)]
    if variants:  # none in the usual folder: no stem key is computed then
        exact = {stem_key(stem) for stem, path in listed.items() if path.endswith(SKELETON_SUFFIX)}
        for stem in variants:
            if stem_key(stem) in exact:
                shadowed.append((stem, listed.pop(stem)))
    return listed, shadowed


def skip_gt_variants(
    files: Iterable[tuple[str, str]], gt_stems: Iterable[str]
) -> Iterator[tuple[str, str]]:
    """The skeleton ``files`` (``(stem, path)`` pairs) Measure Length reads beside the GT
    skeletons of ``gt_stems`` (the stems of the GT files that have an image).

    A file whose suffix is in another case (``B_fake.PNG``) is left out for an image with a
    GT skeleton file, used or not, and is not even a duplicate: Measure did not see such a
    file before #3, so that image is measured as it was (its ``Image`` name, ``Source`` and
    value, or no row). #3 adds rows only, for images without a GT skeleton (owner, #3).
    Applied to the files before :func:`group_skeletons`, or to what it lists and to what it
    shadows, it leaves the same files.
    """
    gt_keys: set[str] | None = None
    for stem, path in files:
        if not path.endswith(SKELETON_SUFFIX):
            if gt_keys is None:  # only a variant needs them: none in the usual folder
                gt_keys = {stem_key(gt_stem) for gt_stem in gt_stems}
            if stem_key(stem) in gt_keys:
                continue
        yield stem, path


@dataclass
class _Skips:
    """What a ranked scan did not list: ``(key, file name)`` of each file that lost its
    stem to another, and the names with a :data:`LEGACY_IMAGE_EXTENSIONS` suffix."""

    losers: list[tuple[str, str]] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)


def _scan_ranked(
    directory: str, extensions: tuple[str, ...], skips: _Skips | None = None
) -> dict[str, str]:
    """``stem -> path`` for files with ``extensions``, one file per stem without case.

    Stems that differ only in case are one stem (:func:`app.naming.stem_key`, O7): the
    lowest ``(extension rank, file name)`` wins (earlier ``extensions`` first, then the
    name that sorts first) and the result keeps the winner's real stem. With ``skips``
    the same listing also collects the losers and the unsupported image files.

    One ``os.scandir`` in a single loop (R6: 85 -> 59 ms for 46k files). The stem
    and extension follow ``os.path.splitext``: the last dot splits, unless only
    dots come before it (``.png`` and ``..png`` have no extension).
    """
    rank = {ext.lower(): i for i, ext in enumerate(extensions)}
    best: dict[str, tuple[int, str, str, str]] = {}  # key -> (rank, name, stem, path)
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                name = entry.name
                dot = name.rfind(".")
                if dot <= 0:
                    continue
                r = rank.get(name[dot:].lower())
                if r is None:
                    if (
                        skips is not None
                        and name[dot:].lower() in LEGACY_IMAGE_EXTENSIONS
                        and name[:dot].lstrip(".")
                        and entry.is_file()
                    ):
                        skips.unsupported.append(name)
                    continue
                if not name[:dot].lstrip(".") or not entry.is_file():
                    continue
                stem = name[:dot]
                key = stem_key(stem)
                held = best.get(key)
                if held is None or (r, name) < held[:2]:
                    best[key] = (r, name, stem, entry.path)
                    if held is not None and skips is not None:
                        skips.losers.append((key, held[1]))
                elif skips is not None:
                    skips.losers.append((key, name))
    except FileNotFoundError, NotADirectoryError:
        return {}
    return {stem: path for _, _, stem, path in best.values()}


# Folders known to ignore case (one probe each): a lookup that misses there lists nothing.
_CASE_FOLDING_DIRS: dict[str, bool] = {}


def _ignores_case(directory: str) -> bool:
    """True when ``directory``'s file system ignores case (Windows, macOS by default).

    Probed once per folder: its own name with the case of every letter swapped names the
    same folder. A folder whose name has no letters counts as case-sensitive.
    """
    directory = os.path.normpath(directory)  # a trailing separator would hide its name
    known = _CASE_FOLDING_DIRS.get(directory)
    if known is not None:
        return known
    head, name = os.path.split(directory)
    swapped = os.path.join(head, name.swapcase())
    try:
        here = os.stat(directory)
    except OSError:
        return False  # missing: nothing to find, and nothing worth remembering
    try:
        folds = swapped != directory and os.path.samestat(here, os.stat(swapped))
    except OSError:
        folds = False
    _CASE_FOLDING_DIRS[directory] = folds
    return folds


# Case-sensitive folders a lookup listed: ``(folder, endings, fold_endings)`` -> (the folder's
# ``(st_dev, st_ino, st_mtime_ns)`` before that listing, the stem key of every name with one
# of the endings). Only atomic get / set / clear: the GUI thread and the workers share it.
_LISTED_KEYS: dict[tuple[str, tuple[str, ...], bool], tuple[tuple[int, int, int], set[str]]] = {}
_LISTED_MAX = 8
# A listing is remembered only when the folder had not changed for this long (git's racy
# timestamp rule): a change in the same timestamp tick as the listing is never missed.
_SETTLED_NS = 2_000_000_000


def _find_without_case(
    directory: str,
    stem: str,
    endings: tuple[str, ...],
    *,
    fold_endings: bool,
    measure_order: bool = False,
) -> str | None:
    """A file of ``directory`` named ``<stem'><ending>`` where ``stem'`` has ``stem``'s
    :func:`~app.naming.stem_key`; the lowest ``(ending rank, file name)`` wins, as in the
    ranked scan. ``fold_endings`` matches the endings without case too (extensions; their
    ASCII letters only, as :func:`skeleton_stem`). ``measure_order`` ranks as Measure Length
    keeps a skeleton (:func:`group_skeletons`, then ``split_inputs``): an ending in its exact
    case first, then ``stem`` as given, then the name that sorts first.

    One ``os.scandir``, skipped where the file system ignores case: there the exact-name
    lookup has already found any case variant. Skipped too while the folder is unchanged
    since a listing that had no name with that key: one ``os.stat`` then (P2-01).
    """
    if _ignores_case(directory):
        return None
    key = stem_key(stem)
    memo = (os.path.normpath(directory), endings, fold_endings)
    started = time.time_ns()
    try:
        st = os.stat(directory)  # before the listing: a change during it shows next time
    except FileNotFoundError, NotADirectoryError:
        return None
    stamp = (st.st_dev, st.st_ino, st.st_mtime_ns)
    listed = _LISTED_KEYS.get(memo)
    keys: set[str] | None = None
    if listed is not None and listed[0] == stamp:
        if key not in listed[1]:
            return None
        # a name has the key: list for it (file test, ranking), the memo stays as it is
    elif st.st_mtime_ns < started - _SETTLED_NS:
        keys = set()
    rank = {(e.lower() if fold_endings else e): i for i, e in enumerate(endings)}
    best: tuple[tuple[int, bool, bool, str], str] | None = None
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                name = entry.name
                for ending, r in rank.items():
                    tail = name[-len(ending) :]
                    # isascii: str.lower() maps U+212A KELVIN SIGN to "k"
                    if (tail.lower() if fold_endings and tail.isascii() else tail) != ending:
                        continue
                    name_stem = name[: -len(ending)]
                    name_key = stem_key(name_stem)
                    if keys is not None:
                        keys.add(name_key)  # every name, file or not: a key it lacks has none
                    if name_key != key or not entry.is_file():
                        continue
                    if measure_order:
                        order = (r, tail not in endings, name_stem != stem, name)
                    else:
                        order = (r, False, False, name)
                    if best is None or order < best[0]:
                        best = (order, entry.path)
    except FileNotFoundError, NotADirectoryError:
        return None
    if keys is not None:
        if len(_LISTED_KEYS) >= _LISTED_MAX:
            _LISTED_KEYS.clear()
        _LISTED_KEYS[memo] = (stamp, keys)
    return best[1] if best is not None else None


def find_by_stem(directory: str, stem: str, suffix: str) -> str | None:
    """The file ``<directory>/<stem><suffix>``, else one whose stem, or ``suffix``, differs
    only in case (O7, #3), picked as Measure Length picks it (the exact suffix, then the
    stem as given, then the name that sorts first); None when there is neither."""
    path = os.path.join(directory, stem + suffix)
    if os.path.isfile(path):
        return path
    return _find_without_case(directory, stem, (suffix,), fold_endings=True, measure_order=True)


def few_keys(found: Sized, stems: Sized) -> bool:
    """True when ``found`` holds under an eighth as many names as ``stems``: a remap onto the
    Library's stems then keeps the strings of ``found`` (copies of a few names cost next to
    nothing); a larger one hands back the strings of ``stems``, so its caller holds no second
    copy of them (4 MiB for the masks of 46k images)."""
    return 8 * len(found) < len(stems)


def key_to_stems[V](found: Mapping[str, V], stems: Iterable[str]) -> dict[str, V]:
    """``found`` with each key replaced by the stem of ``stems`` with the same
    :func:`~app.naming.stem_key` (``{"a": m}`` with stems ``["A"]`` -> ``{"A": m}``).

    Keys without such a stem are kept as they are. A mask or skeleton whose name differs
    from its image's only in case is then found under the image's stem (O7). ``stems`` holds
    one stem per stem key, as a scan lists them: a key that is one of them is its own stem, so
    when every key is (outputs named as their images), no stem-key table of ``stems`` is built.
    The result's keys are then still the strings of ``stems`` (in ``found``'s order), unless
    they are a few (:func:`few_keys`), which keep their own strings.
    """
    if not found:
        return {}
    names = stems.keys() if isinstance(stems, Mapping) else stems
    if isinstance(names, AbstractSet) and names >= found.keys():
        if few_keys(found, names):
            return dict(found)  # a few keys: their copies cost next to nothing
        canon = dict(zip(names, names, strict=True))  # each stem to its own string
        return dict(zip(map(canon.__getitem__, found), found.values(), strict=True))
    by_key = {stem_key(stem): stem for stem in stems}
    return {by_key.get(stem_key(key), key): value for key, value in found.items()}


@dataclass(frozen=True)
class ImageScan:
    """One listing of an image folder: what the Library shows, and what it does not.

    ``images`` is ``stem -> path`` (as :meth:`ProjectLayout.scan_images`); ``hidden``
    holds ``(file not listed, file listed instead)`` for each file whose stem another
    file took (sorted); ``unsupported`` the file names with a
    :data:`~app.naming.LEGACY_IMAGE_EXTENSIONS` suffix, in any case (sorted).
    """

    images: dict[str, str]
    hidden: tuple[tuple[str, str], ...] = ()
    unsupported: tuple[str, ...] = ()


@dataclass(frozen=True)
class GtStems:
    """The stems with GT, one set per kind (:meth:`ProjectLayout.scan_gt_stems`).

    Each set holds the stems as the scans list them, the winners' real stems (not their
    :func:`~app.naming.stem_key`): one per name without case (O7).
    """

    mask: frozenset[str]
    skeleton: frozenset[str]
    labels: frozenset[str]


@dataclass(frozen=True)
class ProjectLayout:
    """Paths of one project folder (``root`` holds the images).

    Methods that take a ``stem`` also accept an image path or file name: they strip an image
    extension from what they are given (:meth:`stem`). Hand them the image path or file name
    when there is one. A stem the Library lists may itself end in an image extension (``X.png``
    is the stem of ``X.png.jpg``): stripped again it names another image (``X``), so such a
    stem goes to the ``*_for_stem`` methods, which take it as it is (final review D1).
    """

    root: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", os.path.normpath(os.fspath(self.root)))

    @classmethod
    def for_image(cls, image_path: str | os.PathLike[str]) -> ProjectLayout:
        """Layout of the folder that contains ``image_path``."""
        return cls(os.path.dirname(os.fspath(image_path)))

    @staticmethod
    def stem(path_or_name: str | os.PathLike[str]) -> str:
        """File name without its image extension (see :func:`app.naming.image_stem`)."""
        return image_stem(path_or_name)

    def _path(self, *parts: str) -> str:
        return os.path.normpath(os.path.join(self.root, *parts))

    # ------------------------------------------------------------------ folders
    @property
    def mask_dir(self) -> str:
        return self._path(MASK_DIRNAME)

    @property
    def skeletons_dir(self) -> str:
        return self._path(SKELETONS_DIRNAME)

    @property
    def gt_dir(self) -> str:
        return self._path(GT_DIRNAME)

    @property
    def gt_mask_dir(self) -> str:
        """``gt/mask/``: the GT of ``mask/``, same ``<stem>.png`` names."""
        return self._path(GT_DIRNAME, MASK_DIRNAME)

    @property
    def gt_skeletons_dir(self) -> str:
        """``gt/skeletons/``: the GT of ``skeletons/``, ``<stem>.png`` (no ``_fake``)."""
        return self._path(GT_DIRNAME, SKELETONS_DIRNAME)

    @property
    def output_dirs(self) -> tuple[str, ...]:
        """Every folder the app writes into: where killed writes leave temp files (R8).

        ``mask/`` and ``skeletons/`` (generation writes there, and Measure writes its CSV
        and ``.provenance.json`` next to what it measured), ``gt/``, ``gt/mask/`` and
        ``gt/skeletons/``. Not the image folder (nothing writes there); nothing is read
        from disk.
        """
        return (
            self.mask_dir,
            self.skeletons_dir,
            self.gt_dir,
            self.gt_mask_dir,
            self.gt_skeletons_dir,
        )

    @staticmethod
    def is_legacy_gt_temp(name: str) -> bool:
        """True for ``classes.json.tmp`` / ``<stem>.png.tmp``: what a killed GT save left
        in :attr:`gt_dir` before R8, when the GT writers used ``<path>.tmp``."""
        if not name.endswith(_LEGACY_GT_TEMP_SUFFIX):
            return False
        target = name[: -len(_LEGACY_GT_TEMP_SUFFIX)]
        return target == GT_CLASSES_FILENAME or target.endswith(".png")

    # ------------------------------------------------------------------ images
    def scan_images(self) -> dict[str, str]:
        """``stem -> path`` of the source images in ``root`` (one scandir).

        PNG and JPEG only (``IMAGE_EXTENSIONS``, any letter case; O6). One file per
        stem without case (O7): when two share it (``a.png`` and ``A.jpg``), ``.png``
        wins over ``.jpg`` over ``.jpeg``, then the name that sorts first. That is the
        image the Library shows, so mask and skeleton generation read it too (R7).
        """
        return _scan_ranked(self.root, IMAGE_EXTENSIONS)

    def scan_image_folder(self) -> ImageScan:
        """:meth:`scan_images` plus the files it does not list (the same one scandir)."""
        skips = _Skips()
        images = _scan_ranked(self.root, IMAGE_EXTENSIONS, skips)
        hidden: tuple[tuple[str, str], ...] = ()
        if skips.losers:  # the names of the files listed instead: only when a stem was shared
            winners = {stem_key(stem): os.path.basename(path) for stem, path in images.items()}
            hidden = tuple(sorted((name, winners[key]) for key, name in skips.losers))
        return ImageScan(images=images, hidden=hidden, unsupported=tuple(sorted(skips.unsupported)))

    # ------------------------------------------------------------------ masks
    def mask_path(self, stem: str) -> str:
        """Canonical path ``mask/<stem>.png`` of the prediction (the editor never writes it)."""
        return self._path(MASK_DIRNAME, self.stem(stem) + ".png")

    def mask_candidates(self, stem: str) -> tuple[str, ...]:
        """Every file name a mask of ``stem`` may have, in lookup order (``.png`` first)."""
        stem = self.stem(stem)
        return tuple(self._path(MASK_DIRNAME, stem + ext) for ext in _MASK_LOOKUP)

    def find_mask(self, stem: str) -> str | None:
        """Existing mask file for ``stem`` (``.png`` first, then ``.jpg`` / ``.jpeg``), else
        None. A file named exactly ``<stem>.<ext>`` wins; else one whose stem differs only
        in case (O7), ranked as :meth:`scan_masks` ranks them."""
        found = next((p for p in self.mask_candidates(stem) if os.path.isfile(p)), None)
        if found is None:
            found = _find_without_case(
                self.mask_dir, self.stem(stem), MASK_EXTENSIONS, fold_endings=True
            )
        return found

    def scan_masks(
        self, directory: str | None = None, *, extensions: tuple[str, ...] = MASK_EXTENSIONS
    ) -> dict[str, str]:
        """``stem -> path`` of the masks in ``directory`` (default ``mask/``).

        The same scan as the images: one mask per stem without case, a ``.png`` wins
        over a ``.jpg``/``.jpeg``. Pass ``extensions=(".png",)`` to keep only PNG masks
        (what Measure reads).
        """
        return _scan_ranked(directory or self.mask_dir, extensions)

    def scan_mask_stems(self) -> set[str]:
        """Stems that have a mask in ``mask/`` (one scandir)."""
        return set(self.scan_masks())

    # ------------------------------------------------------------------ skeletons
    def skeleton_path(self, stem: str) -> str:
        """Generated skeleton ``skeletons/<stem>_fake.png``."""
        return self._path(SKELETONS_DIRNAME, self.stem(stem) + SKELETON_SUFFIX)

    def find_skeleton(self, stem: str) -> str | None:
        """Existing ``skeletons/<stem>_fake.png``, its stem and suffix compared without case
        (O7, #3)."""
        return find_by_stem(self.skeletons_dir, self.stem(stem), SKELETON_SUFFIX)

    def scan_skeleton_files(
        self, directory: str | None = None
    ) -> tuple[dict[str, str], list[tuple[str, str]]]:
        """``(stem -> path, shadowed)`` of the skeletons in ``directory`` (default
        ``skeletons/``): what Measure Length reads (:func:`skeleton_stem`,
        :func:`group_skeletons`; one scandir)."""
        return group_skeletons(_scan(directory or self.skeletons_dir, skeleton_stem))

    def scan_skeletons(self, directory: str | None = None) -> dict[str, str]:
        """``stem -> path`` of the skeletons in ``directory`` (default ``skeletons/``): the
        ``_fake.png`` suffix in any letter case, one file per image (:meth:`scan_skeleton_files`)."""
        return self.scan_skeleton_files(directory)[0]

    def scan_skeleton_stems(self) -> set[str]:
        return set(self.scan_skeletons())

    # ------------------------------------------------------------------ GT
    def gt_path(self, stem: str) -> str:
        """GT class labels ``gt/<stem>.png`` (same as ``app.gt.store.gt_path``)."""
        return self._path(GT_DIRNAME, self.stem(stem) + ".png")

    def gt_mask_path(self, stem: str) -> str:
        """GT mask ``gt/mask/<stem>.png`` (what a saved mask edit writes)."""
        return self.gt_mask_path_for_stem(self.stem(stem))

    def gt_skeleton_path(self, stem: str) -> str:
        """GT skeleton ``gt/skeletons/<stem>.png`` (what a saved skeleton edit writes)."""
        return self.gt_skeleton_path_for_stem(self.stem(stem))

    def gt_mask_path_for_stem(self, stem: str) -> str:
        """:meth:`gt_mask_path` of ``stem`` as it is: never stripped (``X.png`` stays)."""
        return self._path(GT_DIRNAME, MASK_DIRNAME, stem + ".png")

    def gt_skeleton_path_for_stem(self, stem: str) -> str:
        """:meth:`gt_skeleton_path` of ``stem`` as it is: never stripped (``X.png`` stays)."""
        return self._path(GT_DIRNAME, SKELETONS_DIRNAME, stem + ".png")

    @staticmethod
    def _find_gt_png(directory: str, stem: str) -> str | None:
        """``<directory>/<stem>.png`` when it exists, else a ``.png`` file (any case of the
        extension) whose stem differs only in case: what :func:`_scan_ranked` lists.
        ``stem`` is taken as it is."""
        path = os.path.join(directory, stem + ".png")
        if os.path.isfile(path):
            return path
        return _find_without_case(directory, stem, (".png",), fold_endings=True)

    def find_gt_mask(self, stem: str) -> str | None:
        """Existing ``gt/mask/<stem>.png``, found as :meth:`scan_gt_masks` lists it (O7)."""
        return self._find_gt_png(self.gt_mask_dir, self.stem(stem))

    def find_gt_skeleton(self, stem: str) -> str | None:
        """Existing ``gt/skeletons/<stem>.png``, found as :meth:`scan_gt_skeletons` lists
        it (O7)."""
        return self.find_gt_skeleton_for_stem(self.stem(stem))

    def find_gt_skeleton_for_stem(self, stem: str) -> str | None:
        """:meth:`find_gt_skeleton` of ``stem`` as it is: never stripped (``X.png`` stays)."""
        return self._find_gt_png(self.gt_skeletons_dir, stem)

    def find_gt_labels(self, stem: str) -> str | None:
        """Existing ``gt/<stem>.png``, found as :meth:`scan_gt_labels` lists it (O7)."""
        return self._find_gt_png(self.gt_dir, self.stem(stem))

    def scan_gt_masks(self) -> dict[str, str]:
        """``stem -> path`` of ``gt/mask/*.png``: one file per stem without case."""
        return _scan_ranked(self.gt_mask_dir, (".png",))

    def scan_gt_skeletons(self) -> dict[str, str]:
        """``stem -> path`` of ``gt/skeletons/*.png``: one file per stem without case."""
        return _scan_ranked(self.gt_skeletons_dir, (".png",))

    def scan_gt_labels(self) -> dict[str, str]:
        """``stem -> path`` of ``gt/*.png`` (files only: ``classes.json`` and the two
        folders under ``gt/`` are not labels)."""
        return _scan_ranked(self.gt_dir, (".png",))

    def scan_gt_stems(self) -> GtStems:
        """The stems with GT of each kind: three scans, one scandir each."""
        return GtStems(
            mask=frozenset(self.scan_gt_masks()),
            skeleton=frozenset(self.scan_gt_skeletons()),
            labels=frozenset(self.scan_gt_labels()),
        )

    # ------------------------------------------------------------------ metric CSVs
    def find_lengths_csv(self) -> str | None:
        """Existing ``root_lengths.csv``: ``skeletons/`` (where Measure writes it), else the
        project folder; None when neither has it (two stats at most)."""
        for path in (self._path(SKELETONS_DIRNAME, ROOT_LENGTHS_CSV), self._path(ROOT_LENGTHS_CSV)):
            if os.path.isfile(path):
                return path
        return None

    def find_areas_csv(self) -> str | None:
        """Existing ``mask/root_areas.csv`` (where Measure writes it), else None."""
        path = self._path(MASK_DIRNAME, ROOT_AREAS_CSV)
        return path if os.path.isfile(path) else None


def project_layout_of(manager: object) -> ProjectLayout | None:
    """Layout of ``manager``'s project: ``original_folder``, else the first image's folder.

    ``manager`` is the window's ``app.gui.image_manager.ImageManager`` or any stand-in with
    ``original_folder`` / ``images``; None when nothing is loaded. Generate,
    Measure, Visualize and the window all resolve the project through here, so
    they can never disagree about it.
    """
    folder = getattr(manager, "original_folder", None)
    if folder:
        return ProjectLayout(folder)
    images = getattr(manager, "images", None)
    if images:
        return ProjectLayout.for_image(next(iter(images.values())))
    return None

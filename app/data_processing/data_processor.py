"""Load a root-metric CSV (length or area) into a typed DataFrame for the dashboards."""

import logging
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from app.config import LENGTH_VALUE_COLUMN
from app.naming import parse_image_name

logger = logging.getLogger(__name__)

#: Columns every chart row needs besides the metric itself.
KEY_COLUMNS = ("Date", "Tube", "Position")
#: Image-name convention the Measure step reads (see app.naming.parse_image_name).
NAME_FORMAT_HINT = "<field>_T<tube>_L<position>_YYYY.MM.DD_HHMMSS_..."
#: A UTF-8 byte-order mark (EF BB BF) read as latin-1.
_LATIN1_BOM = "\u00ef\u00bb\u00bf"
#: The notice's and the exports' words for the rows :func:`repeated_scans` leaves out.
REPEATED_SCANS = "repeated scans of a position on one date (the last image is kept)"
_NAT = np.datetime64("NaT", "ns")


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _join(parts: list[str]) -> str:
    """'a', 'a and b', 'a, b and c'."""
    return parts[0] if len(parts) == 1 else f"{', '.join(parts[:-1])} and {parts[-1]}"


def _parse_dates(text: pd.Series) -> pd.Series:
    """``YYYY.MM.DD`` cells as nanosecond timestamps; anything else is NaT.

    As before pandas 3, which parses dates to microseconds: nanoseconds keep the figures'
    date strings and the dashboards' dtypes as they were, and a date outside the
    nanosecond range (years 1677-2262, e.g. a mistyped year in a file name) is NaT, so only
    its row is left out (counted as unreadable), not the whole CSV.

    Each distinct cell is parsed once (a season has a few dozen dates in 46k rows: 2.5 ms
    instead of 8.5 ms); an empty cell is NaT.
    """
    codes, distinct = pd.factorize(text)
    dates = pd.to_datetime(pd.Series(distinct), format="%Y.%m.%d", errors="coerce")
    inside = dates.between(pd.Timestamp.min, pd.Timestamp.max)
    values = np.append(dates.where(inside).astype("datetime64[ns]").to_numpy(), _NAT)
    return pd.Series(values[codes], index=text.index)  # code -1 (empty cell): the NaT


def _numbers(cells: pd.Series) -> pd.Series:
    """``cells`` as numbers; text, empty cells and infinities (``inf``, ``-inf``) are NaN.

    Measure never writes an infinity, but a hand-edited CSV can hold one: the charts and
    exports would show it, and the depth intervals cannot place it (the Depth Profile Grid
    failed with "cannot convert float infinity to integer"). Its row is left out as
    unreadable, like a non-numeric cell.
    """
    numbers = pd.to_numeric(cells, errors="coerce")
    if pd.api.types.is_float_dtype(numbers):
        numbers = numbers.mask(np.isinf(numbers))
    return numbers


def repeated_scans(
    chart: pd.DataFrame, names: pd.Series | None, times: pd.Series | None
) -> pd.Series:
    """Rows of ``chart`` that image a position again on the same date (True = leave out).

    A tube scanned twice in a day has two images of each position; summed, its total for
    that date is about doubled. Rows are repeats of each other when their image names give
    the same field (``app.naming``: ``24EFCAM1`` and ``24EFCAM2`` are one field, ``24EF``)
    and their tube, position and date are equal; a row whose name gives no tube, position
    and date is never a repeat. Of each set the last by capture time (``times``, text
    ``HH:MM:SS``; a row without one counts as earlier than every row with one, so a timed
    capture is kept over an untimed one), then by image name (the last name), is kept: the
    later scan of a position replaces the earlier one.

    ``chart`` holds only usable rows (int64 Tube and Position, datetime64[ns] Date);
    ``names`` and ``times`` are the CSV's Image and Time columns on the same index (or
    None). Only rows whose tube, position and date occur more than once have their names
    parsed.
    """
    repeats = np.zeros(len(chart), dtype=bool)
    if names is not None and len(chart):
        shared = np.flatnonzero(chart.duplicated(list(KEY_COLUMNS), keep=False).to_numpy())
        if len(shared):
            labels = chart.index[shared]
            columns = (chart[key].to_numpy()[shared].tolist() for key in KEY_COLUMNS)
            cells = (names.loc[labels].tolist(), _cells(times, labels))
            candidates = []  # (order, key, row): a few rows per repeated scan, not every row
            rows = zip(shared.tolist(), *columns, *cells, strict=True)
            for row, date, tube, position, name, time in rows:
                parsed = parse_image_name(str(name))
                if parsed.is_complete:
                    text = "" if pd.isna(time) else str(time).strip()
                    order = (bool(text), text, str(name))  # untimed < timed; then time, name
                    candidates.append((order, (parsed.field, date, tube, position), row))
            candidates.sort(key=lambda candidate: candidate[0], reverse=True)  # last first
            seen = set()
            for _order, key, row in candidates:
                repeats[row] = key in seen
                seen.add(key)
    return pd.Series(repeats, index=chart.index)


def _cells(column: pd.Series | None, labels: pd.Index) -> list:
    """The cells of ``column`` at ``labels`` (all missing when there is no column)."""
    return [None] * len(labels) if column is None else column.loc[labels].tolist()


@dataclass(frozen=True)
class RowExclusions:
    """Rows of a metric CSV that the dashboards leave out, by cause (FR-15).

    Attributes:
        failed: measurements that failed (non-empty ``Error`` cell).
        bad_name: image names ``app.naming`` cannot read a tube, position and
            date from (not in :data:`NAME_FORMAT_HINT` form, e.g. ``IMG_0001``).
        unreadable: other rows with a missing or non-numeric field or value (an infinite
            number included).
        repeated: usable rows of a position imaged again on the same date (a tube scanned
            twice in a day); the last image is kept (:func:`repeated_scans`).
    """

    failed: int = 0
    bad_name: int = 0
    unreadable: int = 0
    repeated: int = 0

    @classmethod
    def from_mask(
        cls,
        excluded: pd.Series,
        errors: pd.Series | None = None,
        names: pd.Series | None = None,
    ) -> "RowExclusions":
        """Classify the ``excluded`` rows (boolean mask) using their Error and Image cells."""
        if not excluded.any():
            return cls()
        rest = excluded
        failed = 0
        if errors is not None:
            has_error = errors.notna() & errors.astype(str).str.strip().ne("")
            failed = int((excluded & has_error).sum())
            rest = excluded & ~has_error
        bad_name = 0
        if names is not None:
            bad_name = sum(
                not parse_image_name(str(name)).is_complete for name in names[rest].tolist()
            )
        return cls(failed=failed, bad_name=bad_name, unreadable=int(rest.sum()) - bad_name)

    @property
    def total(self) -> int:
        """Number of rows left out of the charts."""
        return self.failed + self.bad_name + self.unreadable + self.repeated

    def message(self) -> str | None:
        """One user-facing sentence explaining the exclusions (``None`` if none)."""
        if not self.total:
            return None
        verb = "is" if self.total == 1 else "are"
        return f"{_plural(self.total, 'row')} {verb} not shown: {self.causes()}."

    def causes(self) -> str:
        """The causes with their counts, joined ('' if none): the notice's and the
        exports' words."""
        parts = []
        if self.failed:
            parts.append(
                f"{_plural(self.failed, 'failed measurement')} (see the CSV's Error column)"
            )
        if self.bad_name:
            parts.append(
                f"{_plural(self.bad_name, 'image name')} not in the {NAME_FORMAT_HINT} format"
            )
        if self.unreadable:
            parts.append(
                f"{_plural(self.unreadable, 'row')} with a missing or non-numeric tube, "
                "position, date or value"
            )
        if self.repeated:
            parts.append(f"{_plural(self.repeated, 'row')} from {REPEATED_SCANS}")
        return _join(parts) if parts else ""


class MetricDataProcessor:
    """
    Loads and prepares a root-metric CSV (length or area) for visualization.

    The length and area pipelines previously had two near-identical processor
    classes; they now share this base, parameterized only by the metric column
    and its display label/unit.

    After loading, ``df`` holds one row per measured image and only the columns
    ``Date`` (datetime64), ``Tube``/``Position`` (int64) and ``value_column``
    (float64); rows missing any of them are dropped and counted in ``exclusions``, and so
    are repeated scans of a position on one date (:func:`repeated_scans`): one image per
    field, tube, position and date is kept, so every total sums each position once.
    """

    #: name of the numeric metric column in the CSV (e.g. "Length (mm)")
    value_column = LENGTH_VALUE_COLUMN
    #: human-readable metric name used in chart titles/hover (e.g. "Length")
    metric_label = "Length"
    #: display unit of ``value_column`` (e.g. "mm")
    unit = "mm"

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.exclusions = RowExclusions()
        self._load_problem: str | None = None  # why the file could not be used at all
        self.df = self._load_and_prepare_data()
        self._compute_unique_caches()

    @property
    def dropped_rows(self) -> int:
        """Rows left out of ``df``: a key field or the value was unreadable, or a repeat."""
        return self.exclusions.total

    def empty_reason(self) -> str | None:
        """Why ``df`` has no row to chart (one user-facing sentence); None when it has rows."""
        if not self.df.empty:
            return None
        if self._load_problem:
            return self._load_problem
        if self.exclusions.total:
            return f"Nothing to chart: {self.exclusions.message()}"
        return "The CSV has no measurement rows."

    def _compute_unique_caches(self):
        """Compute the sorted unique lookups once instead of on every accessor call."""
        self._unique_tubes = sorted(self.df["Tube"].unique()) if "Tube" in self.df.columns else []
        self._unique_dates = sorted(self.df["Date"].unique()) if "Date" in self.df.columns else []
        self._unique_positions = (
            sorted(self.df["Position"].unique()) if "Position" in self.df.columns else []
        )

    def _read_csv(self):
        """Read the metric CSV; writers emit UTF-8 (see write_metric_csv)."""
        # Exact "utf-8" is not redundant: pandas 2 and 3.0 hand its raw bytes to the C parser,
        # which reads some damaged files (a NUL byte in a field) that "utf-8-sig" rejects.
        for encoding in ("utf-8-sig", "utf-8"):
            try:
                return pd.read_csv(self.csv_path, encoding=encoding)
            except UnicodeDecodeError:
                pass
        df = pd.read_csv(self.csv_path, encoding="latin-1")  # decodes any byte
        # pandas 3.1 decodes such a damaged UTF-8 file in full, so it is read here: its BOM
        # then reads as _LATIN1_BOM and would hide the first column (Image) from the row
        # counts.
        first = str(df.columns[0]) if len(df.columns) else ""
        if first.startswith(_LATIN1_BOM):
            df = df.rename(columns={df.columns[0]: first[len(_LATIN1_BOM) :]})
        return df

    def _resolve_value_column(self, df):
        """Map the metric column when legacy encodings mangled the header."""
        if self.value_column in df.columns:
            return df
        for col in df.columns:
            if col.replace("\ufeff", "").replace("Â", "") == self.value_column:
                return df.rename(columns={col: self.value_column})
        # e.g. "Area (mm^2)" for "Area (mm²)"; only this processor's own metric,
        # so a length dashboard never charts an area column as length.
        same_metric = [c for c in df.columns if c.startswith(self.metric_label) and "mm" in c]
        if len(same_metric) == 1:
            return df.rename(columns={same_metric[0]: self.value_column})
        return df

    def _load_and_prepare_data(self):
        """Read, type and filter the CSV; returns an empty frame on any failure."""
        try:
            df = self._read_csv()
            df = self._resolve_value_column(df)
            if self.value_column not in df.columns:
                logger.error(
                    "CSV %s missing metric column %r (columns: %s)",
                    self.csv_path,
                    self.value_column,
                    list(df.columns),
                )
                self._load_problem = f"The CSV has no {self.value_column!r} column."
                return pd.DataFrame()
            missing = [column for column in KEY_COLUMNS if column not in df.columns]
            if missing:
                logger.error("CSV %s missing key column(s) %s", self.csv_path, missing)
                names = ", ".join(repr(column) for column in missing)
                self._load_problem = f"The CSV has no {names} column{'s' * (len(missing) > 1)}."
                return pd.DataFrame()
            # Only the columns the charts read, typed in a frame of their own: the Image,
            # Time and Error text (80 % of the memory) is neither kept while the dashboard
            # is open nor copied by the steps below (dropna copies every column).
            chart = pd.DataFrame(
                {
                    "Date": _parse_dates(df["Date"]),
                    # No integer downcast: int8 positions overflow in the depth-interval
                    # arithmetic (124 -> interval end 130 wraps to -126, PV-01).
                    "Tube": _numbers(df["Tube"]),
                    "Position": _numbers(df["Position"]),
                    self.value_column: _numbers(df[self.value_column]).astype("float64"),
                },
                index=df.index,
            )
            exclusions = RowExclusions.from_mask(
                chart.isna().any(axis=1), df.get("Error"), df.get("Image")
            )
            chart = chart.dropna()
            chart["Tube"] = chart["Tube"].astype("int64")
            chart["Position"] = chart["Position"].astype("int64")
            # One image per field, tube, position and date: a second scan of a tube on
            # the same day would otherwise count each position twice in every total.
            repeats = repeated_scans(chart, df.get("Image"), df.get("Time"))
            if repeats.any():
                chart = chart[~repeats.to_numpy()]
                exclusions = replace(exclusions, repeated=int(repeats.sum()))
            self.exclusions = exclusions
            if self.exclusions.total:
                logger.warning("%s: %s", self.csv_path, self.exclusions.message())
            return chart
        except Exception as exc:
            logger.exception("Failed to load CSV %s", self.csv_path)
            self._load_problem = f"The CSV could not be read: {exc}"
            return pd.DataFrame()

    def get_unique_tubes(self):
        """Sorted tube numbers present in ``df``."""
        return self._unique_tubes

    def get_unique_dates(self):
        """Sorted measurement dates (Timestamps) present in ``df``."""
        return self._unique_dates

    def get_unique_positions(self):
        """Sorted depth positions (L numbers) present in ``df``."""
        return self._unique_positions


class DataProcessor(MetricDataProcessor):
    """Root-length data processor."""

    value_column = LENGTH_VALUE_COLUMN
    metric_label = "Length"
    unit = "mm"

"""A plain-Python reference of the dashboards' numbers, read straight from a metric CSV.

The oracle of ``tests/test_viz_dashboard_numbers.py``. It shares no code and no library path
with the app's aggregation: no pandas, no numpy, no ``app`` import, only ``csv``, ``math`` and
``datetime``, and it follows the rules as the dashboards state them, not as they are coded:

- a row is charted when its Tube and Position are whole numbers, its Date is ``YYYY.MM.DD`` and
  its value a finite number; every other row (a failed measurement, a name the parser cannot
  read, an infinite value) is left out;
- intervals are fixed bins of 10 positions (L1-L10 ends at 10, L11-L20 at 20, ...); an interval's
  statistics are the count, mean, min, max and sample SD (none for one value) of its rows;
- a tube's total on a date is the sum of its rows that day (Totals by Tube, Growth Over Time);
- the field average merges measurement dates within 3 days: dates are visited in order, each
  date not yet merged anchors a window of the unmerged dates within +-3 days, and the window
  takes the date of it with the most tubes (the earliest on a tie); a tube measured on two
  merged dates counts with the mean of its totals; per merged date the field mean, the sample SD
  across tubes (none for one tube) and the number of tubes.

Sums are ``math.fsum`` (correctly rounded); the app uses Kahan sums, so the two can differ in the
last bits: callers compare with a relative tolerance.
"""

from __future__ import annotations

import csv
import datetime as dt
import math
from collections import defaultdict

INTERVAL = 10
MERGE_DAYS = 3


def _whole(text: str) -> int | None:
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number) or number != int(number):
        return None
    return int(number)


def _value(text: str) -> float | None:
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _date(text: str) -> dt.date | None:
    try:
        return dt.datetime.strptime(text, "%Y.%m.%d").date()
    except ValueError:
        return None


def charted_rows(csv_path: str, value_column: str) -> list[tuple[int, int, dt.date, float]]:
    """The rows the dashboards chart, as (tube, position, date, value), in CSV order."""
    rows = []
    with open(csv_path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            tube, position = _whole(row["Tube"]), _whole(row["Position"])
            date, value = _date(row["Date"]), _value(row[value_column])
            if None not in (tube, position, date, value):
                rows.append((tube, position, date, value))
    return rows


def interval_end(position: int) -> int:
    return ((position - 1) // INTERVAL + 1) * INTERVAL


def stats(values: list[float]) -> dict:
    """count, mean, min, max and sample SD (None for one value) of ``values``."""
    n = len(values)
    mean = math.fsum(values) / n
    sd = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else None
    return {"count": n, "mean": mean, "min": min(values), "max": max(values), "sd": sd}


def totals(rows) -> dict[tuple[int, dt.date], float]:
    """(tube, date) -> the tube's total that day."""
    groups = defaultdict(list)
    for tube, _position, date, value in rows:
        groups[tube, date].append(value)
    return {key: math.fsum(values) for key, values in groups.items()}


def intervals(rows) -> dict[tuple[int, dt.date, int], dict]:
    """(tube, date, interval end) -> stats of the tube's rows in that interval that day."""
    groups = defaultdict(list)
    for tube, position, date, value in rows:
        groups[tube, date, interval_end(position)].append(value)
    return {key: stats(values) for key, values in groups.items()}


def field_intervals(rows) -> dict[tuple[dt.date, int], dict]:
    """(date, interval end) -> stats of every tube's rows in that interval that day."""
    groups = defaultdict(list)
    for _tube, position, date, value in rows:
        groups[date, interval_end(position)].append(value)
    return {key: stats(values) for key, values in groups.items()}


def merged_dates(tubes_per_date: dict[dt.date, int]) -> dict[dt.date, dt.date]:
    """Each measurement date -> the date of its field-average campaign (see the docstring)."""
    mapping: dict[dt.date, dt.date] = {}
    window = dt.timedelta(days=MERGE_DAYS)
    for anchor in sorted(tubes_per_date):
        if anchor in mapping:
            continue
        nearby = [
            d
            for d in sorted(tubes_per_date)
            if d not in mapping and anchor - window <= d <= anchor + window
        ]
        best = max(tubes_per_date[d] for d in nearby)
        representative = next(d for d in nearby if tubes_per_date[d] == best)
        for d in nearby:
            mapping[d] = representative
    return mapping


def field_growth(rows) -> dict[dt.date, dict]:
    """merged date -> {"mean", "sd" (None for one tube), "tubes"} of the per-tube totals."""
    by_tube_date = totals(rows)
    tubes_per_date = defaultdict(int)
    for _tube, date in by_tube_date:
        tubes_per_date[date] += 1
    mapping = merged_dates(dict(tubes_per_date))
    per_tube = defaultdict(list)
    for (tube, date), total in by_tube_date.items():
        per_tube[mapping[date], tube].append(total)
    per_date = defaultdict(list)
    for (date, _tube), values in per_tube.items():
        per_date[date].append(math.fsum(values) / len(values))
    out = {}
    for date, values in per_date.items():
        s = stats(values)
        out[date] = {"mean": s["mean"], "sd": s["sd"], "tubes": s["count"]}
    return out

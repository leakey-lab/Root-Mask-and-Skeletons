"""
Data caching and interval calculation utilities for Dash visualizations.

The cache is built once from the DataProcessor's DataFrame in ``__init__`` /
``_cache_data``. The DataFrame MUST NOT be mutated after a DataCache is
constructed; if the underlying data changes, rebuild via ``refresh()`` (or
create a new DataCache) so memoized interval data is not stale.
"""

import copy
import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class DataCache:
    """Manages cached data for Dash visualizations."""

    def __init__(self, data_processor: Any):
        """
        Initialize the data cache.

        Args:
            data_processor: DataProcessor instance with loaded data
        """
        self.data_processor = data_processor
        self.tubes: Optional[Tuple[int, ...]] = None
        self.dates: Optional[Tuple[pd.Timestamp, ...]] = None
        self.tube_date_groups = None
        self.position_groups = None
        # Per-instance memo for get_interval_data. Keyed by (tube, date,
        # interval_size). Per-instance (not @lru_cache) so it does not pin the
        # DataCache in a module-level cache (memory leak) nor share / return
        # stale results across instances (F-012).
        self._interval_cache: Dict[Tuple[int, Any, int], dict] = {}
        self._cache_data()

    def _cache_data(self) -> None:
        """Cache frequently used data to improve performance."""
        self.tubes = tuple(sorted(self.data_processor.get_unique_tubes()))
        self.dates = tuple(sorted(self.data_processor.get_unique_dates()))

        # Pre-compute grouped data for different views
        df = self.data_processor.df
        self.tube_date_groups = df.groupby(["Tube", "Date"])["Length (mm)"].sum()
        self.position_groups = df.groupby(["Tube", "Position", "Date"])[
            "Length (mm)"
        ].agg(["mean", "min", "max", "std", "count"])
        # Invalidate any previously memoized interval data.
        self._interval_cache.clear()

    def refresh(self) -> None:
        """Rebuild all cached structures from the (possibly changed) DataFrame.

        Call this if ``data_processor.df`` is replaced/mutated so callers do not
        observe stale interval data.
        """
        self._cache_data()

    def get_interval_data(
        self, tube: int, date: pd.Timestamp, interval_size: int = 10
    ) -> dict:
        """
        Get cached interval data for a specific tube and date with standardized intervals.
        Returns data in fixed intervals (1-10, 11-20, etc.) regardless of starting position.

        A deep copy is returned so callers cannot mutate the cached structure.

        Args:
            tube: Tube number
            date: Date timestamp
            interval_size: Size of each interval (default 10)

        Returns:
            dict: Interval statistics keyed by interval end position
        """
        cache_key = (tube, date, interval_size)
        cached = self._interval_cache.get(cache_key)
        if cached is not None:
            return copy.deepcopy(cached)

        computed = self._compute_interval_data(tube, date, interval_size)
        self._interval_cache[cache_key] = computed
        return copy.deepcopy(computed)

    def _compute_interval_data(
        self, tube: int, date: pd.Timestamp, interval_size: int
    ) -> dict:
        """Compute (uncached) standardized-interval statistics for one tube/date."""
        df = self.data_processor.df
        tube_data = df[(df["Tube"] == tube) & (df["Date"] == date)]

        if tube_data.empty:
            return {}

        # Get actual positions
        positions = sorted(tube_data["Position"].unique())
        min_pos = min(positions)
        max_pos = max(positions)

        # Calculate standardized intervals
        first_interval_end = ((min_pos - 1) // interval_size + 1) * interval_size
        last_interval_end = ((max_pos - 1) // interval_size + 1) * interval_size

        interval_stats = {}
        for interval_end in range(
            first_interval_end, last_interval_end + 1, interval_size
        ):
            interval_start = interval_end - interval_size + 1
            interval_data = tube_data[
                (tube_data["Position"] >= interval_start)
                & (tube_data["Position"] <= interval_end)
            ]

            if not interval_data.empty:
                stats = {
                    "avg": interval_data["Length (mm)"].mean(),
                    "min": interval_data["Length (mm)"].min(),
                    "max": interval_data["Length (mm)"].max(),
                    "std": interval_data["Length (mm)"].std(),
                    "count": len(interval_data),
                    "interval_start": interval_start,
                    "interval_end": interval_end,
                }
                interval_stats[interval_end] = stats
            else:
                interval_stats[interval_end] = {
                    "avg": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                    "std": float("nan"),
                    "count": 0,
                    "interval_start": interval_start,
                    "interval_end": interval_end,
                }

        return interval_stats

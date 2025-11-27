"""
Data caching and interval calculation utilities for Dash visualizations.
"""

from functools import lru_cache
from typing import Any
import pandas as pd


class DataCache:
    """Manages cached data for Dash visualizations."""
    
    def __init__(self, data_processor: Any):
        """
        Initialize the data cache.
        
        Args:
            data_processor: DataProcessor instance with loaded data
        """
        self.data_processor = data_processor
        self.tubes = None
        self.dates = None
        self.tube_date_groups = None
        self.position_groups = None
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

    @lru_cache(maxsize=32)
    def get_interval_data(
        self, tube: int, date: pd.Timestamp, interval_size: int = 10
    ) -> dict:
        """
        Get cached interval data for a specific tube and date with standardized intervals.
        Returns data in fixed intervals (1-10, 11-20, etc.) regardless of starting position.
        
        Args:
            tube: Tube number
            date: Date timestamp
            interval_size: Size of each interval (default 10)
            
        Returns:
            dict: Interval statistics keyed by interval end position
        """
        tube_data = self.data_processor.df[
            (self.data_processor.df["Tube"] == tube)
            & (self.data_processor.df["Date"] == date)
        ]

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


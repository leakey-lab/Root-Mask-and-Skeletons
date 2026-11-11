"""Shared fixture data of the dashboard data tests (``tests/test_viz_data_*.py`` and others).

``ROWS`` is a small metric CSV whose every aggregate can be derived by hand (the table is in
``tests/test_viz_data_cache.py``'s docstring); ``write_metric_csv`` writes rows of
``(tube, position, date, value)`` with the app's CSV headers; ``Charts`` is the minimal host of
the ``DashVisualizations`` mixin (the attributes ``MetricDashApp`` provides). Holds no assert.

``tests/viz_dash_helpers.py`` has a ``write_metric_csv`` of its own for the Dash tests: it
takes one value column and rows of ``(image, tube, position, date, value)``.
"""

import csv

from app.visualization.dash_data_cache import DataCache
from app.visualization.dash_visualizations import DashVisualizations

ROWS = [
    (1, 5, "2024.07.18", 2.0),
    (1, 7, "2024.07.18", 4.0),
    (1, 25, "2024.07.18", 10.0),
    (2, 12, "2024.07.18", 6.0),
    (2, 18, "2024.07.18", 8.0),
    (1, 3, "2024.08.07", 1.0),
    (1, 14, "2024.08.07", 3.0),
    (2, 9, "2024.08.07", 5.0),
]


def write_metric_csv(path, headers, rows=ROWS):
    """Write ``rows`` (tube, position, date, value) with the app's CSV headers."""
    value_column = headers[5]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for tube, position, date, value in rows:
            writer.writerow(
                {
                    "Image": f"CAM_T{tube:03d}_L{position:03d}_{date}_120000_001",
                    "Tube": tube,
                    "Position": position,
                    "Date": date,
                    "Time": "12:00:00",
                    value_column: value,
                    "Error": "",
                }
            )
    return str(path)


class Charts(DashVisualizations):
    """Minimal host for the DashVisualizations mixin (the attributes DashApp provides)."""

    def __init__(self, processor):
        self.data_processor = processor
        self.data_cache = DataCache(processor)
        self.available_images = {}

"""``DataProcessor`` / ``DataProcessorArea`` load a Measure CSV (parked here from the retired
``tests/test_regression.py`` for W2-viz to merge into its own files).
"""

import csv

from app.data_processing.data_processor import DataProcessor


def test_data_processor_loads_and_filters(tmp_path):
    csv_path = tmp_path / "root_lengths.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["Image", "Tube", "Position", "Date", "Time", "Length (mm)"]
        )
        w.writeheader()
        w.writerow(
            {
                "Image": "a",
                "Tube": 1,
                "Position": 10,
                "Date": "2024.01.01",
                "Time": "00:00:01",
                "Length (mm)": 5.5,
            }
        )
        w.writerow(
            {
                "Image": "b",
                "Tube": 2,
                "Position": 20,
                "Date": "2024.01.02",
                "Time": "00:00:02",
                "Length (mm)": 7.0,
            }
        )
    dp = DataProcessor(str(csv_path))
    assert list(dp.get_unique_tubes()) == [1, 2]
    assert len(dp.get_unique_positions()) == 2


def test_data_processor_bad_path_returns_empty():
    dp = DataProcessor("does_not_exist_12345.csv")
    assert dp.df.empty


def test_data_processor_area_loads_utf8_csv(tmp_path):
    """Area CSVs are written UTF-8 with a superscript-2 header; must not read as latin-1."""
    from app.config import AREA_CSV_HEADERS
    from app.data_processing.data_processor_area import DataProcessorArea

    csv_path = tmp_path / "root_areas.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=AREA_CSV_HEADERS)
        w.writeheader()
        w.writerow(
            {
                "Image": "img",
                "Tube": 1,
                "Position": 2,
                "Date": "2024.01.01",
                "Time": "00:00:01",
                "Area (mm²)": 3.25,
                "Error": "",
            }
        )
    dp = DataProcessorArea(str(csv_path))
    assert not dp.df.empty
    assert list(dp.get_unique_tubes()) == [1]
    assert dp.get_unique_positions() == [2]

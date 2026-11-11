"""``MetricDataProcessor`` (``app/data_processing/data_processor.py``): a Measure CSV as the
dashboards chart it.

Promises: one row per image with ``Date`` (datetime64[ns]), ``Tube`` / ``Position`` (int64) and
the value (float64), and nothing else kept; a row with a missing, non-numeric or infinite key or
value, or a date that does not parse, is left out and counted by cause (failed: an ``Error``
cell; bad name; unreadable) and logged once (FR-15); ``empty_reason()`` says why nothing is
charted; UTF-8 (with or without a BOM), then latin-1 for a legacy or damaged file; a mangled
metric header is mapped, but only to the processor's own metric; a file that cannot be read
gives an empty frame, never an exception.
"""

import csv
import gc
import logging
import weakref

import pandas as pd
import pytest

from app.config import AREA_CSV_HEADERS, LENGTH_CSV_HEADERS
from app.data_processing.data_processor import DataProcessor, RowExclusions
from app.data_processing.data_processor_area import DataProcessorArea
from tests.viz_dash_helpers import AREA_COLUMN, LENGTH_COLUMN

METRICS = ["length", "area"]


def _write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LENGTH_CSV_HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h, "") for h in LENGTH_CSV_HEADERS})
    return str(path)


def _row(image, tube, position, date="2024.07.18", length=1.5, error=""):
    return {
        "Image": image,
        "Tube": tube,
        "Position": position,
        "Date": date,
        "Time": "12:00:00",
        "Length (mm)": length,
        "Error": error,
    }


def _write_table(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return str(path)


GOOD = [
    _row("CAM_T001_L010_2024.07.18_120000_001", 1, 10),
    _row("CAM_T001_L011_2024.07.18_120002_001", 1, 11),
]
# What the Measure workers write for an image they could not process.
FAILED = _row("CAM_T001_L012_2024.07.18_120004", "", "", date="", length=0, error="boom")
# What they write for a name the parser could not read (no Error text).
BAD_NAME = _row("IMG_0001", "", "", date="")
LATER = _row("CAM_T002_L003_2024.07.19_120000_001", "2.0", "3.0", "2024.07.19", 4)


# ---- the chart frame ------------------------------------------------------------------


def test_kept_rows_keep_their_csv_row_numbers_and_types(tmp_path):
    dp = DataProcessor(_write(tmp_path / "l.csv", [FAILED, *GOOD, BAD_NAME, LATER]))

    assert dp.df.index.tolist() == [1, 2, 4]
    assert [str(dtype) for dtype in dp.df.dtypes] == [
        "datetime64[ns]",
        "int64",
        "int64",
        "float64",
    ]
    assert dp.df["Tube"].tolist() == [1, 1, 2]  # "2.0" read as a number, then an integer
    assert dp.df["Position"].tolist() == [10, 11, 3]
    assert dp.df["Length (mm)"].tolist() == [1.5, 1.5, 4.0]
    assert dp.df["Date"].tolist() == [
        pd.Timestamp("2024-07-18"),
        pd.Timestamp("2024-07-18"),
        pd.Timestamp("2024-07-19"),
    ]
    assert dp.exclusions == RowExclusions(failed=1, bad_name=1)


def test_a_date_outside_the_nanosecond_range_leaves_out_only_its_row(tmp_path):
    # pandas 3 parses 3024.07.18 (microseconds); the chart frame keeps nanoseconds, as
    # pandas 2 did, so such a row is unreadable instead of the whole CSV failing to load.
    rows = [
        *GOOD,
        _row("CAM_T001_L013_3024.07.18_120000_001", 1, 13, "3024.07.18"),
        _row("CAM_T001_L014_1601.01.01_120000_001", 1, 14, "1601.01.01"),
        LATER,
    ]
    dp = DataProcessor(_write(tmp_path / "l.csv", rows))

    assert dp.empty_reason() is None
    assert dp.df.index.tolist() == [0, 1, 4]
    assert str(dp.df["Date"].dtype) == "datetime64[ns]"
    assert dp.exclusions == RowExclusions(unreadable=2)


@pytest.mark.parametrize("date", ["2024-07-18", "18.07.2024", "2024.13.40", ""])
def test_a_date_that_does_not_parse_leaves_out_only_its_row(tmp_path, date):
    # A mistyped date in a hand-edited CSV: the row is counted as unreadable, the rest charts.
    rows = [*GOOD, _row("CAM_T001_L013_2024.07.18_120000_001", 1, 13, date), LATER]
    dp = DataProcessor(_write(tmp_path / "l.csv", rows))

    assert dp.empty_reason() is None
    assert dp.df.index.tolist() == [0, 1, 3]
    assert dp.exclusions == RowExclusions(unreadable=1)


def test_a_csv_without_left_out_rows_keeps_its_default_row_numbers(tmp_path):
    dp = DataProcessor(_write(tmp_path / "l.csv", [*GOOD, LATER]))

    assert isinstance(dp.df.index, pd.RangeIndex)
    assert dp.df.index.tolist() == [0, 1, 2]
    assert dp.exclusions == RowExclusions()


def test_the_frame_read_from_the_csv_is_not_kept(tmp_path, monkeypatch):
    # The Image, Time and Error text goes with the frame read from the CSV.
    read = []
    original = DataProcessor._read_csv

    def keep_a_weak_reference(self):
        frame = original(self)
        read.append(weakref.ref(frame))
        return frame

    monkeypatch.setattr(DataProcessor, "_read_csv", keep_a_weak_reference)
    dp = DataProcessor(_write(tmp_path / "l.csv", [FAILED, *GOOD]))
    gc.collect()

    assert len(dp.df) == 2
    assert read and read[0]() is None


# ---- rows left out, by cause (FR-15) ------------------------------------------------------


def test_clean_csv_excludes_nothing(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="app.data_processing.data_processor"):
        dp = DataProcessor(_write(tmp_path / "l.csv", GOOD))

    assert dp.dropped_rows == 0
    assert dp.exclusions == RowExclusions()
    assert dp.exclusions.message() is None
    assert not caplog.records


def test_excluded_rows_are_counted_by_cause_and_logged(tmp_path, caplog):
    rows = GOOD + [FAILED, BAD_NAME, dict(BAD_NAME, Image="IMG_0002")]
    with caplog.at_level(logging.WARNING, logger="app.data_processing.data_processor"):
        dp = DataProcessor(_write(tmp_path / "l.csv", rows))

    assert len(dp.df) == 2
    assert dp.dropped_rows == 3
    assert dp.exclusions == RowExclusions(failed=1, bad_name=2)
    assert dp.exclusions.message() == (
        "3 rows are not shown: 1 failed measurement (see the CSV's Error column) and "
        "2 image names not in the <field>_T<tube>_L<position>_YYYY.MM.DD_HHMMSS_... format."
    )
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    assert dp.exclusions.message() in caplog.records[0].getMessage()


def test_readable_name_with_a_non_numeric_value_is_unreadable(tmp_path):
    odd = _row("CAM_T001_L013_2024.07.18_120006_001", 1, 13, length="n/a")
    dp = DataProcessor(_write(tmp_path / "l.csv", GOOD + [odd]))

    assert dp.exclusions == RowExclusions(unreadable=1)
    assert dp.exclusions.message() == (
        "1 row is not shown: 1 row with a missing or non-numeric tube, position, date or value."
    )


def test_csv_without_error_column_still_classifies_names(tmp_path):
    path = tmp_path / "legacy.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(LENGTH_CSV_HEADERS[:6])
        writer.writerow(["CAM_T001_L010_2024.07.18_120000_001", 1, 10, "2024.07.18", "", 1.0])
        writer.writerow(["IMG_0001", "", "", "", "", 0])

    dp = DataProcessor(str(path))

    assert dp.exclusions == RowExclusions(bad_name=1)


def test_message_lists_every_cause_in_order():
    assert RowExclusions(failed=2, bad_name=1, unreadable=1).message() == (
        "4 rows are not shown: 2 failed measurements (see the CSV's Error column), "
        "1 image name not in the <field>_T<tube>_L<position>_YYYY.MM.DD_HHMMSS_... format "
        "and 1 row with a missing or non-numeric tube, position, date or value."
    )


@pytest.mark.parametrize("cell", ["inf", "-inf", "Infinity"])
@pytest.mark.parametrize("metric", METRICS)
def test_a_non_finite_value_is_left_out_as_unreadable(tmp_path, metric, cell):
    from app.data_processing.data_processor import DataProcessor
    from app.data_processing.data_processor_area import DataProcessorArea

    column, processor = {
        "length": (LENGTH_COLUMN, DataProcessor),
        "area": (AREA_COLUMN, DataProcessorArea),
    }[metric]
    path = tmp_path / "metric.csv"
    path.write_text(
        f"Image,Tube,Position,Date,Time,{column},Error\n"
        "A_T001_L005_2024.07.18_120000,1,5,2024.07.18,120000,2.5,\n"
        f"A_T002_L007_2024.07.18_120000,2,7,2024.07.18,120000,{cell},\n",
        encoding="utf-8",
    )

    loaded = processor(str(path))

    assert list(loaded.df["Tube"]) == [1]
    assert loaded.exclusions.unreadable == 1
    assert loaded.exclusions.message() == (
        "1 row is not shown: 1 row with a missing or non-numeric tube, position, date or value."
    )


@pytest.mark.parametrize("column", ["Tube", "Position"])
def test_a_non_finite_tube_or_position_is_left_out_as_unreadable(tmp_path, column):
    from app.data_processing.data_processor import DataProcessor

    cells = {"Tube": "2", "Position": "7", column: "inf"}
    path = tmp_path / "root_lengths.csv"
    path.write_text(
        f"Image,Tube,Position,Date,Time,{LENGTH_COLUMN},Error\n"
        "A_T001_L005_2024.07.18_120000,1,5,2024.07.18,120000,2.5,\n"
        f"A_T002_L007_2024.07.18_120000,{cells['Tube']},{cells['Position']},2024.07.18,"
        "120000,1.0,\n",
        encoding="utf-8",
    )

    loaded = DataProcessor(str(path))

    assert list(loaded.df["Tube"]) == [1]  # the file still loads: only the row is left out
    assert loaded.exclusions.unreadable == 1


# ---- headers and encodings ------------------------------------------------------------


def test_area_processor_maps_a_mangled_legacy_area_header(tmp_path):
    from app.data_processing.data_processor_area import DataProcessorArea

    path = tmp_path / "root_areas.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Tube", "Position", "Date", "Time", "Area (mm^2)", "Error"])
        writer.writerow(["CAM_T001_L010_2024.07.18_120000_001", 1, 10, "2024.07.18", "", 3.25, ""])

    dp = DataProcessorArea(str(path))

    assert dp.df["Area (mm²)"].tolist() == [3.25]


_HEADER = b"Image,Tube,Position,Date,Time,Length (mm),Error\n"
_MEASURED = "Feldé_T001_L005_2024.07.18_120000_001,1,5,2024.07.18,120000,1.5,\n"
_BAD_NAME_ROW = b"IMG_0002,,5,2024.07.18,120000,2.0,\n"


def test_a_legacy_csv_that_is_not_utf8_is_read_as_latin1(tmp_path):
    path = tmp_path / "root_lengths.csv"
    path.write_bytes(_HEADER + _MEASURED.encode("cp1252") + _BAD_NAME_ROW)

    dp = DataProcessor(str(path))

    assert dp.df["Length (mm)"].tolist() == [1.5]
    assert dp.exclusions == RowExclusions(bad_name=1)


def test_a_damaged_utf8_csv_keeps_its_header_and_names(tmp_path):
    # A NUL byte before a byte UTF-8 cannot decode: "utf-8-sig" refuses the file, the exact
    # "utf-8" attempt reads it (pandas' C parser stops each field at the NUL) and strips the
    # BOM. Read as latin-1, the header would start with "ï»¿" and the Image column be lost.
    path = tmp_path / "root_lengths.csv"
    damaged = b"IMG_0001,1,5,2024.07.18,120000,,x\x00\xff\n"
    path.write_bytes(b"\xef\xbb\xbf" + _HEADER + _MEASURED.encode() + damaged + _BAD_NAME_ROW)

    dp = DataProcessor(str(path))

    assert dp.df["Length (mm)"].tolist() == [1.5]
    assert dp.exclusions == RowExclusions(failed=1, bad_name=1)


def test_an_area_csv_as_measure_writes_it_charts_its_value(tmp_path):
    # The header holds "²" in UTF-8; the encoding fallback itself is
    # test_a_damaged_utf8_csv_keeps_its_header_and_names'.
    csv_path = tmp_path / "root_areas.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=AREA_CSV_HEADERS)
        w.writeheader()
        w.writerow(
            {
                "Image": "CAM_T001_L002_2024.01.01_000001_001",
                "Tube": 1,
                "Position": 2,
                "Date": "2024.01.01",
                "Time": "00:00:01",
                "Area (mm²)": 3.25,
                "Error": "",
            }
        )

    dp = DataProcessorArea(str(csv_path))

    assert dp.df["Area (mm²)"].tolist() == [3.25]
    assert dp.df["Tube"].tolist() == [1]
    assert dp.df["Position"].tolist() == [2]
    assert dp.df["Date"].tolist() == [pd.Timestamp("2024-01-01")]
    assert dp.exclusions == RowExclusions()


# ---- a CSV the dashboards cannot chart says why (V3) ------------------------------------


def test_a_csv_with_rows_has_no_empty_reason(tmp_path):
    csv_path = _write_table(
        tmp_path / "l.csv",
        LENGTH_CSV_HEADERS,
        [["CAM_T001_L010_2024.07.18_120000_001", 1, 10, "2024.07.18", "12:00:00", 1.5, ""]],
    )
    assert DataProcessor(csv_path).empty_reason() is None


def test_only_a_header_is_reported_as_no_rows(tmp_path):
    processor = DataProcessor(_write_table(tmp_path / "l.csv", LENGTH_CSV_HEADERS, []))
    assert processor.df.empty
    assert processor.empty_reason() == "The CSV has no measurement rows."


def test_an_empty_file_says_it_could_not_be_read(tmp_path):
    path = tmp_path / "root_lengths.csv"
    path.write_bytes(b"")

    dp = DataProcessor(str(path))

    assert dp.df.empty
    assert dp.empty_reason().startswith("The CSV could not be read: ")


def test_a_csv_whose_measurements_all_failed_says_so(tmp_path):
    failed = [
        [f"CAM_T001_L{i:03d}", "", "", "", "", 0, "cannot identify image file"] for i in range(3)
    ]
    processor = DataProcessor(_write_table(tmp_path / "l.csv", LENGTH_CSV_HEADERS, failed))

    reason = processor.empty_reason()

    assert processor.df.empty
    assert "3 failed measurements" in reason
    assert "Error column" in reason


def test_a_csv_without_the_metric_column_names_it(tmp_path):
    # e.g. a root_areas.csv copied over root_lengths.csv
    header = ["Image", "Tube", "Position", "Date", "Time", "Area (mm²)", "Error"]
    processor = DataProcessor(
        _write_table(tmp_path / "l.csv", header, [["x", 1, 1, "2024.07.18", "12:00:00", 1.0, ""]])
    )
    assert processor.df.empty
    assert processor.empty_reason() == "The CSV has no 'Length (mm)' column."


def test_a_csv_without_a_key_column_names_it(tmp_path):
    header = ["Image", "Tube", "Position", "Time", "Area (mm²)", "Error"]
    processor = DataProcessorArea(
        _write_table(tmp_path / "a.csv", header, [["x", 1, 1, "12:00:00", 1.0, ""]])
    )
    assert processor.df.empty
    assert processor.empty_reason() == "The CSV has no 'Date' column."


def test_a_missing_csv_gives_no_rows_and_says_it_could_not_be_read(tmp_path):
    missing = tmp_path / "root_lengths.csv"

    dp = DataProcessor(str(missing))

    assert dp.df.empty
    reason = dp.empty_reason()
    assert reason.startswith("The CSV could not be read: ")
    assert "root_lengths.csv" in reason

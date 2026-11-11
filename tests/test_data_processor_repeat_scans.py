"""A tube position imaged twice on one date counts once in the dashboards.

An operator who scans a tube again on the same day leaves two images of each position;
summed as they were, the tube's total for that date was about doubled. The dashboards'
frame (``MetricDataProcessor``) keeps the last image by capture time, then by name, of
each field, tube, position and date (a row without a time counts as earlier than every
row with one), counts the others as ``repeated`` and names them in the notice and in the
exports' header. The metric CSV keeps every row.
"""

from __future__ import annotations

import io
import os

import pandas as pd
import pytest

from app.data_processing.data_processor import DataProcessor, RowExclusions
from app.data_processing.data_processor_area import DataProcessorArea
from app.visualization import dash_export
from app.visualization import dash_metric_views as views
from app.visualization.dash_data_cache import DataCache
from app.visualization.dash_image_utils import ImageIndex
from app.visualization.dash_shell import exclusions_note

CAUSE = "repeated scans of a position on one date (the last image is kept)"
HEADER = "Image,Tube,Position,Date,Time,{column},Error"


def _write(path, rows, column="Length (mm)"):
    """``rows``: (image, tube, position, date, time, value[, error]) tuples."""
    lines = [HEADER.format(column=column)]
    for row in rows:
        image, tube, position, date, time, value, *error = row
        lines.append(f"{image},{tube},{position},{date},{time},{value},{''.join(error)}")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(lines) + "\n")
    return str(path)


def _name(field, tube, position, date, hhmmss):
    return f"{field}_T{tube:03d}_L{position:03d}_{date}_{hhmmss}_001_SMM"


def _row(field, tube, position, hhmmss, value, date="2024.07.18"):
    time = f"{hhmmss[:2]}:{hhmmss[2:4]}:{hhmmss[4:]}"
    return (_name(field, tube, position, date, hhmmss), tube, position, date, time, value)


def _untimed(row):
    """``row`` with its Time cell empty."""
    image, tube, position, date, _time, value = row
    return (image, tube, position, date, "", value)


# A tube scanned twice on 2024.07.18 (the second pass 20 minutes later, the other camera of
# the same field), and once on 2024.08.07. The second pass comes first in the CSV.
SECOND_PASS = [
    _row("24EFCAM2", 1, 5, "113000", 9.0),
    _row("24EFCAM2", 1, 6, "113002", 8.0),
]
FIRST_PASS = [
    _row("24EFCAM1", 1, 5, "111000", 1.5),
    _row("24EFCAM1", 1, 6, "111002", 2.5),
    _row("24EFCAM1", 1, 15, "111010", 0.5),
]
LATER = [_row("24EFCAM1", 1, 5, "101000", 4.0, date="2024.08.07")]
ROWS = SECOND_PASS + FIRST_PASS + LATER


def _kept(folder, rows):
    """The length processor over ``rows`` and the values of the rows it keeps."""
    dp = DataProcessor(_write(folder / "root_lengths.csv", rows))
    return dp, dp.df["Length (mm)"].tolist()


# ---- the frame -------------------------------------------------------------------


def test_the_last_image_by_capture_time_is_kept(tmp_path):
    dp = DataProcessor(_write(tmp_path / "root_lengths.csv", ROWS))

    assert dp.exclusions == RowExclusions(repeated=2)
    assert dp.dropped_rows == 2
    july = dp.df[dp.df["Date"] == pd.Timestamp("2024-07-18")]
    assert sorted(zip(july["Position"], july["Length (mm)"], strict=True)) == [
        (5, 9.0),
        (6, 8.0),
        (15, 0.5),
    ]
    assert len(dp.df) == 4


def test_of_three_captures_the_last_is_kept(tmp_path):
    rows = [
        _row("24EFCAM1", 1, 5, "112000", 2.0),
        _row("24EFCAM1", 1, 5, "113000", 3.0),
        _row("24EFCAM1", 1, 5, "111000", 1.0),
    ]
    dp, kept = _kept(tmp_path, rows)

    assert kept == [3.0]
    assert dp.exclusions == RowExclusions(repeated=2)


def test_the_kept_rows_keep_their_csv_order(tmp_path):
    # Every statistic is computed over the rows in CSV order (dash_data_cache).
    dp = DataProcessor(_write(tmp_path / "root_lengths.csv", ROWS))

    assert dp.df.index.tolist() == [0, 1, 4, 5]


def test_the_tube_date_total_counts_each_position_once(tmp_path):
    cache = DataCache(DataProcessor(_write(tmp_path / "root_lengths.csv", ROWS)))

    totals = cache.get_tube_totals(1)
    assert totals[pd.Timestamp("2024-07-18")] == pytest.approx(17.5)  # was 21.5
    assert totals[pd.Timestamp("2024-08-07")] == pytest.approx(4.0)
    table = cache.get_interval_data(1, pd.Timestamp("2024-07-18"), 10)
    assert table[10]["count"] == 2  # was 4
    assert table[10]["avg"] == pytest.approx(8.5)


def test_equal_times_fall_back_to_the_last_image_name(tmp_path):
    rows = [
        _row("24EFCAM2", 1, 5, "111000", 9.0),
        _row("24EFCAM1", 1, 5, "111000", 1.5),
    ]
    dp, kept = _kept(tmp_path, rows)

    assert kept == [9.0]  # 24EFCAM2_... sorts last
    assert dp.exclusions == RowExclusions(repeated=1)


def test_a_timed_capture_is_kept_over_an_untimed_one(tmp_path):
    # The untimed row has the later time in its name and the later name: it still loses.
    untimed = _untimed(_row("24EFCAM2", 1, 5, "120000", 9.0))
    dp, kept = _kept(tmp_path, [_row("24EFCAM1", 1, 5, "111000", 1.5), untimed])

    assert kept == [1.5]
    assert dp.exclusions == RowExclusions(repeated=1)


def test_untimed_rows_fall_back_to_the_last_name(tmp_path):
    rows = [
        _untimed(_row("24EFCAM2", 1, 5, "111000", 9.0)),
        _untimed(_row("24EFCAM1", 1, 5, "113000", 1.5)),
    ]
    _dp, kept = _kept(tmp_path, rows)

    assert kept == [9.0]


def test_a_csv_without_a_time_column_keeps_the_last_name(tmp_path):
    path = tmp_path / "root_lengths.csv"
    path.write_text(
        "Image,Tube,Position,Date,Length (mm)\n"
        f"{_name('24EFCAM2', 1, 5, '2024.07.18', '113000')},1,5,2024.07.18,9.0\n"
        f"{_name('24EFCAM1', 1, 5, '2024.07.18', '111000')},1,5,2024.07.18,1.5\n",
        encoding="utf-8",
    )
    dp = DataProcessor(str(path))

    assert dp.df["Length (mm)"].tolist() == [9.0]
    assert dp.exclusions == RowExclusions(repeated=1)


def test_two_fields_with_the_same_tube_number_keep_both_rows(tmp_path):
    rows = [
        _row("24EFCAM1", 2, 5, "121618", 3.0),
        _row("ROSCAM1", 2, 5, "131618", 7.0),
    ]
    dp, kept = _kept(tmp_path, rows)

    assert dp.exclusions == RowExclusions()
    assert kept == [3.0, 7.0]


def test_rows_whose_name_cannot_be_read_are_not_repeats(tmp_path):
    # A legacy row with its tube, position and date in their columns but not in its name.
    rows = [("IMG_0001", 1, 5, "2024.07.18", "", 9.0), _row("24EFCAM1", 1, 5, "111000", 1.5)]
    dp, kept = _kept(tmp_path, rows)

    assert dp.exclusions == RowExclusions()
    assert kept == [9.0, 1.5]


def test_left_out_rows_are_not_counted_as_repeats(tmp_path):
    failed = (_row("24EFCAM2", 1, 5, "113000", 0)[0], "", "", "", "", 0, "boom")
    dp, _kept_values = _kept(tmp_path, [failed, *FIRST_PASS])

    assert dp.exclusions == RowExclusions(failed=1)


def test_the_area_dashboard_follows_the_same_rule(tmp_path):
    dp = DataProcessorArea(_write(tmp_path / "root_areas.csv", ROWS, column="Area (mm²)"))

    assert dp.exclusions == RowExclusions(repeated=2)
    assert dp.df["Area (mm²)"].tolist() == [9.0, 8.0, 0.5, 4.0]


# ---- the notice ------------------------------------------------------------------


def test_the_notice_names_the_repeated_rows(tmp_path):
    dp = DataProcessor(_write(tmp_path / "root_lengths.csv", ROWS))

    assert exclusions_note(dp) == f"2 rows are not shown: 2 rows from {CAUSE}."


def test_the_notice_lists_repeats_after_the_other_causes():
    assert RowExclusions(failed=1, bad_name=1, unreadable=1, repeated=1).message() == (
        "4 rows are not shown: 1 failed measurement (see the CSV's Error column), "
        "1 image name not in the <field>_T<tube>_L<position>_YYYY.MM.DD_HHMMSS_... format, "
        "1 row with a missing or non-numeric tube, position, date or value "
        f"and 1 row from {CAUSE}."
    )


def test_the_page_shows_the_notice(tmp_path):
    from app.visualization.dash_app import DashApp
    from tests.viz_dash_helpers import FakeImageManager, component_ids, walk_components

    dp = DataProcessor(_write(tmp_path / "root_lengths.csv", ROWS))
    dash = DashApp(dp, str(tmp_path), FakeImageManager())

    assert "viz-data-notice" in component_ids(dash.app.layout)
    (notice,) = [
        node
        for node in walk_components(dash.app.layout)
        if getattr(node, "id", None) == "viz-data-notice"
    ]
    assert notice.children == f"2 rows are not shown: 2 rows from {CAUSE}."


# ---- the exports -----------------------------------------------------------------


def _header(text):
    return dict(line[2:].split(": ", 1) for line in text.splitlines() if line.startswith("# "))


@pytest.mark.parametrize(
    "view, args",
    [
        ("stacked", (None, [], [], [], [])),
        ("time", (None, [1], ["show_field_avg"], [], [])),
        ("lines", (1, [], [], [], [])),
        ("faceted", (None, [], [], [1], [])),
    ],
)
def test_every_export_counts_the_rows_used_and_left_out(tmp_path, view, args):
    from app.visualization.dash_app import DashApp
    from tests.viz_dash_helpers import FakeImageManager

    csv_path = _write(tmp_path / "root_lengths.csv", ROWS)
    dash = DashApp(DataProcessor(csv_path), str(tmp_path), FakeImageManager())
    table = views.export_table(dash, view, *args)
    text = dash_export.render_csv(
        table, source_csv=csv_path, metric_column="Length (mm)", view=view
    )

    header = _header(text)
    assert header["rows_used"] == "4 of 6"
    assert header["rows_left_out"] == f"2 rows from {CAUSE}"
    assert not pd.read_csv(io.StringIO(text), comment="#").empty


def test_the_stacked_export_sums_each_position_once(tmp_path):
    from app.visualization.dash_app import DashApp
    from tests.viz_dash_helpers import FakeImageManager

    csv_path = _write(tmp_path / "root_lengths.csv", ROWS)
    dash = DashApp(DataProcessor(csv_path), str(tmp_path), FakeImageManager())
    frame = views.export_table(dash, "stacked", None, [], [], [], []).frame

    assert frame["Total Length (mm)"].tolist() == pytest.approx([17.5, 4.0])


def test_an_export_without_repeats_has_no_row_count_lines(tmp_path):
    # Without repeated scans the export is the same file as before (byte for byte).
    from app.visualization.dash_app import DashApp
    from tests.viz_dash_helpers import FakeImageManager

    csv_path = _write(tmp_path / "root_lengths.csv", FIRST_PASS)
    dash = DashApp(DataProcessor(csv_path), str(tmp_path), FakeImageManager())
    table = views.export_table(dash, "stacked", None, [], [], [], [])
    lines = dash_export.provenance_lines(
        table, source_csv=csv_path, metric_column="Length (mm)", view="stacked"
    )

    assert not [line for line in lines if line.startswith("# rows_")]
    assert table.row_counts == {}


# ---- the photo hover ---------------------------------------------------------------


def _paths(names):
    return {name: os.path.join("C:/photos", f"{name}.jpg") for name in names}


def _shown(names):
    return os.path.basename(ImageIndex(_paths(names))[(1, 5, "2024.07.18")])


def test_the_hover_shows_the_kept_image():
    early = _name("24EFCAM1", 1, 5, "2024.07.18", "111000")
    middle = _name("24EFCAM2", 1, 5, "2024.07.18", "112000")
    late = _name("24EFCAM1", 1, 5, "2024.07.18", "113000")
    for names in ([late, early, middle], [early, middle, late], [middle, late, early]):
        assert _shown(names) == f"{late}.jpg"
        assert len(ImageIndex(_paths(names))) == 1


def test_the_hover_falls_back_to_the_last_name():
    first_name = _name("24EFCAM1", 1, 5, "2024.07.18", "111000")
    last_name = _name("24EFCAM2", 1, 5, "2024.07.18", "111000")

    assert _shown([last_name, first_name]) == f"{last_name}.jpg"
    assert _shown([first_name, last_name]) == f"{last_name}.jpg"


def test_the_hover_puts_a_name_without_a_time_before_the_timed_ones():
    timeless = "24EFCAM9_T001_L005_2024.07.18"  # no time token; the last name
    timed = _name("24EFCAM1", 1, 5, "2024.07.18", "111000")

    assert _shown([timed, timeless]) == f"{timed}.jpg"
    assert _shown([timeless, timed]) == f"{timed}.jpg"
    assert _shown([timeless]) == f"{timeless}.jpg"

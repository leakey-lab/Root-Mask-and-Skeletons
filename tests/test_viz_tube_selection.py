"""The Growth Over Time tube selection of both dashboards (``dash_metric_callbacks``
``manage_tube_selection`` / ``display_selected_tubes``, ``parse_tube_selection``).

Promises: the Multi-Select dropdown replaces the selection (an emptied one clears it); a typed
list or a From / To range adds only tubes the trial has, a span matched against them and never
expanded; Select All takes every tube with data, Clear All none; the typed entry is added on
Enter, not per keystroke; the chosen tubes are listed as chips (50 at most). Callbacks go
through Dash's real dispatch (Flask test client).
"""

from __future__ import annotations

import json
import tracemalloc

import pytest

from tests.viz_dash_helpers import build_length_app, call_callback


def test_removing_the_last_multiselect_tube_clears_the_selection(tmp_path):
    # Multi-Select replaces the selection with the dropdown's tubes; removing the
    # last one used to leave it plotted (and listed) under an empty dropdown.
    dashboard = build_length_app(tmp_path)

    _, out = call_callback(
        dashboard.app,
        "selected-tubes-store.data",
        {"tube-multiselect.value": [], "selected-tubes-store.data": [1]},
        changed=["tube-multiselect.value"],
    )

    assert out["selected-tubes-store"]["data"] == []


def test_multiselect_still_replaces_the_selection(tmp_path):
    dashboard = build_length_app(tmp_path)

    _, out = call_callback(
        dashboard.app,
        "selected-tubes-store.data",
        {"tube-multiselect.value": [2, 3], "selected-tubes-store.data": [1]},
        changed=["tube-multiselect.value"],
    )

    assert out["selected-tubes-store"]["data"] == [2, 3]


def test_typed_or_ranged_tubes_without_data_are_not_selected(tmp_path):
    # "2-9" on a trial with tubes 1-3 said "8 tubes selected" (and the CSV export's
    # provenance listed 8 tubes) while the chart could only plot two of them.
    dashboard = build_length_app(tmp_path)  # tubes 1, 2 and 3

    _, typed = call_callback(
        dashboard.app,
        "selected-tubes-store.data",
        {"tube-manual-entry.value": "2-9", "selected-tubes-store.data": []},
        changed=["tube-manual-entry.value"],
    )
    _, ranged = call_callback(
        dashboard.app,
        "selected-tubes-store.data",
        {
            "add-range-btn.n_clicks": 1,
            "tube-range-from.value": 1,
            "tube-range-to.value": 50,
            "selected-tubes-store.data": [],
        },
        changed=["add-range-btn.n_clicks"],
    )

    assert typed["selected-tubes-store"]["data"] == [2, 3]
    assert ranged["selected-tubes-store"]["data"] == [1, 2, 3]


@pytest.mark.parametrize(
    "values, trigger, expected",
    [
        ({"tube-manual-entry.value": "2-2000000"}, "tube-manual-entry.value", [2, 3]),
        (
            {"add-range-btn.n_clicks": 1, "tube-range-from.value": 1,
             "tube-range-to.value": 2_000_000},
            "add-range-btn.n_clicks",
            [1, 2, 3],
        ),
    ],
    ids=["typed", "ranged"],
)  # fmt: skip
def test_a_wide_tube_span_is_matched_against_the_trial_not_expanded(
    tmp_path, values, trigger, expected
):
    # A typo with extra digits ("1-100000000") listed every number of the span before
    # the trial's tubes were kept: gigabytes and seconds on the Dash server thread, which
    # holds the GIL, so the whole app froze. (Two million here: a regression fails this
    # test instead of exhausting the machine.)
    dashboard = build_length_app(tmp_path)  # tubes 1, 2 and 3

    # A trace that already runs (PYTHONTRACEMALLOC, a profiler) stays on; the peak is
    # counted above what is held when the measurement starts.
    tracing = tracemalloc.is_tracing()
    if tracing:
        tracemalloc.reset_peak()
    else:
        tracemalloc.start()
    held = tracemalloc.get_traced_memory()[0]
    try:
        _, out = call_callback(
            dashboard.app,
            "selected-tubes-store.data",
            {**values, "selected-tubes-store.data": []},
            changed=[trigger],
        )
        peak = tracemalloc.get_traced_memory()[1] - held
    finally:
        if not tracing:
            tracemalloc.stop()

    assert out["selected-tubes-store"]["data"] == expected
    assert peak < 16 * 2**20, f"{peak / 2**20:.0f} MiB at peak for a 2-million span"


@pytest.mark.parametrize(
    "low, high, expected",
    [(2.1, 2.9, [2]), (2.9, 2.1, []), (-5, 2, [1, 2]), (3, 3, [3]), (4, 9, [])],
)
def test_add_range_truncates_its_bounds_and_needs_from_up_to_to(tmp_path, low, high, expected):
    dashboard = build_length_app(tmp_path)  # tubes 1, 2 and 3

    _, out = call_callback(
        dashboard.app,
        "selected-tubes-store.data",
        {
            "add-range-btn.n_clicks": 1,
            "tube-range-from.value": low,
            "tube-range-to.value": high,
            "selected-tubes-store.data": [],
        },
        changed=["add-range-btn.n_clicks"],
    )

    assert out["selected-tubes-store"]["data"] == expected


def test_parse_tube_selection_keeps_only_the_given_tubes():
    from app.visualization.dash_visualizations import parse_tube_selection

    assert parse_tube_selection("3,1-2000000", within={2, 7, 500}) == [2, 7, 500]
    assert parse_tube_selection("7, 9, 400-600, x", within={2, 7, 500}) == [7, 500]
    assert parse_tube_selection("5-3", within={4}) == []
    # Without ``within``: every number the text names, as before.
    assert parse_tube_selection("1,3,5,10-12") == [1, 3, 5, 10, 11, 12]
    assert parse_tube_selection("1,foo,3") == [1, 3]  # F-037: a malformed part is skipped
    assert parse_tube_selection("  ") == []


def _selection(dashboard, trigger: str, current=None, **values) -> list:
    _, out = call_callback(
        dashboard.app,
        "selected-tubes-store.data",
        {trigger: 1, "selected-tubes-store.data": current, **values},
        changed=[trigger],
    )
    return out["selected-tubes-store"]["data"]


def test_select_all_takes_every_tube_with_data(tmp_path):
    dashboard = build_length_app(tmp_path)  # tubes 1, 2 and 3

    assert _selection(dashboard, "select-all-btn.n_clicks", current=[2]) == [1, 2, 3]


def test_clear_all_empties_the_selection(tmp_path):
    dashboard = build_length_app(tmp_path)

    assert _selection(dashboard, "clear-all-btn.n_clicks", current=[1, 3]) == []


def test_the_typed_entry_is_added_on_enter_not_per_keystroke(tmp_path):
    # Per keystroke, typing "12" first added tube 1 (selections accumulate).
    layout = json.loads(
        build_length_app(tmp_path).app.server.test_client().get("/_dash-layout").data
    )
    (entry,) = [n for n in _nodes(layout) if n.get("props", {}).get("id") == "tube-manual-entry"]

    assert entry["props"]["debounce"] is True


def _nodes(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _nodes(value)
    elif isinstance(node, list):
        for value in node:
            yield from _nodes(value)


@pytest.mark.parametrize(
    "count, chips, more", [(0, 0, None), (3, 3, None), (51, 50, "... and 1 more")]
)
def test_the_selected_tubes_are_listed_as_at_most_fifty_chips(tmp_path, count, chips, more):
    dashboard = build_length_app(tmp_path)
    _, out = call_callback(
        dashboard.app,
        "selected-tubes-display.children",
        {"selected-tubes-store.data": list(range(1, count + 1))},
        changed=["selected-tubes-store.data"],
    )
    shown = out["selected-tubes-display"]["children"]
    texts = [
        n["props"]["children"] for n in _nodes(shown) if "props" in n and "className" in n["props"]
    ]

    if count == 0:
        assert texts == ["No tubes selected"]
        return
    badges = [t for t in texts if isinstance(t, str) and t.startswith("Tube ")]
    assert badges == [f"Tube {t}" for t in range(1, chips + 1)]
    assert (more in texts) if more else not any(str(t).startswith("...") for t in texts)


def test_the_grid_selection_is_stored_sorted_and_its_availability_listed(tmp_path):
    dashboard = build_length_app(tmp_path)  # tube 3 has data on 2024-08-07 only
    _, out = call_callback(
        dashboard.app,
        "faceted-selection-info.children",
        {"faceted-tube-selector.value": [3, 1],
         "faceted-date-selector.value": ["2024-08-07T00:00:00", "2024-07-18T00:00:00"]},
        changed=["faceted-tube-selector.value"],
    )  # fmt: skip

    assert out["faceted-tubes-store"]["data"] == [1, 3]
    assert [d[:10] for d in out["faceted-dates-store"]["data"]] == ["2024-07-18", "2024-08-07"]
    table = [n for n in _nodes(out["tube-date-availability"]["children"]) if n.get("type") == "Tr"]
    cells = [[c["props"]["children"] for c in row["props"]["children"]] for row in table]
    assert cells == [["Tube/Date", "07/18", "08/07"], ["T1", "✓", "✓"], ["T3", "✗", "✓"]]

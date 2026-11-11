"""The dashboard page's exclusions notice (FR-15).

What is left of an integration pass's file: its other tests moved to their units' homes
(test-suite review 2026-11-10). This one goes to the dashboard shell's tests in wave 2.
"""

from __future__ import annotations


# --------------------------------------------------------------------------- FR-15
def _find(component, component_id):
    if getattr(component, "id", None) == component_id:
        return component
    children = getattr(component, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            found = _find(child, component_id)
            if found is not None:
                return found
    elif children is not None and not isinstance(children, str):
        return _find(children, component_id)
    return None


def test_dashboard_page_shows_the_exclusions_notice_only_when_rows_were_left_out():
    from app.data_processing.data_processor import RowExclusions
    from app.visualization.dash_metric_layout import LENGTH_SHELL
    from app.visualization.dash_shell import DashboardShell, exclusions_note

    class Processor:
        exclusions = RowExclusions(failed=2, bad_name=0, unreadable=0)

    note = exclusions_note(Processor())
    assert note and note.startswith("2 rows are not shown")
    assert exclusions_note(object()) is None

    shell = DashboardShell(LENGTH_SHELL)
    with_note = shell.page(controls=[], chart=[], notice=note)
    assert _find(with_note, "viz-data-notice").children == note
    assert _find(shell.page(controls=[], chart=[]), "viz-data-notice") is None

"""Bar value labels take plotly.js 4's text colour (owner decision #51, 2026-10-31).

plotly.js 4 picks the colour of a label inside a bar by WCAG contrast instead of
tinycolor's brightness, so labels on some mid-blue date colours turn from dark grey
(``#444``) to white. The owner chose plotly.js 4's white labels: ``theme.bar_labels``
sets no text colour (it kept plotly.js 3's dark grey on those bars from 2309d67 until
then). The labels that change are on the fourth ``DATE_RAMP`` stop, ``#4f97d1``, and the
colours blended next to it: trials with 5, 8-10 or 12 or more dates.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.config import AREA_CSV_HEADERS, LENGTH_CSV_HEADERS
from app.data_processing.data_processor import DataProcessor
from app.data_processing.data_processor_area import DataProcessorArea
from app.visualization import theme
from app.visualization.dash_app_area import DashAppArea
from app.visualization.dash_data_cache import DataCache
from app.visualization.dash_visualizations import stacked_bar_figure
from tests.test_viz_data_cache import write_metric_csv

BUNDLE = Path(__file__).resolve().parents[1] / "app" / "assets" / "plotly-basic.min.js"


def _rgb(colour):
    return tuple(int(colour[i : i + 2], 16) for i in (1, 3, 5))


def _plotly3_pick(colour):  # tinycolor isDark: brightness < 128
    r, g, b = _rgb(colour)
    return "#fff" if (r * 299 + g * 587 + b * 114) / 1000 < 128 else "#444"


def _luminance(colour):
    def linear(v):
        v /= 255
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = (linear(v) for v in _rgb(colour))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _plotly4_pick(colour):  # Color.isDark of plotly.js 4.1.1 (culori wcagContrast)
    def ratio(a, b):
        hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
        return (hi + 0.05) / (lo + 0.05)

    return "#fff" if ratio(colour, "#ffffff") > ratio(colour, "#444444") else "#444"


def test_bar_labels_set_no_text_colour():
    assert theme.bar_labels(["1.00"], 1) == {"text": ["1.00"], "textposition": "auto"}
    assert theme.bar_labels(["1.00"], theme.MAX_LABELLED_BARS + 1) == {"textposition": "none"}
    assert not hasattr(theme, "inside_label_colour")  # the plotly.js 3 override is gone


def _five_date_csv(tmp_path, headers):
    dates = ("2024.06.03", "2024.06.24", "2024.07.16", "2024.08.07", "2024.08.28")
    rows = [(tube, 15, date, 1.5 + tube + d) for tube in (1, 2) for d, date in enumerate(dates)]
    return write_metric_csv(tmp_path / "metric.csv", headers, rows)


def test_the_dashboards_bars_leave_the_label_colour_to_plotly_js(tmp_path):
    # Five dates: the second one is #4f97d1, whose labels plotly.js 4 draws white.
    cache = DataCache(DataProcessor(_five_date_csv(tmp_path, LENGTH_CSV_HEADERS)))
    area = DashAppArea(DataProcessorArea(_five_date_csv(tmp_path, AREA_CSV_HEADERS)), "")
    for figure in (stacked_bar_figure(cache, "Length", "mm"), area.create_stacked_bar_chart()):
        traces = figure.to_plotly_json()["data"]
        assert traces[1]["marker"]["color"] == "#4f97d1"
        for trace in traces:
            assert trace["textposition"] == "auto" and trace["text"] is not None
            assert "insidetextfont" not in trace and "textfont" not in trace


def test_the_bundled_plotly_js_picks_label_colours_by_wcag_contrast():
    # What the reference picks above model: a label inside a bar without a text colour
    # of its own is Color.contrast(bar colour), which is the background (#fff) on a dark
    # bar and the default line colour (#444) on a light one, and isDark compares WCAG
    # ratios against those two.
    bundle = BUNDLE.read_text(encoding="utf-8")
    assert "plotly.js (basic - minified) v4.1.1" in bundle[:200]
    is_dark = re.search(r"isDark:([\w$]+)", bundle)[1]
    wcag = re.search(r"wcagContrast:([\w$]+)\}", bundle)[1]
    background, line = re.search(r"\{background:([\w$]+),defaultLine:([\w$]+),", bundle).groups()
    assert f"{is_dark}=e=>{wcag}(e,{background})>{wcag}(e,{line})" in bundle
    assert re.search(
        r'defaultLine="#444";[\w$]+\.lightLine="#eee";[\w$]+\.background="#fff"', bundle
    )
    contrast = re.search(r"contrast:([\w$]+),", bundle)[1]
    names = {k: re.escape(v) for k, v in dict(c=contrast, d=is_dark, b=background, ln=line).items()}
    assert re.search(  # contrast(colour) with no shift: dark -> background, else defaultLine
        rf"([\w$]+)=[\w$]+\([\w$]+\({names['b']}\)\),([\w$]+)=[\w$]+\([\w$]+\({names['ln']}\)\),"
        rf"{names['c']}=\(e,r,t\)=>[^;]*?{names['d']}\(e\)\?r\?[\w$]+\(e,r\):\1:"
        rf"t\?[\w$]+\(e,-t\):\2\)",
        bundle,
    )
    # The bar trace's inside-label rule (bar/style.js determineInsideTextFont); the pie
    # trace has a look-alike rule of its own, which this pattern does not match.
    bar_rule = (
        r"_input\.textfont===void 0\|\|[\w$]+\._input\.textfont\.color===void 0[^;]*;"
        r"return [\w$]+&&\([\w$]+=\{color:[\w$]+\.contrast\([\w$]+\),"
    )
    assert len(re.findall(bar_rule, bundle)) == 1


def test_exactly_the_labels_on_the_fourth_ramp_blue_and_its_blends_turn_white():
    changed = {}
    for dates in range(1, 61):
        for number, colour in enumerate(theme.date_colors(dates), start=1):
            if _plotly3_pick(colour) != _plotly4_pick(colour):
                assert _plotly4_pick(colour) == "#fff", colour  # dark grey to white only
                changed.setdefault(dates, []).append((number, colour))

    assert sorted(changed) == [5, 8, 9, 10, *range(12, 61)]
    assert {d: changed[d] for d in (5, 8, 9, 10, 12, 13)} == {
        5: [(2, "#4f97d1")],
        8: [(3, "#4f97d1")],
        9: [(3, "#4f97d1")],
        10: [(3, "#4f97d1")],
        12: [(4, "#4f97d1")],
        13: [(4, "#4f97d1")],
    }
    blends = {colour for found in changed.values() for _, colour in found}
    assert all("#4490cd" <= colour <= "#4f97d1" for colour in blends)  # next to #4f97d1


def test_the_owners_eleven_dates_keep_their_label_colours():
    # The owner's trial has 11 dates: no label changes colour there.
    assert all(_plotly3_pick(c) == _plotly4_pick(c) for c in theme.date_colors(11))

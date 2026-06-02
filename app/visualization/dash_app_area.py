"""Dash application for root area visualization (F-014 hardened)."""
import logging
import os

from dash import Dash, dcc, html, Input, Output
import dash
import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd

from app.config import DASH_AREA_PORT
from . import theme
theme.use("sprouts")

logger = logging.getLogger(__name__)

# Shared graph-container style to avoid duplication (F-087).
_GRAPH_STYLE = {
    "borderRadius": "8px",
    "padding": "20px",
    "height": "800px",
    "width": "100%",
    "marginBottom": "20px",
}


class DashAppArea:
    """Manages the Dash application for root area visualization."""

    def __init__(self, data_processor, save_directory):
        self.data_processor = data_processor
        self.save_directory = save_directory

        self.app = Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY],
            assets_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets"),
            suppress_callback_exceptions=True,
            update_title=None,
            compress=True,
        )
        self.app.index_string = '''<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
{%favicon%}{%css%}</head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>'''

        self._setup_layout()
        self._setup_callbacks()

    def _setup_layout(self):
        """Define the Dash app layout."""
        self.app.layout = html.Div(
            className="sprouts-app",
            style={
                "display": "flex",
                "flexDirection": "column",
                "alignItems": "center",
                "minHeight": "100vh",
                "width": "100%",
                "padding": "20px",
            },
            children=[
                # Title
                html.H1(
                    "Root Area Visualization",
                    className="sprouts-title",
                    style={
                        "textAlign": "center",
                        "width": "100%",
                        "marginBottom": "30px",
                        "fontSize": "36px",
                    },
                ),
                # Container
                dbc.Container(
                    [
                        # Controls
                        dbc.Row(
                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.H4(
                                                "View Options",
                                                className="card-title text-center",
                                                style={"fontSize": "24px", "marginBottom": "20px"},
                                            ),
                                            dbc.RadioItems(
                                                id="view-selector",
                                                options=[
                                                    {
                                                        "label": "Stacked Bar View",
                                                        "value": "stacked",
                                                    },
                                                    {
                                                        "label": "Growth Over Time",
                                                        "value": "time",
                                                    },
                                                    {
                                                        "label": "Area Profile by Position",
                                                        "value": "profile",
                                                    },
                                                ],
                                                value="stacked",
                                                class_name="btn-group",
                                                input_class_name="btn-check",
                                                label_class_name="btn btn-outline-secondary",
                                            ),
                                            dcc.Dropdown(
                                                id="tube-selector",
                                                placeholder="Select Tube",
                                                className="mb-3",
                                                style={"display": "none", "fontSize": "16px"},
                                            ),
                                        ]
                                    ),
                                    style={"marginBottom": "30px", "padding": "20px"},
                                ),
                                md=12,
                                className="mx-auto",
                            ),
                            className="mb-4 justify-content-center",
                        ),
                        # Graph
                        dbc.Row(
                            dbc.Col(
                                [
                                    dcc.Loading(
                                        type="default",
                                        color="#5fd6a0",
                                        children=dcc.Graph(
                                            id="main-graph",
                                            className="sprouts-graph-card",
                                            config={
                                                "scrollZoom": True,
                                                "doubleClick": "reset",
                                                "showTips": False,
                                                "displayModeBar": True,
                                                "watermark": False,
                                                "responsive": True,
                                                "autosizable": True,
                                                "toImageButtonOptions": {
                                                    "format": "svg",
                                                    "filename": "root_area_plot",
                                                    "scale": 2,
                                                },
                                            },
                                        ),
                                    ),
                                    html.Div(
                                        id="click-data",
                                        className="text-center my-3",
                                        style={"fontSize": "18px", "padding": "10px"},
                                    ),
                                ],
                                className="d-flex flex-column align-items-center",
                            )
                        ),
                        # Hidden store for caching
                        dcc.Store(id="cached-data"),
                    ],
                    fluid=True,
                    className="px-4",
                ),
            ],
        )

    def _setup_callbacks(self):
        """Define Dash callbacks."""
        @self.app.callback(
            [
                Output("main-graph", "figure"),
                Output("click-data", "children"),
                Output("tube-selector", "style"),
                Output("tube-selector", "options"),
                Output("main-graph", "style"),
            ],
            [
                Input("view-selector", "value"),
                Input("tube-selector", "value"),
            ],
        )
        def update_visualization(view_type, selected_tube):
            ctx = dash.callback_context
            if not ctx.triggered:
                # Initial load - show stacked view by default
                return (
                    self.create_stacked_bar_chart(),
                    "",
                    {"display": "none"},
                    [],
                    dict(_GRAPH_STYLE, overflow="hidden"),
                )

            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

            try:
                if view_type == "stacked":
                    return (
                        self.create_stacked_bar_chart(),
                        "",
                        {"display": "none"},
                        [],
                        _GRAPH_STYLE,
                    )

                elif view_type == "profile":
                    tubes = self.data_processor.get_unique_tubes()
                    tube_options = [
                        {"label": f"Tube {tube}", "value": tube}
                        for tube in tubes
                    ]

                    if trigger_id == "view-selector":
                        if not tubes:
                            logger.warning("profile view: no tubes available")
                            return dash.no_update
                        first_tube = tubes[0]
                        return (
                            self.show_area_profile(first_tube),
                            "",
                            {"display": "block"},
                            tube_options,
                            _GRAPH_STYLE,
                        )

                    if trigger_id == "tube-selector" and selected_tube:
                        return (
                            self.show_area_profile(selected_tube),
                            "",
                            {"display": "block"},
                            tube_options,
                            _GRAPH_STYLE,
                        )

                elif view_type == "time":
                    return (
                        self.show_growth_over_time(),
                        "",
                        {"display": "none"},
                        [],
                        _GRAPH_STYLE,
                    )

            except (KeyError, IndexError, ValueError) as exc:
                logger.error("update_visualization(%s): %s", view_type, exc, exc_info=True)
                return dash.no_update
            except Exception as exc:  # noqa: BLE001 — last-resort guard
                logger.exception("update_visualization(%s): unexpected error", view_type)
                return dash.no_update

            return dash.no_update

    def create_stacked_bar_chart(self):
        """Create a stacked bar chart showing root area by tube and date."""
        df = self.data_processor.df

        try:
            # Group data by tube and date
            grouped = df.groupby(["Tube", "Date"])["Area (mm²)"].sum().reset_index()

            # Sort dates chronologically
            grouped["Date"] = pd.to_datetime(grouped["Date"])
            dates = sorted(grouped["Date"].unique())

            # Prepare data for plotting
            tubes = sorted(grouped["Tube"].unique())
            traces = []

            # Pivot once (tube x date) to avoid O(dates*tubes) per-cell filtering.
            pivot = (
                grouped.pivot(index="Tube", columns="Date", values="Area (mm²)")
                .reindex(index=tubes)
                .fillna(0)
            )

            # Create a bar trace for each date
            for date in dates:
                trace_data = pivot[date].to_numpy()

                traces.append(
                    go.Bar(
                        name=date.strftime("%Y-%m-%d"),
                        x=[f"Tube {int(tube)}" for tube in tubes],
                        y=trace_data,
                        text=[f"{v:.2f}" for v in trace_data],
                        textposition="auto",
                        hovertemplate=theme.hover("%{x}", [
                            ("Date", date.strftime("%Y-%m-%d")),
                            ("Area", "%{y:.2f} mm²"),
                        ]),
                    )
                )

            fig = go.Figure(data=traces)

            fig.update_layout(
                barmode="stack",
                title_text="Root Area Growth by Tube and Date",
                xaxis_title_text="Tube",
                yaxis_title_text="Total Area (mm²)",
                legend_title_text="Measurement Date",
                hovermode="x unified",
            )

            return fig

        except (KeyError, ValueError) as exc:
            logger.error("create_stacked_bar_chart: %s", exc, exc_info=True)
            return go.Figure()
        except Exception as exc:  # noqa: BLE001
            logger.exception("create_stacked_bar_chart: unexpected error")
            return go.Figure()

    def show_area_profile(self, selected_tube):
        """Generate area profile by position for a selected tube."""
        df = self.data_processor.df
        fig = go.Figure()

        try:
            tube_data = df[df["Tube"] == selected_tube].copy()
            dates = sorted(tube_data["Date"].unique())

            colors = px.colors.qualitative.Plotly

            for i, date in enumerate(dates):
                date_data = tube_data[tube_data["Date"] == date]
                position_data = (
                    date_data.groupby("Position")["Area (mm²)"].mean().reset_index()
                )

                if not position_data.empty:
                    fig.add_trace(
                        go.Bar(
                            x=[f"L{int(pos)}" for pos in position_data["Position"]],
                            y=position_data["Area (mm²)"],
                            name=date.strftime("%Y-%m-%d"),
                            marker_color=colors[i % len(colors)],
                            text=position_data["Area (mm²)"].round(2),
                            textposition="auto",
                            hovertemplate=theme.hover("Position: %{x}", [
                                ("Area", "%{y:.2f} mm²"),
                                ("Date", date.strftime("%Y-%m-%d")),
                            ]),
                        )
                    )

            fig.update_layout(
                title_text=f"Root Area Profile - Tube {int(selected_tube)}",
                xaxis_title_text="Position",
                yaxis_title_text="Root Area (mm²)",
                legend_title_text="Measurement Dates",
                barmode="group",
            )
        except (KeyError, ValueError) as exc:
            logger.error("show_area_profile(tube=%s): %s", selected_tube, exc, exc_info=True)
        except Exception as exc:  # noqa: BLE001
            logger.exception("show_area_profile(tube=%s): unexpected error", selected_tube)

        return fig

    def show_growth_over_time(self):
        """Generate growth over time figure."""
        df = self.data_processor.df
        fig = go.Figure()
        try:
            for tube in self.data_processor.get_unique_tubes():
                tube_data = df[df["Tube"] == tube].sort_values("Date")
                grouped = tube_data.groupby("Date")["Area (mm²)"].sum().reset_index()
                fig.add_trace(
                    go.Scatter(
                        x=grouped["Date"],
                        y=grouped["Area (mm²)"],
                        mode="lines+markers",
                        name=f"Tube {int(tube)}",
                        hovertemplate=theme.hover("Tube %{fullData.name}", [
                            ("Date", "%{x|%Y-%m-%d}"),
                            ("Total Area", "%{y:.2f} mm²"),
                        ]),
                    )
                )

            fig.update_layout(
                title_text="Root Area Growth Over Time",
                xaxis_title_text="Date",
                yaxis_title_text="Total Area (mm²)",
            )
            return fig
        except (KeyError, ValueError) as exc:
            logger.error("show_growth_over_time: %s", exc, exc_info=True)
            return go.Figure()
        except Exception as exc:  # noqa: BLE001
            logger.exception("show_growth_over_time: unexpected error")
            return go.Figure()

    def run_server(self):
        """Run the Dash server (legacy entry point — use DashServerThreadArea in production)."""
        try:
            self.app.run(debug=False, port=DASH_AREA_PORT, host="127.0.0.1", threaded=True)
        except OSError as exc:
            logger.error("run_server: port %d unavailable: %s", DASH_AREA_PORT, exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception("run_server: unexpected error")

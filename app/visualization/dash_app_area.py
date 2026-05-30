"""Dash application for root area visualization (F-014 hardened)."""
import logging

from dash import Dash, dcc, html, Input, Output
import dash
import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd

from app.config import DASH_AREA_PORT

logger = logging.getLogger(__name__)

# Shared graph-container style to avoid duplication (F-087).
_GRAPH_STYLE = {
    "backgroundColor": "white",
    "borderRadius": "8px",
    "padding": "20px",
    "boxShadow": "0 4px 8px rgba(0,0,0,0.15)",
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
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True,
            update_title=None,
        )

        self._setup_layout()
        self._setup_callbacks()

    def _setup_layout(self):
        """Define the Dash app layout."""
        self.app.layout = html.Div(
            style={
                "display": "flex",
                "flexDirection": "column",
                "alignItems": "center",
                "minHeight": "100vh",
                "width": "100%",
                "backgroundColor": "#f8f9fa",
                "padding": "20px",
            },
            children=[
                # Title
                html.H1(
                    "Root Area Visualization",
                    style={
                        "textAlign": "center",
                        "width": "100%",
                        "marginBottom": "30px",
                        "color": "#2c3e50",
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
                                            dcc.Dropdown(
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
                                                className="mb-3",
                                                style={"fontSize": "16px"},
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
                                    dcc.Graph(
                                        id="main-graph",
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
                                    html.Div(
                                        id="click-data",
                                        className="text-center my-3",
                                        style={"color": "#2c3e50", "fontSize": "18px", "padding": "10px"},
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

            # Create a bar trace for each date
            for date in dates:
                date_data = grouped[grouped["Date"] == date]
                trace_data = []

                for tube in tubes:
                    value = date_data[date_data["Tube"] == tube]["Area (mm²)"].values
                    trace_data.append(value[0] if len(value) > 0 else 0)

                traces.append(
                    go.Bar(
                        name=date.strftime("%Y-%m-%d"),
                        x=[f"Tube {int(tube)}" for tube in tubes],
                        y=trace_data,
                        text=[f"{v:.2f}" for v in trace_data],
                        textposition="auto",
                        hovertemplate="<b>%{x}</b><br>"
                        + "Date: "
                        + date.strftime("%Y-%m-%d")
                        + "<br>"
                        + "Area: %{y:.2f} mm²<br>"
                        + "<extra></extra>",
                    )
                )

            fig = go.Figure(data=traces)

            # Update layout for stacked bars
            fig.update_layout(
                barmode="stack",
                title={
                    "text": "Root Area Growth by Tube and Date",
                    "x": 0.5,
                    "xanchor": "center",
                    "font": {"size": 24},
                },
                xaxis=dict(
                    title="Tube",
                    titlefont=dict(size=18),
                    tickfont=dict(size=14)
                ),
                yaxis=dict(
                    title="Total Area (mm²)",
                    titlefont=dict(size=18),
                    tickfont=dict(size=14)
                ),
                legend_title=dict(text="Measurement Date", font=dict(size=18)),
                autosize=True,
                showlegend=True,
                hovermode="x unified",
                plot_bgcolor="white",
                bargap=0.2,
                bargroupgap=0.1,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.02,
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="black",
                    borderwidth=1,
                    font=dict(size=14),
                ),
                margin=dict(l=60, r=150, t=60, b=40),
            )

            # Update axes
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor="lightgray",
                showline=True,
                linewidth=2,
                linecolor="black",
            )

            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor="lightgray",
                showline=True,
                linewidth=2,
                linecolor="black",
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
                            hovertemplate="<b>Position: %{x}</b><br>"
                            + "Area: %{y:.2f} mm²<br>"
                            + "Date: "
                            + date.strftime("%Y-%m-%d")
                            + "<extra></extra>",
                        )
                    )

            fig.update_layout(
                title=dict(
                    text=f"Root Area Profile - Tube {int(selected_tube)}",
                    x=0.5,
                    xanchor="center",
                    font=dict(size=24),
                ),
                xaxis=dict(
                    title="Position",
                    showgrid=True,
                    gridcolor="lightgray",
                    titlefont=dict(size=18),
                    tickfont=dict(size=14),
                ),
                yaxis=dict(
                    title="Root Area (mm²)",
                    showgrid=True,
                    gridcolor="lightgray",
                    titlefont=dict(size=18),
                    tickfont=dict(size=14),
                ),
                plot_bgcolor="white",
                legend=dict(
                    title=dict(text="Measurement Dates", font=dict(size=18)),
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.02,
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="black",
                    borderwidth=1,
                    font=dict(size=14),
                ),
                barmode="group",
                autosize=True,
                margin=dict(t=60, b=40, l=60, r=100),
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
                        hovertemplate="<b>Tube %{fullData.name}</b><br>"
                        + "Date: %{x|%Y-%m-%d}<br>"
                        + "Total Area: %{y:.2f} mm²<br>"
                        + "<extra></extra>",
                    )
                )

            fig.update_layout(
                title=dict(
                    text="Root Area Growth Over Time",
                    font=dict(size=24)
                ),
                xaxis=dict(
                    title="Date",
                    titlefont=dict(size=18),
                    tickfont=dict(size=14)
                ),
                yaxis=dict(
                    title="Total Area (mm²)",
                    titlefont=dict(size=18),
                    tickfont=dict(size=14)
                ),
                showlegend=True,
                autosize=True,
                margin=dict(l=60, r=100, t=60, b=40),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="white",
                legend=dict(font=dict(size=14)),
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

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash
import dash_bootstrap_components as dbc
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QUrl
from threading import Thread
import plotly.express as px
from functools import lru_cache
import numpy as np


class RootLengthVisualization(QMainWindow):
    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path
        self.df = self._prepare_data(csv_path)

        self.app = Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            update_title=None,
            suppress_callback_exceptions=True,
        )
        self.setup_dash_layout()
        self.setup_dash_callbacks()
        self.setup_pyqt_ui()

    @staticmethod
    def _prepare_data(csv_path):
        """Optimized data preparation with better data types"""
        df = pd.read_csv(csv_path)

        # Convert Date to datetime more efficiently
        df["Date"] = pd.to_datetime(df["Date"], format="%Y.%m.%d", cache=True)

        # Convert numeric columns to appropriate types
        df["Tube"] = pd.to_numeric(df["Tube"], downcast="integer")
        df["Position"] = pd.to_numeric(df["Position"], downcast="integer")
        df["Length (mm)"] = pd.to_numeric(df["Length (mm)"], downcast="float")

        # Pre-compute identifiers
        df["tube_date"] = df.apply(
            lambda x: f"Tube {x['Tube']} ({x['Date'].strftime('%Y-%m-%d')})", axis=1
        )
        df["tube_position"] = df.apply(
            lambda x: f"Tube {x['Tube']}_L{x['Position']}", axis=1
        )

        return df

    def setup_dash_layout(self):
        # Updated layout with centered content
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
                # Title Section
                html.H1(
                    "Root Length Visualization",
                    style={
                        "textAlign": "center",
                        "width": "100%",
                        "marginBottom": "30px",
                        "color": "#2c3e50",
                    },
                ),
                # Main Content Container
                dbc.Container(
                    [
                        # Controls Row
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardBody(
                                                    [
                                                        html.H4(
                                                            "View Options",
                                                            className="card-title text-center",
                                                        ),
                                                        dcc.Dropdown(
                                                            id="view-selector",
                                                            options=[
                                                                {
                                                                    "label": "Combined View",
                                                                    "value": "combined",
                                                                },
                                                                {
                                                                    "label": "Separate by Date",
                                                                    "value": "separate",
                                                                },
                                                                {
                                                                    "label": "Growth Over Time",
                                                                    "value": "time",
                                                                },
                                                                {
                                                                    "label": "Growth Lines",
                                                                    "value": "lines",
                                                                },
                                                            ],
                                                            value="separate",
                                                            className="mb-3",
                                                        ),
                                                        dcc.Dropdown(
                                                            id="tube-selector",
                                                            placeholder="Select Tube",
                                                            className="mb-3",
                                                            style={"display": "none"},
                                                        ),
                                                    ]
                                                )
                                            ],
                                            className="mb-4 shadow-sm",
                                            style={"backgroundColor": "white"},
                                        )
                                    ],
                                    md=8,
                                    className="mx-auto",
                                )
                            ],
                            className="mb-4 justify-content-center",
                        ),
                        # Graph Row
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Div(
                                            [
                                                dcc.Graph(
                                                    id="main-graph",
                                                    config={
                                                        "scrollZoom": True,
                                                        "doubleClick": "reset",
                                                        "showTips": False,
                                                        "displayModeBar": True,
                                                        "watermark": False,
                                                    },
                                                    style={
                                                        "backgroundColor": "white",
                                                        "borderRadius": "8px",
                                                        "padding": "15px",
                                                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                                                    },
                                                ),
                                            ],
                                            style={
                                                "width": "100%",
                                                "maxWidth": "1200px",
                                            },
                                        ),
                                        html.Div(
                                            id="click-data",
                                            className="text-center my-3",
                                            style={"color": "#2c3e50"},
                                        ),
                                        dbc.Button(
                                            "Back",
                                            id="back-button",
                                            className="mt-3 d-none",
                                            style={
                                                "backgroundColor": "#2c3e50",
                                                "borderColor": "#2c3e50",
                                            },
                                        ),
                                    ],
                                    className="d-flex flex-column align-items-center",
                                )
                            ],
                            className="justify-content-center",
                        ),
                    ],
                    fluid=True,
                    className="px-4",
                ),
            ],
        )

    def update_graph_layout(self, fig):
        # Update the layout of any graph to ensure it's centered and properly sized
        fig.update_layout(
            autosize=True,
            margin=dict(l=50, r=50, t=50, b=50),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="white",
            height=700,  # Fixed height for consistency
            width=1000,  # Fixed width for consistency
        )
        return fig

    @lru_cache(maxsize=32)
    def get_cached_overview_data(self, view_type, date_str=None):
        """Cache overview calculations"""
        if view_type == "separate":
            return self.df.groupby("tube_date")["Length (mm)"].sum().reset_index()
        return self.df.groupby("Tube")["Length (mm)"].mean().reset_index()

    @lru_cache(maxsize=32)
    def get_cached_section_data(self, tube, date_str=None):
        """Cache section data calculations"""
        if date_str:
            date = pd.to_datetime(date_str)
            section_data = self.df[
                (self.df["Tube"] == tube) & (self.df["Date"] == date)
            ]
        else:
            section_data = self.df[self.df["Tube"] == tube]
        return section_data.groupby("Position")["Length (mm)"].mean().reset_index()

    def setup_dash_callbacks(self):
        @self.app.callback(
            [
                Output("main-graph", "figure"),
                Output("click-data", "children"),
                Output("back-button", "className"),
                Output("tube-selector", "style"),
                Output("tube-selector", "options"),
            ],
            [
                Input("view-selector", "value"),
                Input("tube-selector", "value"),
                Input("main-graph", "clickData"),
                Input("back-button", "n_clicks"),
            ],
            [State("main-graph", "figure")],
        )
        def update_visualization(
            view_type, selected_tube, click_data, n_clicks, current_figure
        ):
            ctx = dash.callback_context
            if not ctx.triggered:
                return (
                    self.show_overview(view_type),
                    "",
                    "mt-2 d-none",
                    {"display": "none"},
                    [],
                )

            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

            # Optimize lines view handling
            if view_type == "lines":
                tube_options = [
                    {"label": f"Tube {tube}", "value": tube}
                    for tube in sorted(self.df["Tube"].unique())
                ]

                if trigger_id == "view-selector":
                    return (
                        self.show_growth_lines(tube_options[0]["value"]),
                        "",
                        "mt-2",
                        {"display": "block"},
                        tube_options,
                    )

                if trigger_id == "tube-selector" and selected_tube:
                    return (
                        self.show_growth_lines(selected_tube),
                        "",
                        "mt-2",
                        {"display": "block"},
                        tube_options,
                    )

            # Handle other view types efficiently
            if trigger_id == "view-selector":
                if view_type == "time":
                    return (
                        self.show_growth_over_time(),
                        "",
                        "mt-2",
                        {"display": "none"},
                        [],
                    )
                return (
                    self.show_overview(view_type),
                    "",
                    "mt-2 d-none",
                    {"display": "none"},
                    [],
                )

            # Optimize click handling
            if click_data:
                clicked_value = click_data["points"][0]["x"]
                if "Overview" in current_figure["layout"]["title"]["text"]:
                    return (
                        self.show_sections(clicked_value),
                        f"Showing sections for {clicked_value}",
                        "mt-2",
                        {"display": "none"},
                        [],
                    )
                elif "Sections" in current_figure["layout"]["title"]["text"]:
                    tube_info = current_figure["layout"]["title"]["text"].split("in ")[
                        1
                    ]
                    position = int(clicked_value.split("L")[1])
                    return (
                        self.show_time_series(tube_info, position),
                        "",
                        "mt-2",
                        {"display": "none"},
                        [],
                    )

            return dash.no_update

    def show_overview(self, view_type):
        # Create the overview figure as before
        if view_type == "separate":
            lengths = self.df.groupby("tube_date")["Length (mm)"].sum().reset_index()
            x_values = lengths["tube_date"]
        else:
            lengths = self.df.groupby("Tube")["Length (mm)"].mean().reset_index()
            x_values = lengths["Tube"].apply(lambda x: f"Tube {x}")

        fig = go.Figure(
            data=[
                go.Bar(
                    x=x_values,
                    y=lengths["Length (mm)"],
                    text=lengths["Length (mm)"].round(2),
                    textposition="auto",
                )
            ]
        )

        # Apply the common layout updates
        fig.update_layout(
            title="Root Length Overview",
            xaxis_title="Tube Information",
            yaxis_title="Total Length (mm)",
            clickmode="event+select",
            xaxis_tickangle=-45,
            autosize=True,
            margin=dict(l=50, r=50, t=50, b=50),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="white",
            height=700,
            width=1000,
        )

        return fig

    def show_growth_lines(self, selected_tube):
        """Vertical tube visualization with smoothed data and 10-position intervals."""
        fig = go.Figure()

        # Filter data for the selected tube
        tube_data = self.df[self.df["Tube"] == selected_tube].copy()

        # Get unique dates and positions
        dates = sorted(tube_data["Date"].unique())
        all_positions = sorted(tube_data["Position"].unique())

        # Calculate interval positions (every 10th position)
        interval_positions = all_positions[::10]

        # Function to calculate moving average with interpolation
        def smooth_data(positions, lengths, window=5):
            # Interpolate missing values
            interp_positions = np.arange(min(positions), max(positions) + 1)
            interp_lengths = np.interp(interp_positions, positions, lengths)

            # Apply moving average
            kernel = np.ones(window) / window
            smoothed = np.convolve(interp_lengths, kernel, mode="valid")
            smoothed_positions = interp_positions[
                window // 2 : -(window // 2) if window // 2 > 0 else None
            ]

            # Interpolate back to original interval positions
            final_lengths = np.interp(interval_positions, smoothed_positions, smoothed)
            return final_lengths

        # Define color scheme
        colors = px.colors.qualitative.Plotly

        # Plot traces for each date
        for i, date in enumerate(dates):
            date_data = tube_data[tube_data["Date"] == date]
            position_data = (
                date_data.groupby("Position")["Length (mm)"].mean().reset_index()
            )

            if not position_data.empty:
                # Smooth the data
                smoothed_lengths = smooth_data(
                    position_data["Position"].values,
                    position_data["Length (mm)"].values,
                )

                # Create hover information arrays
                hover_info = []

                # Calculate statistics for each interval
                for pos in interval_positions:
                    # Get raw values for this interval (±5 positions)
                    interval_values = date_data[
                        (date_data["Position"] >= pos - 5)
                        & (date_data["Position"] <= pos + 5)
                    ]["Length (mm)"]

                    if not interval_values.empty:
                        avg_length = interval_values.mean()
                        max_length = interval_values.max()
                        min_length = interval_values.min()
                        std_dev = interval_values.std()
                        n_measurements = len(interval_values)

                        hover_text = (
                            f"Interval L{pos-5}-L{pos+5}:<br>"
                            f"Average: {avg_length:.2f} mm<br>"
                            f"Range: {min_length:.2f} - {max_length:.2f} mm<br>"
                            f"Std Dev: {std_dev:.2f}<br>"
                            f"Measurements: {n_measurements}"
                        )
                    else:
                        hover_text = f"No data for interval L{pos-5}-L{pos+5}"

                    hover_info.append(hover_text)

                fig.add_trace(
                    go.Scatter(
                        x=smoothed_lengths,
                        y=interval_positions,
                        mode="lines+markers",
                        name=date.strftime("%Y-%m-%d"),
                        line=dict(
                            color=colors[i % len(colors)],
                            width=2,
                            shape="spline",
                            smoothing=0.3,
                        ),
                        marker=dict(
                            size=8,
                            opacity=0.8,
                            symbol="circle",
                        ),
                        hovertemplate=(
                            "<b>Position: L%{y}</b><br>"
                            "<b>Smoothed length: %{x:.2f} mm</b><br>"
                            "%{text}<br>"
                            "<b>Date: "
                            + date.strftime("%Y-%m-%d")
                            + "</b><extra></extra>"
                        ),
                        text=hover_info,
                    )
                )

        # Add vertical line representing the tube
        fig.add_shape(
            type="line",
            x0=0,
            x1=0,
            y0=min(interval_positions),
            y1=max(interval_positions),
            line=dict(color="black", width=2, dash="dot"),
        )

        # Calculate the maximum length for x-axis range
        max_length = tube_data["Length (mm)"].max()

        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Root Growth - Tube {selected_tube}",
                x=0.5,
                xanchor="center",
                font=dict(size=20),
            ),
            xaxis=dict(
                title="Root Length (mm)",
                showgrid=True,
                gridcolor="lightgray",
                zeroline=True,
                zerolinecolor="black",
                zerolinewidth=2,
                range=[-1, max_length * 1.1],
                tickformat=".1f",
            ),
            yaxis=dict(
                title="Position (L)",
                showgrid=True,
                gridcolor="lightgray",
                zeroline=False,
                autorange="reversed",
                tickmode="array",
                ticktext=[f"L{pos}" for pos in interval_positions],
                tickvals=interval_positions,
                dtick=10,
            ),
            plot_bgcolor="white",
            legend=dict(
                title=dict(text="Measurement Dates", font=dict(size=12)),
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=10),
            ),
            hovermode="closest",
            height=900,
            width=800,
            margin=dict(t=80, b=60, l=80, r=120),
        )

        fig.update_layout(
            autosize=True,
            margin=dict(l=50, r=50, t=50, b=50),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="white",
            height=700,
            width=1000,
        )
        return fig

    def show_sections(self, tube_info):
        # Create the sections figure as before
        if "(" in tube_info:
            tube = int(tube_info.split(" ")[1])
            date_str = tube_info.split("(")[1].rstrip(")")
            date = pd.to_datetime(date_str)
            section_data = self.df[
                (self.df["Tube"] == tube) & (self.df["Date"] == date)
            ]
        else:
            tube = int(tube_info.split(" ")[1])
            section_data = self.df[self.df["Tube"] == tube]

        section_lengths = (
            section_data.groupby("Position")["Length (mm)"].mean().reset_index()
        )

        fig = go.Figure(
            data=[
                go.Bar(
                    x=[f"L{pos}" for pos in section_lengths["Position"]],
                    y=section_lengths["Length (mm)"],
                    text=section_lengths["Length (mm)"].round(2),
                    textposition="auto",
                )
            ]
        )

        # Apply the common layout updates
        fig.update_layout(
            title=f"Root Length by Sections in {tube_info}",
            xaxis_title="Position",
            yaxis_title="Length (mm)",
            clickmode="event+select",
            autosize=True,
            margin=dict(l=50, r=50, t=50, b=50),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="white",
            height=700,
            width=1000,
        )

        return fig

    def show_time_series(self, tube_info, position):
        tube = int(
            tube_info.split(" ")[1] if "(" not in tube_info else tube_info.split(" ")[1]
        )
        time_series = self.df[
            (self.df["Tube"] == tube) & (self.df["Position"] == position)
        ].sort_values("Date")

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=time_series["Date"],
                    y=time_series["Length (mm)"],
                    mode="lines+markers",
                    text=time_series["Length (mm)"].round(2),
                    textposition="top center",
                )
            ]
        )

        # Apply the common layout updates
        fig.update_layout(
            title=f"Growth Over Time - Tube {tube}, Position L{position}",
            xaxis_title="Date",
            yaxis_title="Length (mm)",
            autosize=True,
            margin=dict(l=50, r=50, t=50, b=50),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="white",
            height=700,
            width=1000,
        )

        return fig

    def show_growth_over_time(self):
        # Create the growth over time figure as before
        fig = go.Figure()

        for tube in self.df["Tube"].unique():
            tube_data = self.df[self.df["Tube"] == tube].sort_values("Date")
            fig.add_trace(
                go.Scatter(
                    x=tube_data["Date"],
                    y=tube_data.groupby("Date")["Length (mm)"].sum(),
                    mode="lines+markers",
                    name=f"Tube {tube}",
                )
            )

        # Apply the common layout updates
        fig.update_layout(
            title="Root Growth Over Time",
            xaxis_title="Date",
            yaxis_title="Total Length (mm)",
            showlegend=True,
            autosize=True,
            margin=dict(l=50, r=50, t=50, b=50),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="white",
            height=700,
            width=1000,
        )

        return fig

    def run_dash_app(self):
        self.app.run_server(debug=False, port=8050, threaded=True)

    def setup_pyqt_ui(self):
        """Optimized PyQt UI setup"""
        self.setWindowTitle("Root Length Visualization")
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.web_view = QWebEngineView()
        # Enable hardware acceleration
        self.web_view.page().settings().setAttribute(
            self.web_view.page().settings().WebAttribute.Accelerated2dCanvasEnabled,
            True,
        )
        layout.addWidget(self.web_view)

        # Start server in separate thread
        self.thread = Thread(target=self.run_dash_app, daemon=True)
        self.thread.start()

        self.web_view.load(QUrl("http://localhost:8050"))

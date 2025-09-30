from dash import Dash, dcc, html, Input, Output, State
import dash
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
from functools import lru_cache
from typing import Any
import warnings
import base64
from PIL import Image
import io
import re
import os


class DashApp:
    """Manages the Dash application."""

    def __init__(self, data_processor: Any, save_directory: str, image_manager=None):
        print("\n=== Initialization Debug ===")
        print(f"Save directory: {save_directory}")

        self.data_processor = data_processor
        self.save_directory = save_directory
        self.image_manager = image_manager

        # Store available images info for debugging
        self.available_images = {}
        if image_manager and image_manager.images:
            print(f"\nFound {len(image_manager.images)} total images")
            for name, path in image_manager.images.items():
                tube_match = re.search(r"T(\d+)", name)
                pos_match = re.search(r"L(\d+)", name)
                date_match = re.search(r"(\d{4})\.(\d{2})\.(\d{2})", name)

                if all([tube_match, pos_match, date_match]):
                    tube = int(tube_match.group(1))
                    position = int(pos_match.group(1))
                    date_str = f"{date_match.group(1)}.{date_match.group(2)}.{date_match.group(3)}"

                    key = (tube, position, date_str)
                    self.available_images[key] = path
                    if tube not in self.available_images.get("tubes", set()):
                        self.available_images.setdefault("tubes", set()).add(tube)
                    if position not in self.available_images.get("positions", set()):
                        self.available_images.setdefault("positions", set()).add(
                            position
                        )
                    if date_str not in self.available_images.get("dates", set()):
                        self.available_images.setdefault("dates", set()).add(date_str)

        # Cache commonly used data
        self._cache_data()

        # Initialize Dash app
        self.app = Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True,
            update_title=None,
            compress=True,
        )

        # Set up layout and callbacks
        self._setup_layout()
        self._setup_callbacks()

    def _cache_data(self) -> None:
        """Cache frequently used data to improve performance."""
        self.tubes = tuple(sorted(self.data_processor.get_unique_tubes()))
        self.dates = tuple(sorted(self.data_processor.get_unique_dates()))

        # Pre-compute grouped data for different views
        df = self.data_processor.df
        self.tube_date_groups = df.groupby(["Tube", "Date"])["Length (mm)"].sum()
        self.position_groups = df.groupby(["Tube", "Position", "Date"])[
            "Length (mm)"
        ].agg(["mean", "min", "max", "std", "count"])

    @lru_cache(maxsize=32)
    def _get_interval_data(
        self, tube: int, date: pd.Timestamp, interval_size: int = 10
    ) -> dict:
        """
        Get cached interval data for a specific tube and date with standardized intervals.
        Returns data in fixed intervals (1-10, 11-20, etc.) regardless of starting position.
        """
        tube_data = self.data_processor.df[
            (self.data_processor.df["Tube"] == tube)
            & (self.data_processor.df["Date"] == date)
        ]

        if tube_data.empty:
            return {}

        # Get actual positions
        positions = sorted(tube_data["Position"].unique())
        min_pos = min(positions)
        max_pos = max(positions)

        # Calculate standardized intervals
        first_interval_end = ((min_pos - 1) // interval_size + 1) * interval_size
        last_interval_end = ((max_pos - 1) // interval_size + 1) * interval_size

        interval_stats = {}
        for interval_end in range(
            first_interval_end, last_interval_end + 1, interval_size
        ):
            interval_start = interval_end - interval_size + 1
            interval_data = tube_data[
                (tube_data["Position"] >= interval_start)
                & (tube_data["Position"] <= interval_end)
            ]

            if not interval_data.empty:
                stats = {
                    "avg": interval_data["Length (mm)"].mean(),
                    "min": interval_data["Length (mm)"].min(),
                    "max": interval_data["Length (mm)"].max(),
                    "std": interval_data["Length (mm)"].std(),
                    "count": len(interval_data),
                    "interval_start": interval_start,
                    "interval_end": interval_end,
                }
                interval_stats[interval_end] = stats
            else:
                interval_stats[interval_end] = {
                    "avg": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                    "std": float("nan"),
                    "count": 0,
                    "interval_start": interval_start,
                    "interval_end": interval_end,
                }

        return interval_stats

    def get_encoded_image(self, tube: int, position: int, date: pd.Timestamp) -> str:
        """Get base64 encoded image with enhanced debugging."""
        try:
            date_str = date.strftime("%Y.%m.%d")
            key = (tube, position, date_str)

            if key in self.available_images:
                path = self.available_images[key]

                try:
                    # Check if file exists and is readable
                    if not os.path.exists(path):
                        print(f"Error: File does not exist: {path}")
                        return ""

                    if not os.access(path, os.R_OK):
                        print(f"Error: No read permission for file: {path}")
                        return ""

                    with Image.open(path) as img:

                        # Convert to RGB if needed
                        if img.mode not in ("RGB", "RGBA"):
                            img = img.convert("RGB")

                        img.thumbnail((200, 150), Image.Resampling.LANCZOS)

                        buffered = io.BytesIO()
                        img.save(buffered, format="PNG", optimize=True)

                        img_str = base64.b64encode(buffered.getvalue()).decode()

                        return f"data:image/png;base64,{img_str}"

                except Exception as img_error:
                    print(f"Error processing image: {str(img_error)}")
                    print(f"Error type: {type(img_error)}")
                    import traceback

                    print(f"Traceback: {traceback.format_exc()}")
                    return ""
            else:
                print("No matching image found in available_images")
                return ""

        except Exception as e:
            print(f"Error in get_encoded_image: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback

            print(f"Traceback: {traceback.format_exc()}")
            return ""

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
                html.H1(
                    "Root Length Visualization",
                    style={
                        "textAlign": "center",
                        "width": "100%",
                        "marginBottom": "30px",
                        "color": "#2c3e50",
                    },
                ),
                dbc.Container(
                    [
                        dbc.Row(
                            dbc.Col(
                                dbc.Card(
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
                                                        "label": "Stacked Bar View",
                                                        "value": "stacked",
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
                                ),
                                md=8,
                                className="mx-auto",
                            ),
                            className="mb-4 justify-content-center",
                        ),
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
                                            "toImageButtonOptions": {
                                                "format": "svg",
                                                "filename": "root_length_plot",
                                                "height": 700,
                                                "width": 1000,
                                                "scale": 2,
                                            },
                                            "modeBarButtonsToAdd": ["downloadCsv"],
                                        },
                                        style={
                                            "backgroundColor": "white",
                                            "borderRadius": "8px",
                                            "padding": "15px",
                                            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                                        },
                                    ),
                                    html.Div(
                                        id="click-data",
                                        className="text-center my-3",
                                        style={"color": "#2c3e50"},
                                    ),
                                ],
                                className="d-flex flex-column align-items-center",
                            )
                        ),
                        # Container for dynamically displayed images on hover
                        html.Div(
                            id="image-container",
                            style={
                                "width": "100%",
                                "display": "flex",
                                "flexWrap": "wrap",
                                "justifyContent": "center",
                                "marginTop": "20px",
                            },
                        ),
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
            ],
            [
                Input("view-selector", "value"),
                Input("tube-selector", "value"),
                Input("main-graph", "clickData"),
            ],
        )
        def update_visualization(view_type, selected_tube, click_data):
            ctx = dash.callback_context
            if not ctx.triggered:
                # Initial load - show stacked view by default
                return (
                    self.create_stacked_bar_chart(),
                    "",
                    {"display": "none"},
                    [],
                )

            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

            try:
                if view_type == "stacked":
                    return (
                        self.create_stacked_bar_chart(),
                        "",
                        {"display": "none"},
                        [],
                    )

                elif view_type == "lines":
                    tube_options = [
                        {"label": f"Tube {tube}", "value": tube}
                        for tube in self.data_processor.get_unique_tubes()
                    ]

                    if trigger_id == "view-selector":
                        first_tube = self.data_processor.get_unique_tubes()[0]
                        fig = self.show_growth_lines(first_tube)
                        return (
                            fig,
                            "",
                            {"display": "block"},
                            tube_options,
                        )

                    if trigger_id == "tube-selector" and selected_tube:
                        fig = self.show_growth_lines(selected_tube)
                        return (
                            fig,
                            "",
                            {"display": "block"},
                            tube_options,
                        )

                elif view_type == "time":
                    return (
                        self.show_growth_over_time(),
                        "",
                        {"display": "none"},
                        [],
                    )

            except Exception as e:
                print(f"Error in callback: {e}")
                return dash.no_update

            return dash.no_update

        # Callback to display the 10 images for hovered interval in Growth Lines view
        @self.app.callback(
            Output("image-container", "children"),
            Input("main-graph", "hoverData"),
            State("tube-selector", "value"),
            State("view-selector", "value"),
        )
        def display_hover_images(hoverData, selected_tube, view_type):
            if (
                view_type != "lines"
                or not hoverData
                or "points" not in hoverData
                or not hoverData["points"]
                or not selected_tube
            ):
                return []

            try:
                point = hoverData["points"][0]
                print(point)
                # Extract the date directly from the hover data
                date_str = point["text"].split("<br>")[5].split(":")[1].strip()
                date = pd.to_datetime(date_str, format="%Y.%m.%d")

                hovered_y = int(point["y"])

                interval_start = hovered_y - 9
                interval_end = hovered_y

                images = []
                successful_encodings = 0

                for pos in range(interval_start, interval_end + 1):
                    img_src = self.get_encoded_image(selected_tube, pos, date)
                    if img_src:
                        successful_encodings += 1
                        images.append(
                            html.Div(
                                [
                                    html.Img(
                                        src=img_src,
                                        style={
                                            "width": "100%",
                                            "height": "auto",
                                            "objectFit": "contain",
                                            "border": "1px solid #ddd",
                                            "borderRadius": "4px",
                                        },
                                    ),
                                    html.P(
                                        f"Position L{pos}",
                                        style={
                                            "textAlign": "center",
                                            "fontWeight": "bold",
                                            "margin": "5px 0",
                                        },
                                    ),
                                ],
                                key=f"image-{pos}-{date_str}",  # Unique key
                                style={
                                    "width": "10%",
                                    "padding": "2px",
                                    "boxSizing": "border-box",
                                    "display": "inline-block",
                                    "verticalAlign": "top",
                                },
                            )
                        )

                if not images:
                    return [
                        html.Div(
                            "No images available for this interval",
                            style={"textAlign": "center", "padding": "20px"},
                        )
                    ]

                return images

            except Exception as e:
                print(f"Error in display_hover_images: {str(e)}")
                import traceback

                print(f"Traceback: {traceback.format_exc()}")
                return []

    def show_growth_lines(self, selected_tube: int) -> go.Figure:
        """Generate growth lines figure with debug information."""
        print(f"\n=== Growth Lines Debug for Tube {selected_tube} ===")
        fig = go.Figure()
        colors = px.colors.qualitative.Plotly

        try:
            interval_size = 10
            max_length = 0

            print("\nProcessing dates:")
            for i, date in enumerate(self.dates):
                print(f"\nDate: {date.strftime('%Y-%m-%d')}")
                interval_data = self._get_interval_data(
                    selected_tube, date, interval_size
                )

                if not interval_data:
                    continue

                positions = sorted(interval_data.keys())
                print(f"Positions found: {positions}")
                averages = []
                hover_info = []

                for pos in positions:
                    stats = interval_data[pos]
                    print(f"\nPosition {pos}:")
                    print(f"Stats: {stats}")

                    # Check if we have corresponding image
                    date_str = date.strftime("%Y.%m.%d")
                    has_image = (selected_tube, pos, date_str) in self.available_images
                    print(f"Image available: {has_image}")

                    if stats:
                        avg = stats["avg"]
                        averages.append(avg)
                        max_length = max(max_length, stats["max"])
                        hover_text = (
                            f"Interval L{pos-interval_size+1}-L{pos}:<br>"
                            f"Average: {avg:.2f} mm<br>"
                            f"Range: {stats['min']:.2f} - {stats['max']:.2f} mm<br>"
                            f"Std Dev: {stats['std']:.2f}<br>"
                            f"Measurements: {stats['count']}<br>"
                            f"Date : {date_str}<br>"
                            f"{'Image Available' if has_image else 'No Image'}"
                        )
                    else:
                        averages.append(float("nan"))
                        hover_text = (
                            f"No data for interval L{pos-interval_size+1}-L{pos}"
                        )

                    hover_info.append(hover_text)

                # Rest of the function remains the same...
                fig.add_trace(
                    go.Scatter(
                        x=averages,
                        y=positions,
                        mode="lines+markers",
                        name=date.strftime("%Y-%m-%d"),
                        line={
                            "color": colors[i % len(colors)],
                            "width": 2,
                            "shape": "spline",
                            "smoothing": 0.3,
                        },
                        marker={"size": 8, "opacity": 0.8},
                        hovertemplate=(
                            "<b>Position: L%{y}</b><br>"
                            "<b>Average: %{x:.2f} mm</b><br>"
                            "%{text}<br>"
                        ),
                        text=hover_info,
                    )
                )

            # Update layout with optimized settings
            fig.update_layout(
                title={
                    "text": f"Root Growth - Tube {int(selected_tube)}",
                    "x": 0.5,
                    "xanchor": "center",
                    "font": {"size": 20},
                },
                xaxis={
                    "title": "Root Length (mm)",
                    "showgrid": True,
                    "gridcolor": "lightgray",
                    "zeroline": True,
                    "zerolinecolor": "black",
                    "zerolinewidth": 2,
                    "range": [-1, max_length * 1.1],
                    "tickformat": ".1f",
                },
                yaxis={
                    "title": "Position (L)",
                    "showgrid": True,
                    "gridcolor": "lightgray",
                    "zeroline": False,
                    "autorange": "reversed",
                    "tickmode": "array",
                    "ticktext": [f"L{pos}" for pos in positions],
                    "tickvals": positions,
                    "dtick": 10,
                },
                plot_bgcolor="white",
                legend={
                    "title": {"text": "Measurement Dates", "font": {"size": 12}},
                    "yanchor": "top",
                    "y": 0.99,
                    "xanchor": "left",
                    "x": 1.02,
                    "bgcolor": "rgba(255, 255, 255, 0.8)",
                    "bordercolor": "black",
                    "borderwidth": 1,
                    "font": {"size": 10},
                },
                hovermode="closest",
                height=700,
                width=1000,
                margin={"t": 80, "b": 60, "l": 80, "r": 120},
            )

            # Add reference line
            fig.add_shape(
                type="line",
                x0=0,
                x1=0,
                y0=min(positions),
                y1=max(positions),
                line={"color": "black", "width": 2, "dash": "dot"},
            )

        except Exception as e:
            warnings.warn(f"Error generating growth lines: {e}")
            fig = go.Figure()

        return fig

    def show_growth_over_time(self):
        """Generate growth over time figure."""
        df = self.data_processor.df
        fig = go.Figure()
        try:
            for tube in self.data_processor.get_unique_tubes():
                tube_data = df[df["Tube"] == tube].sort_values("Date")
                grouped = tube_data.groupby("Date")["Length (mm)"].sum().reset_index()
                fig.add_trace(
                    go.Scatter(
                        x=grouped["Date"],
                        y=grouped["Length (mm)"],
                        mode="lines+markers",
                        name=f"Tube {int(tube)}",
                    )
                )

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
        except Exception as e:
            print(f"Error generating growth over time: {e}")
            return go.Figure()

    def run_server(self):
        """Run the Dash server."""
        try:
            self.app.run_server(debug=True, port=8050, threaded=True, processes=1)
        except Exception as e:
            print(f"Error running Dash server: {e}")

    def create_stacked_bar_chart(self) -> go.Figure:
        """Create an optimized stacked bar chart."""
        try:
            traces = []
            for date in self.dates:
                trace_data = []
                for tube in self.tubes:
                    value = self.tube_date_groups.get((tube, date), 0)
                    trace_data.append(value)

                traces.append(
                    go.Bar(
                        name=date.strftime("%Y-%m-%d"),
                        x=[f"Tube {int(tube)}" for tube in self.tubes],
                        y=trace_data,
                        text=[f"{v:.2f}" for v in trace_data],
                        textposition="auto",
                        hovertemplate=(
                            "<b>%{x}</b><br>"
                            f"Date: {date.strftime('%Y-%m-%d')}<br>"
                            "Length: %{y:.2f} mm<br>"
                            "<extra></extra>"
                        ),
                    )
                )

            fig = go.Figure(data=traces)
            fig.update_layout(
                barmode="stack",
                title={
                    "text": "Root Length Growth by Tube and Date",
                    "x": 0.5,
                    "xanchor": "center",
                    "font": {"size": 20},
                },
                xaxis_title="Tube",
                yaxis_title="Total Length (mm)",
                legend_title="Measurement Date",
                height=700,
                width=1600,
                showlegend=True,
                hovermode="x unified",
                plot_bgcolor="white",
                bargap=0.1,
                bargroupgap=0.5,
                legend={
                    "yanchor": "top",
                    "y": 0.99,
                    "xanchor": "left",
                    "x": 1.02,
                    "bgcolor": "rgba(255, 255, 255, 0.8)",
                    "bordercolor": "black",
                    "borderwidth": 1,
                },
                margin={"l": 50, "r": 150, "t": 80, "b": 50},
            )

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

        except Exception as e:
            warnings.warn(f"Error generating stacked bar chart: {e}")
            return go.Figure()

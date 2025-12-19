"""
Main Dash application for root length visualization.
Orchestrates layout, callbacks, and visualization components.
"""


from dash import Dash, dcc, html, Input, Output, State
import dash
import dash_bootstrap_components as dbc
import pandas as pd
import warnings
from typing import Any

from .dash_data_cache import DataCache
from .dash_image_utils import build_available_images_map, get_encoded_image
from .dash_visualizations import DashVisualizations




class DashApp(DashVisualizations):
    """Manages the Dash application."""

    def __init__(self, data_processor: Any, save_directory: str, image_manager=None):
        try:
            self.data_processor = data_processor
            self.save_directory = save_directory
            self.image_manager = image_manager

            # Build available images map for hover display
            self.available_images = build_available_images_map(image_manager)

            # Initialize data cache
            self.data_cache = DataCache(data_processor)

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
        except Exception as e:
            raise

    def get_encoded_image(self, tube: int, position: int, date: pd.Timestamp) -> str:
        """Get base64 encoded image for hover display."""
        return get_encoded_image(self.available_images, tube, position, date)

    def _setup_layout(self):
        """Define the Dash app layout."""
        try:
            options = [
                {"label": "Stacked Bar View", "value": "stacked"},
                {"label": "Growth Over Time", "value": "time"},
                {"label": "Growth Lines", "value": "lines"},
            ]
            
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
                        "fontSize": "36px",
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
                                                style={"fontSize": "24px", "marginBottom": "20px"},
                                            ),
                                            dcc.Dropdown(
                                                id="view-selector",
                                                options=options,
                                                value="stacked",
                                                className="mb-3",
                                                style={"fontSize": "16px"},
                                            ),
                                            # Tube selector (for Growth Lines view)
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
                                                "filename": "root_length_plot",
                                                "scale": 2,
                                            },
                                            "modeBarButtonsToAdd": ["downloadCsv"],
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
                        # Container for dynamically displayed images on hover
                        dbc.Row(
                            dbc.Col(
                                html.Div(
                                    id="image-container",
                                ),
                                md=12,
                            ),
                            className="mt-4",
                        ),
                        dcc.Store(id="cached-data"),
                    ],
                    fluid=True,
                    className="px-4",
                ),
            ],
        )
        except Exception as e:
            raise

    def _setup_callbacks(self):
        """Define Dash callbacks."""

        # Main visualization callback
        @self.app.callback(
            [
                Output("main-graph", "figure"),
                Output("click-data", "children"),
                Output("tube-selector", "style"),
                Output("tube-selector", "options"),
                Output("main-graph", "style"),
                Output("image-container", "style"),
            ],
            [
                Input("view-selector", "value"),
                Input("tube-selector", "value"),
                Input("main-graph", "clickData"),
            ],
        )
        def update_visualization(view_type, selected_tube, click_data):
            ctx = dash.callback_context
            default_style = {
                "backgroundColor": "white",
                "borderRadius": "8px",
                "padding": "20px",
                "boxShadow": "0 4px 8px rgba(0,0,0,0.15)",
                "height": "800px",
                "width": "100%",
                "marginBottom": "20px",
                "overflow": "hidden",
            }
            
            hidden_images_style = {
                "width": "100%",
                "display": "none",
                "flexWrap": "wrap",
                "justifyContent": "center",
                "marginTop": "20px",
                "padding": "20px",
                "backgroundColor": "#f8f9fa",
                "borderRadius": "8px",
                "minHeight": "200px",
            }
            
            if not ctx.triggered:
                return (
                    self.create_stacked_bar_chart(),
                    "",
                    {"display": "none"},
                    [],
                    default_style,
                    hidden_images_style,
                )

            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

            try:
                if view_type == "stacked":
                    return (
                        self.create_stacked_bar_chart(),
                        "",
                        {"display": "none"},
                        [],
                        default_style,
                        hidden_images_style,
                    )

                elif view_type == "lines":
                    tube_options = [
                        {"label": f"Tube {tube}", "value": tube}
                        for tube in self.data_processor.get_unique_tubes()
                    ]
                    
                    lines_graph_style = default_style.copy()
                    lines_graph_style["height"] = "600px"

                    visible_images_style = {
                        "width": "100%",
                        "display": "flex",
                        "flexWrap": "wrap",
                        "justifyContent": "center",
                        "marginTop": "20px",
                        "padding": "20px",
                        "backgroundColor": "#f8f9fa",
                        "borderRadius": "8px",
                        "minHeight": "200px",
                    }

                    if trigger_id == "view-selector":
                        first_tube = self.data_processor.get_unique_tubes()[0]
                        fig = self.show_growth_lines(first_tube)
                        return (
                            fig,
                            "",
                            {"display": "block"},
                            tube_options,
                            lines_graph_style,
                            visible_images_style,
                        )

                    if trigger_id == "tube-selector" and selected_tube:
                        fig = self.show_growth_lines(selected_tube)
                        return (
                            fig,
                            "",
                            {"display": "block"},
                            tube_options,
                            lines_graph_style,
                            visible_images_style,
                        )

                elif view_type == "time":
                    return (
                        self.show_growth_over_time(),
                        "",
                        {"display": "none"},
                        [],
                        default_style,
                        hidden_images_style,
                    )

            except Exception as e:
                warnings.warn(f"Error in visualization: {e}")
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

                # Extract the date directly from the hover data
                text_lines = point.get("text", "").split("<br>")
                date_str = None
                for line in text_lines:
                    if "Date :" in line or "Date:" in line:
                        date_str = line.split(":")[-1].strip()
                        break

                if not date_str:
                    return []

                date = pd.to_datetime(date_str, format="%Y.%m.%d")
                hovered_y = int(point["y"])

                interval_start = hovered_y - 9
                interval_end = hovered_y

                images = []

                for pos in range(interval_start, interval_end + 1):
                    img_src = self.get_encoded_image(selected_tube, pos, date)
                    if img_src:
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
                                            "fontSize": "14px",
                                        },
                                    ),
                                ],
                                key=f"image-{pos}-{date_str}",
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
                            style={"textAlign": "center", "padding": "20px", "fontSize": "18px"},
                        )
                    ]

                return images

            except Exception as e:
                return []

    def run_server(self):
        """Run the Dash server."""
        try:
            self.app.run_server(debug=False, port=8050, threaded=True)
        except Exception:
            pass

"""
Main Dash application for root length visualization.
Orchestrates layout, callbacks, and visualization components.
"""

import logging
from dash import Dash, dcc, html, Input, Output, State
import dash
import dash_bootstrap_components as dbc
import pandas as pd
import warnings
from typing import Any

from .dash_data_cache import DataCache
from .dash_image_utils import build_available_images_map, get_encoded_image
from .dash_visualizations import DashVisualizations

logger = logging.getLogger(__name__)


class DashApp(DashVisualizations):
    """Manages the Dash application."""

    def __init__(self, data_processor: Any, save_directory: str, image_manager=None):
        logger.info(f"DashApp.__init__ called with save_directory={save_directory}")
        try:
            self.data_processor = data_processor
            self.save_directory = save_directory
            self.image_manager = image_manager

            # Build available images map for hover display
            logger.debug("Building available images map")
            self.available_images = build_available_images_map(image_manager)
            logger.debug(f"Available images map built: {len(self.available_images) if self.available_images else 0} images")

            # Initialize data cache
            logger.debug("Initializing DataCache")
            self.data_cache = DataCache(data_processor)
            logger.debug("DataCache initialized")

            # Initialize Dash app
            logger.debug("Creating Dash application")
            self.app = Dash(
                __name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True,
                update_title=None,
                compress=True,
            )
            logger.info("Dash application created")

            # Set up layout and callbacks
            logger.debug("Setting up layout")
            self._setup_layout()
            logger.debug("Setting up callbacks")
            self._setup_callbacks()
            logger.info("DashApp initialization completed successfully")
        except Exception as e:
            logger.error(f"Error in DashApp.__init__: {e}", exc_info=True)
            raise

    def get_encoded_image(self, tube: int, position: int, date: pd.Timestamp) -> str:
        """Get base64 encoded image for hover display."""
        return get_encoded_image(self.available_images, tube, position, date)

    def _setup_layout(self):
        """Define the Dash app layout."""
        logger.debug("_setup_layout() called")
        try:
            # Check if experimental data is available to show extra options
            has_experiments = "Treatment" in self.data_processor.df.columns
            logger.debug(f"Experimental data available: {has_experiments}")

            options = [
                {"label": "Stacked Bar View", "value": "stacked"},
                {"label": "Growth Over Time", "value": "time"},
                {"label": "Growth Lines", "value": "lines"},
            ]
            
            if has_experiments:
                options.extend([
                    {"label": "Treatment Analysis (Box Plot)", "value": "treatment_box"},
                    {"label": "Genotype Performance (Bar Chart)", "value": "genotype_bar"},
                    {"label": "Field Heatmap", "value": "field_heatmap"},
                    {"label": "Genotype Comparison (Interactive)", "value": "genotype_comparison"},
                    {"label": "Depth Profile Comparison", "value": "depth_profile"},
                    {"label": "Growth Trajectory Comparison", "value": "growth_trajectory"},
                ])

            # Get unique genotypes and treatments for dynamic controls
            genotypes = self.data_processor.get_unique_genotypes() if has_experiments else []
            treatments = self.data_processor.get_unique_treatments() if has_experiments else []
            date_options = [{"label": d.strftime("%Y-%m-%d"), "value": d.strftime("%Y-%m-%d")} for d in self.data_cache.dates]
            date_options.insert(0, {"label": "Average All Dates", "value": "all"})

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
                                            
                                            # Dynamic Controls Container for Interactive Views
                                            html.Div(
                                                id="dynamic-controls",
                                                style={"display": "none"},
                                                children=[
                                                    # Mode Toggle
                                                    html.Div([
                                                        html.Label("Comparison Mode:", style={"fontWeight": "bold", "marginBottom": "5px"}),
                                                        dcc.RadioItems(
                                                            id="comparison-mode",
                                                            options=[
                                                                {"label": " Compare Genotypes in one Treatment", "value": "genotypes_in_treatment"},
                                                                {"label": " Compare one Genotype across Treatments", "value": "genotype_across_treatments"},
                                                            ],
                                                            value="genotypes_in_treatment",
                                                            inline=False,
                                                            style={"marginBottom": "15px"},
                                                            labelStyle={"display": "block", "marginBottom": "5px"}
                                                        ),
                                                    ], className="mb-3"),
                                                    
                                                    # Primary Selector (Treatment or Genotype based on mode)
                                                    html.Div([
                                                        html.Label(id="primary-selector-label", children="Select Treatment:", style={"fontWeight": "bold"}),
                                                        dcc.Dropdown(
                                                            id="primary-selector",
                                                            options=[{"label": t, "value": t} for t in treatments],
                                                            value=treatments[0] if treatments else None,
                                                            className="mb-3",
                                                        ),
                                                    ]),
                                                    
                                                    # Secondary Selector (Multi-select Genotypes or hidden)
                                                    html.Div(
                                                        id="secondary-selector-container",
                                                        children=[
                                                            html.Label("Select Genotypes to Compare:", style={"fontWeight": "bold"}),
                                                            dcc.Dropdown(
                                                                id="secondary-selector",
                                                                options=[{"label": g, "value": g} for g in genotypes],
                                                                value=genotypes[:3] if len(genotypes) >= 3 else genotypes,
                                                                multi=True,
                                                                className="mb-3",
                                                            ),
                                                        ],
                                                    ),
                                                    
                                                    # Date Selector (for Depth Profile)
                                                    html.Div(
                                                        id="date-selector-container",
                                                        style={"display": "none"},
                                                        children=[
                                                            html.Label("Select Date:", style={"fontWeight": "bold"}),
                                                            dcc.Dropdown(
                                                                id="date-selector",
                                                                options=date_options,
                                                                value="all",
                                                                className="mb-3",
                                                            ),
                                                        ],
                                                    ),
                                                ],
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
            logger.info("Layout setup completed")
        except Exception as e:
            logger.error(f"Error in _setup_layout: {e}", exc_info=True)
            raise

    def _setup_callbacks(self):
        """Define Dash callbacks."""
        logger.debug("_setup_callbacks() called")
        try:
            has_experiments = "Treatment" in self.data_processor.df.columns
            logger.debug(f"Setting up callbacks, has_experiments={has_experiments}")
        except Exception as e:
            logger.error(f"Error in _setup_callbacks: {e}", exc_info=True)
            raise

        genotypes = self.data_processor.get_unique_genotypes() if has_experiments else []
        treatments = self.data_processor.get_unique_treatments() if has_experiments else []

        # Callback to update dynamic controls based on view and mode
        @self.app.callback(
            [
                Output("dynamic-controls", "style"),
                Output("primary-selector-label", "children"),
                Output("primary-selector", "options"),
                Output("primary-selector", "value"),
                Output("secondary-selector-container", "style"),
                Output("secondary-selector", "options"),
                Output("secondary-selector", "value"),
                Output("date-selector-container", "style"),
            ],
            [
                Input("view-selector", "value"),
                Input("comparison-mode", "value"),
            ],
        )
        def update_dynamic_controls(view_type, mode):
            interactive_views = ["genotype_comparison", "depth_profile", "growth_trajectory"]
            
            if view_type not in interactive_views:
                return (
                    {"display": "none"},
                    "Select Treatment:",
                    [{"label": t, "value": t} for t in treatments],
                    treatments[0] if treatments else None,
                    {"display": "block"},
                    [{"label": g, "value": g} for g in genotypes],
                    genotypes[:3] if len(genotypes) >= 3 else genotypes,
                    {"display": "none"},
                )
            
            show_date = view_type == "depth_profile"
            
            if mode == "genotypes_in_treatment":
                return (
                    {"display": "block"},
                    "Select Treatment:",
                    [{"label": t, "value": t} for t in treatments],
                    treatments[0] if treatments else None,
                    {"display": "block"},
                    [{"label": g, "value": g} for g in genotypes],
                    genotypes[:3] if len(genotypes) >= 3 else genotypes,
                    {"display": "block"} if show_date else {"display": "none"},
                )
            else:  # genotype_across_treatments
                return (
                    {"display": "block"},
                    "Select Genotype:",
                    [{"label": g, "value": g} for g in genotypes],
                    genotypes[0] if genotypes else None,
                    {"display": "none"},
                    [],
                    [],
                    {"display": "block"} if show_date else {"display": "none"},
                )

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
                Input("comparison-mode", "value"),
                Input("primary-selector", "value"),
                Input("secondary-selector", "value"),
                Input("date-selector", "value"),
            ],
        )
        def update_visualization(view_type, selected_tube, click_data, 
                                 comparison_mode, primary_selection, secondary_selection, date_selection):
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

                # Experimental Views (Static)
                elif view_type == "treatment_box":
                    return (
                        self.show_treatment_comparison(),
                        "",
                        {"display": "none"},
                        [],
                        default_style,
                        hidden_images_style,
                    )

                elif view_type == "genotype_bar":
                    return (
                        self.show_genotype_ranking(),
                        "",
                        {"display": "none"},
                        [],
                        default_style,
                        hidden_images_style,
                    )

                elif view_type == "field_heatmap":
                    return (
                        self.show_field_heatmap(),
                        "",
                        {"display": "none"},
                        [],
                        default_style,
                        hidden_images_style,
                    )

                # Interactive Views
                elif view_type == "genotype_comparison":
                    fig = self.show_genotype_comparison(
                        comparison_mode, primary_selection, secondary_selection
                    )
                    return (
                        fig,
                        "",
                        {"display": "none"},
                        [],
                        default_style,
                        hidden_images_style,
                    )

                elif view_type == "depth_profile":
                    fig = self.show_depth_profile_comparison(
                        comparison_mode, primary_selection, secondary_selection, date_selection
                    )
                    return (
                        fig,
                        "",
                        {"display": "none"},
                        [],
                        default_style,
                        hidden_images_style,
                    )

                elif view_type == "growth_trajectory":
                    fig = self.show_growth_trajectory(
                        comparison_mode, primary_selection, secondary_selection
                    )
                    return (
                        fig,
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
                logger.error(f"Error in display_hover_images callback: {e}", exc_info=True)
                return []

    def run_server(self):
        """Run the Dash server."""
        try:
            self.app.run_server(debug=False, port=8050, threaded=True)
        except Exception:
            pass

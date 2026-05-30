"""
Main Dash application for root length visualization.
Orchestrates layout, callbacks, and visualization components.
"""


from dash import Dash, dcc, html, Input, Output, State
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import warnings
import re
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
                {"label": "Faceted Depth Profile", "value": "faceted"},
            ]
            
            self.app.layout = html.Div(
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "alignItems": "center",
                    "minHeight": "100vh",
                    "width": "100%",
                    "backgroundColor": "#f8f9fa",
                    "padding": "12px",
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
                        # Tube Selection Panel (for Growth Over Time view)
                        dbc.Row(
                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.H4(
                                                "Tube Selection",
                                                className="card-title text-center",
                                                style={"fontSize": "22px", "marginBottom": "20px"},
                                            ),
                                            dcc.Tabs(
                                                id="tube-selection-tabs",
                                                value="tab-multiselect",
                                                children=[
                                                    dcc.Tab(
                                                        label="Multi-Select",
                                                        value="tab-multiselect",
                                                        children=[
                                                            html.Div(
                                                                [
                                                                    html.Label("Select Tubes:", style={"fontWeight": "bold", "marginTop": "15px"}),
                                                                    dcc.Dropdown(
                                                                        id="tube-multiselect",
                                                                        multi=True,
                                                                        placeholder="Select tubes...",
                                                                        style={"fontSize": "14px"},
                                                                    ),
                                                                ],
                                                                style={"padding": "15px"},
                                                            )
                                                        ],
                                                    ),
                                                    dcc.Tab(
                                                        label="Range",
                                                        value="tab-range",
                                                        children=[
                                                            html.Div(
                                                                [
                                                                    html.Label("From:", style={"fontWeight": "bold", "marginTop": "15px", "marginRight": "10px"}),
                                                                    dcc.Input(
                                                                        id="tube-range-from",
                                                                        type="number",
                                                                        placeholder="Start",
                                                                        style={"width": "100px", "marginRight": "20px", "fontSize": "14px"},
                                                                    ),
                                                                    html.Label("To:", style={"fontWeight": "bold", "marginRight": "10px"}),
                                                                    dcc.Input(
                                                                        id="tube-range-to",
                                                                        type="number",
                                                                        placeholder="End",
                                                                        style={"width": "100px", "marginRight": "20px", "fontSize": "14px"},
                                                                    ),
                                                                    html.Button(
                                                                        "Add Range",
                                                                        id="add-range-btn",
                                                                        n_clicks=0,
                                                                        className="btn btn-primary",
                                                                        style={"fontSize": "14px"},
                                                                    ),
                                                                ],
                                                                style={"padding": "15px", "display": "flex", "alignItems": "center"},
                                                            )
                                                        ],
                                                    ),
                                                    dcc.Tab(
                                                        label="Manual Entry",
                                                        value="tab-manual",
                                                        children=[
                                                            html.Div(
                                                                [
                                                                    html.Label(
                                                                        "Enter tube numbers (e.g., 1,3,5,10-15):",
                                                                        style={"fontWeight": "bold", "marginTop": "15px", "marginBottom": "10px"},
                                                                    ),
                                                                    dcc.Input(
                                                                        id="tube-manual-entry",
                                                                        type="text",
                                                                        placeholder="1,3,5,10-15",
                                                                        style={"width": "100%", "fontSize": "14px"},
                                                                    ),
                                                                ],
                                                                style={"padding": "15px"},
                                                            )
                                                        ],
                                                    ),
                                                ],
                                                style={"fontSize": "16px"},
                                            ),
                                            html.Hr(style={"margin": "20px 0"}),
                                            html.Div(
                                                [
                                                    html.Label("Selected Tubes:", style={"fontWeight": "bold", "marginBottom": "10px"}),
                                                    html.Div(
                                                        id="selected-tubes-display",
                                                        style={
                                                            "minHeight": "50px",
                                                            "padding": "10px",
                                                            "backgroundColor": "#f8f9fa",
                                                            "borderRadius": "5px",
                                                            "marginBottom": "15px",
                                                        },
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Button(
                                                                "Clear All",
                                                                id="clear-all-btn",
                                                                n_clicks=0,
                                                                className="btn btn-warning",
                                                                style={"marginRight": "10px", "fontSize": "14px"},
                                                            ),
                                                            html.Button(
                                                                "Select All",
                                                                id="select-all-btn",
                                                                n_clicks=0,
                                                                className="btn btn-info",
                                                                style={"fontSize": "14px"},
                                                            ),
                                                        ],
                                                        style={"display": "flex", "gap": "10px"},
                                                    ),
                                                ]
                                            ),
                                        ]
                                    ),
                                    id="tube-selection-card",
                                    style={"marginBottom": "30px", "padding": "20px", "display": "none"},
                                ),
                                md=12,
                                className="mx-auto",
                            ),
                            className="mb-4 justify-content-center",
                        ),
                        # Visualization Options Panel (for Growth Over Time view)
                        dbc.Row(
                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.H4(
                                                "Visualization Options",
                                                className="card-title text-center",
                                                style={"fontSize": "22px", "marginBottom": "20px"},
                                            ),
                                            dcc.Checklist(
                                                id="growth-viz-options",
                                                options=[
                                                    {"label": " Show Field Average", "value": "show_field_avg"},
                                                    {"label": " Show Field Variance Shading", "value": "show_field_var"},
                                                ],
                                                value=["show_field_avg", "show_field_var"],
                                                style={"fontSize": "16px"},
                                                labelStyle={"display": "block", "marginBottom": "10px"},
                                            ),
                                            html.Hr(style={"margin": "20px 0"}),
                                            html.Button(
                                                "Toggle Legend",
                                                id="toggle-legend-btn",
                                                n_clicks=0,
                                                className="btn btn-secondary",
                                                style={"width": "100%", "fontSize": "16px"},
                                            ),
                                        ]
                                    ),
                                    id="viz-options-card",
                                    style={"marginBottom": "30px", "padding": "20px", "display": "none"},
                                ),
                                md=12,
                                className="mx-auto",
                            ),
                            className="mb-4 justify-content-center",
                        ),
                        # Faceted View Options Panel
                        dbc.Row(
                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.H4(
                                                "Faceted View Options",
                                                className="card-title text-center",
                                                style={"fontSize": "22px", "marginBottom": "20px"},
                                            ),
                                            html.Label(
                                                "Select Dates for Rows:",
                                                style={"fontWeight": "bold", "marginBottom": "10px"}
                                            ),
                                            dcc.Dropdown(
                                                id="faceted-date-selector",
                                                multi=True,
                                                placeholder="Select dates (all if none selected)...",
                                                style={"fontSize": "14px", "marginBottom": "15px"},
                                            ),
                                            html.Hr(style={"margin": "15px 0"}),
                                            html.Label(
                                                "Select Tubes for Columns (up to 6):",
                                                style={"fontWeight": "bold", "marginBottom": "10px"}
                                            ),
                                            dcc.Dropdown(
                                                id="faceted-tube-selector",
                                                multi=True,
                                                placeholder="Select up to 6 tubes...",
                                                style={"fontSize": "14px", "marginBottom": "15px"},
                                            ),
                                            html.Div(
                                                id="faceted-selection-info",
                                                style={
                                                    "padding": "10px",
                                                    "backgroundColor": "#f8f9fa",
                                                    "borderRadius": "5px",
                                                    "marginBottom": "15px",
                                                },
                                            ),
                                            html.Hr(style={"margin": "15px 0"}),
                                            html.Label(
                                                "Tube-Date Availability:",
                                                style={"fontWeight": "bold", "marginBottom": "10px"}
                                            ),
                                            html.Div(
                                                id="tube-date-availability",
                                                style={
                                                    "padding": "10px",
                                                    "backgroundColor": "#f8f9fa",
                                                    "borderRadius": "5px",
                                                    "maxHeight": "200px",
                                                    "overflowY": "auto",
                                                    "fontSize": "12px",
                                                },
                                            ),
                                        ]
                                    ),
                                    id="faceted-options-card",
                                    style={"marginBottom": "30px", "padding": "20px", "display": "none"},
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
                        dcc.Store(id="selected-tubes-store", data=[]),
                        dcc.Store(id="legend-visible-store", data=True),
                        dcc.Store(id="faceted-tubes-store", data=[]),
                        dcc.Store(id="faceted-dates-store", data=[]),
                    ],
                    fluid=True,
                    className="px-2",
                ),
            ],
        )
        except Exception as e:
            raise

    def _setup_callbacks(self):
        """Define Dash callbacks."""

        # Import parse_tube_selection function
        from .dash_visualizations import parse_tube_selection

        # Callback to show/hide tube selection, viz options, and faceted panels based on view type
        @self.app.callback(
            [
                Output("tube-selection-card", "style"),
                Output("viz-options-card", "style"),
                Output("tube-multiselect", "options"),
                Output("faceted-options-card", "style"),
                Output("faceted-tube-selector", "options"),
                Output("faceted-date-selector", "options"),
            ],
            [Input("view-selector", "value")],
        )
        def toggle_view_panels(view_type):
            """Show appropriate panels based on view type."""
            # Get available tubes and dates
            tube_options = [
                {"label": f"Tube {int(tube)}", "value": int(tube)}
                for tube in self.data_processor.get_unique_tubes()
            ]
            
            date_options = [
                {"label": date.strftime("%Y-%m-%d"), "value": date.isoformat()}
                for date in sorted(self.data_processor.get_unique_dates())
            ]
            
            visible_style = {"marginBottom": "30px", "padding": "20px", "display": "block"}
            hidden_style = {"marginBottom": "30px", "padding": "20px", "display": "none"}
            
            if view_type == "time":
                return (
                    visible_style,  # tube-selection-card
                    visible_style,  # viz-options-card
                    tube_options,   # tube-multiselect options
                    hidden_style,   # faceted-options-card
                    [],             # faceted-tube-selector options
                    [],             # faceted-date-selector options
                )
            elif view_type == "faceted":
                return (
                    hidden_style,   # tube-selection-card
                    hidden_style,   # viz-options-card
                    [],             # tube-multiselect options
                    visible_style,  # faceted-options-card
                    tube_options,   # faceted-tube-selector options
                    date_options,   # faceted-date-selector options
                )
            else:
                return (
                    hidden_style,   # tube-selection-card
                    hidden_style,   # viz-options-card
                    [],             # tube-multiselect options
                    hidden_style,   # faceted-options-card
                    [],             # faceted-tube-selector options
                    [],             # faceted-date-selector options
                )

        # Callback to manage faceted selection and show availability
        @self.app.callback(
            [
                Output("faceted-tubes-store", "data"),
                Output("faceted-dates-store", "data"),
                Output("faceted-selection-info", "children"),
                Output("tube-date-availability", "children"),
            ],
            [
                Input("faceted-tube-selector", "value"),
                Input("faceted-date-selector", "value"),
            ],
        )
        def manage_faceted_selection(selected_tubes, selected_dates):
            """Manage faceted selection and show availability info."""
            # Convert dates from ISO format
            if selected_dates:
                selected_dates = [pd.Timestamp(d) for d in selected_dates]
            else:
                selected_dates = []
            
            # Validate tubes
            if not selected_tubes:
                selected_tubes = []
            else:
                selected_tubes = sorted(selected_tubes)
            
            tube_count = len(selected_tubes)
            date_count = len(selected_dates)
            
            # Create selection info message
            if tube_count == 0:
                info_msg = html.Span("No tubes selected. Please select up to 6 tubes.", style={"color": "red"})
            elif tube_count > 6:
                info_msg = html.Span(
                    f"Too many tubes! {tube_count}/6 selected. Please remove {tube_count - 6} tubes.",
                    style={"color": "red"}
                )
            else:
                info_msg = html.Div([
                    html.Span(f"Tubes: {tube_count}/6 selected", style={"color": "green" if tube_count <= 6 else "red"}),
                    html.Br(),
                    html.Span(f"Dates: {date_count} selected (all dates if none selected)", style={"color": "blue"}),
                ])
            
            # Get availability info
            availability = self.get_tube_date_availability()
            
            if selected_tubes and selected_dates:
                # Show availability matrix for selected tubes and dates
                availability_display = []
                availability_display.append(html.Div("Availability Matrix (✓ = has data, ✗ = no data):", 
                                                    style={"fontWeight": "bold", "marginBottom": "10px"}))
                
                # Create table
                table_rows = []
                # Header row
                header = [html.Th("Tube/Date", style={"padding": "5px", "border": "1px solid #ccc"})]
                for date in sorted(selected_dates):
                    header.append(html.Th(date.strftime("%m/%d"), 
                                        style={"padding": "5px", "border": "1px solid #ccc", "fontSize": "10px"}))
                table_rows.append(html.Tr(header))
                
                # Data rows
                for tube in sorted(selected_tubes):
                    row = [html.Td(f"T{int(tube)}", style={"padding": "5px", "border": "1px solid #ccc", "fontWeight": "bold"})]
                    for date in sorted(selected_dates):
                        has_data = availability.get((tube, date), False)
                        symbol = "✓" if has_data else "✗"
                        color = "green" if has_data else "red"
                        row.append(html.Td(symbol, style={"padding": "5px", "border": "1px solid #ccc", 
                                                         "textAlign": "center", "color": color, "fontWeight": "bold"}))
                    table_rows.append(html.Tr(row))
                
                availability_display.append(
                    html.Table(table_rows, style={"borderCollapse": "collapse", "width": "100%"})
                )
            elif selected_tubes:
                # Show which dates have data for selected tubes
                availability_display = []
                availability_display.append(html.Div("Dates with data for selected tubes:", 
                                                    style={"fontWeight": "bold", "marginBottom": "10px"}))
                for tube in sorted(selected_tubes):
                    dates_with_data = [date.strftime("%Y-%m-%d") for tube_id, date in availability.keys() 
                                     if tube_id == tube and availability[(tube_id, date)]]
                    if dates_with_data:
                        availability_display.append(
                            html.Div(f"Tube {int(tube)}: {', '.join(dates_with_data)}", 
                                   style={"fontSize": "11px", "marginBottom": "5px"})
                        )
            else:
                availability_display = html.Div("Select tubes to see availability information.", 
                                               style={"color": "gray", "fontStyle": "italic"})
            
            return selected_tubes, selected_dates, info_msg, availability_display

        # Callback to manage tube selection from multiple sources
        @self.app.callback(
            Output("selected-tubes-store", "data"),
            [
                Input("tube-multiselect", "value"),
                Input("add-range-btn", "n_clicks"),
                Input("tube-manual-entry", "value"),
                Input("clear-all-btn", "n_clicks"),
                Input("select-all-btn", "n_clicks"),
            ],
            [
                State("tube-range-from", "value"),
                State("tube-range-to", "value"),
                State("selected-tubes-store", "data"),
            ],
        )
        def manage_tube_selection(
            multiselect_values,
            range_clicks,
            manual_entry,
            clear_clicks,
            select_all_clicks,
            range_from,
            range_to,
            current_selection,
        ):
            """Manage tube selection from multiple input methods."""
            ctx = dash.callback_context
            
            if not ctx.triggered:
                return current_selection or []
            
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
            
            # Clear all selections
            if trigger_id == "clear-all-btn":
                return []
            
            # Select all tubes
            if trigger_id == "select-all-btn":
                all_tubes = [int(tube) for tube in self.data_processor.get_unique_tubes()]
                return sorted(all_tubes)
            
            # Start with current selection
            selected = set(current_selection or [])
            
            # Add from multiselect dropdown
            if trigger_id == "tube-multiselect" and multiselect_values:
                selected = set(multiselect_values)
            
            # Add from range input
            if trigger_id == "add-range-btn" and range_from is not None and range_to is not None:
                if range_from <= range_to:
                    selected.update(range(int(range_from), int(range_to) + 1))
            
            # Add from manual entry
            if trigger_id == "tube-manual-entry" and manual_entry:
                parsed_tubes = parse_tube_selection(manual_entry)
                selected.update(parsed_tubes)
            
            return sorted(list(selected))

        # Callback to display selected tubes as badges
        @self.app.callback(
            Output("selected-tubes-display", "children"),
            [Input("selected-tubes-store", "data")],
        )
        def display_selected_tubes(selected_tubes):
            """Display selected tubes as removable badges."""
            if not selected_tubes:
                return html.Div(
                    "No tubes selected",
                    style={"color": "#999", "fontStyle": "italic", "padding": "10px"},
                )
            
            badges = []
            for tube in selected_tubes[:50]:  # Limit display to first 50
                badges.append(
                    html.Span(
                        f"Tube {tube}",
                        className="badge bg-primary",
                        style={
                            "marginRight": "5px",
                            "marginBottom": "5px",
                            "fontSize": "14px",
                            "padding": "8px 12px",
                        },
                    )
                )
            
            if len(selected_tubes) > 50:
                badges.append(
                    html.Span(
                        f"... and {len(selected_tubes) - 50} more",
                        style={
                            "color": "#666",
                            "fontStyle": "italic",
                            "marginLeft": "10px",
                        },
                    )
                )
            
            return html.Div(badges, style={"display": "flex", "flexWrap": "wrap"})

        # Callback to toggle legend visibility
        @self.app.callback(
            Output("legend-visible-store", "data"),
            [Input("toggle-legend-btn", "n_clicks")],
            [State("legend-visible-store", "data")],
        )
        def toggle_legend(n_clicks, current_state):
            """Toggle legend visibility."""
            if n_clicks:
                return not current_state
            return current_state

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
                Input("selected-tubes-store", "data"),
                Input("growth-viz-options", "value"),
                Input("legend-visible-store", "data"),
                Input("faceted-tubes-store", "data"),
                Input("faceted-dates-store", "data"),
            ],
        )
        def update_visualization(
            view_type, 
            selected_tube, 
            click_data, 
            selected_tubes, 
            viz_options,
            legend_visible,
            faceted_tubes,
            faceted_dates
        ):
            ctx = dash.callback_context
            default_style = {
                "backgroundColor": "white",
                "borderRadius": "8px",
                "padding": "12px",
                "boxShadow": "0 4px 8px rgba(0,0,0,0.15)",
                "height": "800px",
                "width": "100%",
                "marginBottom": "12px",
                "overflow": "hidden",
            }
            
            hidden_images_style = {
                "width": "100%",
                "display": "none",
                "flexWrap": "wrap",
                "justifyContent": "center",
                "marginTop": "12px",
                "padding": "12px",
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
                    # Parse visualization options
                    viz_options = viz_options or []
                    show_field_avg = "show_field_avg" in viz_options
                    show_field_var = "show_field_var" in viz_options
                    
                    return (
                        self.show_growth_over_time(
                            selected_tubes=selected_tubes,
                            show_field_average=show_field_avg,
                            show_field_variance=show_field_var,
                            show_legend=legend_visible,
                        ),
                        "",
                        {"display": "none"},
                        [],
                        default_style,
                        hidden_images_style,
                    )

                elif view_type == "faceted":
                    # Validate tubes selected (1-6)
                    if not faceted_tubes or len(faceted_tubes) > 6:
                        # Return empty figure with message
                        fig = go.Figure()
                        msg = "Please select between 1 and 6 tubes for the faceted view"
                        if faceted_tubes and len(faceted_tubes) > 6:
                            msg = f"Too many tubes selected ({len(faceted_tubes)}/6). Please remove {len(faceted_tubes) - 6} tubes."
                        fig.add_annotation(
                            text=msg,
                            xref="paper",
                            yref="paper",
                            x=0.5,
                            y=0.5,
                            showarrow=False,
                            font=dict(size=20, color="red"),
                            align="center",
                        )
                        fig.update_layout(
                            plot_bgcolor="white",
                            paper_bgcolor="white",
                            height=600,
                        )
                        return (
                            fig,
                            "",
                            {"display": "none"},
                            [],
                            default_style,
                            hidden_images_style,
                        )
                    
                    # Convert dates to Timestamps if provided
                    selected_dates_ts = None
                    if faceted_dates:
                        selected_dates_ts = [pd.Timestamp(d) if isinstance(d, str) else d 
                                           for d in faceted_dates]
                    
                    # Create faceted depth profile
                    faceted_style = default_style.copy()
                    faceted_style["height"] = "auto"  # Allow dynamic height based on dates
                    faceted_style["padding"] = "6px"
                    faceted_style["marginBottom"] = "8px"
                    
                    return (
                        self.create_faceted_depth_profile(faceted_tubes, selected_dates_ts),
                        "",
                        {"display": "none"},
                        [],
                        faceted_style,
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

                # Extract date from customdata (more reliable than parsing HTML)
                date_str = point.get("customdata")
                
                # Fallback: try parsing from text if customdata is not available
                if not date_str:
                    text_lines = point.get("text", "").split("<br>")
                    for line in text_lines:
                        if "Date:" in line:
                            # Extract date and remove HTML tags
                            date_part = line.split("Date:")[-1].strip()
                            if "<span" in date_part and "</span>" in date_part:
                                start_idx = date_part.find('>') + 1
                                end_idx = date_part.find('</span>')
                                date_str = date_part[start_idx:end_idx].strip()
                            else:
                                date_str = re.sub(r'<[^>]+>', '', date_part).strip()
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
            self.app.run(debug=False, port=8050, threaded=True)
        except Exception:
            pass

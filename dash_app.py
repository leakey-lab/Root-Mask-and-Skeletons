from dash import Dash, dcc, html, Input, Output
import dash
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd


class DashApp:
    """Manages the Dash application."""

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
                    "Root Length Visualization",
                    style={
                        "textAlign": "center",
                        "width": "100%",
                        "marginBottom": "30px",
                        "color": "#2c3e50",
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
        )
        def update_visualization(view_type, selected_tube, click_data, n_clicks):
            ctx = dash.callback_context
            if not ctx.triggered:
                # Initial load - show stacked view by default
                return (
                    self.create_stacked_bar_chart(),  # Changed default view
                    "",
                    "mt-2 d-none",
                    {"display": "none"},
                    [],
                )

            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

            try:
                if view_type == "stacked":
                    return (
                        self.create_stacked_bar_chart(),
                        "",
                        "mt-2 d-none",
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
                        return (
                            self.show_growth_lines(first_tube),
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

                elif view_type == "time":
                    return (
                        self.show_growth_over_time(),
                        "",
                        "mt-2 d-none",
                        {"display": "none"},
                        [],
                    )

            except Exception as e:
                print(f"Error in callback: {e}")
                return dash.no_update

            return dash.no_update

    def show_overview(self, view_type):
        """Generate the overview figure."""
        df = self.data_processor.df
        if view_type == "separate":
            lengths = df.groupby("tube_date")["Length (mm)"].sum().reset_index()
            x_values = lengths["tube_date"]
            title = "Root Length Overview by Date"
        else:
            lengths = df.groupby("Tube")["Length (mm)"].mean().reset_index()
            x_values = lengths["Tube"].apply(lambda x: f"Tube {int(x)}")
            title = "Root Length Overview"

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

        fig.update_layout(
            title=title,
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
        """Generate growth lines figure for a selected tube."""
        df = self.data_processor.df
        fig = go.Figure()

        try:
            tube_data = df[df["Tube"] == selected_tube].copy()
            dates = sorted(tube_data["Date"].unique())
            all_positions = sorted(tube_data["Position"].unique())
            interval_positions = all_positions[::10]  # Every 10th position

            colors = px.colors.qualitative.Plotly

            for i, date in enumerate(dates):
                date_data = tube_data[tube_data["Date"] == date]
                position_data = (
                    date_data.groupby("Position")["Length (mm)"].mean().reset_index()
                )

                if not position_data.empty:
                    smoothed_lengths = self.smooth_data(
                        position_data["Position"].values,
                        position_data["Length (mm)"].values,
                        interval_positions,
                    )

                    hover_info = self.generate_hover_info(date_data, interval_positions)

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

            max_length = tube_data["Length (mm)"].max()

            fig.update_layout(
                title=dict(
                    text=f"Root Growth - Tube {int(selected_tube)}",
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
                height=700,
                width=1000,
                margin=dict(t=80, b=60, l=80, r=120),
            )
        except Exception as e:
            print(f"Error generating growth lines: {e}")

        return fig

    def smooth_data(self, positions, lengths, interval_positions, window=5):
        """Apply moving average smoothing to data."""
        try:
            interp_positions = np.arange(min(positions), max(positions) + 1)
            interp_lengths = np.interp(interp_positions, positions, lengths)

            kernel = np.ones(window) / window
            smoothed = np.convolve(interp_lengths, kernel, mode="valid")
            smoothed_positions = interp_positions[
                window // 2 : -(window // 2) if window // 2 > 0 else None
            ]

            final_lengths = np.interp(interval_positions, smoothed_positions, smoothed)
            return final_lengths
        except Exception as e:
            print(f"Error smoothing data: {e}")
            return lengths

    def generate_hover_info(self, date_data, interval_positions):
        """Generate hover information for growth lines."""
        hover_info = []
        for pos in interval_positions:
            interval_values = date_data[
                (date_data["Position"] >= pos - 5) & (date_data["Position"] <= pos + 5)
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
        return hover_info

    def show_sections(self, tube_info):
        """Generate sections figure based on tube information."""
        df = self.data_processor.df
        try:
            if "(" in tube_info:
                tube = int(tube_info.split(" ")[1])
                date_str = tube_info.split("(")[1].rstrip(")")
                date = pd.to_datetime(date_str)
                section_data = df[(df["Tube"] == tube) & (df["Date"] == date)]
            else:
                tube = int(tube_info.split(" ")[1])
                section_data = df[df["Tube"] == tube]

            section_lengths = (
                section_data.groupby("Position")["Length (mm)"].mean().reset_index()
            )

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=[f"L{int(pos)}" for pos in section_lengths["Position"]],
                        y=section_lengths["Length (mm)"],
                        text=section_lengths["Length (mm)"].round(2),
                        textposition="auto",
                    )
                ]
            )

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
        except Exception as e:
            print(f"Error generating sections: {e}")
            return go.Figure()

    def show_time_series(self, tube_info, position):
        """Generate time series figure for a specific tube and position."""
        df = self.data_processor.df
        try:
            tube = int(tube_info.split(" ")[1])
            time_series = df[
                (df["Tube"] == tube) & (df["Position"] == position)
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
        except Exception as e:
            print(f"Error generating time series: {e}")
            return go.Figure()

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
            self.app.run_server(debug=False, port=8050, threaded=True)
        except Exception as e:
            print(f"Error running Dash server: {e}")

    def create_stacked_bar_chart(self):
        """Create a stacked bar chart showing root length by tube and date."""
        df = self.data_processor.df

        try:
            # Group data by tube and date
            grouped = df.groupby(["Tube", "Date"])["Length (mm)"].sum().reset_index()

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
                    value = date_data[date_data["Tube"] == tube]["Length (mm)"].values
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
                        + "Length: %{y:.2f} mm<br>"
                        + "<extra></extra>",
                    )
                )

            fig = go.Figure(data=traces)

            # Update layout for stacked bars
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
                width=1000,
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
                ),
                margin=dict(l=50, r=150, t=80, b=50),
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

        except Exception as e:
            print(f"Error generating stacked bar chart: {e}")
            return go.Figure()

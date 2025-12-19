"""
Visualization methods for the Dash root length application.
Contains all chart creation methods extracted from DashApp.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import warnings
from typing import Any


class DashVisualizations:
    """
    Mixin class providing visualization methods for DashApp.
    Should be used as a mixin with a class that has:
    - self.data_cache: DataCache instance
    - self.data_processor: DataProcessor instance
    - self.available_images: dict of available images
    """

    def show_growth_lines(self, selected_tube: int) -> go.Figure:
        """Generate growth lines figure for selected tube."""
        fig = go.Figure()
        colors = px.colors.qualitative.Plotly

        try:
            interval_size = 10
            max_length = 0
            positions = []

            for i, date in enumerate(self.data_cache.dates):
                interval_data = self.data_cache.get_interval_data(
                    selected_tube, date, interval_size
                )

                if not interval_data:
                    continue

                positions = sorted(interval_data.keys())
                averages = []
                hover_info = []

                for pos in positions:
                    stats = interval_data[pos]

                    # Check if we have corresponding image
                    date_str = date.strftime("%Y.%m.%d")
                    has_image = (selected_tube, pos, date_str) in self.available_images

                    if stats:
                        avg = stats["avg"]
                        averages.append(avg)
                        max_length = max(max_length, stats["max"])
                        hover_text = (
                            f"Interval L{pos - interval_size + 1}-L{pos}:<br>"
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
                            f"No data for interval L{pos - interval_size + 1}-L{pos}"
                        )

                    hover_info.append(hover_text)

                # Add trace for this date
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
                            "<extra></extra>"
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
                    "font": {"size": 24},
                },
                xaxis={
                    "title": "Root Length (mm)",
                    "showgrid": True,
                    "gridcolor": "lightgray",
                    "zeroline": True,
                    "zerolinecolor": "black",
                    "zerolinewidth": 2,
                    "range": [-1, max_length * 1.1] if max_length > 0 else [-1, 10],
                    "tickformat": ".1f",
                    "titlefont": {"size": 18},
                    "tickfont": {"size": 14},
                },
                yaxis={
                    "title": "Position (L)",
                    "showgrid": True,
                    "gridcolor": "lightgray",
                    "zeroline": False,
                    "autorange": "reversed",
                    "tickmode": "array",
                    "ticktext": [f"L{pos}" for pos in positions] if positions else [],
                    "tickvals": positions if positions else [],
                    "dtick": 10,
                    "titlefont": {"size": 18},
                    "tickfont": {"size": 14},
                },
                plot_bgcolor="white",
                legend={
                    "title": {"text": "Measurement Dates", "font": {"size": 18}},
                    "yanchor": "top",
                    "y": 0.99,
                    "xanchor": "left",
                    "x": 1.02,
                    "bgcolor": "rgba(255, 255, 255, 0.8)",
                    "bordercolor": "black",
                    "borderwidth": 1,
                    "font": {"size": 14},
                },
                hovermode="closest",
                height=520,
                autosize=True,
                margin={"t": 60, "b": 40, "l": 60, "r": 100},
            )

            # Add reference line if we have positions
            if positions:
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

    def show_growth_over_time(self) -> go.Figure:
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
                title=dict(
                    text="Root Growth Over Time",
                    font=dict(size=24)
                ),
                xaxis=dict(
                    title="Date",
                    titlefont=dict(size=18),
                    tickfont=dict(size=14)
                ),
                yaxis=dict(
                    title="Total Length (mm)",
                    titlefont=dict(size=18),
                    tickfont=dict(size=14)
                ),
                showlegend=True,
                autosize=True,
                margin=dict(l=60, r=100, t=60, b=40),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="white",
                height=720,
                legend=dict(font=dict(size=14)),
            )
            return fig
        except Exception:
            return go.Figure()

    def create_stacked_bar_chart(self) -> go.Figure:
        """Create an optimized stacked bar chart."""
        try:
            traces = []
            for date in self.data_cache.dates:
                trace_data = []
                for tube in self.data_cache.tubes:
                    value = self.data_cache.tube_date_groups.get((tube, date), 0)
                    trace_data.append(value)

                traces.append(
                    go.Bar(
                        name=date.strftime("%Y-%m-%d"),
                        x=[f"Tube {int(tube)}" for tube in self.data_cache.tubes],
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
                    "font": {"size": 24},
                },
                xaxis=dict(
                    title="Tube",
                    titlefont=dict(size=18),
                    tickfont=dict(size=14)
                ),
                yaxis=dict(
                    title="Total Length (mm)",
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
                legend={
                    "yanchor": "top",
                    "y": 0.99,
                    "xanchor": "left",
                    "x": 1.02,
                    "bgcolor": "rgba(255, 255, 255, 0.8)",
                    "bordercolor": "black",
                    "borderwidth": 1,
                    "font": {"size": 14},
                },
                margin={"l": 60, "r": 150, "t": 60, "b": 40},
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


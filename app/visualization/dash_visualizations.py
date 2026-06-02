"""
Visualization methods for the Dash root length application.
Contains all chart creation methods extracted from DashApp.
"""

import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from . import theme

logger = logging.getLogger(__name__)


def parse_tube_selection(text: str) -> list[int]:
    """
    Parse tube selection string like '1,3,5,10-15' into list [1,3,5,10,11,12,13,14,15].

    Malformed parts are skipped individually rather than discarding the whole
    selection, so e.g. "1,foo,3" yields [1, 3] (F-037).

    Args:
        text: Comma-separated string containing tube numbers and ranges

    Returns:
        Sorted list of unique tube numbers
    """
    if not text or not text.strip():
        return []

    tubes = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                # Handle range like "10-15"
                range_parts = part.split("-")
                if len(range_parts) == 2:
                    start, end = int(range_parts[0].strip()), int(range_parts[1].strip())
                    if start <= end:
                        tubes.extend(range(start, end + 1))
            else:
                # Handle single number
                tubes.append(int(part))
        except (ValueError, AttributeError):
            # Skip this malformed part but keep any valid ones already parsed.
            logger.debug("Skipping unparseable tube-selection part: %r", part)
            continue

    return sorted(set(tubes))


class DashVisualizations:
    """
    Mixin class providing visualization methods for DashApp.
    Should be used as a mixin with a class that has:
    - self.data_cache: DataCache instance
    - self.data_processor: DataProcessor instance
    - self.available_images: dict of available images
    """

    def get_tube_date_availability(self) -> dict:
        """
        Get information about which tubes have data for which dates.

        The result is memoized per (DataFrame identity) so repeated callback
        invocations do not re-run an O(tubes*dates) scan with a fresh boolean
        mask per cell (F-033). A single groupby computes the present pairs in
        one pass.

        Returns:
            dict: {(tube, date): has_data_bool}
        """
        df = self.data_processor.df

        # Invalidate the memo if the underlying DataFrame object changed.
        cache = getattr(self, "_availability_cache", None)
        if cache is not None and cache[0] is df:
            return cache[1]

        tubes = sorted(df["Tube"].unique())
        dates = sorted(df["Date"].unique())
        # Pairs that actually have at least one row.
        present = set(map(tuple, df[["Tube", "Date"]].drop_duplicates().itertuples(index=False)))

        availability = {(tube, date): (tube, date) in present for tube in tubes for date in dates}

        self._availability_cache = (df, availability)
        return availability

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
                custom_data = []  # Store date string for callback

                for pos in positions:
                    stats = interval_data[pos]

                    # Check if we have corresponding image
                    date_str = date.strftime("%Y.%m.%d")
                    has_image = (selected_tube, pos, date_str) in self.available_images

                    if stats:
                        avg = stats["avg"]
                        averages.append(avg)
                        max_length = max(max_length, stats["max"])
                        img_status = "Available" if has_image else "Not Available"
                        hover_text = (
                            f"Interval L{pos - interval_size + 1}-L{pos}  <b>{date_str}</b><br>"
                            f"Average  <b>{avg:.2f} mm</b><br>"
                            f"Range  <b>{stats['min']:.2f} - {stats['max']:.2f} mm</b><br>"
                            f"Std Dev  <b>+/-{stats['std']:.2f}</b><br>"
                            f"Measurements  <b>{stats['count']}</b><br>"
                            f"Image  <b>{img_status}</b>"
                        )
                        custom_data.append(date_str)
                    else:
                        averages.append(float("nan"))
                        hover_text = (
                            f"No Data  Interval L{pos - interval_size + 1}-L{pos}"
                        )
                        custom_data.append(date_str)

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
                        hovertemplate="%{text}<extra></extra>",
                        text=hover_info,
                        customdata=custom_data,  # Pass date string directly
                    )
                )

            # Update layout with optimized settings
            fig.update_layout(
                title_text=f"Root Growth - Tube {int(selected_tube)}",
                xaxis_title_text="Root Length (mm)",
                yaxis_title_text="Position (L)",
                legend_title_text="Measurement Dates",
                hovermode="closest",
            )
            fig.update_xaxes(
                range=[-1, max_length * 1.1] if max_length > 0 else [-1, 10],
                tickformat=".1f",
            )
            fig.update_yaxes(
                autorange="reversed",
                tickmode="array",
                ticktext=[f"L{pos}" for pos in positions] if positions else [],
                tickvals=positions if positions else [],
                dtick=10,
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

        except Exception:
            logger.exception("Error generating growth lines")
            fig = go.Figure()

        return fig

    def show_growth_over_time(
        self,
        selected_tubes: list[int] | None = None,
        show_field_average: bool = True,
        show_field_variance: bool = True,
        show_legend: bool = True,
    ) -> go.Figure:
        """
        Generate growth over time figure with flexible tube selection and variance options.

        Args:
            selected_tubes: List of tube numbers to display. If None or empty, no individual tubes shown.
            show_field_average: Whether to show the field average line
            show_field_variance: Whether to show shaded variance for field average
            show_legend: Whether to display the legend

        Returns:
            Plotly figure object
        """
        df = self.data_processor.df
        fig = go.Figure()

        try:
            colors = px.colors.qualitative.Plotly

            # Calculate field average statistics if requested
            if show_field_average:
                # First, calculate total length per tube per date
                tube_totals = df.groupby(["Date", "Tube"])["Length (mm)"].sum().reset_index()
                tube_totals.columns = ["Date", "Tube", "Total_Length"]

                # Create date mapping to merge measurement dates within ±3 days
                # (ONLY for the field average). The merge is deterministic:
                #   - dates are processed in ascending order;
                #   - the greedy ±3-day window is anchored on the earliest
                #     not-yet-assigned date;
                #   - the representative date is the one with the most measuring
                #     tubes, breaking ties toward the earliest date (idxmax on a
                #     date-sorted frame returns the first/earliest on a tie).
                # See F-034: this ordering is fixed so results do not depend on
                # row order, and merged tube totals are AVERAGED (below), not
                # summed, so a tube measured on two nearby dates is not double
                # counted.
                date_counts = (
                    tube_totals.groupby("Date").size().reset_index(name="measurement_count")
                )
                date_counts = date_counts.sort_values("Date").reset_index(drop=True)

                # Create a mapping of dates to their representative date
                date_mapping = {}
                processed_dates = set()

                for _, row in date_counts.iterrows():
                    current_date = row["Date"]

                    if current_date in processed_dates:
                        continue

                    # Find not-yet-assigned dates within ±3 days of current_date.
                    date_range_start = current_date - pd.Timedelta(days=3)
                    date_range_end = current_date + pd.Timedelta(days=3)

                    nearby_dates = date_counts[
                        (date_counts["Date"] >= date_range_start)
                        & (date_counts["Date"] <= date_range_end)
                        & (~date_counts["Date"].isin(processed_dates))
                    ]

                    if len(nearby_dates) > 1:
                        # Representative = most measurements; earliest date wins
                        # ties (frame is date-sorted, idxmax returns first max).
                        representative_date = nearby_dates.loc[
                            nearby_dates["measurement_count"].idxmax(), "Date"
                        ]

                        for nearby_date in nearby_dates["Date"]:
                            date_mapping[nearby_date] = representative_date
                            processed_dates.add(nearby_date)
                    else:
                        date_mapping[current_date] = current_date
                        processed_dates.add(current_date)

                # Apply the date mapping to merge dates
                tube_totals["Date"] = tube_totals["Date"].map(date_mapping)

                # Collapse merged dates by AVERAGING each tube's total length
                # across the dates that merged into one representative date
                # (F-034: previously summed, which double-counted re-measured
                # tubes). A tube measured once contributes its single total.
                tube_totals = (
                    tube_totals.groupby(["Date", "Tube"])["Total_Length"].mean().reset_index()
                )

                # Then calculate mean and std of those totals across tubes for each date
                field_stats = (
                    tube_totals.groupby("Date")["Total_Length"]
                    .agg(["mean", "std", "count"])
                    .reset_index()
                )
                field_stats.columns = ["Date", "mean_length", "std_length", "count"]

                # Sort by date
                field_stats = field_stats.sort_values("Date")

                # Add variance shading if requested
                if show_field_variance and field_stats["std_length"].notna().any():
                    # Upper bound
                    fig.add_trace(
                        go.Scatter(
                            x=field_stats["Date"],
                            y=field_stats["mean_length"] + field_stats["std_length"],
                            mode="lines",
                            line=dict(color="rgba(0,0,0,0)"),
                            showlegend=False,
                            hoverinfo="skip",
                            legendgroup="field_avg",
                        )
                    )

                    # Lower bound with fill
                    fig.add_trace(
                        go.Scatter(
                            x=field_stats["Date"],
                            y=field_stats["mean_length"] - field_stats["std_length"],
                            mode="lines",
                            fill="tonexty",
                            fillcolor="rgba(100,100,100,0.2)",
                            line=dict(color="rgba(0,0,0,0)"),
                            name="Field Variance (±1 SD)",
                            legendgroup="field_avg",
                            hovertemplate=theme.hover("Field Variance Range", [
                                ("Date", "%{x|%b %d, %Y}"),
                                ("Range", "+-1 Standard Deviation"),
                            ]),
                        )
                    )

                # Add mean line
                fig.add_trace(
                    go.Scatter(
                        x=field_stats["Date"],
                        y=field_stats["mean_length"],
                        mode="lines+markers",
                        name="Field Average",
                        line=dict(color="black", width=3),
                        marker=dict(size=8, color="black"),
                        legendgroup="field_avg",
                        hovertemplate=theme.hover("Field Average", [
                            ("Date", "%{x|%b %d, %Y}"),
                            ("Avg Length", "%{y:.2f} mm"),
                            ("Std Dev", "+/-%{customdata[0]:.2f} mm"),
                            ("Tubes", "%{customdata[1]:.0f}"),
                        ]),
                        customdata=field_stats[["std_length", "count"]].values,
                    )
                )

            # Add individual tube traces if selected
            if selected_tubes:
                for idx, tube in enumerate(selected_tubes):
                    tube_data = df[df["Tube"] == tube].sort_values("Date")

                    if tube_data.empty:
                        continue

                    # Calculate total length per date for this tube (NO date merging for individual tubes)
                    tube_stats = tube_data.groupby("Date")["Length (mm)"].sum().reset_index()
                    tube_stats.columns = ["Date", "total_length"]

                    color = colors[idx % len(colors)]

                    # Add main line
                    fig.add_trace(
                        go.Scatter(
                            x=tube_stats["Date"],
                            y=tube_stats["total_length"],
                            mode="lines+markers",
                            name=f"Tube {int(tube)}",
                            line=dict(color=color, width=2),
                            marker=dict(size=6),
                            hovertemplate=theme.hover(f"Tube {int(tube)}", [
                                ("Date", "%{x|%b %d, %Y}"),
                                ("Total Length", "%{y:.2f} mm"),
                            ]),
                        )
                    )

            # Update layout
            fig.update_layout(
                title_text="Root Growth Over Time",
                xaxis_title_text="Date",
                yaxis_title_text="Total Length (mm)",
                showlegend=show_legend,
                hovermode="closest",
            )

            return fig

        except Exception:
            logger.exception("Error generating growth over time")
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
                        hovertemplate=theme.hover("%{x}", [
                            ("Date", date.strftime('%b %d, %Y')),
                            ("Length", "%{y:.2f} mm"),
                        ]),
                    )
                )

            fig = go.Figure(data=traces)
            fig.update_layout(
                barmode="stack",
                title_text="Root Length Growth by Tube and Date",
                xaxis_title_text="Tube",
                yaxis_title_text="Total Length (mm)",
                legend_title_text="Measurement Date",
                hovermode="x unified",
            )

            return fig

        except Exception:
            logger.exception("Error generating stacked bar chart")
            return go.Figure()

    def create_faceted_depth_profile(
        self, selected_tubes: list[int], selected_dates: list[pd.Timestamp] | None = None
    ) -> go.Figure:
        """
        Create a faceted depth profile visualization showing horizontal bar charts
        of root length by depth for each date-tube combination with 10-position binning.

        Args:
            selected_tubes: List of tube numbers to display (up to 6)
            selected_dates: List of dates to display (if None, uses all available dates)

        Returns:
            Plotly figure with subplots in a grid (rows=dates, cols=tubes)
        """
        try:
            df = self.data_processor.df
            interval_size = 10

            # Validate input
            if not selected_tubes or len(selected_tubes) > 6:
                fig = go.Figure()
                fig.add_annotation(
                    text="Please select between 1 and 6 tubes for the faceted view",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=20, color="#9498ad"),
                )
                return fig

            # Handle date selection
            if selected_dates is None:
                selected_dates = sorted(df["Date"].unique())
            else:
                selected_dates = sorted(selected_dates)

            # Filter out dates that have no data for any selected tube
            # Use tube date availability to check which dates have data
            availability = self.get_tube_date_availability()
            dates_with_data = []
            for date in selected_dates:
                # Check if at least one selected tube has data for this date
                has_any_data = any(availability.get((tube, date), False) for tube in selected_tubes)
                if has_any_data:
                    dates_with_data.append(date)

            selected_dates = dates_with_data
            num_dates = len(selected_dates)
            num_tubes = len(selected_tubes)

            if num_dates == 0:
                fig = go.Figure()
                fig.add_annotation(
                    text="No dates with data for selected tubes",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=20, color="#9498ad"),
                )
                return fig

            # Create color scale (plasma)
            plasma_colors = px.colors.sequential.Plasma

            # Create subplots with titles only on first row
            # Subplot titles: show tube names only in first row, empty strings for other rows
            subplot_titles = []
            for date_idx in range(num_dates):
                for tube in selected_tubes:
                    if date_idx == 0:
                        # First row: show tube name
                        subplot_titles.append(f"Tube {int(tube)}")
                    else:
                        # Other rows: empty string
                        subplot_titles.append("")

            fig = make_subplots(
                rows=num_dates,
                cols=num_tubes,
                shared_xaxes=False,
                shared_yaxes=True,
                vertical_spacing=0.005,  # Tighter vertical spacing
                horizontal_spacing=0.008,  # Tighter horizontal spacing
                subplot_titles=subplot_titles,
                row_titles=[date.strftime("%Y-%m-%d") for date in selected_dates],
            )
            theme.style(fig, title="Faceted Depth Profile - Field Average (Left) vs Individual Tube (Right)")

            # Track legends and max values globally
            field_avg_added_to_legend = False
            tube_bar_added_to_legend = False
            global_max_depth = 0  # Track maximum depth position
            global_max_val = 0  # Track max value across all graphs

            # Calculate field average intervals for all dates (across ALL tubes, not just selected)
            field_intervals_by_date = {}
            for date in selected_dates:
                # Get all data for this date (all tubes)
                date_data = df[df["Date"] == date]
                if not date_data.empty:
                    positions = sorted(date_data["Position"].unique())
                    min_pos, max_pos = min(positions), max(positions)
                    first_interval_end = ((min_pos - 1) // interval_size + 1) * interval_size
                    last_interval_end = ((max_pos - 1) // interval_size + 1) * interval_size

                    field_intervals = {}
                    for interval_end in range(
                        first_interval_end, last_interval_end + 1, interval_size
                    ):
                        interval_start = interval_end - interval_size + 1
                        interval_data = date_data[
                            (date_data["Position"] >= interval_start)
                            & (date_data["Position"] <= interval_end)
                        ]

                        if not interval_data.empty:
                            field_intervals[interval_end] = {
                                "mean": interval_data["Length (mm)"].mean(),
                                "std": interval_data["Length (mm)"].std(),
                                "interval_start": interval_start,
                                "interval_end": interval_end,
                            }
                    field_intervals_by_date[date] = field_intervals

            # Add traces for each date-tube combination
            for date_idx, date in enumerate(selected_dates, start=1):
                for tube_idx, tube in enumerate(selected_tubes, start=1):
                    # Get interval data for this tube-date combination
                    tube_intervals = self.data_cache.get_interval_data(tube, date, interval_size)
                    field_intervals = field_intervals_by_date.get(date, {})

                    # Check if we have data for this combination
                    has_data = bool(
                        tube_intervals
                        and any(
                            not pd.isna(stats["avg"]) and stats["count"] > 0
                            for stats in tube_intervals.values()
                        )
                    )

                    if not has_data:
                        # Plot an empty graph with "No Data" text annotation
                        # Add a dummy invisible trace to establish the subplot
                        fig.add_trace(
                            go.Scatter(
                                x=[0],
                                y=[50],  # Middle of typical range
                                mode="text",
                                text=["No Data"],
                                textfont=dict(size=40, color="#9498ad"),
                                showlegend=False,
                                hoverinfo="skip",
                            ),
                            row=date_idx,
                            col=tube_idx,
                        )
                        continue

                    # Get sorted intervals
                    interval_ends = sorted(tube_intervals.keys())

                    # Prepare data for bars
                    tube_means = []
                    tube_stds = []
                    field_means = []
                    field_stds = []
                    y_positions = []
                    colors = []

                    for interval_end in interval_ends:
                        tube_stat = tube_intervals[interval_end]
                        if not pd.isna(tube_stat["avg"]) and tube_stat["count"] > 0:
                            tube_means.append(tube_stat["avg"])
                            tube_stds.append(
                                tube_stat["std"] if not pd.isna(tube_stat["std"]) else 0
                            )
                            y_positions.append(interval_end)

                            # Get field average for this interval
                            if interval_end in field_intervals:
                                field_means.append(field_intervals[interval_end]["mean"])
                                field_stds.append(
                                    field_intervals[interval_end]["std"]
                                    if not pd.isna(field_intervals[interval_end]["std"])
                                    else 0
                                )
                            else:
                                field_means.append(0)
                                field_stds.append(0)

                            # Color based on depth
                            norm_pos = (
                                (interval_end - interval_ends[0])
                                / (interval_ends[-1] - interval_ends[0])
                                if len(interval_ends) > 1
                                else 0.5
                            )
                            color_idx = int(norm_pos * (len(plasma_colors) - 1))
                            colors.append(plasma_colors[color_idx])

                    if not y_positions:
                        continue

                    # Track max values for axis scaling globally (across all graphs)
                    tube_max = max(field_means + tube_means, default=0)
                    global_max_val = max(global_max_val, tube_max)

                    global_max_depth = max(global_max_depth, max(y_positions, default=0))

                    # Add field average bars (extending LEFT from center).
                    # Show the legend entry on the FIRST data-bearing subplot
                    # only, and set the dedup flag right after the trace is added
                    # so a leading empty tube/date does not silence the legend
                    # forever (F-011).
                    show_field_legend = not field_avg_added_to_legend

                    fig.add_trace(
                        go.Bar(
                            x=[-val for val in field_means],  # Negative values to extend left
                            y=y_positions,  # Same y-position as tube bars (same depth)
                            orientation="h",
                            width=8,  # Increased bar width for thicker bins
                            base=0,
                            marker=dict(
                                color="lightgray",
                                line=dict(color="black", width=1),
                            ),
                            name="Field Avg",
                            showlegend=show_field_legend,
                            legendgroup="field_avg",
                            hovertemplate=(
                                f"<b>Field Average - {date.strftime('%Y-%m-%d')}</b><br>"
                                "Interval: L%{customdata[0]}-%{customdata[1]}<br>"
                                "Avg Length: %{customdata[2]:.2f} mm<br>"
                                "<extra></extra>"
                            ),
                            customdata=[
                                [
                                    tube_intervals[pos]["interval_start"],
                                    pos,
                                    field_means[i],
                                    field_stds[i],
                                ]
                                for i, pos in enumerate(y_positions)
                            ],
                        ),
                        row=date_idx,
                        col=tube_idx,
                    )
                    # Mark the field-avg legend entry as emitted now that a real
                    # trace carrying it has been added (F-011).
                    field_avg_added_to_legend = True

                    # Add tube bars (extending RIGHT from center). Same dedup
                    # handling as the field bars above (F-011).
                    show_tube_legend = not tube_bar_added_to_legend

                    fig.add_trace(
                        go.Bar(
                            x=tube_means,  # Positive values to extend right
                            y=y_positions,  # Same y-position as field bars (same depth)
                            orientation="h",
                            width=8,  # Increased bar width for thicker bins
                            base=0,
                            marker=dict(
                                color=colors,
                                line=dict(color="black", width=1),
                            ),
                            name="Tube",
                            showlegend=show_tube_legend,
                            legendgroup="tube",
                            hovertemplate=(
                                f"<b>Tube {int(tube)} - {date.strftime('%Y-%m-%d')}</b><br>"
                                "Interval: L%{customdata[0]}-%{customdata[1]}<br>"
                                "Avg Length: %{x:.2f} mm<br>"
                                "Count: %{customdata[2]}<br>"
                                "<extra></extra>"
                            ),
                            customdata=[
                                [
                                    tube_intervals[pos]["interval_start"],
                                    pos,
                                    tube_intervals[pos]["count"],
                                ]
                                for pos in y_positions
                            ],
                        ),
                        row=date_idx,
                        col=tube_idx,
                    )
                    # Mark the tube legend entry as emitted (F-011).
                    tube_bar_added_to_legend = True

            # Set y-axis range based on actual data
            y_max = global_max_depth + 10 if global_max_depth > 0 else 120

            # Update axes - only show labels on edges
            for date_idx in range(1, num_dates + 1):
                for tube_idx in range(1, num_tubes + 1):
                    # Y-axis: only show title on leftmost column, reversed for depth
                    fig.update_yaxes(
                        range=[y_max, 0],  # Reversed: start from 0 at top, go to max at bottom
                        title=dict(
                            text="Vertical Depth (cm)"
                            if tube_idx == 1 and date_idx == (num_dates + 1) // 2
                            else "",
                            font=dict(size=20),
                        ),
                        showgrid=True,
                        gridcolor="lightgray",
                        tickfont=dict(size=18),
                        row=date_idx,
                        col=tube_idx,
                    )

            # Update overall layout
            fig.update_layout(
                showlegend=True,
                height=max(1100, num_dates * 420),  # Larger height: ~420px per facet row
                hovermode="closest",
                barmode="overlay",  # Allow bars at same y-position to display independently
            )
            fig.update_annotations(font=dict(size=18))

            # Set y-axis range based on actual data
            y_max = global_max_depth + 10 if global_max_depth > 0 else 120

            # Calculate normalized x-axis range globally (use highest value across all graphs)
            # Round up to nearest 10th digit (20, 30, 40, etc.)
            def round_to_nearest_10(val):
                """Round up to nearest 10th digit."""
                if val <= 0:
                    return 10
                return int(np.ceil(val / 10.0) * 10)

            # Use global max value for all graphs
            if global_max_val > 0:
                # Add 10% padding and round to nearest 10
                padded_max = global_max_val * 1.1
                global_max_range = round_to_nearest_10(padded_max)
            else:
                global_max_range = 10

            # Update all x-axes to use global range and show labels only on bottom row of each column
            for date_idx in range(1, num_dates + 1):
                for tube_idx in range(1, num_tubes + 1):
                    # Calculate subplot number (row-major order for plotly subplots)
                    subplot_num = (date_idx - 1) * num_tubes + tube_idx
                    xaxis_key = f"xaxis{subplot_num}" if subplot_num > 1 else "xaxis"

                    # Add background rectangle and border for each mini graph
                    # Use domain coordinates so the box stays within each subplot
                    x_domain_ref = f"x{subplot_num} domain" if subplot_num > 1 else "x domain"
                    y_domain_ref = f"y{subplot_num} domain" if subplot_num > 1 else "y domain"
                    fig.add_shape(
                        type="rect",
                        xref=x_domain_ref,
                        yref=y_domain_ref,
                        x0=0,
                        y0=0,
                        x1=1,
                        y1=1,
                        fillcolor="rgba(40,42,55,0.55)",
                        line=dict(color="#2a2c39", width=1),
                        layer="below",  # Place behind the data
                    )

                    if xaxis_key in fig.layout:
                        # Create tick values using nearest 10th digit numbers (10, 20, 30, 40, etc.)
                        # Generate ticks at 10-unit intervals from -max to +max
                        tick_step = 10
                        tickvals = []
                        # Generate ticks from -max to +max at 10-unit intervals
                        # Round max_val down to nearest 10 for cleaner ticks
                        max_tick = (global_max_range // tick_step) * tick_step
                        for val in range(-max_tick, max_tick + tick_step, tick_step):
                            tickvals.append(val)
                        tickvals = sorted(set(tickvals))

                        # Show absolute values as labels, but hide the first/last labels
                        ticktext = []
                        for idx, val in enumerate(tickvals):
                            if idx == 0 or idx == len(tickvals) - 1:
                                ticktext.append("")
                            else:
                                ticktext.append(f"{abs(val):.0f}")

                        # Only show x-axis labels on bottom row of each column
                        show_labels = date_idx == num_dates

                        fig.layout[xaxis_key].tickmode = "array"
                        fig.layout[xaxis_key].tickvals = tickvals
                        fig.layout[xaxis_key].ticktext = ticktext
                        fig.layout[xaxis_key].range = [-global_max_range, global_max_range]
                        fig.layout[xaxis_key].showticklabels = show_labels
                        fig.layout[xaxis_key].showgrid = True
                        fig.layout[xaxis_key].gridcolor = "lightgray"
                        fig.layout[xaxis_key].zeroline = True
                        fig.layout[xaxis_key].zerolinecolor = "black"
                        fig.layout[xaxis_key].zerolinewidth = 2
                        fig.layout[xaxis_key].showline = show_labels
                        fig.layout[xaxis_key].linecolor = "black"
                        fig.layout[xaxis_key].linewidth = 2
                        fig.layout[xaxis_key].matches = "x"
                        fig.layout[xaxis_key].side = "bottom"
                        fig.layout[xaxis_key].ticks = "outside" if show_labels else ""
                        fig.layout[xaxis_key].ticklen = 6 if show_labels else 0
                        fig.layout[xaxis_key].tickfont = dict(size=16)

                    yaxis_key = f"yaxis{subplot_num}" if subplot_num > 1 else "yaxis"
                    if yaxis_key in fig.layout:
                        fig.layout[yaxis_key].showline = True
                        fig.layout[yaxis_key].linecolor = "black"
                        fig.layout[yaxis_key].linewidth = 2
                        fig.layout[yaxis_key].matches = "y"
                        # Hide 0 and max labels on the y-axis to reduce clutter
                        y_tick_step = 20
                        y_tickvals = list(range(0, int(y_max) + 1, y_tick_step))
                        if y_tickvals[-1] != int(y_max):
                            y_tickvals.append(int(y_max))
                        y_ticktext = []
                        for idx, val in enumerate(y_tickvals):
                            if val == 0 or val == int(y_max):
                                y_ticktext.append("")
                            else:
                                y_ticktext.append(f"{val}")
                        fig.layout[yaxis_key].tickmode = "array"
                        fig.layout[yaxis_key].tickvals = y_tickvals
                        fig.layout[yaxis_key].ticktext = y_ticktext

            return fig

        except Exception as e:
            logger.exception("Error generating faceted depth profile")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error: {str(e)}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16, color="#9498ad"),
            )
            return fig

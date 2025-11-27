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

    def show_treatment_comparison(self) -> go.Figure:
        """Box plot of Total Root Length by Treatment"""
        df = self.data_processor.df
        tube_totals = df.groupby(["Tube", "Treatment", "Date"])["Length (mm)"].sum().reset_index()
        tube_totals["DateStr"] = tube_totals["Date"].dt.strftime("%Y-%m-%d")
        
        fig = px.box(
            tube_totals, 
            x="Treatment", 
            y="Length (mm)", 
            color="Treatment",
            points="all",
            hover_data=["Tube", "DateStr"],
            title="Impact of Conditions on Total Root Length",
            color_discrete_map={
                "Drought": "#e74c3c",
                "Water (Control)": "#3498db",
                "Padding": "#95a5a6",
                "Unknown": "#7f8c8d"
            },
            facet_col="DateStr"
        )
        fig.update_layout(height=720)
        return fig

    def show_genotype_ranking(self) -> go.Figure:
        """Bar chart of Average Root Length by Genotype, split by Treatment"""
        df = self.data_processor.df
        tube_totals = df.groupby(["Tube", "Treatment", "Genotype", "Date"])["Length (mm)"].sum().reset_index()
        geno_stats = tube_totals.groupby(["Genotype", "Treatment", "Date"])["Length (mm)"].mean().reset_index()
        geno_stats["DateStr"] = geno_stats["Date"].dt.strftime("%Y-%m-%d")
        
        fig = px.bar(
            geno_stats,
            x="Genotype",
            y="Length (mm)",
            color="Treatment",
            barmode="group",
            facet_row="DateStr",
            title="Genotype Performance by Condition",
            labels={"Length (mm)": "Average Total Length (mm)"},
            color_discrete_map={
                "Drought": "#e74c3c",
                "Water (Control)": "#3498db",
                "Padding": "#95a5a6",
                "Unknown": "#7f8c8d"
            }
        )
        fig.update_layout(height=800)
        return fig

    def show_field_heatmap(self) -> go.Figure:
        """Heatmap of root length laid out physically by Field Row/Col"""
        df = self.data_processor.df
        tube_date_sums = df.groupby(["Tube", "Rng", "Col", "Date"])["Length (mm)"].sum().reset_index()
        plot_totals = tube_date_sums.groupby(["Tube", "Rng", "Col"])["Length (mm)"].max().reset_index()
        
        try:
            grid = plot_totals.pivot(index="Rng", columns="Col", values="Length (mm)")
            
            fig = px.imshow(
                grid,
                title="Spatial Distribution of Max Root Growth (Field Map)",
                labels=dict(x="Field Column", y="Field Row", color="Max Length (mm)"),
                color_continuous_scale="Viridis"
            )
            fig.update_layout(height=720)
            return fig
        except Exception as e:
            warnings.warn(f"Could not generate heatmap: {e}")
            return go.Figure()

    def show_genotype_comparison(self, mode: str, primary: str, secondary: list) -> go.Figure:
        """Interactive genotype comparison view."""
        df = self.data_processor.df
        
        if mode == "genotypes_in_treatment":
            # Compare multiple genotypes within one treatment
            if not primary or not secondary:
                return go.Figure().add_annotation(text="Please select a treatment and genotypes", showarrow=False)
            
            filtered = df[df["Treatment"] == primary]
            if isinstance(secondary, str):
                secondary = [secondary]
            filtered = filtered[filtered["Genotype"].isin(secondary)]
            
            # Aggregate per tube per date
            tube_totals = filtered.groupby(["Tube", "Genotype", "Date"])["Length (mm)"].sum().reset_index()
            
            # Calculate mean per genotype per date
            geno_means = tube_totals.groupby(["Genotype", "Date"])["Length (mm)"].agg(["mean", "std"]).reset_index()
            geno_means.columns = ["Genotype", "Date", "Mean", "Std"]
            geno_means["Std"] = geno_means["Std"].fillna(0)
            
            fig = go.Figure()
            colors = px.colors.qualitative.Set2
            
            for i, geno in enumerate(secondary):
                geno_data = geno_means[geno_means["Genotype"] == geno].sort_values("Date")
                color = colors[i % len(colors)]
                
                fig.add_trace(go.Scatter(
                    x=geno_data["Date"],
                    y=geno_data["Mean"],
                    mode="lines+markers",
                    name=geno,
                    line=dict(color=color, width=2),
                    marker=dict(size=8),
                ))
                
                # Add shaded error band
                fig.add_trace(go.Scatter(
                    x=pd.concat([geno_data["Date"], geno_data["Date"][::-1]]),
                    y=pd.concat([geno_data["Mean"] + geno_data["Std"], 
                                (geno_data["Mean"] - geno_data["Std"])[::-1]]),
                    fill='toself',
                    fillcolor=color.replace(')', ', 0.2)').replace('rgb', 'rgba') if 'rgb' in color else "rgba(100,100,100,0.2)",
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False,
                ))
            
            fig.update_layout(
                title=f"Genotype Comparison under {primary}",
                xaxis_title="Date",
                yaxis_title="Mean Total Root Length (mm)",
                height=720,
                plot_bgcolor="white",
                hovermode="x unified",
            )
            return fig
            
        else:  # genotype_across_treatments
            # Compare one genotype across all treatments
            if not primary:
                return go.Figure().add_annotation(text="Please select a genotype", showarrow=False)
            
            filtered = df[df["Genotype"] == primary]
            
            # Aggregate per tube per treatment per date
            tube_totals = filtered.groupby(["Tube", "Treatment", "Date"])["Length (mm)"].sum().reset_index()
            
            # Calculate mean per treatment per date
            treat_means = tube_totals.groupby(["Treatment", "Date"])["Length (mm)"].agg(["mean", "std"]).reset_index()
            treat_means.columns = ["Treatment", "Date", "Mean", "Std"]
            treat_means["Std"] = treat_means["Std"].fillna(0)
            
            fig = go.Figure()
            color_map = {
                "Drought": "#e74c3c",
                "Water (Control)": "#3498db",
                "Padding": "#95a5a6",
                "Unknown": "#7f8c8d"
            }
            
            for treatment in treat_means["Treatment"].unique():
                treat_data = treat_means[treat_means["Treatment"] == treatment].sort_values("Date")
                color = color_map.get(treatment, "#333333")
                
                fig.add_trace(go.Scatter(
                    x=treat_data["Date"],
                    y=treat_data["Mean"],
                    mode="lines+markers",
                    name=treatment,
                    line=dict(color=color, width=3),
                    marker=dict(size=10),
                ))
                
                # Add shaded error band
                upper = treat_data["Mean"] + treat_data["Std"]
                lower = treat_data["Mean"] - treat_data["Std"]
                
                fig.add_trace(go.Scatter(
                    x=pd.concat([treat_data["Date"], treat_data["Date"][::-1]]),
                    y=pd.concat([upper, lower[::-1]]),
                    fill='toself',
                    fillcolor=color.replace('#', '').replace(')', ', 0.2)'),
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False,
                    opacity=0.3,
                ))
            
            fig.update_layout(
                title=f"Genotype '{primary}' Across Treatments",
                xaxis_title="Date",
                yaxis_title="Mean Total Root Length (mm)",
                height=720,
                plot_bgcolor="white",
                hovermode="x unified",
            )
            return fig

    def show_depth_profile_comparison(self, mode: str, primary: str, secondary: list, date_selection: str) -> go.Figure:
        """Depth profile comparison with shaded variance."""
        df = self.data_processor.df
        interval_size = 10
        
        # Filter by date if specified
        if date_selection and date_selection != "all":
            target_date = pd.to_datetime(date_selection)
            df = df[df["Date"] == target_date]
        
        if df.empty:
            return go.Figure().add_annotation(text="No data for selected date", showarrow=False)
        
        fig = go.Figure()
        
        if mode == "genotypes_in_treatment":
            # Compare genotypes within one treatment
            if not primary or not secondary:
                return go.Figure().add_annotation(text="Please select a treatment and genotypes", showarrow=False)
            
            filtered = df[df["Treatment"] == primary]
            if isinstance(secondary, str):
                secondary = [secondary]
            
            colors = px.colors.qualitative.Set2
            
            for i, geno in enumerate(secondary):
                geno_data = filtered[filtered["Genotype"] == geno]
                if geno_data.empty:
                    continue
                
                # Calculate interval statistics
                positions = sorted(geno_data["Position"].unique())
                if not positions:
                    continue
                    
                min_pos = min(positions)
                max_pos = max(positions)
                first_interval_end = ((min_pos - 1) // interval_size + 1) * interval_size
                last_interval_end = ((max_pos - 1) // interval_size + 1) * interval_size
                
                interval_positions = []
                means = []
                stds = []
                
                for interval_end in range(first_interval_end, last_interval_end + 1, interval_size):
                    interval_start = interval_end - interval_size + 1
                    interval_data = geno_data[
                        (geno_data["Position"] >= interval_start) & 
                        (geno_data["Position"] <= interval_end)
                    ]
                    
                    if not interval_data.empty:
                        interval_positions.append(interval_end)
                        means.append(interval_data["Length (mm)"].mean())
                        stds.append(interval_data["Length (mm)"].std() if len(interval_data) > 1 else 0)
                
                if not interval_positions:
                    continue
                
                color = colors[i % len(colors)]
                means = np.array(means)
                stds = np.array(stds)
                stds = np.nan_to_num(stds, nan=0.0)
                
                # Main line
                fig.add_trace(go.Scatter(
                    x=means,
                    y=interval_positions,
                    mode="lines+markers",
                    name=geno,
                    line=dict(color=color, width=2),
                    marker=dict(size=8),
                ))
                
                # Shaded area for std
                upper = means + stds
                lower = means - stds
                lower = np.maximum(lower, 0)  # No negative lengths
                
                fig.add_trace(go.Scatter(
                    x=np.concatenate([upper, lower[::-1]]),
                    y=np.concatenate([interval_positions, interval_positions[::-1]]),
                    fill='toself',
                    fillcolor="rgba(100,100,100,0.15)",
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False,
                ))
            
            title = f"Depth Profile: Genotypes under {primary}"
            
        else:  # genotype_across_treatments
            # Compare treatments for one genotype
            if not primary:
                return go.Figure().add_annotation(text="Please select a genotype", showarrow=False)
            
            filtered = df[df["Genotype"] == primary]
            
            color_map = {
                "Drought": "#e74c3c",
                "Water (Control)": "#3498db",
                "Padding": "#95a5a6",
                "Unknown": "#7f8c8d"
            }
            
            for treatment in filtered["Treatment"].unique():
                treat_data = filtered[filtered["Treatment"] == treatment]
                if treat_data.empty:
                    continue
                
                positions = sorted(treat_data["Position"].unique())
                if not positions:
                    continue
                    
                min_pos = min(positions)
                max_pos = max(positions)
                first_interval_end = ((min_pos - 1) // interval_size + 1) * interval_size
                last_interval_end = ((max_pos - 1) // interval_size + 1) * interval_size
                
                interval_positions = []
                means = []
                stds = []
                
                for interval_end in range(first_interval_end, last_interval_end + 1, interval_size):
                    interval_start = interval_end - interval_size + 1
                    interval_data = treat_data[
                        (treat_data["Position"] >= interval_start) & 
                        (treat_data["Position"] <= interval_end)
                    ]
                    
                    if not interval_data.empty:
                        interval_positions.append(interval_end)
                        means.append(interval_data["Length (mm)"].mean())
                        stds.append(interval_data["Length (mm)"].std() if len(interval_data) > 1 else 0)
                
                if not interval_positions:
                    continue
                
                color = color_map.get(treatment, "#333333")
                means = np.array(means)
                stds = np.array(stds)
                stds = np.nan_to_num(stds, nan=0.0)
                
                # Main line
                fig.add_trace(go.Scatter(
                    x=means,
                    y=interval_positions,
                    mode="lines+markers",
                    name=treatment,
                    line=dict(color=color, width=3),
                    marker=dict(size=10),
                ))
                
                # Shaded area for std
                upper = means + stds
                lower = means - stds
                lower = np.maximum(lower, 0)
                
                fig.add_trace(go.Scatter(
                    x=np.concatenate([upper, lower[::-1]]),
                    y=np.concatenate([interval_positions, interval_positions[::-1]]),
                    fill='toself',
                    fillcolor="rgba(100,100,100,0.15)",
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False,
                ))
            
            title = f"Depth Profile: '{primary}' across Treatments"
        
        date_label = date_selection if date_selection != "all" else "All Dates Averaged"
        
        fig.update_layout(
            title=f"{title} ({date_label})",
            xaxis_title="Mean Root Length (mm)",
            yaxis_title="Depth Position",
            yaxis=dict(autorange="reversed"),
            height=720,
            plot_bgcolor="white",
            hovermode="closest",
        )
        
        return fig

    def show_growth_trajectory(self, mode: str, primary: str, secondary: list) -> go.Figure:
        """Growth trajectory comparison over time."""
        df = self.data_processor.df
        
        if mode == "genotypes_in_treatment":
            if not primary or not secondary:
                return go.Figure().add_annotation(text="Please select a treatment and genotypes", showarrow=False)
            
            filtered = df[df["Treatment"] == primary]
            if isinstance(secondary, str):
                secondary = [secondary]
            filtered = filtered[filtered["Genotype"].isin(secondary)]
            
            # Aggregate: total length per tube per date, then mean per genotype
            tube_totals = filtered.groupby(["Tube", "Genotype", "Date"])["Length (mm)"].sum().reset_index()
            geno_means = tube_totals.groupby(["Genotype", "Date"])["Length (mm)"].agg(["mean", "std"]).reset_index()
            geno_means.columns = ["Genotype", "Date", "Mean", "Std"]
            geno_means["Std"] = geno_means["Std"].fillna(0)
            
            fig = go.Figure()
            colors = px.colors.qualitative.Set2
            
            for i, geno in enumerate(secondary):
                geno_data = geno_means[geno_means["Genotype"] == geno].sort_values("Date")
                if geno_data.empty:
                    continue
                    
                color = colors[i % len(colors)]
                
                fig.add_trace(go.Scatter(
                    x=geno_data["Date"],
                    y=geno_data["Mean"],
                    mode="lines+markers",
                    name=geno,
                    line=dict(color=color, width=2),
                    marker=dict(size=8),
                    error_y=dict(
                        type='data',
                        array=geno_data["Std"],
                        visible=True,
                        color=color,
                        thickness=1.5,
                        width=3,
                    )
                ))
            
            fig.update_layout(
                title=f"Growth Trajectory: Genotypes under {primary}",
                xaxis_title="Date",
                yaxis_title="Mean Total Root Length (mm)",
                height=720,
                plot_bgcolor="white",
                hovermode="x unified",
            )
            return fig
            
        else:  # genotype_across_treatments
            if not primary:
                return go.Figure().add_annotation(text="Please select a genotype", showarrow=False)
            
            filtered = df[df["Genotype"] == primary]
            
            tube_totals = filtered.groupby(["Tube", "Treatment", "Date"])["Length (mm)"].sum().reset_index()
            treat_means = tube_totals.groupby(["Treatment", "Date"])["Length (mm)"].agg(["mean", "std"]).reset_index()
            treat_means.columns = ["Treatment", "Date", "Mean", "Std"]
            treat_means["Std"] = treat_means["Std"].fillna(0)
            
            fig = go.Figure()
            color_map = {
                "Drought": "#e74c3c",
                "Water (Control)": "#3498db",
                "Padding": "#95a5a6",
                "Unknown": "#7f8c8d"
            }
            
            for treatment in treat_means["Treatment"].unique():
                treat_data = treat_means[treat_means["Treatment"] == treatment].sort_values("Date")
                if treat_data.empty:
                    continue
                    
                color = color_map.get(treatment, "#333333")
                
                fig.add_trace(go.Scatter(
                    x=treat_data["Date"],
                    y=treat_data["Mean"],
                    mode="lines+markers",
                    name=treatment,
                    line=dict(color=color, width=3),
                    marker=dict(size=10),
                    error_y=dict(
                        type='data',
                        array=treat_data["Std"],
                        visible=True,
                        color=color,
                        thickness=1.5,
                        width=3,
                    )
                ))
            
            fig.update_layout(
                title=f"Growth Trajectory: '{primary}' across Treatments",
                xaxis_title="Date",
                yaxis_title="Mean Total Root Length (mm)",
                height=720,
                plot_bgcolor="white",
                hovermode="x unified",
            )
            return fig

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


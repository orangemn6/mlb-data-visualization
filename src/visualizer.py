"""
Interactive visualization engine for baseball spray charts.

This module creates Plotly-based interactive visualizations with baseball field
overlays, stadium dimensions, and advanced filtering capabilities.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import streamlit as st
import math
from .coordinate_transform import get_field_boundary_coordinates
from .stadium_simulator import get_stadium_simulator


class SprayChartVisualizer:
    """Creates interactive spray chart visualizations with stadium overlays."""

    def __init__(self):
        """Initialize the visualizer."""
        self.stadium_simulator = get_stadium_simulator()
        self.color_scales = {
            "exit_velocity": "Viridis",
            "launch_angle": "RdYlBu",
            "distance": "Plasma",
            "woba": "Cividis"
        }

    def create_spray_chart(
        self,
        hit_data: pd.DataFrame,
        color_by: str = "launch_speed",
        size_by: str = "launch_angle",
        stadium_overlays: List[str] = None,
        show_field_lines: bool = True,
        chart_title: str = None,
        point_size_range: Tuple[int, int] = (4, 20),
        opacity: float = 0.7
    ) -> go.Figure:
        """
        Create an interactive spray chart visualization.

        Args:
            hit_data: DataFrame with hit coordinate data
            color_by: Column to use for color coding
            size_by: Column to use for point sizing
            stadium_overlays: List of stadiums to overlay
            show_field_lines: Whether to show field boundary lines
            chart_title: Custom chart title
            point_size_range: (min_size, max_size) for point sizing
            opacity: Point opacity (0.0 to 1.0)

        Returns:
            Plotly figure object
        """
        fig = go.Figure()

        if hit_data.empty:
            fig.add_annotation(
                text="No data to display",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            return fig

        # Add field boundary lines if requested
        if show_field_lines:
            fig = self._add_field_boundaries(fig)

        # Add stadium overlays if requested
        if stadium_overlays:
            for stadium_key in stadium_overlays:
                fig = self._add_stadium_overlay(fig, stadium_key)

        # Prepare data for visualization
        viz_data = self._prepare_visualization_data(hit_data, color_by, size_by)

        if viz_data.empty:
            return fig

        # Create scatter plot
        scatter_trace = self._create_scatter_trace(
            viz_data, color_by, size_by, point_size_range, opacity
        )

        fig.add_trace(scatter_trace)

        # Configure layout
        fig = self._configure_layout(fig, chart_title, hit_data)

        return fig

    def _prepare_visualization_data(
        self,
        hit_data: pd.DataFrame,
        color_by: str,
        size_by: str
    ) -> pd.DataFrame:
        """Prepare and validate data for visualization."""
        viz_data = hit_data.copy()

        # Ensure coordinate columns exist
        if 'x_viz' not in viz_data.columns or 'y_viz' not in viz_data.columns:
            st.error("Hit coordinate data not found. Please ensure data includes 'x_viz' and 'y_viz' columns.")
            return pd.DataFrame()

        # Filter out invalid coordinates
        viz_data = viz_data[
            (viz_data['x_viz'].notna()) &
            (viz_data['y_viz'].notna())
        ].copy()

        # Validate color column
        if color_by not in viz_data.columns:
            st.warning(f"Color column '{color_by}' not found. Using default coloring.")
            viz_data['_color_default'] = 1
            color_by = '_color_default'

        # Validate size column
        if size_by not in viz_data.columns:
            st.warning(f"Size column '{size_by}' not found. Using default sizing.")
            viz_data['_size_default'] = 10
            size_by = '_size_default'

        return viz_data

    def _create_scatter_trace(
        self,
        viz_data: pd.DataFrame,
        color_by: str,
        size_by: str,
        point_size_range: Tuple[int, int],
        opacity: float
    ) -> go.Scatter:
        """Create the main scatter plot trace."""
        # Prepare hover text
        hover_text = self._create_hover_text(viz_data)

        # Prepare color values
        color_values = viz_data[color_by]
        if pd.api.types.is_numeric_dtype(color_values):
            color_values = color_values.fillna(0)
        else:
            # Convert categorical to numeric
            unique_vals = color_values.dropna().unique()
            color_map = {val: i for i, val in enumerate(unique_vals)}
            color_values = color_values.map(color_map).fillna(-1)

        # Prepare size values
        size_values = viz_data[size_by]
        if pd.api.types.is_numeric_dtype(size_values):
            size_values = size_values.fillna(size_values.median())
            # Normalize size values to point_size_range
            if size_values.std() > 0:
                size_normalized = (size_values - size_values.min()) / (size_values.max() - size_values.min())
                size_final = point_size_range[0] + size_normalized * (point_size_range[1] - point_size_range[0])
            else:
                size_final = [point_size_range[0]] * len(size_values)
        else:
            size_final = [point_size_range[0]] * len(viz_data)

        # Choose color scale
        color_scale = self.color_scales.get(color_by, "Viridis")

        scatter_trace = go.Scatter(
            x=viz_data['x_viz'],
            y=viz_data['y_viz'],
            mode='markers',
            marker=dict(
                size=size_final,
                color=color_values,
                colorscale=color_scale,
                opacity=opacity,
                line=dict(width=0.5, color='rgba(200, 200, 200, 0.3)'),
                colorbar=dict(
                    title=dict(
                        text=self._get_column_display_name(color_by),
                        font=dict(color="white")
                    ),
                    tickfont=dict(color="white"),
                    bgcolor='rgba(14, 17, 23, 0.8)',
                    bordercolor='rgba(128, 128, 128, 0.5)',
                    borderwidth=1
                )
            ),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            name='Batted Balls'
        )

        return scatter_trace

    def _create_hover_text(self, viz_data: pd.DataFrame) -> List[str]:
        """Create detailed hover text for each data point."""
        hover_texts = []

        for _, row in viz_data.iterrows():
            text_parts = []

            # Basic hit info
            if 'events' in row and pd.notna(row['events']):
                text_parts.append(f"<b>Result:</b> {row['events'].title()}")

            if 'game_date' in row and pd.notna(row['game_date']):
                date_str = pd.to_datetime(row['game_date']).strftime('%Y-%m-%d')
                text_parts.append(f"<b>Date:</b> {date_str}")

            # Ball flight data
            if 'launch_speed' in row and pd.notna(row['launch_speed']):
                text_parts.append(f"<b>Exit Velocity:</b> {row['launch_speed']:.1f} mph")

            if 'launch_angle' in row and pd.notna(row['launch_angle']):
                text_parts.append(f"<b>Launch Angle:</b> {row['launch_angle']:.1f}°")

            if 'hit_distance_sc' in row and pd.notna(row['hit_distance_sc']):
                text_parts.append(f"<b>Distance:</b> {row['hit_distance_sc']:.0f} ft")

            # Calculated values
            if 'distance_feet' in row and pd.notna(row['distance_feet']):
                text_parts.append(f"<b>Calculated Distance:</b> {row['distance_feet']:.0f} ft")

            if 'angle_degrees' in row and pd.notna(row['angle_degrees']):
                direction = "Right" if row['angle_degrees'] > 0 else "Left"
                text_parts.append(f"<b>Direction:</b> {abs(row['angle_degrees']):.1f}° {direction}")

            # Pitch info
            if 'pitch_type' in row and pd.notna(row['pitch_type']):
                text_parts.append(f"<b>Pitch Type:</b> {row['pitch_type']}")

            # Game situation
            if 'balls' in row and 'strikes' in row:
                if pd.notna(row['balls']) and pd.notna(row['strikes']):
                    text_parts.append(f"<b>Count:</b> {int(row['balls'])}-{int(row['strikes'])}")

            # Stadium-specific data
            stadium_hrs = []
            for col in viz_data.columns:
                if col.startswith('hr_prob_') and pd.notna(row[col]) and row[col] > 0:
                    stadium_name = col.replace('hr_prob_', '').replace('_', ' ').title()
                    stadium_hrs.append(stadium_name)

            if stadium_hrs:
                text_parts.append(f"<b>Home Run In:</b> {', '.join(stadium_hrs)}")

            hover_texts.append('<br>'.join(text_parts))

        return hover_texts

    def _add_field_boundaries(self, fig: go.Figure) -> go.Figure:
        """Add baseball field boundary lines to the figure."""
        field_coords = get_field_boundary_coordinates(max_distance=450)

        # Add foul lines
        fig.add_trace(go.Scatter(
            x=field_coords['left_foul_line']['x'],
            y=field_coords['left_foul_line']['y'],
            mode='lines',
            line=dict(color='#32CD32', width=2, dash='dash'),  # Bright green for dark mode
            name='Left Foul Line',
            showlegend=False,
            hoverinfo='skip'
        ))

        fig.add_trace(go.Scatter(
            x=field_coords['right_foul_line']['x'],
            y=field_coords['right_foul_line']['y'],
            mode='lines',
            line=dict(color='#32CD32', width=2, dash='dash'),  # Bright green for dark mode
            name='Right Foul Line',
            showlegend=False,
            hoverinfo='skip'
        ))

        # Add outfield arc
        fig.add_trace(go.Scatter(
            x=field_coords['outfield_arc']['x'],
            y=field_coords['outfield_arc']['y'],
            mode='lines',
            line=dict(color='#D2B48C', width=1, dash='dot'),  # Tan/wheat color for dark mode
            name='Field Boundary',
            showlegend=False,
            hoverinfo='skip'
        ))

        return fig

    def _add_stadium_overlay(self, fig: go.Figure, stadium_key: str) -> go.Figure:
        """Add stadium dimension overlay to the figure."""
        stadium_info = self.stadium_simulator.get_stadium_info(stadium_key)
        if not stadium_info:
            return fig

        dimensions = stadium_info.get("dimensions", {})

        # Create stadium boundary points
        stadium_points = self._calculate_stadium_boundary_points(dimensions)

        if stadium_points:
            stadium_name = stadium_info.get("name", stadium_key)
            color = self._get_stadium_color(stadium_key)

            fig.add_trace(go.Scatter(
                x=stadium_points['x'],
                y=stadium_points['y'],
                mode='lines',
                line=dict(color=color, width=3),
                name=f"{stadium_name} Walls",
                hoverinfo='name',
                showlegend=True
            ))

        return fig

    def _calculate_stadium_boundary_points(self, dimensions: Dict) -> Optional[Dict]:
        """Calculate points for drawing stadium boundary."""
        try:
            # Key dimensions
            lf_line = dimensions.get("left_field_line", 330)
            lf_center = dimensions.get("left_center", 375)
            center = dimensions.get("center_field", 400)
            rf_center = dimensions.get("right_center", 375)
            rf_line = dimensions.get("right_field_line", 330)

            # Calculate angles for each segment
            angles = np.array([-45, -22.5, 0, 22.5, 45])  # degrees from center
            distances = np.array([lf_line, lf_center, center, rf_center, rf_line])

            # Convert to coordinates
            x_coords = []
            y_coords = []

            for angle, distance in zip(angles, distances):
                angle_rad = np.radians(angle)
                x = distance * np.sin(angle_rad)
                y = distance * np.cos(angle_rad)
                x_coords.append(x)
                y_coords.append(y)

            # Close the shape by connecting back to start
            x_coords.append(x_coords[0])
            y_coords.append(y_coords[0])

            return {'x': x_coords, 'y': y_coords}

        except Exception:
            return None

    def _get_stadium_color(self, stadium_key: str) -> str:
        """Get a consistent color for each stadium."""
        # Bright colors that work well on dark backgrounds
        colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#FFA500', '#FF4757', '#9B59B6', '#3498DB',
            '#1ABC9C', '#2ECC71', '#F1C40F', '#E67E22', '#E91E63',
            '#00D2D3', '#FF9FF3', '#54A0FF', '#5F27CD', '#FD79A8'
        ]

        # Use hash of stadium key to get consistent color
        color_index = hash(stadium_key) % len(colors)
        return colors[color_index]

    def _configure_layout(
        self,
        fig: go.Figure,
        chart_title: str = None,
        hit_data: pd.DataFrame = None
    ) -> go.Figure:
        """Configure the chart layout and styling."""
        # Determine chart bounds
        if hit_data is not None and not hit_data.empty:
            x_range = self._calculate_axis_range(hit_data['x_viz'])
            y_range = self._calculate_axis_range(hit_data['y_viz'])
        else:
            x_range = [-250, 250]
            y_range = [0, 450]

        # Generate title
        if chart_title is None:
            if hit_data is not None and 'player_name' in hit_data.columns:
                player_names = hit_data['player_name'].dropna().unique()
                if len(player_names) == 1:
                    chart_title = f"Spray Chart - {player_names[0]}"
                else:
                    chart_title = "Spray Chart - Multiple Players"
            else:
                chart_title = "Baseball Spray Chart"

        # Dark theme configuration
        fig.update_layout(
            title=dict(
                text=chart_title,
                x=0.5,
                font=dict(size=20, family="Arial, sans-serif", color="white")
            ),
            xaxis=dict(
                title=dict(
                    text="Distance from Center (feet)",
                    font=dict(color="white")
                ),
                range=x_range,
                scaleanchor="y",
                scaleratio=1,
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.3)',
                gridwidth=1,
                zeroline=True,
                zerolinecolor='rgba(255, 255, 255, 0.8)',
                zerolinewidth=2,
                tickfont=dict(color="white"),
                linecolor='rgba(128, 128, 128, 0.5)'
            ),
            yaxis=dict(
                title=dict(
                    text="Distance from Home Plate (feet)",
                    font=dict(color="white")
                ),
                range=y_range,
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.3)',
                gridwidth=1,
                zeroline=True,
                zerolinecolor='rgba(255, 255, 255, 0.8)',
                zerolinewidth=2,
                tickfont=dict(color="white"),
                linecolor='rgba(128, 128, 128, 0.5)'
            ),
            plot_bgcolor='rgba(14, 17, 23, 1)',  # Dark background matching Streamlit
            paper_bgcolor='rgba(14, 17, 23, 1)',  # Dark paper background
            font=dict(color="white"),
            width=800,
            height=700,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.01,
                bgcolor='rgba(14, 17, 23, 0.8)',
                bordercolor='rgba(128, 128, 128, 0.5)',
                borderwidth=1,
                font=dict(color="white")
            )
        )

        return fig

    def _calculate_axis_range(self, values: pd.Series, padding: float = 0.1) -> List[float]:
        """Calculate appropriate axis range with padding."""
        if values.empty:
            return [-250, 250]

        min_val = values.min()
        max_val = values.max()
        range_span = max_val - min_val

        if range_span == 0:
            return [min_val - 50, max_val + 50]

        padding_amount = range_span * padding
        return [min_val - padding_amount, max_val + padding_amount]

    def _get_column_display_name(self, column: str) -> str:
        """Get user-friendly display name for column."""
        display_names = {
            'launch_speed': 'Exit Velocity (mph)',
            'launch_angle': 'Launch Angle (°)',
            'hit_distance_sc': 'Distance (ft)',
            'woba_value': 'wOBA',
            'estimated_woba_using_speedangle': 'Est. wOBA',
            'distance_feet': 'Distance (ft)',
            'angle_degrees': 'Angle (°)'
        }

        return display_names.get(column, column.replace('_', ' ').title())

    def create_comparison_chart(
        self,
        hit_data: pd.DataFrame,
        stadium_keys: List[str],
        chart_type: str = "home_runs"
    ) -> go.Figure:
        """
        Create a comparison chart showing differences across stadiums.

        Args:
            hit_data: DataFrame with stadium comparison data
            stadium_keys: List of stadiums to compare
            chart_type: Type of comparison ('home_runs', 'park_factors')

        Returns:
            Plotly figure object
        """
        if chart_type == "home_runs":
            return self._create_home_run_comparison(hit_data, stadium_keys)
        elif chart_type == "park_factors":
            return self._create_park_factor_chart(hit_data, stadium_keys)
        else:
            st.error(f"Unknown chart type: {chart_type}")
            return go.Figure()

    def _create_home_run_comparison(
        self,
        hit_data: pd.DataFrame,
        stadium_keys: List[str]
    ) -> go.Figure:
        """Create bar chart comparing home runs across stadiums."""
        stadium_hr_counts = []
        stadium_names = []

        for stadium_key in stadium_keys:
            hr_col = f'hr_prob_{stadium_key}'
            if hr_col in hit_data.columns:
                hr_count = hit_data[hr_col].sum()
                stadium_info = self.stadium_simulator.get_stadium_info(stadium_key)
                stadium_name = stadium_info.get("name", stadium_key) if stadium_info else stadium_key

                stadium_hr_counts.append(hr_count)
                stadium_names.append(stadium_name)

        fig = go.Figure(data=[
            go.Bar(
                x=stadium_names,
                y=stadium_hr_counts,
                marker_color=[self._get_stadium_color(key) for key in stadium_keys[:len(stadium_names)]],
                text=stadium_hr_counts,
                textposition='auto',
                textfont=dict(color='white')
            )
        ])

        fig.update_layout(
            title=dict(
                text="Home Runs by Stadium",
                font=dict(color="white")
            ),
            xaxis=dict(
                title=dict(
                    text="Stadium",
                    font=dict(color="white")
                ),
                tickfont=dict(color="white"),
                linecolor='rgba(128, 128, 128, 0.5)'
            ),
            yaxis=dict(
                title=dict(
                    text="Total Home Runs",
                    font=dict(color="white")
                ),
                tickfont=dict(color="white"),
                linecolor='rgba(128, 128, 128, 0.5)',
                gridcolor='rgba(128, 128, 128, 0.3)'
            ),
            plot_bgcolor='rgba(14, 17, 23, 1)',
            paper_bgcolor='rgba(14, 17, 23, 1)',
            font=dict(color="white"),
            showlegend=False
        )

        return fig

    def _create_park_factor_chart(
        self,
        hit_data: pd.DataFrame,
        stadium_keys: List[str]
    ) -> go.Figure:
        """Create chart showing park factors for each stadium."""
        # This is a placeholder implementation for park factors
        # In a real implementation, you would calculate park factors
        # based on league average home run rates vs stadium-specific rates

        fig = go.Figure()

        # Add a note that this feature is coming soon
        fig.add_annotation(
            text="Park Factor Analysis<br>Coming Soon",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="white"),
            bgcolor='rgba(14, 17, 23, 0.8)',
            bordercolor='rgba(128, 128, 128, 0.5)',
            borderwidth=1
        )

        fig.update_layout(
            plot_bgcolor='rgba(14, 17, 23, 1)',
            paper_bgcolor='rgba(14, 17, 23, 1)',
            font=dict(color="white"),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            width=800,
            height=400
        )

        return fig


# Global visualizer instance
_visualizer = None


def get_visualizer() -> SprayChartVisualizer:
    """Get global visualizer instance."""
    global _visualizer
    if _visualizer is None:
        _visualizer = SprayChartVisualizer()
    return _visualizer
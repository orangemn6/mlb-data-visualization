"""
Advanced stadium simulator for baseball analytics.

This module handles stadium dimension loading, trajectory calculations,
and multi-stadium comparison for "would it be a home run" analysis.
"""

import json
import os
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, List, Optional, Tuple, Union
import math


class StadiumSimulator:
    """Advanced stadium simulator with multi-park comparison capabilities."""

    def __init__(self):
        """Initialize the stadium simulator."""
        self.current_stadiums = self._load_stadium_data("data/stadiums/current_stadiums.json")
        self.historical_stadiums = self._load_stadium_data("data/stadiums/historical_stadiums.json")
        self.custom_stadiums = self._load_stadium_data("data/stadiums/custom_stadiums.json")

    def _load_stadium_data(self, file_path: str) -> Dict:
        """Load stadium data from JSON file."""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f).get("stadiums", {})
            return {}
        except (json.JSONDecodeError, IOError):
            st.warning(f"Could not load stadium data from {file_path}")
            return {}

    def get_all_stadiums(self, include_historical: bool = True, include_custom: bool = True) -> Dict:
        """
        Get all available stadiums.

        Args:
            include_historical: Include historical stadiums
            include_custom: Include custom stadiums

        Returns:
            Dictionary of all stadiums
        """
        all_stadiums = self.current_stadiums.copy()

        if include_historical:
            all_stadiums.update(self.historical_stadiums)

        if include_custom:
            all_stadiums.update(self.custom_stadiums)

        return all_stadiums

    def get_stadium_info(self, stadium_key: str) -> Optional[Dict]:
        """Get information for a specific stadium."""
        all_stadiums = self.get_all_stadiums()
        return all_stadiums.get(stadium_key)

    def calculate_home_run_probability(
        self,
        hit_data: pd.DataFrame,
        stadium_key: str,
        use_trajectory: bool = True
    ) -> pd.DataFrame:
        """
        Calculate home run probability for hits in a specific stadium.

        Args:
            hit_data: DataFrame with hit coordinate and flight data
            stadium_key: Stadium identifier
            use_trajectory: Whether to use launch angle/velocity for trajectory

        Returns:
            DataFrame with added home run probability columns
        """
        stadium_info = self.get_stadium_info(stadium_key)
        if not stadium_info:
            st.error(f"Stadium '{stadium_key}' not found")
            return hit_data

        result_df = hit_data.copy()

        # Calculate for each hit
        hr_probabilities = []
        wall_heights_cleared = []
        distances_to_wall = []

        for _, row in hit_data.iterrows():
            if pd.isna(row.get('x_viz')) or pd.isna(row.get('y_viz')):
                hr_probabilities.append(0.0)
                wall_heights_cleared.append(0.0)
                distances_to_wall.append(0.0)
                continue

            x_viz = row['x_viz']
            y_viz = row['y_viz']

            # Calculate angle from center field
            angle_from_center = math.atan2(x_viz, y_viz) if y_viz != 0 else 0
            angle_degrees = math.degrees(angle_from_center)

            # Determine which wall section this hit would encounter
            wall_distance, wall_height = self._get_wall_distance_and_height(
                stadium_info, angle_degrees, x_viz, y_viz
            )

            distances_to_wall.append(wall_distance)

            # Calculate hit distance
            hit_distance = math.sqrt(x_viz**2 + y_viz**2)

            # Basic home run determination: distance > wall distance
            basic_hr = hit_distance > wall_distance

            if use_trajectory and 'launch_angle' in row and 'launch_speed' in row:
                if pd.notna(row['launch_angle']) and pd.notna(row['launch_speed']):
                    # Calculate trajectory and height at wall
                    height_at_wall = self._calculate_trajectory_height(
                        row['launch_speed'],
                        row['launch_angle'],
                        wall_distance
                    )
                    wall_heights_cleared.append(max(0, height_at_wall - wall_height))

                    # Home run if ball clears wall
                    hr_prob = 1.0 if (basic_hr and height_at_wall > wall_height) else 0.0
                else:
                    wall_heights_cleared.append(0.0)
                    hr_prob = 1.0 if basic_hr else 0.0
            else:
                wall_heights_cleared.append(0.0)
                hr_prob = 1.0 if basic_hr else 0.0

            hr_probabilities.append(hr_prob)

        result_df[f'hr_prob_{stadium_key}'] = hr_probabilities
        result_df[f'wall_height_cleared_{stadium_key}'] = wall_heights_cleared
        result_df[f'distance_to_wall_{stadium_key}'] = distances_to_wall

        return result_df

    def _get_wall_distance_and_height(
        self,
        stadium_info: Dict,
        angle_degrees: float,
        x_viz: float,
        y_viz: float
    ) -> Tuple[float, float]:
        """
        Get wall distance and height for a given angle.

        Args:
            stadium_info: Stadium dimension data
            angle_degrees: Angle from center field in degrees
            x_viz: X coordinate
            y_viz: Y coordinate

        Returns:
            Tuple of (wall_distance, wall_height)
        """
        dimensions = stadium_info.get("dimensions", {})

        # Define angle ranges for different wall sections
        # Angles: negative = left field, positive = right field
        if angle_degrees <= -35:  # Left field line area
            distance = dimensions.get("left_field_line", 330)
            height = dimensions.get("left_field_wall_height", 8)
        elif -35 < angle_degrees <= -15:  # Left center area
            distance = dimensions.get("left_center", 375)
            height = dimensions.get("left_field_wall_height", 8)
        elif -15 < angle_degrees <= 15:  # Center field area
            distance = dimensions.get("center_field", 400)
            height = dimensions.get("center_field_wall_height", 8)
        elif 15 < angle_degrees <= 35:  # Right center area
            distance = dimensions.get("right_center", 375)
            height = dimensions.get("right_field_wall_height", 8)
        else:  # Right field line area
            distance = dimensions.get("right_field_line", 330)
            height = dimensions.get("right_field_wall_height", 8)

        return distance, height

    def _calculate_trajectory_height(
        self,
        exit_velocity: float,
        launch_angle: float,
        horizontal_distance: float
    ) -> float:
        """
        Calculate ball height at given horizontal distance using trajectory physics.

        Args:
            exit_velocity: Exit velocity in mph
            launch_angle: Launch angle in degrees
            horizontal_distance: Horizontal distance in feet

        Returns:
            Height in feet at the given distance
        """
        # Convert units
        v0_fps = exit_velocity * 1.467  # mph to feet per second
        angle_rad = math.radians(launch_angle)

        # Physics constants
        g = 32.2  # ft/s^2 (gravity)

        # Initial velocity components
        v0_x = v0_fps * math.cos(angle_rad)
        v0_y = v0_fps * math.sin(angle_rad)

        # Time to reach horizontal distance
        if v0_x == 0:
            return 0

        t = horizontal_distance / v0_x

        # Height at time t (starting at 3 feet - typical contact height)
        height = 3.0 + v0_y * t - 0.5 * g * t**2

        return max(0, height)  # Can't be below ground

    def compare_stadiums(
        self,
        hit_data: pd.DataFrame,
        stadium_keys: List[str],
        use_trajectory: bool = True
    ) -> pd.DataFrame:
        """
        Compare home run probabilities across multiple stadiums.

        Args:
            hit_data: DataFrame with hit data
            stadium_keys: List of stadium identifiers
            use_trajectory: Whether to use trajectory calculations

        Returns:
            DataFrame with comparison data for all stadiums
        """
        result_df = hit_data.copy()

        for stadium_key in stadium_keys:
            result_df = self.calculate_home_run_probability(
                result_df, stadium_key, use_trajectory
            )

        # Add summary statistics
        hr_columns = [col for col in result_df.columns if col.startswith('hr_prob_')]

        if hr_columns:
            result_df['total_stadiums_hr'] = result_df[hr_columns].sum(axis=1)
            result_df['hr_probability_avg'] = result_df[hr_columns].mean(axis=1)
            result_df['hr_probability_std'] = result_df[hr_columns].std(axis=1)

        return result_df

    def generate_stadium_report(
        self,
        hit_data: pd.DataFrame,
        stadium_keys: List[str]
    ) -> Dict:
        """
        Generate a comprehensive comparison report for multiple stadiums.

        Args:
            hit_data: DataFrame with hit data
            stadium_keys: List of stadiums to compare

        Returns:
            Dictionary with report data
        """
        comparison_data = self.compare_stadiums(hit_data, stadium_keys)

        report = {
            "stadiums": {},
            "summary": {},
            "player_stats": {}
        }

        # Per-stadium statistics
        for stadium_key in stadium_keys:
            hr_col = f'hr_prob_{stadium_key}'
            if hr_col in comparison_data.columns:
                stadium_hrs = comparison_data[hr_col].sum()
                stadium_info = self.get_stadium_info(stadium_key)

                report["stadiums"][stadium_key] = {
                    "name": stadium_info.get("name", stadium_key),
                    "total_home_runs": int(stadium_hrs),
                    "home_run_rate": stadium_hrs / len(comparison_data) if len(comparison_data) > 0 else 0,
                    "dimensions": stadium_info.get("dimensions", {}),
                    "foul_territory": stadium_info.get("foul_territory", "unknown")
                }

        # Summary statistics
        if 'total_stadiums_hr' in comparison_data.columns:
            report["summary"] = {
                "total_hits_analyzed": len(comparison_data),
                "avg_home_runs_per_stadium": comparison_data['total_stadiums_hr'].mean(),
                "most_hr_friendly": max(report["stadiums"].items(), key=lambda x: x[1]["total_home_runs"])[0] if report["stadiums"] else None,
                "least_hr_friendly": min(report["stadiums"].items(), key=lambda x: x[1]["total_home_runs"])[0] if report["stadiums"] else None,
                "stadium_variance": comparison_data['hr_probability_std'].mean() if 'hr_probability_std' in comparison_data.columns else 0
            }

        return report

    def create_custom_stadium(
        self,
        name: str,
        dimensions: Dict[str, float],
        description: str = ""
    ) -> str:
        """
        Create a custom stadium configuration.

        Args:
            name: Stadium name
            dimensions: Dictionary of field dimensions
            description: Optional description

        Returns:
            Stadium key for the created stadium
        """
        # Generate unique key
        stadium_key = name.lower().replace(" ", "_").replace("-", "_")

        # Ensure required dimensions
        required_dims = [
            "left_field_line", "left_center", "center_field",
            "right_center", "right_field_line"
        ]

        for dim in required_dims:
            if dim not in dimensions:
                raise ValueError(f"Missing required dimension: {dim}")

        # Add default wall heights if not provided
        wall_height_defaults = {
            "left_field_wall_height": 12,
            "center_field_wall_height": 12,
            "right_field_wall_height": 12
        }

        for height_key, default_height in wall_height_defaults.items():
            if height_key not in dimensions:
                dimensions[height_key] = default_height

        stadium_data = {
            "name": name,
            "description": description,
            "custom": True,
            "dimensions": dimensions
        }

        # Add to custom stadiums
        self.custom_stadiums[stadium_key] = stadium_data

        # Save to file
        self._save_custom_stadiums()

        return stadium_key

    def _save_custom_stadiums(self):
        """Save custom stadiums to file."""
        try:
            os.makedirs("data/stadiums", exist_ok=True)
            with open("data/stadiums/custom_stadiums.json", 'w') as f:
                json.dump({"stadiums": self.custom_stadiums}, f, indent=2)
        except Exception as e:
            st.error(f"Failed to save custom stadiums: {str(e)}")

    def get_stadium_categories(self) -> Dict[str, List[str]]:
        """
        Get stadiums organized by category.

        Returns:
            Dictionary with stadium categories
        """
        return {
            "Current MLB": list(self.current_stadiums.keys()),
            "Historical": list(self.historical_stadiums.keys()),
            "Custom": list(self.custom_stadiums.keys())
        }

    def get_park_factors(
        self,
        hit_data: pd.DataFrame,
        reference_stadium: str = "perfect_symmetry"
    ) -> Dict[str, float]:
        """
        Calculate park factors relative to a reference stadium.

        Args:
            hit_data: DataFrame with hit data
            reference_stadium: Reference stadium for comparison

        Returns:
            Dictionary of park factors (>1.0 = hitter friendly)
        """
        all_stadiums = list(self.get_all_stadiums().keys())
        comparison_data = self.compare_stadiums(hit_data, all_stadiums)

        reference_col = f'hr_prob_{reference_stadium}'
        if reference_col not in comparison_data.columns:
            st.error(f"Reference stadium '{reference_stadium}' not found")
            return {}

        reference_hrs = comparison_data[reference_col].sum()
        if reference_hrs == 0:
            return {}

        park_factors = {}
        for stadium_key in all_stadiums:
            hr_col = f'hr_prob_{stadium_key}'
            if hr_col in comparison_data.columns:
                stadium_hrs = comparison_data[hr_col].sum()
                park_factors[stadium_key] = stadium_hrs / reference_hrs if reference_hrs > 0 else 0

        return park_factors


# Global simulator instance
_stadium_simulator = None


def get_stadium_simulator() -> StadiumSimulator:
    """Get global stadium simulator instance."""
    global _stadium_simulator
    if _stadium_simulator is None:
        _stadium_simulator = StadiumSimulator()
    return _stadium_simulator


@st.cache_data
def calculate_stadium_comparison(
    hit_data_dict: Dict,
    stadium_keys: List[str],
    use_trajectory: bool = True
) -> Dict:
    """
    Cached wrapper for stadium comparison.

    Args:
        hit_data_dict: Hit data as dictionary (for caching)
        stadium_keys: List of stadiums to compare
        use_trajectory: Whether to use trajectory calculations

    Returns:
        Comparison results as dictionary
    """
    simulator = get_stadium_simulator()
    hit_data = pd.DataFrame(hit_data_dict)
    result_df = simulator.compare_stadiums(hit_data, stadium_keys, use_trajectory)
    return result_df.to_dict('records')
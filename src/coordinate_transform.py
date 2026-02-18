"""
Coordinate transformation utilities for baseball field visualization.

This module implements the Bill Petti transformation for converting Statcast
hit coordinate data to a standardized visualization coordinate system.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def statcast_to_viz_coords(hc_x: float, hc_y: float) -> Tuple[float, float]:
    """
    Transform Statcast hit coordinates to visualization coordinates.

    Uses the Bill Petti transformation:
    - x_adj = hc_x - 125.42
    - y_adj = 198.27 - hc_y

    Args:
        hc_x: Statcast horizontal coordinate
        hc_y: Statcast vertical coordinate

    Returns:
        Tuple of (x_viz, y_viz) coordinates for visualization
    """
    x_viz = hc_x - 125.42
    y_viz = 198.27 - hc_y
    return x_viz, y_viz


def transform_statcast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform all hit coordinates in a Statcast DataFrame.

    Args:
        df: DataFrame with 'hc_x' and 'hc_y' columns

    Returns:
        DataFrame with added 'x_viz' and 'y_viz' columns
    """
    df = df.copy()

    # Apply transformation to all rows
    coords = df[['hc_x', 'hc_y']].apply(
        lambda row: statcast_to_viz_coords(row['hc_x'], row['hc_y']) if pd.notna(row['hc_x']) and pd.notna(row['hc_y']) else (np.nan, np.nan),
        axis=1
    )

    df['x_viz'] = coords.apply(lambda x: x[0])
    df['y_viz'] = coords.apply(lambda x: x[1])

    return df


def is_fair_ball(x_viz: float, y_viz: float) -> bool:
    """
    Determine if a ball at given coordinates is fair or foul.

    Uses 45-degree foul lines from home plate (origin).

    Args:
        x_viz: X coordinate in visualization system
        y_viz: Y coordinate in visualization system

    Returns:
        True if fair, False if foul
    """
    if y_viz <= 0:  # Behind home plate
        return False

    # Foul lines are at 45 degrees from home plate
    # Left foul line: x = -y, Right foul line: x = y
    if x_viz < -abs(y_viz) or x_viz > abs(y_viz):
        return False

    return True


def filter_fair_balls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to only include fair balls.

    Args:
        df: DataFrame with 'x_viz' and 'y_viz' columns

    Returns:
        DataFrame filtered to fair balls only
    """
    if 'x_viz' not in df.columns or 'y_viz' not in df.columns:
        raise ValueError("DataFrame must have 'x_viz' and 'y_viz' columns")

    fair_mask = df.apply(
        lambda row: is_fair_ball(row['x_viz'], row['y_viz']) if pd.notna(row['x_viz']) and pd.notna(row['y_viz']) else False,
        axis=1
    )

    return df[fair_mask].copy()


def get_field_boundary_coordinates(max_distance: float = 450) -> dict:
    """
    Generate coordinates for drawing baseball field boundaries.

    Args:
        max_distance: Maximum distance for field boundary (feet)

    Returns:
        Dictionary with coordinates for field elements
    """
    # Foul lines (45 degree angles)
    foul_line_distance = max_distance * np.sqrt(2) / 2

    field_coords = {
        'left_foul_line': {
            'x': [-foul_line_distance, 0],
            'y': [0, foul_line_distance]
        },
        'right_foul_line': {
            'x': [foul_line_distance, 0],
            'y': [0, foul_line_distance]
        },
        'outfield_arc': {
            'x': [],
            'y': []
        }
    }

    # Create outfield arc (semicircle)
    angles = np.linspace(-np.pi/4, np.pi/4, 100)
    for angle in angles:
        x = max_distance * np.sin(angle)
        y = max_distance * np.cos(angle)
        field_coords['outfield_arc']['x'].append(x)
        field_coords['outfield_arc']['y'].append(y)

    return field_coords


def calculate_distance_from_home(x_viz: float, y_viz: float) -> float:
    """
    Calculate the distance from home plate to the hit location.

    Args:
        x_viz: X coordinate in visualization system
        y_viz: Y coordinate in visualization system

    Returns:
        Distance in feet from home plate
    """
    return np.sqrt(x_viz**2 + y_viz**2)


def calculate_angle_from_center(x_viz: float, y_viz: float) -> float:
    """
    Calculate the angle from center field (in degrees).

    Positive angles are toward right field, negative toward left field.

    Args:
        x_viz: X coordinate in visualization system
        y_viz: Y coordinate in visualization system

    Returns:
        Angle in degrees from center field (-45 to +45)
    """
    if y_viz == 0:
        return 0 if x_viz == 0 else (90 if x_viz > 0 else -90)

    angle_rad = np.arctan(x_viz / y_viz)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def add_distance_and_angle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add distance and angle calculations to DataFrame.

    Args:
        df: DataFrame with 'x_viz' and 'y_viz' columns

    Returns:
        DataFrame with added 'distance_feet' and 'angle_degrees' columns
    """
    df = df.copy()

    df['distance_feet'] = df.apply(
        lambda row: calculate_distance_from_home(row['x_viz'], row['y_viz']) if pd.notna(row['x_viz']) and pd.notna(row['y_viz']) else np.nan,
        axis=1
    )

    df['angle_degrees'] = df.apply(
        lambda row: calculate_angle_from_center(row['x_viz'], row['y_viz']) if pd.notna(row['x_viz']) and pd.notna(row['y_viz']) else np.nan,
        axis=1
    )

    return df


def scale_coordinates_for_plot(df: pd.DataFrame, scale_factor: float = 1.0) -> pd.DataFrame:
    """
    Scale visualization coordinates for plotting.

    Args:
        df: DataFrame with coordinate columns
        scale_factor: Scaling factor for coordinates

    Returns:
        DataFrame with scaled coordinates
    """
    df = df.copy()

    if 'x_viz' in df.columns:
        df['x_viz_scaled'] = df['x_viz'] * scale_factor

    if 'y_viz' in df.columns:
        df['y_viz_scaled'] = df['y_viz'] * scale_factor

    return df
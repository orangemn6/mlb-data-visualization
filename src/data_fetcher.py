"""
Data fetching pipeline for MLB Statcast data.

This module handles player lookup and Statcast data retrieval with intelligent
caching and error handling.
"""

import pandas as pd
import streamlit as st
from typing import Optional, List, Dict, Tuple
import pybaseball
from datetime import datetime, timedelta
import warnings
import time
from .search_engine import get_search_engine

# Suppress pybaseball warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class DataFetcher:
    """Handles MLB data fetching with caching and error handling."""

    def __init__(self):
        """Initialize the data fetcher."""
        # Cache directory will be handled by Streamlit
        self.search_engine = get_search_engine()

    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def lookup_player_ids(_self, name: str, fuzzy_threshold: float = 0.8) -> List[Dict]:
        """
        Look up player IDs by name with fuzzy matching.

        Args:
            name: Player name to search for
            fuzzy_threshold: Minimum similarity for fuzzy matching

        Returns:
            List of player dictionaries with IDs and metadata
        """
        try:
            # First try exact pybaseball lookup
            pybaseball_results = pybaseball.playerid_lookup(name.split()[-1], name.split()[0])

            if not pybaseball_results.empty:
                players = []
                for _, row in pybaseball_results.iterrows():
                    player_data = {
                        "id": row.get("key_mlbam"),
                        "name": f"{row.get('name_first', '')} {row.get('name_last', '')}".strip(),
                        "first_name": row.get("name_first", ""),
                        "last_name": row.get("name_last", ""),
                        "birth_year": row.get("birth_year"),
                        "mlb_debut": row.get("mlb_played_first"),
                        "mlb_last": row.get("mlb_played_last"),
                        "teams": [],
                        "positions": [],
                        "source": "pybaseball"
                    }

                    # Add to search engine database
                    if player_data["id"]:
                        _self.search_engine.add_player_to_database(
                            player_id=player_data["id"],
                            name=player_data["name"],
                            first_name=player_data["first_name"],
                            last_name=player_data["last_name"],
                            birth_year=player_data["birth_year"],
                            mlb_debut=player_data["mlb_debut"]
                        )

                    players.append(player_data)

                _self.search_engine.save_to_cache()
                return players

        except Exception as e:
            st.warning(f"PyBaseball lookup failed: {str(e)}")

        # Fall back to search engine fuzzy search
        fuzzy_results = _self.search_engine.fuzzy_name_search(name, threshold=fuzzy_threshold)

        players = []
        for player_id, player_name, score in fuzzy_results:
            player_info = _self.search_engine.player_db["players"][str(player_id)]
            players.append({
                "id": player_id,
                "name": player_name,
                "first_name": player_info["first_name"],
                "last_name": player_info["last_name"],
                "birth_year": player_info.get("birth_year"),
                "mlb_debut": player_info.get("mlb_debut"),
                "teams": player_info.get("teams", []),
                "positions": player_info.get("positions", []),
                "relevance_score": score,
                "source": "cache"
            })

        return players

    @st.cache_data(ttl=3600, max_entries=10)  # Cache for 1 hour, max 10 entries
    def fetch_statcast_data(
        _self,
        player_id: int,
        start_date: str,
        end_date: str,
        pitch_types: Optional[List[str]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch Statcast data for a player within date range.

        Args:
            player_id: MLB player ID
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            pitch_types: Optional list of pitch types to filter

        Returns:
            DataFrame with Statcast data or None if failed
        """
        try:
            with st.spinner(f"Fetching Statcast data for player {player_id}..."):
                # Add small delay to be respectful to API
                time.sleep(0.5)

                data = pybaseball.statcast_batter(start_date, end_date, player_id)

                if data.empty:
                    return None

                # Filter for batted ball events only
                batted_ball_events = [
                    'single', 'double', 'triple', 'home_run',
                    'field_out', 'force_out', 'grounded_into_double_play',
                    'field_error', 'fielders_choice', 'fielders_choice_out',
                    'sac_fly', 'sac_bunt', 'double_play'
                ]

                # Filter for events with hit coordinate data
                data = data[
                    (data['events'].isin(batted_ball_events)) &
                    (data['hc_x'].notna()) &
                    (data['hc_y'].notna())
                ].copy()

                # Filter by pitch types if specified
                if pitch_types:
                    data = data[data['pitch_type'].isin(pitch_types)].copy()

                # Select relevant columns
                columns_to_keep = [
                    'game_date', 'player_name', 'batter', 'pitcher',
                    'events', 'description', 'des',
                    'hc_x', 'hc_y',  # Hit coordinates
                    'launch_speed', 'launch_angle',  # Ball flight
                    'hit_distance_sc',  # Distance
                    'pitch_type', 'pitch_name',  # Pitch info
                    'balls', 'strikes', 'inning', 'inning_topbot',  # Game situation
                    'home_team', 'away_team',  # Teams
                    'bb_type',  # Batted ball type
                    'woba_value', 'estimated_woba_using_speedangle'  # Advanced metrics
                ]

                # Only keep columns that exist in the data
                available_columns = [col for col in columns_to_keep if col in data.columns]
                data = data[available_columns].copy()

                # Convert date column
                if 'game_date' in data.columns:
                    data['game_date'] = pd.to_datetime(data['game_date'])

                # Sort by date (most recent first)
                if 'game_date' in data.columns:
                    data = data.sort_values('game_date', ascending=False)

                return data

        except Exception as e:
            st.error(f"Failed to fetch Statcast data: {str(e)}")
            return None

    @st.cache_data(ttl=86400)  # Cache for 24 hours
    def get_available_pitch_types(_self, player_id: int, start_date: str, end_date: str) -> List[str]:
        """
        Get list of pitch types faced by a player in the date range.

        Args:
            player_id: MLB player ID
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            List of pitch type codes
        """
        try:
            data = pybaseball.statcast_batter(start_date, end_date, player_id)
            if data.empty:
                return []

            pitch_types = data['pitch_type'].dropna().unique().tolist()
            return sorted(pitch_types)

        except Exception:
            return []

    def get_date_range_suggestions(self) -> Dict[str, Tuple[str, str]]:
        """
        Get suggested date ranges for data fetching.

        Returns:
            Dictionary of range names to (start_date, end_date) tuples
        """
        today = datetime.now()
        current_year = today.year

        # Determine if we're in season or off-season
        if today.month >= 3 and today.month <= 10:
            # In season - current season dates
            season_start = f"{current_year}-03-01"
            season_end = today.strftime("%Y-%m-%d")
        else:
            # Off season - use previous season
            if today.month <= 2:
                season_year = current_year - 1
            else:  # November-December
                season_year = current_year

            season_start = f"{season_year}-03-01"
            season_end = f"{season_year}-10-31"

        ranges = {
            "Current/Recent Season": (season_start, season_end),
            "Last 30 Days": (
                (today - timedelta(days=30)).strftime("%Y-%m-%d"),
                today.strftime("%Y-%m-%d")
            ),
            "Last 7 Days": (
                (today - timedelta(days=7)).strftime("%Y-%m-%d"),
                today.strftime("%Y-%m-%d")
            ),
            "Previous Season": (
                f"{current_year - 1}-03-01",
                f"{current_year - 1}-10-31"
            ),
            "2023 Season": ("2023-03-01", "2023-10-31"),
            "2022 Season": ("2022-03-01", "2022-10-31"),
            "2021 Season": ("2021-03-01", "2021-10-31")
        }

        return ranges

    def validate_date_range(self, start_date: str, end_date: str) -> Tuple[bool, str]:
        """
        Validate date range for data fetching.

        Args:
            start_date: Start date string
            end_date: End date string

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")

            if start_dt > end_dt:
                return False, "Start date must be before end date"

            if end_dt > datetime.now():
                return False, "End date cannot be in the future"

            # Check if range is too large (more than 1 year)
            if (end_dt - start_dt).days > 365:
                return False, "Date range cannot exceed 1 year"

            # Check if dates are too far in the past (before 2008 when Statcast began)
            if start_dt.year < 2008:
                return False, "Statcast data is only available from 2008 onwards"

            return True, ""

        except ValueError:
            return False, "Invalid date format. Use YYYY-MM-DD"

    @st.cache_data(ttl=3600)
    def get_player_season_stats(_self, player_id: int, year: int) -> Optional[Dict]:
        """
        Get basic season statistics for a player.

        Args:
            player_id: MLB player ID
            year: Season year

        Returns:
            Dictionary with basic stats or None
        """
        try:
            start_date = f"{year}-03-01"
            end_date = f"{year}-10-31"

            data = _self.fetch_statcast_data(player_id, start_date, end_date)

            if data is None or data.empty:
                return None

            # Calculate basic stats
            total_batted_balls = len(data)
            hits = data[data['events'].isin(['single', 'double', 'triple', 'home_run'])]
            home_runs = data[data['events'] == 'home_run']

            stats = {
                "total_batted_balls": total_batted_balls,
                "hits": len(hits),
                "home_runs": len(home_runs),
                "avg_exit_velocity": data['launch_speed'].mean() if 'launch_speed' in data.columns else None,
                "avg_launch_angle": data['launch_angle'].mean() if 'launch_angle' in data.columns else None,
                "max_exit_velocity": data['launch_speed'].max() if 'launch_speed' in data.columns else None,
                "avg_distance": data['hit_distance_sc'].mean() if 'hit_distance_sc' in data.columns else None
            }

            return stats

        except Exception:
            return None


# Global data fetcher instance
_data_fetcher = None


def get_data_fetcher() -> DataFetcher:
    """Get global data fetcher instance."""
    global _data_fetcher
    if _data_fetcher is None:
        _data_fetcher = DataFetcher()
    return _data_fetcher
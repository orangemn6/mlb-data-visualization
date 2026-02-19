"""
Advanced player search engine for MLB Statcast data.

This module provides comprehensive player search functionality with support for
multi-criteria filtering, fuzzy name matching, and intelligent caching.
"""

import json
import os
from typing import List, Dict, Optional, Tuple
import pandas as pd
from difflib import SequenceMatcher
import streamlit as st
import pybaseball
import warnings
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class PlayerSearchEngine:
    """Advanced search engine for MLB players with caching and fuzzy matching."""

    def __init__(self, cache_file: str = None):
        """
        Initialize the search engine.

        Args:
            cache_file: Path to cached player database
        """
        if cache_file is None:
            # Use absolute path to ensure persistence between app runs
            cache_file = os.path.join(os.getcwd(), "data", "players", "player_database.json")

        self.cache_file = cache_file
        self.player_db = self._load_or_create_database()
        self._last_update_check = None

        # Ensure cache directory exists
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

        # Initialize with basic players if database is empty
        if not self.player_db["players"]:
            self._initialize_with_popular_players()

    def _initialize_with_popular_players(self):
        """Initialize with a set of popular players for immediate usability."""
        popular_players = [
            {"id": 592450, "name": "Aaron Judge", "first_name": "Aaron", "last_name": "Judge"},
            {"id": 545361, "name": "Mike Trout", "first_name": "Mike", "last_name": "Trout"},
            {"id": 660271, "name": "Ronald Acuña Jr.", "first_name": "Ronald", "last_name": "Acuña Jr."},
            {"id": 665742, "name": "Shohei Ohtani", "first_name": "Shohei", "last_name": "Ohtani"},
            {"id": 596059, "name": "Nolan Arenado", "first_name": "Nolan", "last_name": "Arenado"},
            {"id": 608324, "name": "Mookie Betts", "first_name": "Mookie", "last_name": "Betts"},
            {"id": 493316, "name": "Bryce Harper", "first_name": "Bryce", "last_name": "Harper"},
            {"id": 665804, "name": "Juan Soto", "first_name": "Juan", "last_name": "Soto"},
            {"id": 646240, "name": "Vladimir Guerrero Jr.", "first_name": "Vladimir", "last_name": "Guerrero Jr."},
            {"id": 645277, "name": "Fernando Tatis Jr.", "first_name": "Fernando", "last_name": "Tatis Jr."},
        ]

        for player in popular_players:
            self.add_player_to_database(
                player_id=player["id"],
                name=player["name"],
                first_name=player["first_name"],
                last_name=player["last_name"],
                years_active=list(range(2018, 2025))  # Recent years
            )

        self._save_database()

    def _load_or_create_database(self) -> Dict:
        """Load existing player database or create empty one."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        # Create empty database structure
        return {
            "players": {},
            "last_updated": None,
            "search_index": {
                "by_name": {},
                "by_team": {},
                "by_position": {},
                "by_year": {}
            }
        }

    def _save_database(self):
        """Save player database to cache file."""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'w') as f:
            json.dump(self.player_db, f, indent=2)

    def add_player_to_database(
        self,
        player_id: int,
        name: str,
        first_name: str,
        last_name: str,
        teams: List[str] = None,
        positions: List[str] = None,
        years_active: List[int] = None,
        mlb_debut: str = None,
        birth_year: int = None
    ):
        """
        Add or update a player in the database.

        Args:
            player_id: MLB player ID
            name: Full name
            first_name: First name
            last_name: Last name
            teams: List of team abbreviations
            positions: List of positions played
            years_active: List of years player was active
            mlb_debut: MLB debut date
            birth_year: Birth year
        """
        player_data = {
            "id": player_id,
            "name": name,
            "first_name": first_name,
            "last_name": last_name,
            "teams": teams or [],
            "positions": positions or [],
            "years_active": years_active or [],
            "mlb_debut": mlb_debut,
            "birth_year": birth_year,
            "search_variants": self._generate_search_variants(name, first_name, last_name)
        }

        self.player_db["players"][str(player_id)] = player_data
        self._update_search_indices(player_id, player_data)

    def _generate_search_variants(self, full_name: str, first_name: str, last_name: str) -> List[str]:
        """Generate common search variants for a player name."""
        variants = [
            full_name.lower(),
            f"{first_name} {last_name}".lower(),
            f"{last_name}, {first_name}".lower(),
            f"{last_name}".lower(),
            f"{first_name[0]}. {last_name}".lower() if first_name else "",
        ]

        # Add common nickname patterns
        if first_name:
            # Common nickname mappings
            nickname_map = {
                "william": ["bill", "billy"],
                "robert": ["bob", "bobby"],
                "richard": ["rick", "dick"],
                "michael": ["mike"],
                "christopher": ["chris"],
                "andrew": ["andy"],
                "anthony": ["tony"],
                "francisco": ["paco"],
                "alexander": ["alex"],
                "benjamin": ["ben"],
                "matthew": ["matt"],
                "joseph": ["joe"],
                "daniel": ["dan", "danny"]
            }

            first_lower = first_name.lower()
            if first_lower in nickname_map:
                for nickname in nickname_map[first_lower]:
                    variants.append(f"{nickname} {last_name}".lower())

        # Remove empty strings and duplicates
        return list(set(filter(None, variants)))

    def _update_search_indices(self, player_id: int, player_data: Dict):
        """Update search indices for fast lookup."""
        pid_str = str(player_id)

        # Index by name variants
        for variant in player_data["search_variants"]:
            if variant not in self.player_db["search_index"]["by_name"]:
                self.player_db["search_index"]["by_name"][variant] = []
            if pid_str not in self.player_db["search_index"]["by_name"][variant]:
                self.player_db["search_index"]["by_name"][variant].append(pid_str)

        # Index by team
        for team in player_data["teams"]:
            if team not in self.player_db["search_index"]["by_team"]:
                self.player_db["search_index"]["by_team"][team] = []
            if pid_str not in self.player_db["search_index"]["by_team"][team]:
                self.player_db["search_index"]["by_team"][team].append(pid_str)

        # Index by position
        for position in player_data["positions"]:
            if position not in self.player_db["search_index"]["by_position"]:
                self.player_db["search_index"]["by_position"][position] = []
            if pid_str not in self.player_db["search_index"]["by_position"][position]:
                self.player_db["search_index"]["by_position"][position].append(pid_str)

        # Index by year
        for year in player_data["years_active"]:
            if year not in self.player_db["search_index"]["by_year"]:
                self.player_db["search_index"]["by_year"][year] = []
            if pid_str not in self.player_db["search_index"]["by_year"][year]:
                self.player_db["search_index"]["by_year"][year].append(pid_str)

    @st.cache_data(ttl=86400)  # Cache for 24 hours
    def fetch_mlb_players(_self, last_name: str = None, first_name: str = None) -> pd.DataFrame:
        """
        Fetch MLB players from pybaseball.

        Args:
            last_name: Optional last name filter
            first_name: Optional first name filter

        Returns:
            DataFrame with player information
        """
        try:
            if last_name and first_name:
                # Search for specific player
                result = pybaseball.playerid_lookup(last_name, first_name)
            elif last_name:
                # Search by last name only
                result = pybaseball.playerid_lookup(last_name, fuzzy=True)
            else:
                # This is a workaround - pybaseball doesn't have a "get all players" function
                # We'll build our database incrementally through searches
                return pd.DataFrame()

            return result
        except Exception as e:
            st.warning(f"Error fetching player data: {str(e)}")
            return pd.DataFrame()

    def search_and_cache_players(self, name_query: str) -> List[Dict]:
        """
        Search for players using pybaseball and cache results.

        Args:
            name_query: Player name to search for

        Returns:
            List of player dictionaries
        """
        if not name_query or len(name_query) < 2:
            return []

        # Parse name query - handle various formats
        name_query = name_query.strip()
        name_parts = name_query.split()

        search_attempts = []

        if len(name_parts) == 1:
            # Single name - try as both first and last name
            search_attempts = [
                (name_parts[0], None),  # As last name only
                (None, name_parts[0])   # As first name only
            ]
        elif len(name_parts) == 2:
            # Two parts - try "First Last" and "Last, First" formats
            search_attempts = [
                (name_parts[1], name_parts[0]),  # Last, First
                (name_parts[0], name_parts[1])   # First, Last (backup)
            ]
        else:
            # Multiple parts - assume "First ... Last"
            first_name = name_parts[0]
            last_name = name_parts[-1]
            search_attempts = [(last_name, first_name)]

        new_players = []

        for last_name, first_name in search_attempts:
            if len(new_players) >= 10:  # Limit results per search
                break

            try:
                with st.spinner(f"Searching MLB database for '{name_query}'..."):
                    df = self.fetch_mlb_players(last_name, first_name)

                    if df.empty:
                        continue

                    for _, row in df.iterrows():
                        # Extract player data
                        player_id = row.get('key_mlbam')
                        if pd.isna(player_id) or player_id <= 0:
                            continue

                        player_id = int(player_id)

                        # Skip if already in database
                        if str(player_id) in self.player_db["players"]:
                            continue

                        first_name_db = str(row.get('name_first', '')).strip()
                        last_name_db = str(row.get('name_last', '')).strip()

                        if not first_name_db or not last_name_db:
                            continue

                        full_name = f"{first_name_db} {last_name_db}".strip()

                        # Create player data
                        years_active = self._extract_years_active(row)

                        # Include all players with any MLB experience (not just Statcast era)
                        player_data = {
                            "id": player_id,
                            "name": full_name,
                            "first_name": first_name_db,
                            "last_name": last_name_db,
                            "birth_year": row.get('birth_year') if not pd.isna(row.get('birth_year')) else None,
                            "mlb_debut": row.get('mlb_played_first') if not pd.isna(row.get('mlb_played_first')) else None,
                            "mlb_last": row.get('mlb_played_last') if not pd.isna(row.get('mlb_played_last')) else None,
                            "teams": [],  # Will be populated from other sources if needed
                            "positions": [],  # Will be populated from other sources if needed
                            "years_active": years_active,
                            "source": "pybaseball"
                        }

                        # Add to database cache
                        self.add_player_to_database(
                            player_id=player_id,
                            name=full_name,
                            first_name=first_name_db,
                            last_name=last_name_db,
                            birth_year=player_data["birth_year"],
                            mlb_debut=player_data["mlb_debut"],
                            years_active=years_active
                        )

                        new_players.append(player_data)

            except Exception as e:
                st.warning(f"Search attempt failed for '{last_name}, {first_name}': {str(e)}")
                continue

        # Save updated database
        if new_players:
            self.save_to_cache()

        return new_players

    def _extract_years_active(self, player_row) -> List[int]:
        """Extract years active from player row."""
        years = []

        debut = player_row.get('mlb_played_first')
        last_year = player_row.get('mlb_played_last')

        if debut and last_year:
            try:
                start_year = int(debut) if isinstance(debut, (int, float, str)) else None
                end_year = int(last_year) if isinstance(last_year, (int, float, str)) else None

                if start_year and end_year:
                    # Include all MLB years, but prioritize recent years for Statcast
                    end_year = min(end_year, datetime.now().year)

                    if start_year <= end_year:
                        # Generate full career range
                        all_years = list(range(start_year, end_year + 1))

                        # If player has recent activity (2008+), include those years
                        statcast_years = [y for y in all_years if y >= 2008]

                        # Return Statcast era years if available, otherwise all career years
                        years = statcast_years if statcast_years else all_years
            except:
                pass

        return years

    def fuzzy_name_search(self, query: str, threshold: float = 0.6) -> List[Tuple[int, str, float]]:
        """
        Perform fuzzy name matching.

        Args:
            query: Search query
            threshold: Minimum similarity score (0.0 to 1.0)

        Returns:
            List of (player_id, name, similarity_score) tuples
        """
        query_lower = query.lower().strip()
        matches = []

        for name_variant, player_ids in self.player_db["search_index"]["by_name"].items():
            similarity = SequenceMatcher(None, query_lower, name_variant).ratio()

            if similarity >= threshold:
                for pid in player_ids:
                    player_data = self.player_db["players"][pid]
                    matches.append((
                        int(pid),
                        player_data["name"],
                        similarity
                    ))

        # Sort by similarity score (highest first)
        matches.sort(key=lambda x: x[2], reverse=True)

        # Remove duplicates while preserving order
        seen = set()
        unique_matches = []
        for match in matches:
            if match[0] not in seen:
                seen.add(match[0])
                unique_matches.append(match)

        return unique_matches

    def search_players(
        self,
        name_query: str = None,
        team: str = None,
        position: str = None,
        year_min: int = None,
        year_max: int = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        Advanced multi-criteria player search with live MLB data.

        Args:
            name_query: Player name (supports fuzzy matching)
            team: Team abbreviation
            position: Position code
            year_min: Minimum year active
            year_max: Maximum year active
            limit: Maximum results to return

        Returns:
            List of player dictionaries matching criteria
        """
        all_results = []

        # First, search cached database
        cached_results = self._search_cached_players(name_query, team, position, year_min, year_max, limit)
        all_results.extend(cached_results)

        # If we have a name query and found few results, search MLB database
        if name_query and len(cached_results) < 3:
            try:
                live_results = self.search_and_cache_players(name_query)

                # Filter live results by criteria
                for player in live_results:
                    # Skip if already in cached results
                    if any(r['id'] == player['id'] for r in all_results):
                        continue

                    # Apply filters
                    if team and team not in player.get('teams', []):
                        continue

                    if position and position not in player.get('positions', []):
                        continue

                    if year_min is not None or year_max is not None:
                        player_years = player.get('years_active', [])
                        if player_years:  # Only filter if we have year data
                            min_year = min(player_years)
                            max_year = max(player_years)

                            if year_min is not None and max_year < year_min:
                                continue
                            if year_max is not None and min_year > year_max:
                                continue

                    # Add relevance score
                    if name_query:
                        player_name = player.get('name', '').lower()
                        query_lower = name_query.lower()
                        similarity = SequenceMatcher(None, query_lower, player_name).ratio()
                        player['relevance_score'] = similarity
                    else:
                        player['relevance_score'] = 1.0

                    all_results.append(player)

            except Exception as e:
                st.warning(f"Live search encountered an error: {str(e)}")

        # If still no results and we have a name query, try a more flexible search
        if name_query and len(all_results) == 0:
            # Try searching with partial names or different formats
            flexible_results = self._flexible_name_search(name_query)
            all_results.extend(flexible_results)

        # Sort by relevance and limit results
        all_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return all_results[:limit]

    def _flexible_name_search(self, name_query: str) -> List[Dict]:
        """Try more flexible search patterns when exact searches fail."""
        results = []

        # Try searching with just parts of the name
        parts = name_query.split()
        if len(parts) > 1:
            for part in parts:
                if len(part) >= 3:  # Only search with meaningful parts
                    try:
                        partial_results = self.search_and_cache_players(part)
                        for result in partial_results:
                            # Check if the result seems relevant to the original query
                            result_name = result.get('name', '').lower()
                            query_lower = name_query.lower()
                            similarity = SequenceMatcher(None, query_lower, result_name).ratio()

                            if similarity >= 0.3:  # Lower threshold for partial matches
                                result['relevance_score'] = similarity
                                results.append(result)
                    except:
                        continue

        return results

    def _search_cached_players(
        self,
        name_query: str = None,
        team: str = None,
        position: str = None,
        year_min: int = None,
        year_max: int = None,
        limit: int = 50
    ) -> List[Dict]:
        """Search only cached players (original implementation)."""
        candidate_ids = set()

        # Start with name search if provided
        if name_query:
            fuzzy_matches = self.fuzzy_name_search(name_query, threshold=0.4)
            candidate_ids.update(match[0] for match in fuzzy_matches[:limit * 2])
        else:
            # If no name query, start with all players
            candidate_ids.update(int(pid) for pid in self.player_db["players"].keys())

        # Filter by team
        if team:
            team_players = set()
            if team in self.player_db["search_index"]["by_team"]:
                team_players.update(int(pid) for pid in self.player_db["search_index"]["by_team"][team])
            candidate_ids &= team_players

        # Filter by position
        if position:
            position_players = set()
            if position in self.player_db["search_index"]["by_position"]:
                position_players.update(int(pid) for pid in self.player_db["search_index"]["by_position"][position])
            candidate_ids &= position_players

        # Filter by year range
        if year_min is not None or year_max is not None:
            year_players = set()
            for year in range(year_min or 2008, (year_max or 2024) + 1):
                if year in self.player_db["search_index"]["by_year"]:
                    year_players.update(int(pid) for pid in self.player_db["search_index"]["by_year"][year])
            candidate_ids &= year_players

        # Build result list
        results = []
        for player_id in candidate_ids:
            if len(results) >= limit:
                break

            player_data = self.player_db["players"][str(player_id)].copy()
            player_data["id"] = player_id

            # Add relevance score if name query was used
            if name_query:
                fuzzy_matches = self.fuzzy_name_search(name_query, threshold=0.0)
                for match_id, _, score in fuzzy_matches:
                    if match_id == player_id:
                        player_data["relevance_score"] = score
                        break
                else:
                    player_data["relevance_score"] = 0.0
            else:
                player_data["relevance_score"] = 1.0

            results.append(player_data)

        return results

    def get_popular_players(self, limit: int = 20) -> List[Dict]:
        """
        Get a list of popular/well-known players for suggestions.

        Args:
            limit: Maximum number of players to return

        Returns:
            List of popular player dictionaries
        """
        all_players = []

        # Get all cached players
        for player_id, player_data in self.player_db["players"].items():
            player_copy = player_data.copy()
            player_copy["id"] = int(player_id)
            all_players.append(player_copy)

        # If we have few players, return all available
        if len(all_players) <= limit:
            return all_players

        # Otherwise, prioritize recent players
        def get_recent_score(player):
            years = player.get('years_active', [])
            if not years:
                return 0
            # Score based on most recent year and career length
            recent_year = max(years)
            career_length = len(years)
            return recent_year * 10 + career_length

        all_players.sort(key=get_recent_score, reverse=True)
        return all_players[:limit]

    def get_teams_list(self) -> List[str]:
        """Get list of all teams in database."""
        return sorted(self.player_db["search_index"]["by_team"].keys())

    def get_positions_list(self) -> List[str]:
        """Get list of all positions in database."""
        return sorted(self.player_db["search_index"]["by_position"].keys())

    def get_years_range(self) -> Tuple[int, int]:
        """Get the range of years covered in database."""
        years = list(self.player_db["search_index"]["by_year"].keys())
        if not years:
            return 2020, 2024
        # Convert string years to integers
        int_years = [int(year) for year in years]
        return min(int_years), max(int_years)

    def save_to_cache(self):
        """Save current database state to cache file."""
        self._save_database()


# Global search engine instance
_search_engine = None


def get_search_engine() -> PlayerSearchEngine:
    """Get global search engine instance with caching."""
    global _search_engine
    if _search_engine is None:
        _search_engine = PlayerSearchEngine()
    return _search_engine


@st.cache_data(ttl=300)  # Cache for 5 minutes
def search_players_cached(
    name_query: str = None,
    team: str = None,
    position: str = None,
    year_min: int = None,
    year_max: int = None,
    limit: int = 50
) -> List[Dict]:
    """Cached wrapper for player search with live MLB data."""
    engine = get_search_engine()
    return engine.search_players(name_query, team, position, year_min, year_max, limit)


@st.cache_data
def get_popular_players_cached(limit: int = 20) -> List[Dict]:
    """Cached wrapper for popular players."""
    engine = get_search_engine()
    return engine.get_popular_players(limit)


def initialize_search_engine():
    """Initialize search engine - now handled automatically in constructor."""
    # This function is kept for backward compatibility but does nothing
    # The search engine now initializes itself with popular players
    pass
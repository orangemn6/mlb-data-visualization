# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Statcast Spray Chart Pro is a comprehensive baseball analytics dashboard built with Streamlit that transforms MLB Statcast data into interactive visualizations. The application allows users to search for MLB players, visualize hitting patterns on spray charts, and perform "Would it be a HR?" analysis across different ballparks.

## Development Commands

### Environment Setup
```bash
# Quick start (handles everything automatically)
./start.sh

# Manual setup
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Application
```bash
# Production start
streamlit run app.py

# Development with auto-reload
streamlit run app.py --server.runOnSave true
```

### Testing and Verification
```bash
# Run comprehensive verification tests
python verify_installation.py

# Test core functionality without UI
python -c "from src.search_engine import get_search_engine; print('Core imports working')"
```

## Architecture Overview

### Core Data Pipeline
The application follows a 5-stage data processing pipeline:

1. **Player Search** (`src/search_engine.py`) - Live MLB database access via pybaseball with intelligent caching
2. **Data Fetching** (`src/data_fetcher.py`) - Statcast API integration with Streamlit caching
3. **Coordinate Transform** (`src/coordinate_transform.py`) - Bill Petti standardization for field visualization
4. **Stadium Analysis** (`src/stadium_simulator.py`) - Physics-based trajectory calculations for multi-ballpark comparison
5. **Visualization** (`src/visualizer.py`) - Interactive Plotly charts with real-time filtering

### Key Architectural Patterns

**Singleton Pattern**: Core modules use global instance getters (`get_search_engine()`, `get_data_fetcher()`, etc.) to ensure single instances across the application.

**Streamlit Caching Strategy**: Critical functions use `@st.cache_data` with appropriate TTL values:
- Player searches: 5 minutes (frequent updates)
- Stadium data: 24 hours (static data)
- Coordinate transformations: No expiration (pure functions)

**Coordinate System**: All hit data uses the Bill Petti transformation:
```python
x_viz = hc_x - 125.42
y_viz = 198.27 - hc_y
```

### Module Responsibilities

**search_engine.py**:
- Live MLB player database access via pybaseball
- Fuzzy name matching with SequenceMatcher
- Intelligent caching in `data/players/player_database.json`
- Multi-criteria filtering (team, position, years)

**data_fetcher.py**:
- Statcast data retrieval from pybaseball.statcast_batter()
- Date range validation and suggestion logic
- Batted ball event filtering (singles, doubles, home runs, outs)
- Smart caching with error recovery

**coordinate_transform.py**:
- Bill Petti coordinate transformation (industry standard)
- Fair/foul boundary detection using 45-degree foul lines
- Distance and angle calculations from home plate
- DataFrame optimization for large datasets

**stadium_simulator.py**:
- Multi-ballpark "Would it be a HR?" analysis
- Physics-based trajectory calculations using launch angle/exit velocity
- Support for 47+ ballparks (current, historical, custom)
- Park factor analysis and comparative statistics

**visualizer.py**:
- Interactive Plotly scatter plots with baseball field overlays
- Color coding by exit velocity, launch angle, or outcome
- Stadium dimension overlays with accurate wall heights
- Export capabilities (HTML charts, CSV data)

## Data Structure

### Stadium Database Schema
Stadium data is stored in JSON format in `data/stadiums/`:

```json
{
  "stadium_key": {
    "name": "Stadium Name",
    "team": "ABC",
    "dimensions": {
      "left_field_line": 330,
      "center_field": 400,
      "right_field_line": 330,
      "left_field_wall_height": 12,
      "center_field_wall_height": 12,
      "right_field_wall_height": 12
    }
  }
}
```

### Player Database Schema
Player data cached in `data/players/player_database.json`:
```json
{
  "players": {
    "player_id": {
      "name": "Full Name",
      "years_active": [2020, 2021, 2022],
      "teams": ["NYY", "LAD"],
      "search_variants": ["full name", "last name", "nickname"]
    }
  },
  "search_index": {
    "by_name": {"search_term": ["player_ids"]},
    "by_team": {"NYY": ["player_ids"]},
    "by_year": {2022: ["player_ids"]}
  }
}
```

## Critical Implementation Details

### First-Time Search Performance
The first search for any new player takes 10-20 seconds as pybaseball builds its lookup table. This is expected behavior - subsequent searches are nearly instantaneous due to caching.

### Coordinate System Edge Cases
- Invalid coordinates (NaN values) are filtered out in `filter_fair_balls()`
- Foul balls are detected using the mathematical constraint: `|x_viz| <= |y_viz|` for positive y_viz
- Home plate is at origin (0,0) in the visualization coordinate system

### Stadium Analysis Physics
Home run calculations use realistic trajectory physics:
```python
# Height at distance using projectile motion
height = 3.0 + v0_y * t - 0.5 * g * t^2
# Where t = horizontal_distance / v0_x
```

### Memory Management
Large datasets (>5000 hits) are automatically sampled using stratified sampling to maintain performance while preserving statistical distribution.

## Data Dependencies

**External APIs**:
- pybaseball: MLB Statcast data (requires internet)
- No API keys required - uses public MLB data

**Static Data Files**:
- `data/stadiums/*.json`: Ballpark dimensions (47+ stadiums)
- `data/players/player_database.json`: Cached player search index

## Common Issues and Solutions

**Search Returns No Results**:
- First search for new players requires internet connection
- Player must have played in Statcast era (2008+)
- Check spelling or try last name only

**Slow Performance**:
- Clear Streamlit cache: Add `?clear_cache=true` to URL
- Reduce date range for large datasets
- Use data sampling for visualization (automatic for >5000 points)

**Coordinate Transformation Issues**:
- Ensure hit coordinates (hc_x, hc_y) are valid numbers
- Check that data includes only batted ball events
- Verify fair/foul detection is working correctly

## Development Notes

**Adding New Stadiums**: Add to appropriate JSON file in `data/stadiums/` following the established schema. The stadium simulator will automatically detect and load new stadiums.

**Modifying Visualizations**: The visualizer module uses Plotly's object-oriented API. Color scales and chart layouts are configurable via the `SprayChartVisualizer` class.

**Performance Optimization**: All database queries and API calls should use Streamlit's caching decorators. Cache TTL should match data volatility (player data: 5min, stadium data: 24hrs).

**Stadium Physics**: Trajectory calculations assume standard Earth gravity (32.2 ft/s²) and 3-foot contact height. Wall heights and distances are measured in feet using MLB official specifications.
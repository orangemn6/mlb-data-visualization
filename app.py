"""
Statcast Spray Chart Pro - Main Streamlit Application

A comprehensive baseball analytics dashboard for visualizing and analyzing
MLB Statcast data with interactive spray charts and stadium comparisons.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import plotly.graph_objects as go

# Import our modules
from src.search_engine import get_search_engine, initialize_search_engine
from src.data_fetcher import get_data_fetcher
from src.coordinate_transform import (
    transform_statcast_dataframe,
    filter_fair_balls,
    add_distance_and_angle
)
from src.stadium_simulator import get_stadium_simulator
from src.visualizer import get_visualizer


# Page configuration
st.set_page_config(
    page_title="Statcast Spray Chart Pro",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_app():
    """Initialize the application and cache components."""
    # Initialize search engine (now happens automatically)
    if "search_initialized" not in st.session_state:
        initialize_search_engine()
        st.session_state.search_initialized = True
        st.sidebar.info("🔄 Search engine ready - try searching for any MLB player!")


def create_sidebar():
    """Create the sidebar with search and filter controls."""
    st.sidebar.header("🔍 Player Search")

    # Advanced search section
    with st.sidebar.expander("Advanced Search", expanded=True):
        player_name = st.text_input(
            "Player Name",
            placeholder="e.g., Aaron Judge, Mookie Betts, Shohei Ohtani",
            help="Search any MLB player from the Statcast era (2008+). First search may take 10-20 seconds as we fetch live data from MLB database."
        )

        st.caption("💡 **Tip**: Try searching for any current or recent MLB player! The first search may take 10-20 seconds as we fetch live data.")

        col1, col2 = st.columns(2)
        with col1:
            search_team = st.selectbox(
                "Team",
                options=["All"] + get_search_engine().get_teams_list(),
                index=0
            )

        with col2:
            search_position = st.selectbox(
                "Position",
                options=["All"] + get_search_engine().get_positions_list(),
                index=0
            )

        search_clicked = st.button("🔍 Search Players", type="primary")

    # Player selection
    st.sidebar.header("📊 Selected Player")

    selected_player = None
    if search_clicked or player_name:
        # Debug the search
        search_engine = get_search_engine()
        total_players = len(search_engine.player_db.get("players", {}))
        st.sidebar.caption(f"Searching {total_players} cached players...")

        # Perform search - removed year filters
        search_results = search_engine.search_players(
            name_query=player_name if player_name else None,
            team=search_team if search_team != "All" else None,
            position=search_position if search_position != "All" else None,
            limit=20
        )

        st.sidebar.caption(f"Found {len(search_results)} results")

        if search_results:
            # Display search results
            player_options = {}
            for player in search_results:
                display_name = f"{player['name']} ({player.get('mlb_debut', 'N/A')})"
                if player.get('teams'):
                    display_name += f" - {', '.join(player['teams'])}"
                player_options[display_name] = player

            selected_display = st.sidebar.selectbox(
                "Select Player",
                options=list(player_options.keys()),
                index=0
            )

            selected_player = player_options[selected_display]
            st.sidebar.success(f"✅ Selected: {selected_player['name']}")
        else:
            st.sidebar.warning("No players found matching search criteria")
            # Show what was searched for debugging
            st.sidebar.caption(f"Searched: '{player_name}', Team: {search_team}, Position: {search_position}")
    else:
        # Show popular players as suggestions
        search_engine = get_search_engine()
        popular_players = search_engine.get_popular_players(10)

        if popular_players:
            player_options = {f"{p['name']} ({p.get('mlb_debut', 'N/A')})": p for p in popular_players}
            selected_display = st.sidebar.selectbox(
                "Or select from available players:",
                options=[""] + list(player_options.keys())
            )
            if selected_display:
                selected_player = player_options[selected_display]
        else:
            st.sidebar.info("💡 **Try searching for any MLB player!**\n\nExamples: Mookie Betts, Vladimir Guerrero Jr, Fernando Tatis Jr, Jacob deGrom, Max Scherzer\n\nFirst search takes 10-20 seconds to fetch live MLB data.")

    return selected_player


def create_data_filters():
    """Create data filtering controls in sidebar."""
    st.sidebar.header("📅 Date Range")

    data_fetcher = get_data_fetcher()
    date_ranges = data_fetcher.get_date_range_suggestions()

    # Quick date range selection
    range_selection = st.sidebar.selectbox(
        "Quick Select",
        options=list(date_ranges.keys()),
        index=0
    )

    start_date, end_date = date_ranges[range_selection]

    # Custom date range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        custom_start = st.date_input(
            "Start Date",
            value=datetime.strptime(start_date, "%Y-%m-%d").date(),
            min_value=datetime(2008, 1, 1).date(),
            max_value=datetime.now().date()
        )

    with col2:
        custom_end = st.date_input(
            "End Date",
            value=datetime.strptime(end_date, "%Y-%m-%d").date(),
            min_value=datetime(2008, 1, 1).date(),
            max_value=datetime.now().date()
        )

    # Use custom dates if different from quick select
    final_start = custom_start.strftime("%Y-%m-%d")
    final_end = custom_end.strftime("%Y-%m-%d")

    # Validate date range
    is_valid, error_msg = data_fetcher.validate_date_range(final_start, final_end)
    if not is_valid:
        st.sidebar.error(f"Date range error: {error_msg}")
        return None, None, {}

    # Additional filters
    st.sidebar.header("🎯 Filters")

    filters = {}

    # Exit velocity filter
    filters['min_exit_velocity'] = st.sidebar.slider(
        "Minimum Exit Velocity (mph)",
        min_value=50,
        max_value=120,
        value=70,
        step=1
    )

    # Launch angle filter
    angle_range = st.sidebar.slider(
        "Launch Angle Range (degrees)",
        min_value=-45,
        max_value=60,
        value=(-10, 45),
        step=1
    )
    filters['min_launch_angle'] = angle_range[0]
    filters['max_launch_angle'] = angle_range[1]

    # Pitch type filter
    pitch_types = st.sidebar.multiselect(
        "Pitch Types",
        options=['FF', 'SI', 'FC', 'SL', 'CH', 'CU', 'KC', 'FS', 'KN'],
        default=[],
        help="Leave empty to include all pitch types"
    )
    filters['pitch_types'] = pitch_types if pitch_types else None

    # Event type filter
    event_types = st.sidebar.multiselect(
        "Outcome Types",
        options=[
            'single', 'double', 'triple', 'home_run',
            'field_out', 'force_out', 'field_error'
        ],
        default=[],
        help="Leave empty to include all outcomes"
    )
    filters['event_types'] = event_types if event_types else None

    return final_start, final_end, filters


def create_stadium_controls():
    """Create stadium comparison controls."""
    st.sidebar.header("🏟️ Stadium Analysis")

    stadium_simulator = get_stadium_simulator()
    stadium_categories = stadium_simulator.get_stadium_categories()

    # Stadium category selection
    category = st.sidebar.selectbox(
        "Stadium Category",
        options=["Current MLB", "Historical", "Custom", "All"],
        index=0
    )

    # Get available stadiums based on category
    if category == "All":
        available_stadiums = stadium_simulator.get_all_stadiums()
    else:
        available_stadiums = {
            key: stadium for key, stadium in stadium_simulator.get_all_stadiums().items()
            if key in stadium_categories.get(category, [])
        }

    # Stadium selection
    selected_stadiums = st.sidebar.multiselect(
        "Select Stadiums for Comparison",
        options=list(available_stadiums.keys()),
        default=[],
        format_func=lambda x: available_stadiums[x].get('name', x),
        help="Select up to 5 stadiums for comparison"
    )

    if len(selected_stadiums) > 5:
        st.sidebar.warning("⚠️ Maximum 5 stadiums allowed for performance")
        selected_stadiums = selected_stadiums[:5]

    # Analysis options
    use_trajectory = st.sidebar.checkbox(
        "Use Trajectory Physics",
        value=True,
        help="Include launch angle and exit velocity in home run calculations"
    )

    show_stadium_overlays = st.sidebar.checkbox(
        "Show Stadium Overlays",
        value=True,
        help="Display stadium dimensions on spray chart"
    )

    return selected_stadiums, use_trajectory, show_stadium_overlays


def fetch_and_process_data(
    player_id: int,
    start_date: str,
    end_date: str,
    filters: Dict,
    selected_stadiums: List[str],
    use_trajectory: bool
) -> Optional[pd.DataFrame]:
    """Fetch and process player data with all transformations."""
    data_fetcher = get_data_fetcher()

    # Fetch raw data
    with st.spinner("Fetching Statcast data..."):
        raw_data = data_fetcher.fetch_statcast_data(
            player_id=player_id,
            start_date=start_date,
            end_date=end_date,
            pitch_types=filters.get('pitch_types')
        )

    if raw_data is None or raw_data.empty:
        return None

    # Apply filters
    filtered_data = raw_data.copy()

    # Exit velocity filter
    if 'launch_speed' in filtered_data.columns and filters.get('min_exit_velocity'):
        filtered_data = filtered_data[
            (filtered_data['launch_speed'].isna()) |
            (filtered_data['launch_speed'] >= filters['min_exit_velocity'])
        ]

    # Launch angle filter
    if 'launch_angle' in filtered_data.columns:
        min_angle = filters.get('min_launch_angle', -90)
        max_angle = filters.get('max_launch_angle', 90)
        filtered_data = filtered_data[
            (filtered_data['launch_angle'].isna()) |
            ((filtered_data['launch_angle'] >= min_angle) &
             (filtered_data['launch_angle'] <= max_angle))
        ]

    # Event type filter
    if filters.get('event_types'):
        filtered_data = filtered_data[
            filtered_data['events'].isin(filters['event_types'])
        ]

    if filtered_data.empty:
        return pd.DataFrame()

    # Transform coordinates
    with st.spinner("Processing coordinates..."):
        processed_data = transform_statcast_dataframe(filtered_data)
        processed_data = filter_fair_balls(processed_data)
        processed_data = add_distance_and_angle(processed_data)

    # Stadium analysis
    if selected_stadiums:
        with st.spinner("Analyzing stadium comparisons..."):
            stadium_simulator = get_stadium_simulator()
            processed_data = stadium_simulator.compare_stadiums(
                processed_data,
                selected_stadiums,
                use_trajectory
            )

    return processed_data


def create_main_dashboard(processed_data: pd.DataFrame, selected_player: Dict, selected_stadiums: List[str]):
    """Create the main dashboard with visualizations."""
    st.header(f"⚾ Spray Chart Analysis - {selected_player['name']}")

    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Batted Balls", len(processed_data))

    with col2:
        hits = processed_data[processed_data['events'].isin(['single', 'double', 'triple', 'home_run'])]
        st.metric("Hits", len(hits))

    with col3:
        home_runs = processed_data[processed_data['events'] == 'home_run']
        st.metric("Home Runs", len(home_runs))

    with col4:
        if 'launch_speed' in processed_data.columns:
            avg_exit_velo = processed_data['launch_speed'].mean()
            st.metric("Avg Exit Velocity", f"{avg_exit_velo:.1f} mph" if pd.notna(avg_exit_velo) else "N/A")
        else:
            st.metric("Avg Exit Velocity", "N/A")

    # Visualization controls
    st.subheader("📊 Visualization Controls")

    viz_col1, viz_col2, viz_col3 = st.columns(3)

    with viz_col1:
        color_by = st.selectbox(
            "Color Points By",
            options=['launch_speed', 'launch_angle', 'hit_distance_sc', 'events'],
            index=0,
            format_func=lambda x: {
                'launch_speed': 'Exit Velocity',
                'launch_angle': 'Launch Angle',
                'hit_distance_sc': 'Distance',
                'events': 'Outcome'
            }.get(x, x)
        )

    with viz_col2:
        size_by = st.selectbox(
            "Size Points By",
            options=['launch_angle', 'launch_speed', 'hit_distance_sc'],
            index=0,
            format_func=lambda x: {
                'launch_speed': 'Exit Velocity',
                'launch_angle': 'Launch Angle',
                'hit_distance_sc': 'Distance'
            }.get(x, x)
        )

    with viz_col3:
        chart_opacity = st.slider("Point Opacity", 0.3, 1.0, 0.7, 0.1)

    # Main spray chart
    st.subheader("🎯 Interactive Spray Chart")

    visualizer = get_visualizer()

    spray_chart = visualizer.create_spray_chart(
        hit_data=processed_data,
        color_by=color_by,
        size_by=size_by,
        stadium_overlays=selected_stadiums if st.session_state.get('show_stadium_overlays', True) else None,
        opacity=chart_opacity
    )

    st.plotly_chart(spray_chart, use_container_width=True)

    # Stadium comparison section
    if selected_stadiums:
        st.subheader("🏟️ Stadium Comparison Analysis")

        # Stadium comparison chart
        comparison_chart = visualizer.create_comparison_chart(
            processed_data,
            selected_stadiums,
            chart_type="home_runs"
        )

        st.plotly_chart(comparison_chart, use_container_width=True)

        # Stadium analysis table
        stadium_simulator = get_stadium_simulator()
        stadium_report = stadium_simulator.generate_stadium_report(
            processed_data,
            selected_stadiums
        )

        if stadium_report.get('stadiums'):
            st.subheader("📋 Stadium Statistics")

            # Create comparison table
            comparison_data = []
            for stadium_key, stats in stadium_report['stadiums'].items():
                comparison_data.append({
                    'Stadium': stats['name'],
                    'Home Runs': stats['total_home_runs'],
                    'HR Rate': f"{stats['home_run_rate']:.3f}",
                    'LF Line': f"{stats['dimensions'].get('left_field_line', 'N/A')} ft",
                    'Center': f"{stats['dimensions'].get('center_field', 'N/A')} ft",
                    'RF Line': f"{stats['dimensions'].get('right_field_line', 'N/A')} ft",
                    'Foul Territory': stats.get('foul_territory', 'N/A').replace('_', ' ').title()
                })

            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)

    # Data export section
    with st.expander("💾 Export Data"):
        export_col1, export_col2 = st.columns(2)

        with export_col1:
            if st.button("📊 Download Filtered Data (CSV)"):
                csv_data = processed_data.to_csv(index=False)
                st.download_button(
                    label="💾 Download CSV",
                    data=csv_data,
                    file_name=f"{selected_player['name']}_spray_chart_data.csv",
                    mime="text/csv"
                )

        with export_col2:
            if st.button("📈 Download Chart (HTML)"):
                chart_html = spray_chart.to_html()
                st.download_button(
                    label="💾 Download HTML",
                    data=chart_html,
                    file_name=f"{selected_player['name']}_spray_chart.html",
                    mime="text/html"
                )


def main():
    """Main application entry point."""
    # Initialize the app
    initialize_app()

    # App header
    st.title("⚾ Statcast Spray Chart Pro")
    st.markdown("""
    **Advanced Baseball Analytics Dashboard** - Visualize hitting patterns, compare stadiums, and analyze player performance with interactive spray charts.
    """)

    # Create sidebar controls
    selected_player = create_sidebar()

    if not selected_player:
        # Show welcome screen
        st.info("👈 Use the sidebar to search for any MLB player (2008+) to begin analysis.")

        # Show app features
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🔍 Live MLB Database")
            st.write("Search any MLB player from the Statcast era (2008+) with live data fetching from the official MLB database.")

        with col2:
            st.subheader("📊 Interactive Charts")
            st.write("Visualize hitting patterns with color-coded exit velocity and launch angle data.")

        with col3:
            st.subheader("🏟️ Stadium Analysis")
            st.write("Compare performance across different ballparks with 'Would it be a HR?' simulation.")

        # Show sample searches
        st.subheader("🌟 Try Searching For These Players")

        sample_players = [
            "Aaron Judge", "Mookie Betts", "Vladimir Guerrero Jr",
            "Fernando Tatis Jr", "Ronald Acuña Jr", "Juan Soto",
            "Mike Trout", "Shohei Ohtani", "Manny Machado"
        ]

        cols = st.columns(3)
        for i, player in enumerate(sample_players[:9]):
            with cols[i % 3]:
                st.info(f"**{player}**")

        st.markdown("---")
        st.markdown("**💡 Pro Tip**: The first search for any player takes 10-20 seconds as we fetch live data from the MLB database. Subsequent searches are much faster thanks to intelligent caching!")

        return

    # Get filter controls
    start_date, end_date, filters = create_data_filters()
    if start_date is None:  # Date validation failed
        return

    selected_stadiums, use_trajectory, show_stadium_overlays = create_stadium_controls()

    # Store in session state for visualizer
    st.session_state['show_stadium_overlays'] = show_stadium_overlays

    # Fetch and process data
    processed_data = fetch_and_process_data(
        player_id=selected_player['id'],
        start_date=start_date,
        end_date=end_date,
        filters=filters,
        selected_stadiums=selected_stadiums,
        use_trajectory=use_trajectory
    )

    if processed_data is None:
        st.error("❌ Failed to fetch data. Please check your search criteria and try again.")
        return

    if processed_data.empty:
        st.warning("⚠️ No data found matching your criteria. Try adjusting the filters or date range.")
        return

    # Create main dashboard
    create_main_dashboard(processed_data, selected_player, selected_stadiums)


if __name__ == "__main__":
    main()
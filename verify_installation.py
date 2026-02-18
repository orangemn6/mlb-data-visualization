#!/usr/bin/env python3
"""
Verification script for Statcast Spray Chart Pro

This script tests all major components to ensure the application
is correctly installed and configured.
"""

import sys
import importlib
import traceback
from datetime import datetime


class VerificationTester:
    """Test all application components."""

    def __init__(self):
        self.passed = []
        self.failed = []

    def test_import(self, module_name, description):
        """Test importing a module."""
        try:
            importlib.import_module(module_name)
            self.passed.append(f"✅ {description}")
            return True
        except Exception as e:
            self.failed.append(f"❌ {description}: {str(e)}")
            return False

    def test_function(self, func, description):
        """Test executing a function."""
        try:
            func()
            self.passed.append(f"✅ {description}")
            return True
        except Exception as e:
            self.failed.append(f"❌ {description}: {str(e)}")
            return False

    def run_all_tests(self):
        """Run all verification tests."""
        print("⚾ Statcast Spray Chart Pro - Verification Tests")
        print("=" * 50)
        print()

        # Test core dependencies
        print("📦 Testing Core Dependencies:")
        self.test_import("streamlit", "Streamlit web framework")
        self.test_import("pandas", "Pandas data manipulation")
        self.test_import("numpy", "NumPy numerical computing")
        self.test_import("plotly", "Plotly visualization library")
        self.test_import("pybaseball", "PyBaseball MLB data access")
        self.test_import("requests", "Requests HTTP library")
        self.test_import("sklearn", "Scikit-learn machine learning")
        print()

        # Test custom modules
        print("🔧 Testing Custom Modules:")
        self.test_import("src.search_engine", "Player search engine")
        self.test_import("src.data_fetcher", "Data fetching pipeline")
        self.test_import("src.coordinate_transform", "Coordinate transformation")
        self.test_import("src.stadium_simulator", "Stadium simulator")
        self.test_import("src.visualizer", "Visualization engine")
        self.test_import("src.performance_utils", "Performance utilities")
        print()

        # Test core functionality
        print("⚙️ Testing Core Functionality:")

        def test_search_engine():
            from src.search_engine import get_search_engine, initialize_with_basic_players
            initialize_with_basic_players()
            engine = get_search_engine()
            results = engine.search_players("Judge", limit=5)
            assert len(results) > 0, "No search results found"

        def test_coordinate_transform():
            from src.coordinate_transform import statcast_to_viz_coords
            x_viz, y_viz = statcast_to_viz_coords(150.0, 100.0)
            assert isinstance(x_viz, float), "Invalid coordinate transformation"
            assert isinstance(y_viz, float), "Invalid coordinate transformation"

        def test_stadium_simulator():
            from src.stadium_simulator import get_stadium_simulator
            simulator = get_stadium_simulator()
            stadiums = simulator.get_all_stadiums()
            assert len(stadiums) > 0, "No stadiums loaded"

        def test_visualizer():
            from src.visualizer import get_visualizer
            import pandas as pd
            visualizer = get_visualizer()
            # Test with empty dataframe (should not crash)
            fig = visualizer.create_spray_chart(pd.DataFrame())
            assert fig is not None, "Chart creation failed"

        self.test_function(test_search_engine, "Player search functionality")
        self.test_function(test_coordinate_transform, "Coordinate transformation")
        self.test_function(test_stadium_simulator, "Stadium simulation")
        self.test_function(test_visualizer, "Visualization creation")
        print()

        # Test data files
        print("📁 Testing Data Files:")

        def test_stadium_data():
            import json
            import os

            files_to_check = [
                "data/stadiums/current_stadiums.json",
                "data/stadiums/historical_stadiums.json",
                "data/stadiums/custom_stadiums.json"
            ]

            for file_path in files_to_check:
                assert os.path.exists(file_path), f"Missing file: {file_path}"
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    assert 'stadiums' in data, f"Invalid format in {file_path}"
                    assert len(data['stadiums']) > 0, f"No stadiums in {file_path}"

        self.test_function(test_stadium_data, "Stadium data files")
        print()

        # Test app entry point
        print("🚀 Testing Application Entry Point:")

        def test_app_import():
            import app
            assert hasattr(app, 'main'), "Main function not found in app.py"

        self.test_function(test_app_import, "Main application import")
        print()

        # Summary
        self.print_summary()

    def print_summary(self):
        """Print test summary."""
        print("📊 Test Summary:")
        print("-" * 20)

        if self.passed:
            print(f"\n🎉 Passed Tests ({len(self.passed)}):")
            for test in self.passed:
                print(f"  {test}")

        if self.failed:
            print(f"\n💥 Failed Tests ({len(self.failed)}):")
            for test in self.failed:
                print(f"  {test}")

        print()
        total_tests = len(self.passed) + len(self.failed)
        success_rate = (len(self.passed) / total_tests) * 100 if total_tests > 0 else 0

        if success_rate == 100:
            print("🎊 ALL TESTS PASSED! Your installation is ready to use.")
            print("Run 'streamlit run app.py' to start the application.")
        elif success_rate >= 80:
            print(f"⚠️  {success_rate:.1f}% tests passed. Minor issues detected but app should work.")
        else:
            print(f"❌ {success_rate:.1f}% tests passed. Please check the failed tests above.")

        print()
        print("To start the application:")
        print("  ./start.sh")
        print("  OR")
        print("  streamlit run app.py")


if __name__ == "__main__":
    # Suppress warnings during testing
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    tester = VerificationTester()
    tester.run_all_tests()
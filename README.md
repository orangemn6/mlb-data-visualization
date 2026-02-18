# ⚾ Statcast Spray Chart Pro

A comprehensive baseball analytics dashboard that transforms MLB Statcast data into interactive visualizations. Built with modern Python tools and designed for both casual fans and advanced analysts.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

### 🔍 Advanced Player Search
- **Fuzzy Name Matching**: Search for players with typos and nicknames
- **Multi-Criteria Filtering**: Filter by team, position, active years
- **Smart Suggestions**: Popular player recommendations
- **Comprehensive Database**: Coverage from 2008+ (Statcast era)

### 📊 Interactive Spray Charts
- **Real-time Visualization**: Color-coded by exit velocity, launch angle, or outcome
- **Coordinate Transformation**: Uses Bill Petti's standardized field coordinates
- **Fair/Foul Detection**: Automatic filtering of foul territory hits
- **Rich Hover Details**: Game date, pitch type, count, and more

### 🏟️ Stadium Simulator
- **"Would it be a HR?" Analysis**: Compare hits across 30+ MLB stadiums
- **Historical Ballparks**: Polo Grounds, Ebbets Field, original Yankee Stadium
- **Custom Park Builder**: Design your own ballpark dimensions
- **Trajectory Physics**: Launch angle and exit velocity calculations
- **Multi-Stadium Comparison**: Side-by-side analysis up to 5 parks

### 📈 Advanced Analytics
- **Park Factors**: Quantify ballpark effects on offensive performance
- **Interactive Filtering**: Real-time updates with pitch type, velocity thresholds
- **Export Capabilities**: Download data (CSV) and charts (HTML)
- **Performance Optimization**: Intelligent caching for large datasets

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- Internet connection (for Statcast data)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mlb-data-graphs
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## 📖 Usage Guide

### Getting Started
1. **Search for a Player**: Use the sidebar to search by name (e.g., "Aaron Judge", "Trout")
2. **Set Date Range**: Choose from quick presets or custom date ranges
3. **Apply Filters**: Adjust exit velocity, launch angle, pitch types as needed
4. **Select Stadiums**: Choose ballparks for comparison analysis
5. **Explore**: Interactive spray chart with hover details and export options

### Advanced Features
- **Fuzzy Search**: Try "Juge" instead of "Judge" - it works!
- **Historical Analysis**: Compare performance in vintage ballparks
- **Custom Parks**: Build your dream ballpark with custom dimensions
- **Export Data**: Download your analysis for further research

## 🏗️ Technical Architecture

### Core Components
```
src/
├── search_engine.py      # Advanced player search with fuzzy matching
├── data_fetcher.py       # Statcast data retrieval with caching
├── coordinate_transform.py # Field coordinate standardization
├── stadium_simulator.py  # Multi-ballpark home run analysis
├── visualizer.py         # Interactive Plotly visualizations
└── performance_utils.py  # Caching and optimization
```

### Technology Stack
- **Frontend**: Streamlit for interactive web interface
- **Data Source**: pybaseball for MLB Statcast API integration
- **Processing**: pandas + numpy for data manipulation
- **Visualization**: plotly for interactive scatter plots
- **Ballpark Data**: Custom JSON databases with historical accuracy

### Data Pipeline
1. **Player Search** → Fuzzy matching with cached player database
2. **Data Fetching** → Statcast API with intelligent caching
3. **Coordinate Transform** → Bill Petti standardization + fair/foul detection
4. **Stadium Analysis** → Physics-based trajectory calculations
5. **Visualization** → Interactive Plotly charts with real-time updates

## 📊 Stadium Database

### Current MLB Parks (30 stadiums)
All 30 current MLB ballparks with accurate dimensions and wall heights.

### Historical Ballparks (12+ vintage parks)
- Polo Grounds (1911-1957): 279' LF, 483' CF, 257' RF
- Ebbets Field (1913-1957): 348' LF, 393' CF, 297' RF
- Original Yankee Stadium (1923-2008)
- Tiger Stadium, Forbes Field, Crosley Field, and more

### Custom Parks (5+ templates)
- Perfect Symmetry Park: Completely balanced dimensions
- Home Run Derby Park: Extremely hitter-friendly
- Pitcher's Paradise: Massive dimensions favoring pitchers
- Little League and Softball field templates

## 🎯 Example Analyses

### Player Comparison
Compare how Aaron Judge's 2022 home runs would have performed in different eras:
- Polo Grounds: 73 home runs → 71 home runs (deep CF hurt him)
- Ebbets Field: 73 home runs → 78 home runs (short RF helped)

### Stadium Effects
Quantify park factors across your favorite hitter's career:
- Coors Field: 1.12 park factor (12% more HRs)
- Marlins Park: 0.89 park factor (11% fewer HRs)

### Historical Context
See how modern sluggers would have performed in vintage ballparks with unique dimensions.

## 🤝 Contributing

We welcome contributions! Here are ways to help:

### Bug Reports
- Use GitHub Issues with detailed reproduction steps
- Include sample player names and date ranges that cause problems

### Feature Requests
- Stadium suggestions for historical database
- New analysis types or visualization options
- Performance improvements for large datasets

### Development
- Fork the repository and create feature branches
- Follow existing code style and documentation standards
- Add tests for new functionality

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **pybaseball**: Excellent Python package for MLB data access
- **Bill Petti**: Coordinate transformation methodology
- **Streamlit**: Amazing framework for data applications
- **MLB**: Statcast data that makes this analysis possible

## 📧 Support

Having issues? Here are your options:

1. **Check the FAQ** in the app's help section
2. **Search existing GitHub Issues** for similar problems
3. **Create a new Issue** with detailed information
4. **Join the discussion** for feature requests and general questions

---

Built with ❤️ for baseball analytics enthusiasts
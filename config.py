"""
Configuration settings for MLB Weather Model
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file (if present)
load_dotenv()

# API Keys (look for environment variables first, fall back to default if not found)
OPENWEATHERMAP_API_KEY = os.getenv('OPENWEATHERMAP_API_KEY', 'c9fc686d86d6a22e96844fa975d96f39')
ODDS_API_KEY = os.getenv('ODDS_API_KEY', 'd069cac732787dff5f9e9f476521cbe9')

# Data directory
DATA_DIR = "data"

# Ensure data directory exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Stadium mapping
STADIUM_MAPPING = {
    'LAA': {'name': 'Angel Stadium', 'lat': 33.8003, 'lon': -117.8827},
    'STL': {'name': 'Busch Stadium', 'lat': 38.6226, 'lon': -90.1928},
    'ARI': {'name': 'Chase Field', 'lat': 33.4452, 'lon': -112.0667},
    'NYM': {'name': 'Citi Field', 'lat': 40.7571, 'lon': -73.8458},
    'PHI': {'name': 'Citizens Bank Park', 'lat': 39.9061, 'lon': -75.1665},
    'DET': {'name': 'Comerica Park', 'lat': 42.339, 'lon': -83.0485},
    'COL': {'name': 'Coors Field', 'lat': 39.7559, 'lon': -104.9942},
    'LAD': {'name': 'Dodger Stadium', 'lat': 34.0739, 'lon': -118.24},
    'BOS': {'name': 'Fenway Park', 'lat': 42.3467, 'lon': -71.0972},
    'TEX': {'name': 'Globe Life Field', 'lat': 32.7515, 'lon': -97.0829},
    'CIN': {'name': 'Great American Ball Park', 'lat': 39.0979, 'lon': -84.5067},
    'CHW': {'name': 'Guaranteed Rate Field', 'lat': 41.83, 'lon': -87.6339},
    'KCR': {'name': 'Kauffman Stadium', 'lat': 39.0517, 'lon': -94.4803},
    'MIA': {'name': 'LoanDepot Park', 'lat': 25.7778, 'lon': -80.2197},
    'HOU': {'name': 'Minute Maid Park', 'lat': 29.7573, 'lon': -95.3555},
    'WSN': {'name': 'Nationals Park', 'lat': 38.873, 'lon': -77.0074},
    'OAK': {'name': 'Oakland Coliseum', 'lat': 37.7516, 'lon': -122.2005},
    'SFG': {'name': 'Oracle Park', 'lat': 37.7786, 'lon': -122.3893},
    'BAL': {'name': 'Oriole Park at Camden Yards', 'lat': 39.2838, 'lon': -76.6215},
    'SDP': {'name': 'Petco Park', 'lat': 32.7076, 'lon': -117.1569},
    'PIT': {'name': 'PNC Park', 'lat': 40.4469, 'lon': -80.0057},
    'CLE': {'name': 'Progressive Field', 'lat': 41.4962, 'lon': -81.6852},
    'TOR': {'name': 'Rogers Centre', 'lat': 43.6414, 'lon': -79.3894},
    'SEA': {'name': 'T-Mobile Park', 'lat': 47.5914, 'lon': -122.3425},
    'MIN': {'name': 'Target Field', 'lat': 44.9818, 'lon': -93.2775},
    'TBR': {'name': 'Tropicana Field', 'lat': 27.7682, 'lon': -82.6534},
    'ATL': {'name': 'Truist Park', 'lat': 33.8911, 'lon': -84.468},
    'CHC': {'name': 'Wrigley Field', 'lat': 41.9484, 'lon': -87.6553},
    'NYY': {'name': 'Yankee Stadium', 'lat': 40.8296, 'lon': -73.9262},
    'MIL': {'name': 'American Family Field', 'lat': 43.0280, 'lon': -87.9712},
}

# Team name mapping (full names to abbreviations)
TEAM_NAME_MAP = {
    'New York Yankees': 'NYY',
    'Boston Red Sox': 'BOS',
    'Los Angeles Dodgers': 'LAD',
    'Chicago Cubs': 'CHC',
    'Houston Astros': 'HOU',
    'Atlanta Braves': 'ATL',
    'Philadelphia Phillies': 'PHI',
    'San Diego Padres': 'SDP',
    'San Francisco Giants': 'SFG',
    'New York Mets': 'NYM',
    'St. Louis Cardinals': 'STL',
    'Milwaukee Brewers': 'MIL',
    'Cleveland Guardians': 'CLE',
    'Toronto Blue Jays': 'TOR',
    'Tampa Bay Rays': 'TBR',
    'Seattle Mariners': 'SEA',
    'Minnesota Twins': 'MIN',
    'Detroit Tigers': 'DET',
    'Colorado Rockies': 'COL',
    'Arizona Diamondbacks': 'ARI',
    'Baltimore Orioles': 'BAL',
    'Chicago White Sox': 'CHW',
    'Cincinnati Reds': 'CIN',
    'Kansas City Royals': 'KCR',
    'Los Angeles Angels': 'LAA',
    'Miami Marlins': 'MIA',
    'Oakland Athletics': 'OAK',
    'Pittsburgh Pirates': 'PIT',
    'Texas Rangers': 'TEX',
    'Washington Nationals': 'WSN'
}

# Ballpark factors for run scoring
BALLPARK_FACTORS = {
    'Coors Field': 1.2,  # High altitude
    'Chase Field': 1.1,  # Hot, dry climate
    'Fenway Park': 1.05,  # Small dimensions
    'Yankee Stadium': 1.05,  # HR friendly
    'Great American Ball Park': 1.05,  # HR friendly
    'Wrigley Field': 1.0,  # Varies with wind
    'Oracle Park': 0.9,  # Pitcher friendly
    'Petco Park': 0.9,  # Pitcher friendly
    'T-Mobile Park': 0.95,  # Pitcher friendly
    'Dodger Stadium': 0.95  # Pitcher friendly
}

# Model parameters
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_BET_SIZE = 100.0
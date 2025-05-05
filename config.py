"""
Configuration settings for MLB Weather Model
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file (if present)
load_dotenv()

# API Keys (look for environment variables first, fall back to default if not found)
ODDS_API_KEY = os.getenv('ODDS_API_KEY', 'd069cac732787dff5f9e9f476521cbe9')
OPENWEATHERMAP_API_KEY = os.getenv('OPENWEATHERMAP_API_KEY', 'c9fc686d86d6a22e96844fa975d96f39')

# Open-Meteo API endpoints
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_HISTORY = "https://archive-api.open-meteo.com/v1/archive"

# Weather variables to fetch from Open-Meteo
WEATHER_VARIABLES = [
    "temperature_2m",
    "relativehumidity_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "pressure_msl",
    "cloudcover",
    "windspeed_10m",
    "winddirection_10m"
]

# Data directory
DATA_DIR = "data"

# Stadium mapping with coordinates
STADIUM_MAPPING = {
    'ARI': {'name': 'Chase Field', 'lat': 33.445302, 'lon': -112.066687},
    'ATL': {'name': 'Truist Park', 'lat': 33.890762, 'lon': -84.468208},
    'BAL': {'name': 'Oriole Park at Camden Yards', 'lat': 39.283787, 'lon': -76.621689},
    'BOS': {'name': 'Fenway Park', 'lat': 42.346268, 'lon': -71.095764},
    'CHC': {'name': 'Wrigley Field', 'lat': 41.947902, 'lon': -87.655823},
    'CHW': {'name': 'Guaranteed Rate Field', 'lat': 41.829902, 'lon': -87.633752},
    'CIN': {'name': 'Great American Ball Park', 'lat': 39.097460, 'lon': -84.506790},
    'CLE': {'name': 'Progressive Field', 'lat': 41.495537, 'lon': -81.685278},
    'COL': {'name': 'Coors Field', 'lat': 39.755882, 'lon': -104.994178},
    'DET': {'name': 'Comerica Park', 'lat': 42.339420, 'lon': -83.048942},
    'HOU': {'name': 'Minute Maid Park', 'lat': 29.757697, 'lon': -95.355537},
    'KCR': {'name': 'Kauffman Stadium', 'lat': 39.051672, 'lon': -94.480314},
    'LAA': {'name': 'Angel Stadium', 'lat': 33.800308, 'lon': -117.882732},
    'LAD': {'name': 'Dodger Stadium', 'lat': 34.073851, 'lon': -118.239958},
    'MIA': {'name': 'LoanDepot Park', 'lat': 25.778318, 'lon': -80.219597},
    'MIL': {'name': 'American Family Field', 'lat': 43.028210, 'lon': -87.971252},
    'MIN': {'name': 'Target Field', 'lat': 44.981715, 'lon': -93.277543},
    'NYM': {'name': 'Citi Field', 'lat': 40.757088, 'lon': -73.845821},
    'NYY': {'name': 'Yankee Stadium', 'lat': 40.829643, 'lon': -73.926175},
    'OAK': {'name': 'Oakland Coliseum', 'lat': 37.751595, 'lon': -122.200528},
    'PHI': {'name': 'Citizens Bank Park', 'lat': 39.905569, 'lon': -75.166591},
    'PIT': {'name': 'PNC Park', 'lat': 40.446855, 'lon': -80.005666},
    'SDP': {'name': 'Petco Park', 'lat': 32.707582, 'lon': -117.156999},
    'SEA': {'name': 'T-Mobile Park', 'lat': 47.591391, 'lon': -122.332327},
    'SFG': {'name': 'Oracle Park', 'lat': 37.778595, 'lon': -122.389270},
    'STL': {'name': 'Busch Stadium', 'lat': 38.622619, 'lon': -90.192821},
    'TBR': {'name': 'Tropicana Field', 'lat': 27.768225, 'lon': -82.653392},
    'TEX': {'name': 'Globe Life Field', 'lat': 32.747299, 'lon': -97.082504},
    'TOR': {'name': 'Rogers Centre', 'lat': 43.641438, 'lon': -79.389353},
    'WSN': {'name': 'Nationals Park', 'lat': 38.872861, 'lon': -77.007500}
}

# Ballpark run-scoring factors (1.0 is neutral)
BALLPARK_FACTORS = {
    'Coors Field': 1.15,        # High altitude
    'LoanDepot Park': 0.95,     # Known pitcher's park
    'Oracle Park': 0.95,        # Known pitcher's park
    'Dodger Stadium': 0.98,     # Slight pitcher's park
    'Fenway Park': 1.05,        # Green Monster effect
    'Yankee Stadium': 1.05,     # Short right field
    'Great American Ball Park': 1.08,  # Known hitter's park
}

# Team name mapping for odds API
TEAM_NAME_MAP = {
    'Arizona Diamondbacks': 'ARI',
    'Atlanta Braves': 'ATL',
    'Baltimore Orioles': 'BAL',
    'Boston Red Sox': 'BOS',
    'Chicago Cubs': 'CHC',
    'Chicago White Sox': 'CHW',
    'Cincinnati Reds': 'CIN',
    'Cleveland Guardians': 'CLE',
    'Colorado Rockies': 'COL',
    'Detroit Tigers': 'DET',
    'Houston Astros': 'HOU',
    'Kansas City Royals': 'KCR',
    'Los Angeles Angels': 'LAA',
    'Los Angeles Dodgers': 'LAD',
    'Miami Marlins': 'MIA',
    'Milwaukee Brewers': 'MIL',
    'Minnesota Twins': 'MIN',
    'New York Mets': 'NYM',
    'New York Yankees': 'NYY',
    'Oakland Athletics': 'OAK',
    'Philadelphia Phillies': 'PHI',
    'Pittsburgh Pirates': 'PIT',
    'San Diego Padres': 'SDP',
    'Seattle Mariners': 'SEA',
    'San Francisco Giants': 'SFG',
    'St. Louis Cardinals': 'STL',
    'Tampa Bay Rays': 'TBR',
    'Texas Rangers': 'TEX',
    'Toronto Blue Jays': 'TOR',
    'Washington Nationals': 'WSN'
}
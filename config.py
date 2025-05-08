"""
Configuration settings for MLB Weather Model
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file (if present)
load_dotenv()

# API Keys (look for environment variables first, fall back to default if not found)
ODDS_API_KEY = os.getenv('ODDS_API_KEY', 'd069cac732787dff5f9e9f476521cbe9')

# Data directory
DATA_DIR = "data"

# Ensure data directory exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Create required data directories
for subdir in ['player_data', 'innings_data']:
    dir_path = os.path.join(DATA_DIR, subdir)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Stadium mapping with wind orientations and dimensions
# lat/lon - geographical coordinates
# orientation - angle in degrees where outfield is facing (for wind effects)
# altitude - in feet above sea level (impacts air density)
# dimensions - average distance to outfield walls in feet
STADIUM_MAPPING = {
    'LAA': {'name': 'Angel Stadium', 'lat': 33.8003, 'lon': -117.8827, 'orientation': 45, 'altitude': 152, 'dimensions': {'left': 330, 'center': 396, 'right': 330}},
    'STL': {'name': 'Busch Stadium', 'lat': 38.6226, 'lon': -90.1928, 'orientation': 35, 'altitude': 466, 'dimensions': {'left': 336, 'center': 400, 'right': 335}},
    'ARI': {'name': 'Chase Field', 'lat': 33.4452, 'lon': -112.0667, 'orientation': 0, 'altitude': 1086, 'dimensions': {'left': 334, 'center': 407, 'right': 334}, 'dome': True, 'retractable_roof': True},
    'NYM': {'name': 'Citi Field', 'lat': 40.7571, 'lon': -73.8458, 'orientation': 130, 'altitude': 10, 'dimensions': {'left': 335, 'center': 408, 'right': 330}},
    'PHI': {'name': 'Citizens Bank Park', 'lat': 39.9061, 'lon': -75.1665, 'orientation': 30, 'altitude': 13, 'dimensions': {'left': 329, 'center': 401, 'right': 330}},
    'DET': {'name': 'Comerica Park', 'lat': 42.339, 'lon': -83.0485, 'orientation': 110, 'altitude': 600, 'dimensions': {'left': 345, 'center': 420, 'right': 330}},
    'COL': {'name': 'Coors Field', 'lat': 39.7559, 'lon': -104.9942, 'orientation': 20, 'altitude': 5280, 'dimensions': {'left': 347, 'center': 415, 'right': 350}},
    'LAD': {'name': 'Dodger Stadium', 'lat': 34.0739, 'lon': -118.24, 'orientation': 0, 'altitude': 500, 'dimensions': {'left': 330, 'center': 395, 'right': 330}},
    'BOS': {'name': 'Fenway Park', 'lat': 42.3467, 'lon': -71.0972, 'orientation': 45, 'altitude': 20, 'dimensions': {'left': 310, 'center': 390, 'right': 302}},
    'TEX': {'name': 'Globe Life Field', 'lat': 32.7515, 'lon': -97.0829, 'orientation': 30, 'altitude': 545, 'dimensions': {'left': 329, 'center': 407, 'right': 326}, 'dome': True, 'retractable_roof': True},
    'CIN': {'name': 'Great American Ball Park', 'lat': 39.0979, 'lon': -84.5067, 'orientation': 320, 'altitude': 490, 'dimensions': {'left': 328, 'center': 404, 'right': 325}},
    'CHW': {'name': 'Guaranteed Rate Field', 'lat': 41.83, 'lon': -87.6339, 'orientation': 180, 'altitude': 594, 'dimensions': {'left': 330, 'center': 400, 'right': 335}},
    'KCR': {'name': 'Kauffman Stadium', 'lat': 39.0517, 'lon': -94.4803, 'orientation': 270, 'altitude': 750, 'dimensions': {'left': 330, 'center': 410, 'right': 330}},
    'MIA': {'name': 'LoanDepot Park', 'lat': 25.7778, 'lon': -80.2197, 'orientation': 0, 'altitude': 2, 'dimensions': {'left': 344, 'center': 400, 'right': 335}, 'dome': True, 'retractable_roof': True},
    'HOU': {'name': 'Minute Maid Park', 'lat': 29.7573, 'lon': -95.3555, 'orientation': 0, 'altitude': 46, 'dimensions': {'left': 315, 'center': 409, 'right': 326}, 'dome': True, 'retractable_roof': True},
    'WSN': {'name': 'Nationals Park', 'lat': 38.873, 'lon': -77.0074, 'orientation': 35, 'altitude': 25, 'dimensions': {'left': 337, 'center': 402, 'right': 335}},
    'OAK': {'name': 'Oakland Coliseum', 'lat': 37.7516, 'lon': -122.2005, 'orientation': 125, 'altitude': 10, 'dimensions': {'left': 330, 'center': 400, 'right': 330}},
    'SFG': {'name': 'Oracle Park', 'lat': 37.7786, 'lon': -122.3893, 'orientation': 105, 'altitude': 15, 'dimensions': {'left': 339, 'center': 399, 'right': 309}},
    'BAL': {'name': 'Oriole Park at Camden Yards', 'lat': 39.2838, 'lon': -76.6215, 'orientation': 165, 'altitude': 36, 'dimensions': {'left': 333, 'center': 410, 'right': 318}},
    'SDP': {'name': 'Petco Park', 'lat': 32.7076, 'lon': -117.1569, 'orientation': 0, 'altitude': 20, 'dimensions': {'left': 334, 'center': 396, 'right': 322}},
    'PIT': {'name': 'PNC Park', 'lat': 40.4469, 'lon': -80.0057, 'orientation': 35, 'altitude': 726, 'dimensions': {'left': 325, 'center': 399, 'right': 320}},
    'CLE': {'name': 'Progressive Field', 'lat': 41.4962, 'lon': -81.6852, 'orientation': 70, 'altitude': 653, 'dimensions': {'left': 325, 'center': 405, 'right': 325}},
    'TOR': {'name': 'Rogers Centre', 'lat': 43.6414, 'lon': -79.3894, 'orientation': 180, 'altitude': 266, 'dimensions': {'left': 328, 'center': 400, 'right': 328}, 'dome': True, 'retractable_roof': True},
    'SEA': {'name': 'T-Mobile Park', 'lat': 47.5914, 'lon': -122.3425, 'orientation': 265, 'altitude': 15, 'dimensions': {'left': 331, 'center': 401, 'right': 326}, 'retractable_roof': True},
    'MIN': {'name': 'Target Field', 'lat': 44.9818, 'lon': -93.2775, 'orientation': 0, 'altitude': 840, 'dimensions': {'left': 339, 'center': 407, 'right': 328}},
    'TBR': {'name': 'Tropicana Field', 'lat': 27.7682, 'lon': -82.6534, 'orientation': 180, 'altitude': 45, 'dimensions': {'left': 315, 'center': 404, 'right': 322}, 'dome': True},
    'ATL': {'name': 'Truist Park', 'lat': 33.8911, 'lon': -84.468, 'orientation': 45, 'altitude': 1050, 'dimensions': {'left': 335, 'center': 400, 'right': 325}},
    'CHC': {'name': 'Wrigley Field', 'lat': 41.9484, 'lon': -87.6553, 'orientation': 45, 'altitude': 600, 'dimensions': {'left': 355, 'center': 400, 'right': 353}},
    'NYY': {'name': 'Yankee Stadium', 'lat': 40.8296, 'lon': -73.9262, 'orientation': 0, 'altitude': 43, 'dimensions': {'left': 318, 'center': 408, 'right': 314}},
    'MIL': {'name': 'American Family Field', 'lat': 43.0280, 'lon': -87.9712, 'orientation': 0, 'altitude': 602, 'dimensions': {'left': 344, 'center': 400, 'right': 345}, 'dome': True, 'retractable_roof': True},
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

# Ballpark factors for run scoring with temperature and wind adjustments
# Base values extracted from actual historical data
BALLPARK_FACTORS = {
    'Coors Field': {
        'base': 1.27,  # Highest run environment due to altitude
        'temp_factor': 0.03,  # Additional impact per 10°F above 70°F
        'wind_out_factor': 0.05,  # Additional impact with wind blowing out
        'wind_in_factor': -0.03  # Reduced impact with wind blowing in
    },
    'Great American Ball Park': {
        'base': 1.08,
        'temp_factor': 0.02,
        'wind_out_factor': 0.04,
        'wind_in_factor': -0.02
    },
    'Yankee Stadium': {
        'base': 1.06,
        'temp_factor': 0.015,
        'wind_out_factor': 0.06,  # Short porch in right field significantly affected by wind
        'wind_in_factor': -0.03
    },
    'Fenway Park': {
        'base': 1.05,
        'temp_factor': 0.01,
        'wind_out_factor': 0.03,
        'wind_in_factor': -0.02
    },
    'Wrigley Field': {
        'base': 1.00,  # Neutral base, but highly variable based on wind
        'temp_factor': 0.01,
        'wind_out_factor': 0.07,  # Most wind-affected park in MLB
        'wind_in_factor': -0.06
    },
    'T-Mobile Park': {
        'base': 0.95,
        'temp_factor': 0.01,
        'wind_out_factor': 0.02,
        'wind_in_factor': -0.01
    },
    'Oracle Park': {
        'base': 0.90,
        'temp_factor': 0.01,
        'wind_out_factor': 0.03,
        'wind_in_factor': -0.01
    },
    'Petco Park': {
        'base': 0.92,
        'temp_factor': 0.01,
        'wind_out_factor': 0.02,
        'wind_in_factor': -0.01
    }
}

# Fill in missing stadiums with neutral factors
for stadium in [s['name'] for s in STADIUM_MAPPING.values()]:
    if stadium not in BALLPARK_FACTORS:
        if any(keyword in stadium for keyword in ['Dome', 'Rogers', 'Tropicana', 'Minute Maid']):
            # Indoor stadiums have less weather impact
            BALLPARK_FACTORS[stadium] = {
                'base': 1.0,
                'temp_factor': 0.0,  # No temperature effect with dome closed
                'wind_out_factor': 0.0,  # No wind effect with dome closed
                'wind_in_factor': 0.0  # No wind effect with dome closed
            }
        else:
            # Average outdoor stadium
            BALLPARK_FACTORS[stadium] = {
                'base': 1.0,
                'temp_factor': 0.01,
                'wind_out_factor': 0.03,
                'wind_in_factor': -0.02
            }
            
# Weather impact thresholds
WEATHER_THRESHOLDS = {
    'hot_temp': 85,  # Temperature above which heat significantly impacts play
    'cold_temp': 50,  # Temperature below which cold significantly impacts play
    'high_wind': 10,  # Wind speed (mph) above which wind becomes significant
    'high_humidity': 70,  # Humidity percentage above which air becomes heavy
    'rain_precip': 0.1  # Precipitation (inches) that can affect play
}

# Model parameters
DEFAULT_CONFIDENCE_THRESHOLD = 0.6  # Increased from 0.5 for more selective betting
DEFAULT_BET_SIZE = 100.0
MAX_KELLY_FRACTION = 0.05  # Maximum fraction of bankroll to bet with Kelly
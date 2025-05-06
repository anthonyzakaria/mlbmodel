"""
Main script for running the MLB Weather Model.
This file handles the workflow from data collection to generating betting recommendations.
"""

import os
import argparse
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import re
import sys

# Import configuration
from config import DATA_DIR, STADIUM_MAPPING, BALLPARK_FACTORS

# Try to import advanced components with fallbacks
try:
    from enhanced_model import MLBAdvancedModel as MLBModel
    print("Using enhanced model with advanced features")
except ImportError:
    print("Enhanced model not available. Falling back to basic model.")
    from mlb_weather_model import MLBWeatherModel as MLBModel

# Import visualization modules with proper error handling
try:
    from enhanced_visualization import visualize_betting_opportunities as visualize_mlb_bets, visualize_weather_impact
    print("Using enhanced visualization module")
except ImportError:
    try:
        from visualization import visualize_mlb_bets, visualize_weather_impact
        print("Using standard visualization module")
    except ImportError:
        print("Warning: Visualization modules not available. Visualizations will be disabled.")
        
        # Create dummy visualization functions
        def visualize_mlb_bets(bets_df):
            print("Visualization module not available. Showing text summary instead:")
            print(f"Found {len(bets_df)} betting opportunities.")
            if len(bets_df) > 0:
                for _, bet in bets_df.iterrows():
                    print(f"* {bet['date'].strftime('%Y-%m-%d')}: {bet['away_team']} @ {bet['home_team']} - Bet: {bet['bet_type']}, Confidence: {bet['confidence']:.2f}")
        
        def visualize_weather_impact(df):
            print("Visualization module not available. Skipping weather impact visualization.")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MLB Weather Model for betting predictions')
    
    parser.add_argument('--fetch-data', action='store_true', 
                        help='Fetch new MLB game data')
    parser.add_argument('--start-year', type=int, default=2022, 
                        help='Starting year for data collection')
    parser.add_argument('--end-year', type=int, default=2023, 
                        help='Ending year for data collection')
    parser.add_argument('--use-existing-games', action='store_true',
                        help='Use existing game data files and only fetch missing years')
    parser.add_argument('--use-existing-weather', action='store_true', 
                        help='Use existing weather data and only fetch missing dates')
    parser.add_argument('--train', action='store_true', 
                        help='Train a new model')
    parser.add_argument('--analyze', action='store_true', 
                        help='Analyze feature importance')
    parser.add_argument('--backtest', action='store_true', 
                        help='Run backtest on historical data')
    parser.add_argument('--kelly', action='store_true', 
                        help='Use Kelly criterion for bet sizing in backtest')
    parser.add_argument('--today', action='store_true', 
                        help='Get today\'s betting recommendations')
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='Confidence threshold for betting recommendations')
    
    return parser.parse_args()

def fetch_historical_data(start_year, end_year, use_existing=True):
    """
    Fetch historical MLB game data from reliable sources.
    
    Args:
        start_year (int): First season to collect
        end_year (int): Last season to collect
        use_existing (bool): Whether to use existing game data and only fetch missing years
        
    Returns:
        DataFrame: Processed game data
    """
    print(f"Fetching MLB game data from {start_year} to {end_year}...")
    
    # Check for cached data first
    cache_file = f"{DATA_DIR}/mlb_games_{start_year}_{end_year}.csv"
    
    # If existing file exists with the exact same date range, just load it
    if use_existing and os.path.exists(cache_file):
        print(f"Loading cached game data from {cache_file}")
        return pd.read_csv(cache_file, parse_dates=['date'])
    
    # Check for other existing game data files
    existing_games_df = None
    if use_existing:
        # Find all mlb_games_*.csv files
        game_files = [f for f in os.listdir(DATA_DIR) if f.startswith("mlb_games_") and f.endswith(".csv")]
        
        if game_files:
            # Choose the most comprehensive file
            print("Found existing game data files:")
            for i, file in enumerate(game_files):
                print(f"  {i+1}. {file}")
            
            # Parse the date ranges from filenames
            existing_ranges = []
            for file in game_files:
                match = re.search(r'mlb_games_(\d{4})_(\d{4})\.csv', file)
                if match:
                    file_start_year = int(match.group(1))
                    file_end_year = int(match.group(2))
                    existing_ranges.append((file_start_year, file_end_year, file))
            
            # Load existing data if it overlaps with requested range
            if existing_ranges:
                # Sort by start year and then by end year
                existing_ranges.sort(key=lambda x: (x[0], -x[1]))
                
                # Check which years we already have data for
                years_covered = set()
                for file_start, file_end, file in existing_ranges:
                    for year in range(file_start, file_end + 1):
                        if file_start <= year <= file_end:
                            years_covered.add(year)
                
                # Determine which years are missing
                requested_years = set(range(start_year, end_year + 1))
                missing_years = sorted(requested_years - years_covered)
                
                if not missing_years:
                    print("All requested years already exist in data files")
                    # Load and merge all needed files
                    dfs_to_merge = []
                    for file_start, file_end, file in existing_ranges:
                        if any(file_start <= year <= file_end for year in requested_years):
                            file_path = os.path.join(DATA_DIR, file)
                            df = pd.read_csv(file_path, parse_dates=['date'])
                            # Filter to only include the requested years
                            df['year'] = pd.to_datetime(df['date']).dt.year
                            df = df[df['year'].between(start_year, end_year)]
                            df = df.drop('year', axis=1)
                            dfs_to_merge.append(df)
                    
                    if dfs_to_merge:
                        merged_df = pd.concat(dfs_to_merge, ignore_index=True)
                        # Remove duplicates if any
                        merged_df = merged_df.drop_duplicates(subset=['date', 'home_team', 'away_team'])
                        # Sort by date
                        merged_df = merged_df.sort_values('date')
                        # Save as new cache file for this specific range
                        merged_df.to_csv(cache_file, index=False)
                        print(f"Created merged game data file {cache_file} with {len(merged_df)} games")
                        return merged_df
                else:
                    print(f"Missing years: {missing_years}")
                    # Load existing data for the years we have
                    dfs_to_merge = []
                    for file_start, file_end, file in existing_ranges:
                        if any(file_start <= year <= file_end for year in requested_years):
                            file_path = os.path.join(DATA_DIR, file)
                            df = pd.read_csv(file_path, parse_dates=['date'])
                            # Filter to only include the requested years
                            df['year'] = pd.to_datetime(df['date']).dt.year
                            df = df[df['year'].between(start_year, end_year)]
                            df = df.drop('year', axis=1)
                            dfs_to_merge.append(df)
                    
                    if dfs_to_merge:
                        existing_games_df = pd.concat(dfs_to_merge, ignore_index=True)
                        # Remove duplicates if any
                        existing_games_df = existing_games_df.drop_duplicates(subset=['date', 'home_team', 'away_team'])
                        print(f"Loaded {len(existing_games_df)} existing game records")
                        
                        # Update the years to fetch - only fetch missing years
                        years_to_fetch = missing_years
                    else:
                        years_to_fetch = list(range(start_year, end_year + 1))
            else:
                years_to_fetch = list(range(start_year, end_year + 1))
        else:
            years_to_fetch = list(range(start_year, end_year + 1))
    else:
        years_to_fetch = list(range(start_year, end_year + 1))
    
    # Current year - we shouldn't try to fetch beyond this
    current_year = datetime.now().year
    
    # Validate year range - limit to completed seasons
    years_to_fetch = [year for year in years_to_fetch if year <= current_year]
    
    if not years_to_fetch:
        print("No additional years to fetch")
        return existing_games_df
    
    print(f"Fetching data for years: {years_to_fetch}")
    
    try:
        import pybaseball
        # Enable caching to reduce API calls
        try:
            pybaseball.cache.enable()
        except:
            print("Cache enabling failed, continuing without cache")
    except ImportError:
        print("pybaseball not installed. Installing now...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pybaseball"])
        import pybaseball
    
    # MLB teams
    teams = ['ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 
             'COL', 'DET', 'HOU', 'KCR', 'LAA', 'LAD', 'MIA', 'MIL', 
             'MIN', 'NYM', 'NYY', 'OAK', 'PHI', 'PIT', 'SDP', 'SEA', 
             'SFG', 'STL', 'TBR', 'TEX', 'TOR', 'WSN']
    
    all_games = []
    
    for year in years_to_fetch:
        print(f"Processing {year} season...")
        
        for team in teams:
            try:
                print(f"Fetching schedule for {team} {year}...")
                
                # Get team's schedule (removed parse_dates parameter)
                team_schedule = pybaseball.schedule_and_record(year, team)
                
                # Manually parse dates after retrieving the data
                if 'Date' in team_schedule.columns:
                    date_series = []
                    for date_str in team_schedule['Date']:
                        try:
                            # Standard format
                            date = pd.to_datetime(date_str)
                        except:
                            try:
                                # Try manual parsing
                                match = re.search(r'(\w+), (\w+) (\d+)', date_str)
                                if match:
                                    month_str, day_str = match.group(2), match.group(3)
                                    month_map = {
                                        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
                                        'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
                                        'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                                    }
                                    month_num = month_map.get(month_str, 1)
                                    date = pd.Timestamp(year=year, month=month_num, day=int(day_str))
                                else:
                                    date = pd.NaT
                            except:
                                date = pd.NaT
                        date_series.append(date)
                    
                    team_schedule['parsed_date'] = date_series
                
                # Filter for completed regular season games
                completed_games = team_schedule[
                    team_schedule['R'].notna() & 
                    team_schedule['RA'].notna()
                ]
                
                # Process each game
                for idx, game in completed_games.iterrows():
                    try:
                        # Only process home games to avoid duplicates
                        if game['Home_Away'] == 'Home':
                            # Get game date
                            if 'parsed_date' in team_schedule.columns:
                                game_date = game['parsed_date']
                            else:
                                try:
                                    game_date = pd.to_datetime(game['Date'])
                                except:
                                    print(f"Skipping game with unparseable date: {game['Date']}")
                                    continue
                            
                            # Skip if date is invalid
                            if pd.isna(game_date):
                                continue
                            
                            # Get stadium info
                            stadium_info = STADIUM_MAPPING.get(team, 
                                                {'name': f"{team} Park", 
                                                 'lat': 40.0, 'lon': -75.0})
                            
                            # Extract scores
                            try:
                                home_score = int(game['R'])
                                away_score = int(game['RA'])
                                total_runs = home_score + away_score
                            except:
                                print(f"Invalid score for {team} vs {game['Opp']} on {game_date}")
                                continue
                            
                            # Add to games list
                            all_games.append({
                                'date': game_date,
                                'home_team': team,
                                'away_team': game['Opp'],
                                'home_score': home_score,
                                'away_score': away_score,
                                'total_runs': total_runs,
                                'stadium': stadium_info['name'],
                                'stadium_lat': stadium_info['lat'],
                                'stadium_lon': stadium_info['lon']
                            })
                    except Exception as e:
                        print(f"Error processing game: {e}")
                
                # Add delay to avoid rate limiting
                time.sleep(1.5)
                
            except Exception as e:
                print(f"Error fetching data for {team} {year}: {e}")
    
    # Create DataFrame from all games
    new_games_df = pd.DataFrame(all_games) if all_games else None
    
    # Combine with existing data if available
    if existing_games_df is not None and new_games_df is not None and len(new_games_df) > 0:
        # Ensure date is in datetime format for both DataFrames
        if 'date' in new_games_df.columns:
            new_games_df['date'] = pd.to_datetime(new_games_df['date'])
        
        # Combine the DataFrames
        games_df = pd.concat([existing_games_df, new_games_df], ignore_index=True)
        # Remove duplicates if any
        games_df = games_df.drop_duplicates(subset=['date', 'home_team', 'away_team'])
        print(f"Combined {len(existing_games_df)} existing records with {len(new_games_df)} new records")
    elif new_games_df is not None and len(new_games_df) > 0:
        games_df = new_games_df
    elif existing_games_df is not None:
        games_df = existing_games_df
    else:
        print("No games collected. Please check your network connection and try again.")
        return None
    
    # Ensure date is in datetime format
    games_df['date'] = pd.to_datetime(games_df['date'])
    
    # Sort by date
    games_df = games_df.sort_values('date')
    
    # Save to disk - both specific cache file and a complete file
    games_df.to_csv(cache_file, index=False)
    
    # Also save as a year-specific file for future reuse
    for year in years_to_fetch:
        year_df = games_df[games_df['date'].dt.year == year]
        if len(year_df) > 0:
            year_file = f"{DATA_DIR}/mlb_games_{year}_{year}.csv"
            year_df.to_csv(year_file, index=False)
            print(f"Saved {len(year_df)} games for {year} to {year_file}")
    
    print(f"Saved {len(games_df)} games to {cache_file}")
    
    return games_df

def fetch_weather_data(games_df, use_existing=True):
    """
    Fetch weather data for games using Open-Meteo API.
    
    Args:
        games_df (DataFrame): Games with stadium location information
        use_existing (bool): Whether to use existing weather data file
        
    Returns:
        DataFrame: Weather data for each game
    """
    if games_df is None or len(games_df) == 0:
        print("No game data available for weather fetching.")
        return None
    
    print("Fetching weather data for game locations and dates...")
    
    # Check if we have the dedicated weather module
    try:
        from weather_data import fetch_historical_weather
        print("Using dedicated weather data module...")
        return fetch_historical_weather(games_df, use_existing=use_existing)
    except Exception as e:
        print(f"Error using weather data module: {e}")
        print("Falling back to basic implementation...")
    
    # Basic implementation when enhanced module not available
    weather_list = []
    
    # Check for existing weather data
    existing_weather_df = None
    existing_file = f"{DATA_DIR}/weather_data.csv"
    
    if use_existing and os.path.exists(existing_file):
        print(f"Loading existing weather data from {existing_file}")
        existing_weather_df = pd.read_csv(existing_file)
        print(f"Loaded {len(existing_weather_df)} existing weather records")
        
        # Convert date to datetime if it's not already
        if 'date' in existing_weather_df.columns:
            existing_weather_df['date'] = pd.to_datetime(existing_weather_df['date'])
            
        # Create a set of date-stadium combinations we already have
        existing_combinations = set()
        if existing_weather_df is not None and 'date' in existing_weather_df.columns and 'stadium' in existing_weather_df.columns:
            for _, row in existing_weather_df.iterrows():
                date_str = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
                stadium = row['stadium']
                existing_combinations.add((date_str, stadium))
            
            print(f"Found {len(existing_combinations)} unique date-stadium combinations in existing data")
    
    # Group by stadium to reduce API calls
    stadium_groups = games_df.groupby(['stadium', 'stadium_lat', 'stadium_lon'])
    
    for (stadium, lat, lon), stadium_games in stadium_groups:
        print(f"Processing weather for {stadium} ({len(stadium_games)} games)")
        
        # Get unique dates for this stadium
        if 'date' in stadium_games.columns:
            all_dates = pd.to_datetime(stadium_games['date']).dt.strftime('%Y-%m-%d').unique()
            
            # Filter out dates we already have
            if use_existing and existing_weather_df is not None:
                dates = [date for date in all_dates if (date, stadium) not in existing_combinations]
                print(f"Need to fetch {len(dates)} out of {len(all_dates)} dates for {stadium}")
            else:
                dates = all_dates
        else:
            print(f"No date column found for {stadium}")
            continue
        
        # Skip if we have all dates for this stadium
        if len(dates) == 0:
            print(f"All dates for {stadium} already exist in data, skipping")
            continue
        
        # Process in chunks of 100 dates to avoid too long URLs
        chunk_size = 100
        
        for i in range(0, len(dates), chunk_size):
            date_chunk = dates[i:i+chunk_size]
            start_date = min(date_chunk)
            end_date = max(date_chunk)
            
            print(f"Fetching weather from {start_date} to {end_date}")
            
            try:
                # Build Open-Meteo API URL
                url = f"https://archive-api.open-meteo.com/v1/archive"
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": start_date,
                    "end_date": end_date,
                    "hourly": "temperature_2m,relativehumidity_2m,precipitation,cloudcover,pressure_msl,windspeed_10m,winddirection_10m",
                    "timezone": "America/New_York"
                }
                
                # Make request
                import requests
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract hourly data
                    hourly_times = data.get("hourly", {}).get("time", [])
                    hourly_data = {
                        "temp": data.get("hourly", {}).get("temperature_2m", []),
                        "humidity": data.get("hourly", {}).get("relativehumidity_2m", []),
                        "precip": data.get("hourly", {}).get("precipitation", []),
                        "cloud_cover": data.get("hourly", {}).get("cloudcover", []),
                        "pressure": data.get("hourly", {}).get("pressure_msl", []),
                        "wind_speed": data.get("hourly", {}).get("windspeed_10m", []),
                        "wind_direction": data.get("hourly", {}).get("winddirection_10m", [])
                    }
                    
                    # Process each date
                    for date_str in date_chunk:
                        # Find game-time hours (assume 1pm to 6pm)
                        game_date = pd.to_datetime(date_str).date()
                        game_indices = []
                        
                        for idx, time_str in enumerate(hourly_times):
                            dt = pd.to_datetime(time_str)
                            if dt.date() == game_date and 13 <= dt.hour <= 18:
                                game_indices.append(idx)
                        
                        if game_indices:
                            # Average the values during game time
                            avg_temp = np.mean([hourly_data["temp"][i] for i in game_indices if i < len(hourly_data["temp"])])
                            avg_humidity = np.mean([hourly_data["humidity"][i] for i in game_indices if i < len(hourly_data["humidity"])])
                            avg_precip = np.sum([hourly_data["precip"][i] for i in game_indices if i < len(hourly_data["precip"])])
                            avg_cloud = np.mean([hourly_data["cloud_cover"][i] for i in game_indices if i < len(hourly_data["cloud_cover"])])
                            avg_pressure = np.mean([hourly_data["pressure"][i] for i in game_indices if i < len(hourly_data["pressure"])])
                            avg_wind = np.mean([hourly_data["wind_speed"][i] for i in game_indices if i < len(hourly_data["wind_speed"])])
                            
                            # For wind direction, use vector averaging
                            sin_sum = np.sum([np.sin(np.radians(hourly_data["wind_direction"][i])) for i in game_indices if i < len(hourly_data["wind_direction"])])
                            cos_sum = np.sum([np.cos(np.radians(hourly_data["wind_direction"][i])) for i in game_indices if i < len(hourly_data["wind_direction"])])
                            avg_direction = np.degrees(np.arctan2(sin_sum, cos_sum)) % 360
                            
                            # Determine weather condition based on cloud and precipitation
                            if avg_precip > 0.5:
                                weather_condition = 'Rain'
                            elif avg_precip > 0.1:
                                weather_condition = 'Drizzle'
                            elif avg_cloud > 80:
                                weather_condition = 'Clouds'
                            else:
                                weather_condition = 'Clear'
                            
                            # Add to weather list
                            weather_list.append({
                                'date': date_str,
                                'stadium': stadium,
                                'temperature': avg_temp * 9/5 + 32,  # Convert from C to F
                                'feels_like': (avg_temp * 9/5 + 32) + (avg_humidity / 100 * 5),  # Simple approximation
                                'humidity': avg_humidity,
                                'pressure': avg_pressure / 100,  # Convert from Pa to hPa
                                'wind_speed': avg_wind * 2.237,  # Convert from m/s to mph
                                'wind_direction': avg_direction,
                                'cloud_cover': avg_cloud,
                                'weather_condition': weather_condition,
                                'weather_description': f"{weather_condition} with {avg_cloud:.0f}% cloud cover",
                                'precipitation': avg_precip,
                                'data_source': 'open-meteo'
                            })
                        else:
                            print(f"No hourly data found for {date_str}")
                            # Add synthetic data as fallback
                            weather_list.append(generate_synthetic_weather(date_str, stadium))
                else:
                    print(f"Error fetching weather: {response.status_code}, {response.text}")
                    # Add synthetic data for this date range
                    for date_str in date_chunk:
                        weather_list.append(generate_synthetic_weather(date_str, stadium))
                
                # Add delay between API calls
                time.sleep(1)
                
            except Exception as e:
                print(f"Exception fetching weather for {stadium}: {e}")
                # Add synthetic data for this date range
                for date_str in date_chunk:
                    weather_list.append(generate_synthetic_weather(date_str, stadium))
    
    # Create DataFrame from new weather data
    new_weather_df = pd.DataFrame(weather_list)
    
    # Combine with existing data if available
    if existing_weather_df is not None and len(new_weather_df) > 0:
        # Ensure date is in datetime format for both DataFrames
        if 'date' in new_weather_df.columns:
            new_weather_df['date'] = pd.to_datetime(new_weather_df['date'])
        
        # Combine the DataFrames
        weather_df = pd.concat([existing_weather_df, new_weather_df], ignore_index=True)
        print(f"Combined {len(existing_weather_df)} existing records with {len(new_weather_df)} new records")
    elif len(new_weather_df) > 0:
        weather_df = new_weather_df
    else:
        weather_df = existing_weather_df if existing_weather_df is not None else pd.DataFrame()
    
    # Fill missing values with reasonable estimates
    numeric_cols = ['temperature', 'feels_like', 'humidity', 'pressure', 
                   'wind_speed', 'wind_direction', 'cloud_cover', 'precipitation']
    
    for col in numeric_cols:
        if col in weather_df.columns:
            weather_df[col] = pd.to_numeric(weather_df[col], errors='coerce')
            # Fill missing values with column median
            weather_df[col].fillna(weather_df[col].median(), inplace=True)
    
    # Save to disk
    weather_df.to_csv(existing_file, index=False)
    print(f"Saved weather data for {len(weather_df)} games.")
    
    return weather_df

def generate_synthetic_weather(date_str, stadium):
    """
    Generate synthetic weather data when API calls fail.
    
    Args:
        date_str (str): Date string in YYYY-MM-DD format
        stadium (str): Stadium name
        
    Returns:
        dict: Synthetic weather data
    """
    date_obj = pd.to_datetime(date_str).date()
    month = date_obj.month
    
    # Seasonal temperature adjustment
    if 5 <= month <= 9:  # Summer months
        base_temp = np.random.randint(70, 90)
    elif month in [4, 10]:  # Spring/Fall
        base_temp = np.random.randint(55, 75)
    else:  # Winter (rare for baseball)
        base_temp = np.random.randint(40, 60)
    
    # Determine weather condition
    weather_probs = {
        'Clear': 0.6,
        'Clouds': 0.3,
        'Rain': 0.1
    }
    
    weather_condition = np.random.choice(
        list(weather_probs.keys()),
        p=list(weather_probs.values())
    )
    
    # Adjust precipitation based on weather condition
    if weather_condition == 'Rain':
        precipitation = np.random.uniform(0.1, 1.0)
    elif weather_condition == 'Drizzle':
        precipitation = np.random.uniform(0.01, 0.1)
    else:
        precipitation = 0
    
    # Adjust cloud cover based on weather condition
    if weather_condition == 'Clear':
        cloud_cover = np.random.randint(0, 30)
    elif weather_condition == 'Clouds':
        cloud_cover = np.random.randint(30, 100)
    else:
        cloud_cover = np.random.randint(70, 100)
    
    return {
        'date': date_str,
        'stadium': stadium,
        'temperature': base_temp,
        'feels_like': base_temp + np.random.randint(-5, 5),
        'humidity': np.random.randint(30, 90),
        'pressure': np.random.randint(990, 1030),
        'wind_speed': np.random.randint(0, 20),
        'wind_direction': np.random.randint(0, 360),
        'cloud_cover': cloud_cover,
        'weather_condition': weather_condition,
        'weather_description': f"Synthetic {weather_condition}",
        'precipitation': precipitation,
        'data_source': 'synthetic'
    }

def generate_synthetic_odds(games_df):
    """
    Generate synthetic betting odds for historical games.
    
    Args:
        games_df (DataFrame): Games data
        
    Returns:
        DataFrame: Synthetic odds data
    """
    # Create copy of games_df
    odds_df = games_df[['date', 'home_team', 'away_team', 'total_runs']].copy()
    
    # Generate realistic over/under lines
    # Base on actual total runs plus some noise
    avg_runs = odds_df['total_runs'].mean()
    std_runs = odds_df['total_runs'].std()
    
    # Calculate line as average runs plus some noise, rounded to nearest 0.5
    odds_df['over_under_line'] = np.round((avg_runs + np.random.normal(0, 0.5, len(odds_df))) * 2) / 2
    
    # Generate odds values (typically around -110)
    odds_values = np.random.choice([-115, -110, -105, -100, 100, 105, 110], 
                                  size=(len(odds_df), 2),
                                  p=[0.3, 0.4, 0.1, 0.05, 0.05, 0.05, 0.05])
    
    odds_df['over_odds'] = odds_values[:, 0]
    odds_df['under_odds'] = odds_values[:, 1]
    
    return odds_df

def merge_datasets(games_df, weather_df, odds_df):
    """
    Combine game data, weather data, and odds data into a single dataset.
    
    Args:
        games_df (DataFrame): Game data
        weather_df (DataFrame): Weather data
        odds_df (DataFrame): Odds data
        
    Returns:
        DataFrame: Merged dataset
    """
    print("Merging datasets...")
    
    if games_df is None or weather_df is None or odds_df is None:
        print("Missing required dataset(s) for merging.")
        return None
    
    # Convert dates to string format for merging
    games_df['date_str'] = pd.to_datetime(games_df['date']).dt.strftime('%Y-%m-%d')
    weather_df['date_str'] = pd.to_datetime(weather_df['date']).dt.strftime('%Y-%m-%d')
    odds_df['date_str'] = pd.to_datetime(odds_df['date']).dt.strftime('%Y-%m-%d')
    
    # Merge games with weather
    merged = pd.merge(
        games_df,
        weather_df,
        on=['date_str', 'stadium'],
        how='inner',
        suffixes=('', '_weather')
    )
    
    # Merge with odds
    merged = pd.merge(
        merged,
        odds_df,
        on=['date_str', 'home_team', 'away_team'],
        how='inner',
        suffixes=('', '_odds')
    )
    
    # Clean up duplicate columns
    cols_to_drop = [col for col in merged.columns if col.endswith(('_weather', '_odds'))]
    merged = merged.drop(cols_to_drop, axis=1)
    
    # Remove temporary columns
    if 'date_str' in merged.columns:
        merged = merged.drop('date_str', axis=1)
    
    # Add derived features
    print("Adding derived features...")
    
    # Runs vs line
    merged['runs_vs_line'] = merged['total_runs'] - merged['over_under_line']
    
    # Binary outcome variables
    merged['over_result'] = (merged['total_runs'] > merged['over_under_line']).astype(int)
    merged['under_result'] = (merged['total_runs'] < merged['over_under_line']).astype(int)
    merged['push_result'] = (merged['total_runs'] == merged['over_under_line']).astype(int)
    
    # Temperature squared for non-linear effects
    if 'temperature' in merged.columns:
        merged['temperature_squared'] = merged['temperature'] ** 2
    
    # Weather interaction features
    if 'temperature' in merged.columns and 'humidity' in merged.columns:
        merged['temp_humidity_interaction'] = merged['temperature'] * merged['humidity'] / 100
    
    # Wind direction features
    if 'wind_direction' in merged.columns:
        # Convert wind direction to cardinal direction
        def degrees_to_cardinal(deg):
            if pd.isna(deg):
                return 'Unknown'
            dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
            ix = round(float(deg) / 45) % 8
            return dirs[ix]
        
        merged['wind_cardinal'] = merged['wind_direction'].apply(
            lambda x: degrees_to_cardinal(x) if pd.notnull(x) else 'Unknown'
        )
        
        # One-hot encode wind direction
        wind_dummies = pd.get_dummies(merged['wind_cardinal'], prefix='wind')
        merged = pd.concat([merged, wind_dummies], axis=1)
    
    # Add ballpark factors
    merged['ballpark_factor'] = merged['stadium'].map(
        {name: factors['base'] for name, factors in BALLPARK_FACTORS.items()}
    ).fillna(1.0)
    
    # Month of year
    merged['month'] = pd.to_datetime(merged['date']).dt.month
    
    # Save to disk
    merged.to_csv(f"{DATA_DIR}/merged_data.csv", index=False)
    print(f"Saved merged dataset with {len(merged)} records.")
    
    return merged

def main():
    """Main function to run the MLB Weather Model."""
    args = parse_args()
    
    print("MLB Weather Model")
    print("-" * 30)
    
    # Create data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    # Initialize the model
    mlb_model = MLBModel()
    print("Model initialized successfully!")
    
    # Fetch or load data
    if args.fetch_data:
        # Fetch new data
        games_df = fetch_historical_data(args.start_year, args.end_year, use_existing=args.use_existing_games)
        
        if games_df is not None and not games_df.empty:
            weather_df = fetch_weather_data(games_df, use_existing=args.use_existing_weather)
            odds_df = generate_synthetic_odds(games_df)
            merged_data = merge_datasets(games_df, weather_df, odds_df)
        else:
            print("Failed to fetch game data. Exiting.")
            return
    else:
        # Try to load existing merged data
        merged_file = f"{DATA_DIR}/merged_data.csv"
        if os.path.exists(merged_file):
            print(f"Loading existing merged data from {merged_file}")
            merged_data = pd.read_csv(merged_file, parse_dates=['date'])
        else:
            print("No existing data found. Please run with --fetch-data first.")
            return
    
    # Set the merged data in the model
    mlb_model.merged_data = merged_data
    
    # Train or load model
    if args.train:
        print("\nTraining model...")
        mlb_model.train_model()
    else:
        # Try to load existing model
        try:
            print("\nLoading pre-trained model...")
            mlb_model.load_model()
        except FileNotFoundError:
            print("No pre-trained model found. Training new model...")
            mlb_model.train_model()
    
    # Analyze feature importance if requested
    if args.analyze:
        print("\nAnalyzing feature importance...")
        importance = mlb_model.analyze_feature_importance()
        if importance is not None:
            print("\nTop 10 most important features:")
            print(importance.head(10))
    
    # Run backtest if requested
    if args.backtest:
        print("\nFinding betting opportunities...")
        opportunities = mlb_model.find_betting_opportunities(confidence_threshold=args.threshold)
        
        if opportunities is not None and len(opportunities) > 0:
            print("\nRunning backtest...")
            results = mlb_model.backtest_strategy(kelly=args.kelly)
            
            # Visualize the weather impact on betting success
            print("\nAnalyzing weather impact on betting success...")
            visualize_weather_impact(opportunities)
    
    # Get today's recommendations if requested
    if args.today:
        print("\nGetting today's betting recommendations...")
        today_bets = mlb_model.get_todays_betting_recommendations(confidence_threshold=args.threshold)
        
        if today_bets is not None and len(today_bets) > 0:
            print("\nVisualizing today's recommendations...")
            visualize_mlb_bets(today_bets)
        else:
            print("No betting opportunities found for today.")
    
    print("\nMLB Weather Model completed successfully!")

if __name__ == "__main__":
    main()
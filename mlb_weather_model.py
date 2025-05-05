"""
MLB Weather Model - Main model class for predicting MLB run totals based on weather.
"""

import os
import time
import pickle
import random
import numpy as np
import pandas as pd
import requests
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import configuration
from config import (
    OPENWEATHERMAP_API_KEY, 
    ODDS_API_KEY, 
    DATA_DIR, 
    STADIUM_MAPPING,
    BALLPARK_FACTORS,
    TEAM_NAME_MAP,
    OPEN_METEO_FORECAST,
    OPEN_METEO_HISTORY,
    WEATHER_VARIABLES
)

class MLBWeatherModel:
    def __init__(self, data_dir=DATA_DIR):
        """Initialize the MLB Weather Model."""
        self.data_dir = data_dir
        self.games_df = None
        self.weather_df = None
        self.odds_df = None
        self.merged_data = None
        self.model = None
        self.stadium_mapping = STADIUM_MAPPING
        self.processed_games = set()
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    def fetch_game_data(self, start_year, end_year):
        """
        Fetch real MLB game data using pybaseball with fixed date parsing.

        Parameters:
            start_year (int): First season to collect
            end_year (int): Last season to collect
            
        Returns:
            DataFrame: Processed game data
        """
        print(f"Fetching MLB game data from {start_year} to {end_year}...")

        try:
            import pybaseball
            # Set the cache directory for pybaseball
            pybaseball.cache.enable()
        except ImportError:
            print("pybaseball not installed. Installing now...")
            os.system("pip install pybaseball")
            import pybaseball

        # Initialize empty DataFrame to store all games
        all_games = pd.DataFrame()
        processed_games = set()  # Track processed game IDs

        # MLB teams - using abbreviations that match pybaseball
        teams = ['ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 
                 'COL', 'DET', 'HOU', 'KCR', 'LAA', 'LAD', 'MIA', 'MIL', 
                 'MIN', 'NYM', 'NYY', 'OAK', 'PHI', 'PIT', 'SDP', 'SEA', 
                 'SFG', 'STL', 'TBR', 'TEX', 'TOR', 'WSN']

        for year in range(start_year, end_year + 1):
            print(f"Processing {year} season...")

            try:
                # Use the statcast data for game results
                print(f"Fetching statcast data for {year}...")

                # Get game data from statcast
                start_date = f"{year}-04-01"
                end_date = f"{year}-10-31"

                # Fetch data in smaller chunks to avoid timeouts
                game_chunks = []
                current_date = pd.to_datetime(start_date)
                end_date_dt = pd.to_datetime(end_date)

                while current_date <= end_date_dt:
                    next_date = current_date + pd.DateOffset(days=30)
                    if next_date > end_date_dt:
                        next_date = end_date_dt

                    chunk_start = current_date.strftime('%Y-%m-%d')
                    chunk_end = next_date.strftime('%Y-%m-%d')

                    print(f"Fetching games from {chunk_start} to {chunk_end}...")

                    try:
                        # Use statcast data for date range
                        statcast_data = pybaseball.statcast(start_dt=chunk_start, end_dt=chunk_end)
                        if not statcast_data.empty:
                            game_chunks.append(statcast_data)
                    except Exception as e:
                        print(f"Error fetching statcast data for {chunk_start} to {chunk_end}: {e}")

                    # Move to next chunk
                    current_date = next_date + pd.DateOffset(days=1)

                    # Add delay to avoid rate limiting
                    time.sleep(2)

                # Combine chunks
                if game_chunks:
                    statcast_data = pd.concat(game_chunks)

                    # Process statcast data to extract game information
                    if not statcast_data.empty:
                        # Extract unique game_pk values (each represents a game)
                        game_ids = statcast_data['game_pk'].unique()

                        for game_id in game_ids:
                            try:
                                # Get data for this game
                                game_data = statcast_data[statcast_data['game_pk'] == game_id]

                                # Get home and away teams
                                home_team = game_data['home_team'].iloc[0] if 'home_team' in game_data.columns else None
                                away_team = game_data['away_team'].iloc[0] if 'away_team' in game_data.columns else None

                                # If team info is missing, try to extract from other fields
                                if home_team is None or away_team is None:
                                    if 'inning_topbot' in game_data.columns:
                                        # Find top and bottom of innings to determine teams
                                        top_inning = game_data[game_data['inning_topbot'] == 'Top']
                                        bottom_inning = game_data[game_data['inning_topbot'] == 'Bot']

                                        if not top_inning.empty and 'fielding_team' in top_inning.columns:
                                            home_team = top_inning['fielding_team'].iloc[0]
                                        if not bottom_inning.empty and 'fielding_team' in bottom_inning.columns:
                                            away_team = bottom_inning['fielding_team'].iloc[0]

                                # Skip if we can't determine teams
                                if home_team is None or away_team is None:
                                    continue

                                # Get game date
                                game_date = pd.to_datetime(game_data['game_date'].iloc[0]) if 'game_date' in game_data.columns else None

                                # Skip if date is missing
                                if game_date is None:
                                    continue

                                # Check for duplicate game
                                game_key = f"{game_date.strftime('%Y-%m-%d')}_{home_team}_{away_team}"
                                if game_key in processed_games:
                                    continue

                                processed_games.add(game_key)

                                # Calculate runs per team
                                home_score = 0
                                away_score = 0

                                # Use events to estimate scores
                                if 'events' in game_data.columns and 'inning_topbot' in game_data.columns:
                                    # Score events
                                    score_events = ['home_run', 'sac_fly', 'field_error', 'single', 'double', 'triple']

                                    # Count runs for home and away teams
                                    for _, event_row in game_data.iterrows():
                                        event = event_row['events']
                                        inning = event_row['inning_topbot']

                                        if event in score_events:
                                            # Estimate runs (simplistic approach)
                                            if inning == 'Top':
                                                away_score += 1
                                            else:
                                                home_score += 1

                                total_runs = home_score + away_score

                                # Get stadium info
                                stadium_info = self.stadium_mapping.get(home_team, 
                                                {'name': f"{home_team} Park", 
                                                 'lat': 40.0, 'lon': -75.0})

                                # Add to games list
                                all_games = pd.concat([all_games, pd.DataFrame([{
                                    'date': game_date,
                                    'home_team': home_team,
                                    'away_team': away_team,
                                    'home_score': home_score,
                                    'away_score': away_score,
                                    'total_runs': total_runs,
                                    'stadium': stadium_info['name'],
                                    'stadium_lat': stadium_info['lat'],
                                    'stadium_lon': stadium_info['lon'],
                                    'game_id': game_id
                                }])], ignore_index=True)
                            except Exception as e:
                                print(f"Error processing game {game_id}: {e}")

            except Exception as e:
                print(f"Error processing {year} season: {e}")

        # Alternative approach: use baseball-reference data from pybaseball
        if all_games.empty:
            print("No data collected from statcast. Using team schedules instead...")

            for year in range(start_year, end_year + 1):
                for team in teams:
                    try:
                        print(f"Fetching schedule for {team} {year}...")

                        # Get team's schedule without parsing dates (safer)
                        try:
                            team_schedule = pybaseball.schedule_and_record(year, team, parse_dates=False)
                        except Exception as e:
                            print(f"Error with parse_dates=False: {e}")
                            # Try alternative date parsing
                            team_schedule = pybaseball.schedule_and_record(year, team)

                        # Filter for completed regular season games
                        completed_games = team_schedule[team_schedule['R'].notna() & team_schedule['RA'].notna()]

                        # Process each game
                        for _, game in completed_games.iterrows():
                            try:
                                # Only process home games to avoid duplicates
                                if game['Home_Away'] == 'Home':
                                    # Parse date safely
                                    try:
                                        game_date = pd.to_datetime(game['Date'])
                                    except:
                                        # Manual parsing with regex if needed
                                        import re
                                        date_str = game['Date']
                                        match = re.search(r'(\w+), (\w+) (\d+)', date_str)
                                        if match:
                                            month_str, day_str = match.group(2), match.group(3)
                                            month_map = {
                                                'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
                                                'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
                                                'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                                            }
                                            month_num = month_map.get(month_str, 1)
                                            game_date = pd.Timestamp(year=year, month=month_num, day=int(day_str))
                                        else:
                                            continue

                                    # Get stadium info
                                    stadium_info = self.stadium_mapping.get(team, 
                                                    {'name': f"{team} Park", 
                                                     'lat': 40.0, 'lon': -75.0})

                                    # Extract scores
                                    try:
                                        home_score = int(game['R'])
                                        away_score = int(game['RA'])
                                        total_runs = home_score + away_score
                                    except:
                                        print(f"Invalid score for {team} vs {game['Opponent']} on {game_date}")
                                        continue

                                    # Check for duplicate game before adding
                                    game_date_str = game_date.strftime('%Y-%m-%d')
                                    game_key = f"{game_date_str}_{team}_{game['Opponent']}"
                                    if game_key in processed_games:
                                        continue

                                    processed_games.add(game_key)

                                    # Add to games list
                                    all_games = pd.concat([all_games, pd.DataFrame([{
                                        'date': game_date,
                                        'home_team': team,
                                        'away_team': game['Opponent'],
                                        'home_score': home_score,
                                        'away_score': away_score,
                                        'total_runs': total_runs,
                                        'stadium': stadium_info['name'],
                                        'stadium_lat': stadium_info['lat'],
                                        'stadium_lon': stadium_info['lon']
                                    }])], ignore_index=True)
                            except Exception as e:
                                print(f"Error processing game: {e}")

                        # Add delay to avoid rate limiting
                        time.sleep(1)

                    except Exception as e:
                        print(f"Error fetching data for {team} {year}: {e}")

        # If we still don't have data, generate synthetic data
        if all_games.empty:
            print("No data collected. Generating synthetic data based on team batting stats...")

            for year in range(start_year, end_year + 1):
                try:
                    # Get team batting stats for the year
                    team_batting = pybaseball.team_batting(year)

                    # Create placeholder games for each team
                    for team in teams:
                        if team in team_batting.index:
                            team_games = team_batting.loc[team, 'G']
                            avg_runs = team_batting.loc[team, 'R'] / team_games

                            # Create synthetic games with realistic scores
                            for i in range(int(team_games / 2)):  # Half the games are home games
                                game_date = pd.Timestamp(year=year, month=random.randint(4, 9), day=random.randint(1, 28))
                                away_team = random.choice([t for t in teams if t != team])

                                # Generate realistic scores based on team averages
                                home_score = max(0, int(np.random.poisson(avg_runs)))
                                away_score = max(0, int(np.random.poisson(avg_runs * 0.9)))  # Slight home advantage

                                # Get stadium info
                                stadium_info = self.stadium_mapping.get(team, 
                                                {'name': f"{team} Park", 
                                                 'lat': 40.0, 'lon': -75.0})

                                # Add to games list
                                all_games = pd.concat([all_games, pd.DataFrame([{
                                    'date': game_date,
                                    'home_team': team,
                                    'away_team': away_team,
                                    'home_score': home_score,
                                    'away_score': away_score,
                                    'total_runs': home_score + away_score,
                                    'stadium': stadium_info['name'],
                                    'stadium_lat': stadium_info['lat'],
                                    'stadium_lon': stadium_info['lon'],
                                    'data_source': 'team_stats'
                                }])], ignore_index=True)
                except Exception as e:
                    print(f"Error getting team stats for {year}: {e}")

        # Clean up data
        if not all_games.empty:
            all_games['date'] = pd.to_datetime(all_games['date'])
            all_games = all_games.sort_values('date')
            self.games_df = all_games

            # Save to disk
            self.games_df.to_csv(f"{self.data_dir}/mlb_games_{start_year}_{end_year}.csv", index=False)
            print(f"Saved {len(self.games_df)} games to CSV.")
        else:
            print("Failed to collect any game data. Please check network connection and try again.")

        return self.games_df
    
    def fetch_weather_data(self):
        """
        Fetch weather data for games using Open-Meteo API.
        """
        print("Fetching weather data for game locations and dates...")
        
        if self.games_df is None:
            raise ValueError("Game data must be loaded before fetching weather data.")
        
        weather_list = []
        
        # Group games by stadium to minimize API calls
        stadium_groups = self.games_df.groupby(['stadium', 'stadium_lat', 'stadium_lon'])
        
        for (stadium, lat, lon), games in stadium_groups:
            print(f"Fetching weather data for {stadium}...")
            
            # Get unique dates for this stadium
            dates = games['date'].dt.strftime('%Y-%m-%d').unique()
            dates = sorted(dates)
            
            if len(dates) == 0:
                continue
                
            # Split dates into historical and forecast
            today = pd.Timestamp.now().strftime('%Y-%m-%d')
            historical_dates = [d for d in dates if d < today]
            forecast_dates = [d for d in dates if d >= today]
            
            try:
                # Fetch historical data
                if historical_dates:
                    start_date = min(historical_dates)
                    end_date = max(historical_dates)
                    
                    params = {
                        'latitude': lat,
                        'longitude': lon,
                        'start_date': start_date,
                        'end_date': end_date,
                        'hourly': ','.join(WEATHER_VARIABLES),
                        'timezone': 'America/New_York'
                    }
                    
                    response = requests.get(OPEN_METEO_HISTORY, params=params)
                    
                    if response.status_code == 200:
                        data = response.json()
                        hourly_data = data.get('hourly', {})
                        times = hourly_data.get('time', [])
                        
                        # Process each time point (assuming game time is 1PM local)
                        for date in historical_dates:
                            target_time = f"{date}T13:00"
                            if target_time in times:
                                idx = times.index(target_time)
                                weather_list.append({
                                    'date': date,
                                    'stadium': stadium,
                                    'temperature': hourly_data['temperature_2m'][idx],
                                    'feels_like': hourly_data['apparent_temperature'][idx],
                                    'humidity': hourly_data['relativehumidity_2m'][idx],
                                    'pressure': hourly_data['pressure_msl'][idx],
                                    'wind_speed': hourly_data['windspeed_10m'][idx],
                                    'wind_direction': hourly_data['winddirection_10m'][idx],
                                    'cloud_cover': hourly_data['cloudcover'][idx],
                                    'precipitation': hourly_data['precipitation'][idx],
                                    'data_source': 'historical'
                                })
                
                # Fetch forecast data
                if forecast_dates:
                    params = {
                        'latitude': lat,
                        'longitude': lon,
                        'hourly': ','.join(WEATHER_VARIABLES),
                        'timezone': 'America/New_York'
                    }
                    
                    response = requests.get(OPEN_METEO_FORECAST, params=params)
                    
                    if response.status_code == 200:
                        data = response.json()
                        hourly_data = data.get('hourly', {})
                        times = hourly_data.get('time', [])
                        
                        for date in forecast_dates:
                            target_time = f"{date}T13:00"
                            if target_time in times:
                                idx = times.index(target_time)
                                weather_list.append({
                                    'date': date,
                                    'stadium': stadium,
                                    'temperature': hourly_data['temperature_2m'][idx],
                                    'feels_like': hourly_data['apparent_temperature'][idx],
                                    'humidity': hourly_data['relativehumidity_2m'][idx],
                                    'pressure': hourly_data['pressure_msl'][idx],
                                    'wind_speed': hourly_data['windspeed_10m'][idx],
                                    'wind_direction': hourly_data['winddirection_10m'][idx],
                                    'cloud_cover': hourly_data['cloudcover'][idx],
                                    'precipitation': hourly_data['precipitation'][idx],
                                    'data_source': 'forecast'
                                })
                                
            except Exception as e:
                print(f"Error fetching weather for {stadium}: {e}")
                continue
            
            # Add delay between stadium requests
            time.sleep(1)
        
        # Create DataFrame from weather data
        self.weather_df = pd.DataFrame(weather_list)
        
        # Fill missing values with reasonable estimates
        numeric_cols = ['temperature', 'feels_like', 'humidity', 'pressure', 
                       'wind_speed', 'wind_direction', 'cloud_cover', 'precipitation']
        
        for col in numeric_cols:
            if col in self.weather_df.columns:
                self.weather_df[col] = pd.to_numeric(self.weather_df[col], errors='coerce')
                self.weather_df[col].fillna(self.weather_df[col].median(), inplace=True)
        
        # Save to disk
        self.weather_df.to_csv(f"{self.data_dir}/weather_data.csv", index=False)
        print(f"Saved weather data for {len(self.weather_df)} games.")
        
        return self.weather_df
    
    def fetch_odds_data(self):
        """
        Fetch real betting odds data using the Odds API where possible,
        and generate synthetic odds for historical games.
        
        Returns:
            DataFrame: Betting odds data
        """
        print("Fetching betting odds...")
        
        if self.games_df is None:
            raise ValueError("Game data must be loaded before fetching odds data.")
        
        odds_list = []
        
        # Try to get current MLB odds from the API
        print("Fetching real betting odds data...")
        
        try:
            url = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/odds?apiKey={ODDS_API_KEY}&regions=us&markets=totals"
            
            response = requests.get(url)
            print(f"Odds API status code: {response.status_code}")
            
            real_odds_count = 0
            
            if response.status_code == 200:
                current_odds = response.json()
                print(f"Retrieved {len(current_odds)} current MLB game odds")
                
                # Process current odds
                for game in current_odds:
                    try:
                        # Extract teams
                        home_team = game.get('home_team')
                        away_team = game.get('away_team')
                        
                        # Convert team names to match our format if needed
                        home_team_abbr = self._convert_team_name(home_team)
                        away_team_abbr = self._convert_team_name(away_team)
                        
                        # Extract the over/under line and odds
                        over_under_line = None
                        over_odds = None
                        under_odds = None
                        
                        # Navigate the JSON structure to find totals market
                        for bookmaker in game.get('bookmakers', []):
                            for market in bookmaker.get('markets', []):
                                if market.get('key') == 'totals':
                                    for outcome in market.get('outcomes', []):
                                        if outcome.get('name') == 'Over':
                                            # Explicitly convert to float
                                            over_under_line = float(outcome.get('point', 0))
                                            over_odds = float(outcome.get('price', 0))
                                        elif outcome.get('name') == 'Under':
                                            under_odds = float(outcome.get('price', 0))
                        
                        # Only add games with complete odds data
                        if over_under_line is not None and over_odds is not None and under_odds is not None:
                            # Convert date to proper format
                            game_date = datetime.now().strftime('%Y-%m-%d')
                            if 'commence_time' in game:
                                commence_time = game['commence_time']
                                if isinstance(commence_time, str):
                                    game_date = datetime.fromisoformat(commence_time.replace('Z', '+00:00')).strftime('%Y-%m-%d')
                                elif isinstance(commence_time, (int, float)):
                                    game_date = datetime.fromtimestamp(commence_time).strftime('%Y-%m-%d')
                            
                            odds_list.append({
                                'date': game_date,
                                'home_team': home_team_abbr,
                                'away_team': away_team_abbr,
                                'over_under_line': over_under_line,
                                'over_odds': over_odds,
                                'under_odds': under_odds,
                                'source': 'real_odds_api'
                            })
                            
                            real_odds_count += 1
                    except Exception as e:
                        print(f"Error processing game odds: {e}")
                
                print(f"Successfully processed {real_odds_count} real odds entries")
        
        except Exception as e:
            print(f"Error fetching odds data: {e}")
        
        # Generate synthetic odds for the rest of the games
        print("Generating synthetic odds for remaining games...")
        
        # Calculate average runs per game
        avg_runs = self.games_df['total_runs'].mean()
        print(f"Season average runs per game: {avg_runs:.2f}")
        
        # Create a set of games already covered by the API
        games_with_odds = set()
        for odds in odds_list:
            game_key = f"{odds['date']}_{odds['home_team']}_{odds['away_team']}"
            games_with_odds.add(game_key)
        
        # Fill in missing games with synthetic odds
        for _, game in self.games_df.iterrows():
            game_date = pd.to_datetime(game['date']).strftime('%Y-%m-%d')
            game_key = f"{game_date}_{game['home_team']}_{game['away_team']}"
            
            # Skip if we already have odds for this game
            if game_key in games_with_odds:
                continue
            
            # Generate synthetic odds
            # For over/under line, use the average runs for the season plus some small variation
            base_line = round(avg_runs * 2) / 2  # Round to nearest 0.5
            variation = np.random.choice([-1.0, -0.5, 0, 0.5, 1.0], p=[0.1, 0.2, 0.4, 0.2, 0.1])
            over_under = base_line + variation
            
            # For odds, generate realistic values (typically around -110)
            over_odds = float(np.random.choice([-115, -110, -105, -100, 100], p=[0.2, 0.5, 0.2, 0.05, 0.05]))
            under_odds = float(np.random.choice([-115, -110, -105, -100, 100], p=[0.2, 0.5, 0.2, 0.05, 0.05]))
            
            odds_list.append({
                'date': game_date,
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'over_under_line': over_under,
                'over_odds': over_odds,
                'under_odds': under_odds,
                'source': 'synthetic'
            })
        
        self.odds_df = pd.DataFrame(odds_list)
        
        # Convert date to datetime
        self.odds_df['date'] = pd.to_datetime(self.odds_df['date'])
        
        # Save to disk
        self.odds_df.to_csv(f"{self.data_dir}/odds_data.csv", index=False)
        
        real_count = len(self.odds_df[self.odds_df['source'] == 'real_odds_api'])
        synthetic_count = len(self.odds_df[self.odds_df['source'] == 'synthetic'])
        print(f"Saved odds data for {len(self.odds_df)} games ({real_count} real, {synthetic_count} synthetic).")
        
        return self.odds_df
    
    def merge_data(self):
        """
        Combine game data, weather data, and odds data into a single dataset.
        """
        print("Merging datasets...")
        
        if self.games_df is None or self.odds_df is None:
            raise ValueError("Game data and odds data must be loaded before merging.")
            
        # Initialize empty weather_df if it doesn't exist
        if self.weather_df is None or len(self.weather_df) == 0:
            print("No weather data available. Creating empty weather DataFrame...")
            self.weather_df = pd.DataFrame(columns=[
                'date', 'stadium', 'temperature', 'feels_like', 'humidity',
                'pressure', 'wind_speed', 'wind_direction', 'cloud_cover', 
                'precipitation', 'data_source'
            ])
        
        # Convert dates to datetime
        self.games_df['date'] = pd.to_datetime(self.games_df['date'])
        self.odds_df['date'] = pd.to_datetime(self.odds_df['date'])
        self.weather_df['date'] = pd.to_datetime(self.weather_df['date'])
        
        # Convert dates to string format for merging
        self.games_df['date_str'] = self.games_df['date'].dt.strftime('%Y-%m-%d')
        self.weather_df['date_str'] = self.weather_df['date'].dt.strftime('%Y-%m-%d')
        self.odds_df['date_str'] = self.odds_df['date'].dt.strftime('%Y-%m-%d')
        
        # Merge games with weather on date and stadium
        print("Merging games with weather data...")
        merged = pd.merge(
            self.games_df,
            self.weather_df,
            on=['date_str', 'stadium'],
            how='inner',
            suffixes=('_game', '_weather')
        )
        
        # Merge with odds data
        print("Merging with odds data...")
        merged = pd.merge(
            merged,
            self.odds_df,
            on=['date_str', 'home_team', 'away_team'],
            how='inner',
            suffixes=('', '_odds')
        )
        
        # Clean up columns - check and remove duplicates
        cols = merged.columns.tolist()
        dup_cols = [col for col in cols if cols.count(col) > 1]
        
        if dup_cols:
            print(f"Found duplicate columns: {dup_cols}")
            # Drop duplicate columns
            for col in dup_cols:
                # Find indices of duplicate columns
                indices = [i for i, x in enumerate(cols) if x == col]
                # Keep first occurrence, drop the rest
                for idx in indices[1:]:
                    merged = merged.drop(cols[idx], axis=1)
                    # Update cols list
                    cols.pop(idx)
        
        # Ensure we have the date column
        if 'date_game' in merged.columns:
            merged['date'] = merged['date_game']
            merged = merged.drop('date_game', axis=1)
        elif 'date' not in merged.columns:
            print("WARNING: 'date' column missing, using date_str instead")
            merged['date'] = pd.to_datetime(merged['date_str'])
        
        # Drop temporary columns
        if 'date_str' in merged.columns:
            merged = merged.drop('date_str', axis=1)
        
        # Ensure date is datetime
        merged['date'] = pd.to_datetime(merged['date'])
        
        # Add derived features
        print("Adding derived features...")
        
        # Runs vs line - how many runs over/under the betting line
        merged['runs_vs_line'] = merged['total_runs'] - merged['over_under_line']
        
        # Binary outcome variables
        merged['over_result'] = (merged['total_runs'] > merged['over_under_line']).astype(int)
        merged['under_result'] = (merged['total_runs'] < merged['over_under_line']).astype(int)
        merged['push_result'] = (merged['total_runs'] == merged['over_under_line']).astype(int)
        
        # Temperature squared (for non-linear effects)
        if 'temperature' in merged.columns:
            merged['temperature_squared'] = merged['temperature'] ** 2
        
        # Weather interaction features
        if 'temperature' in merged.columns and 'humidity' in merged.columns:
            merged['temp_humidity_interaction'] = merged['temperature'] * merged['humidity'] / 100
        
        # Wind direction features
        if 'wind_direction' in merged.columns:
            # Convert wind direction degrees to cardinal direction
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
        
        # Add ballpark factors (altitude effect)
        merged['ballpark_factor'] = merged['stadium'].map(BALLPARK_FACTORS).fillna(1.0)
        
        # Month of year as a feature
        merged['month'] = merged['date'].dt.month
        
        self.merged_data = merged
        
        # Save to disk
        self.merged_data.to_csv(f"{self.data_dir}/merged_data.csv", index=False)
        print(f"Saved merged dataset with {len(self.merged_data)} records.")
        
        return self.merged_data
    
    def prepare_features(self):
        """
        Prepare features for modeling, including scaling and encoding.
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test) datasets for model training
        """
        if self.merged_data is None:
            raise ValueError("Data must be merged before preparing features.")
        
        # Select numeric base features first
        base_features = [
            'temperature', 'temperature_squared', 'humidity', 'wind_speed',
            'precipitation', 'cloud_cover', 'pressure', 'ballpark_factor',
            'temp_humidity_interaction', 'month'
        ]
        
        # Add one-hot encoded wind direction columns, but exclude the original columns
        wind_cols = [col for col in self.merged_data.columns if col.startswith('wind_') 
                    and col != 'wind_direction' and col != 'wind_cardinal']
        
        # Combine features without duplicates
        feature_cols = list(set(base_features + wind_cols))
        
        print("Using these features:", feature_cols)
        
        # Create feature matrix making sure all columns exist
        available_cols = [col for col in feature_cols if col in self.merged_data.columns]
        X = self.merged_data[available_cols].copy()
        
        # Fix any non-numeric columns
        for col in X.columns:
            if X[col].dtypes == 'object':
                print(f"WARNING: Column {col} contains non-numeric data. Converting to numeric.")
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Fill any NaN values
        X = X.fillna(X.mean())
        
        # Target variable
        y = self.merged_data['total_runs']
        
        # Split into training and testing sets
        # Use time-based split if possible
        if 'date' in self.merged_data.columns and len(self.merged_data['date'].unique()) > 1:
            print("Using time-based train/test split...")
            # Sort by date
            self.merged_data = self.merged_data.sort_values('date')
            
            # Use 80% for training, 20% for testing
            train_size = int(len(self.merged_data) * 0.8)
            train_idx = self.merged_data.index[:train_size]
            test_idx = self.merged_data.index[train_size:]
            
            X_train = X.loc[train_idx]
            X_test = X.loc[test_idx]
            y_train = y.loc[train_idx]
            y_test = y.loc[test_idx]
        else:
            # Fall back to random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self):
        """
        Train a model to predict total runs based on weather features.
        
        Returns:
            Pipeline: Trained model pipeline
        """
        print("Training model...")
        
        X_train, X_test, y_train, y_test = self.prepare_features()
        
        # Create a pipeline with preprocessing and model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            ))
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"Model performance: MSE = {mse:.4f}, MAE = {mae:.4f}")
        
        self.model = pipeline
        
        # Save model to disk
        with open(f"{self.data_dir}/weather_model.pkl", 'wb') as f:
            pickle.dump(pipeline, f)
        
        return self.model
    
    def analyze_feature_importance(self):
        """
        Analyze which weather features have the biggest impact on run scoring.
        
        Returns:
            DataFrame: Feature importance rankings
        """
        if self.model is None:
            raise ValueError("Model must be trained before analyzing feature importance.")
        
        model = self.model.named_steps['model']
        
        # Get feature importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            X_train, _, _, _ = self.prepare_features()
            feature_names = X_train.columns
            
            # Create DataFrame of feature importances
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Plot feature importances
            plt.figure(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', data=importance_df)
            plt.title('Weather Feature Importance for MLB Run Scoring')
            plt.tight_layout()
            plt.savefig(f"{self.data_dir}/feature_importance.png")
            
            return importance_df
        else:
            return None
    
    def find_betting_opportunities(self, confidence_threshold=0.75):
        """Find potential betting opportunities based on model predictions."""
        if self.model is None:
            raise ValueError("Model must be trained before finding betting opportunities.")
        
        # Prepare features for predictions
        print("Using these features:", self.model.named_steps['scaler'].feature_names_in_.tolist())
        X_train, X_test, y_train, y_test = self.prepare_features()
        
        # Make predictions on test set
        test_idx = X_test.index
        test_data = self.merged_data.loc[test_idx].copy()
        
        # Add predictions to the test data
        test_data['predicted_runs'] = self.model.predict(X_test)
        test_data['pred_diff'] = test_data['predicted_runs'] - test_data['over_under_line']
        
        # Find opportunities where prediction differs from line by more than threshold
        opportunities = test_data[abs(test_data['pred_diff']) > confidence_threshold].copy()
        
        # ...existing code...
        
        # Calculate overall accuracy and print detailed stats
        accuracy = opportunities['bet_correct'].mean()
        print(f"Found {len(opportunities)} potential betting opportunities")
        print(f"Betting model accuracy: {accuracy:.4f}")
        print(f"Average prediction difference: {abs(opportunities['pred_diff']).mean():.2f} runs")
        
        return opportunities

    def backtest_strategy(self, bet_size=100.0, kelly=False):
        """Backtest the betting strategy with realistic parameters."""
        print("\nFinding betting opportunities...")
        opportunities = self.find_betting_opportunities()
        
        if len(opportunities) == 0:
            print("No betting opportunities found.")
            return None
        
        print("\nRunning backtest...")
        # ...existing code...
        
        # Calculate and print detailed performance metrics
        bet_count = len(results_df)
        winning_bets = len(results_df[results_df['profit'] > 0])
        losing_bets = len(results_df[results_df['profit'] < 0])
        push_bets = len(results_df[results_df['profit'] == 0])
        win_rate = winning_bets / bet_count if bet_count > 0 else 0
        roi = (bankroll - initial_bankroll) / initial_bankroll * 100
        
        print(f"\nBacktest Results:")
        print(f"Total Bets: {bet_count}")
        print(f"Winning Bets: {winning_bets}")
        print(f"Losing Bets: {losing_bets}")
        print(f"Push Bets: {push_bets}")
        print(f"Win Rate: {win_rate:.4f}")
        print(f"ROI: {roi:.2f}%")
        print(f"Initial Bankroll: ${initial_bankroll:.2f}")
        print(f"Final Bankroll: ${bankroll:.2f}")
        print(f"Total Profit/Loss: ${bankroll - initial_bankroll:.2f}")
        
        return results_df
    
    def get_todays_betting_recommendations(self, confidence_threshold=0.15):
        """
        Get betting recommendations for today's MLB games using real odds.

        Parameters:
            confidence_threshold (float): Minimum difference between predicted runs and line

        Returns:
            DataFrame: Recommended bets for today's games
        """
        print("Fetching today's MLB odds...")

        # Get today's date
        today = datetime.now().strftime('%Y-%m-%d')

        # Fetch current MLB odds from API
        url = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/odds?apiKey={ODDS_API_KEY}&regions=us&markets=totals"

        try:
            response = requests.get(url)
            print(f"API Response: {response.status_code}")

            if response.status_code == 200:
                games_odds = []
                current_odds = response.json()
                print(f"Found {len(current_odds)} MLB games with odds")

                for game in current_odds:
                    try:
                        # Extract teams
                        home_team = game.get('home_team')
                        away_team = game.get('away_team')

                        # Convert team names to match our format
                        home_team_abbr = self._convert_team_name(home_team)
                        away_team_abbr = self._convert_team_name(away_team)

                        # Get game date/time
                        game_datetime = game.get('commence_time')
                        if game_datetime:
                            game_date = datetime.fromisoformat(game_datetime.replace('Z', '+00:00'))
                        else:
                            game_date = datetime.now()

                        # Get over/under line and odds
                        over_under_line = None
                        over_odds = None
                        under_odds = None

                        # Navigate to find totals market
                        for bookmaker in game.get('bookmakers', []):
                            for market in bookmaker.get('markets', []):
                                if market.get('key') == 'totals':
                                    for outcome in market.get('outcomes', []):
                                        if outcome.get('name') == 'Over':
                                            over_under_line = float(outcome.get('point'))
                                            over_odds = float(outcome.get('price'))
                                        elif outcome.get('name') == 'Under':
                                            under_odds = float(outcome.get('price'))

                        if over_under_line and over_odds and under_odds:
                            # Get stadium info
                            stadium_info = self.stadium_mapping.get(home_team_abbr, None)

                            if stadium_info:
                                # Fetch weather for the stadium
                                lat = stadium_info.get('lat')
                                lon = stadium_info.get('lon')
                                stadium_name = stadium_info.get('name')

                                # Get weather data
                                weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=imperial&appid={OPENWEATHERMAP_API_KEY}"
                                weather_response = requests.get(weather_url)

                                if weather_response.status_code == 200:
                                    weather_data = weather_response.json()

                                    # Build features for prediction
                                    features = {
                                        'date': game_date,
                                        'home_team': home_team_abbr,
                                        'away_team': away_team_abbr,
                                        'over_under_line': over_under_line,
                                        'over_odds': over_odds,
                                        'under_odds': under_odds,
                                        'stadium': stadium_name,
                                        'temperature': weather_data.get('main', {}).get('temp'),
                                        'humidity': weather_data.get('main', {}).get('humidity'),
                                        'wind_speed': weather_data.get('wind', {}).get('speed'),
                                        'wind_direction': weather_data.get('wind', {}).get('deg'),
                                        'pressure': weather_data.get('main', {}).get('pressure'),
                                        'cloud_cover': weather_data.get('clouds', {}).get('all'),
                                        'precipitation': weather_data.get('rain', {}).get('1h', 0) if 'rain' in weather_data else 0,
                                        'month': game_date.month,
                                        'source': 'real_odds_api'
                                    }

                                    # Add ballpark factor
                                    features['ballpark_factor'] = BALLPARK_FACTORS.get(stadium_name, 1.0)

                                    # Append to games list
                                    games_odds.append(features)
                    except Exception as e:
                        print(f"Error processing game: {e}")

                if games_odds:
                    # Convert to DataFrame
                    games_df = pd.DataFrame(games_odds)
                    
                    # Remove duplicate entries based on 'home_team' and 'away_team'
                    games_df = games_df.drop_duplicates(subset=['home_team', 'away_team'], keep='first')

                    # Prepare features for prediction
                    X_pred = games_df.copy()

                    # Add derived features
                    if 'temperature' in X_pred.columns:
                        X_pred['temperature_squared'] = X_pred['temperature'] ** 2
                    if 'temperature' in X_pred.columns and 'humidity' in X_pred.columns:
                        X_pred['temp_humidity_interaction'] = X_pred['temperature'] * X_pred['humidity'] / 100

                    # Convert wind direction to cardinal
                    if 'wind_direction' in X_pred.columns:
                        def degrees_to_cardinal(deg):
                            if pd.isna(deg):
                                return 'Unknown'
                            dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
                            ix = round(float(deg) / 45) % 8
                            return dirs[ix]

                        X_pred['wind_cardinal'] = X_pred['wind_direction'].apply(
                            lambda x: degrees_to_cardinal(x) if pd.notnull(x) else 'Unknown'
                        )

                        # One-hot encode wind direction
                        wind_dummies = pd.get_dummies(X_pred['wind_cardinal'], prefix='wind')
                        X_pred = pd.concat([X_pred, wind_dummies], axis=1)

                    # Get features used by model
                    model_features = self.model.named_steps['scaler'].feature_names_in_
                    available_features = [f for f in model_features if f in X_pred.columns]

                    # If missing features, fill with defaults
                    for feature in model_features:
                        if feature not in X_pred.columns:
                            if feature.startswith('wind_'):
                                X_pred[feature] = 0  # Default for one-hot encoded features
                            else:
                                X_pred[feature] = 0  # Default value

                    # Get prediction features
                    X_features = X_pred[model_features]

                    # Predict runs
                    X_pred['predicted_runs'] = self.model.predict(X_features)

                    # Adjust prediction based on current league run environment
                    X_pred['predicted_runs'] = (X_pred['predicted_runs'] * 0.9)  # Sample adjustment factor

                    # Temporary fix: Halve the predicted runs
                    X_pred['predicted_runs'] = X_pred['predicted_runs'] / 2

                    # Calculate difference from line
                    X_pred['diff_from_line'] = X_pred['predicted_runs'] - X_pred['over_under_line']

                    # Recommend bet based on threshold
                    X_pred['recommended_bet'] = np.where(
                        X_pred['diff_from_line'] > confidence_threshold, 'OVER',
                        np.where(X_pred['diff_from_line'] < -confidence_threshold, 'UNDER', 'PASS')
                    )

                    # Calculate absolute confidence
                    X_pred['confidence'] = abs(X_pred['diff_from_line'])

                    # Filter for actionable bets
                    bets = X_pred[X_pred['recommended_bet'] != 'PASS'].copy()

                    # Sort by confidence level (highest first)
                    bets = bets.sort_values('confidence', ascending=False)

                    # Select display columns
                    display_cols = ['date', 'home_team', 'away_team', 'over_under_line', 
                                   'predicted_runs', 'recommended_bet', 'confidence', 
                                   'over_odds', 'under_odds', 'temperature', 'wind_speed']

                    available_cols = [col for col in display_cols if col in bets.columns]
                    result = bets[available_cols]

                    if len(result) > 0:
                        print(f"Found {len(result)} betting opportunities!")
                        return result
                    else:
                        print("No betting opportunities found with current threshold.")
                        return None
                else:
                    print("No valid games with odds found.")
                    return None
            else:
                print(f"Failed to get odds. Status code: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching odds: {e}")
            return None
    
    def save_model(self, filename='weather_model.pkl'):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Please train a model first.")
        
        with open(f"{self.data_dir}/{filename}", 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {self.data_dir}/{filename}")
    
    def load_model(self, filename='weather_model.pkl'):
        """Load a trained model from disk."""
        model_path = f"{self.data_dir}/{filename}"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found.")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {model_path}")
        return self.model
        
    def _convert_team_name(self, full_name):
        """Convert full team name to abbreviation."""
        # Check if name is in our mapping
        if full_name in TEAM_NAME_MAP:
            return TEAM_NAME_MAP[full_name]
            
        # If no match, return the original name
        return full_name
    
    def _american_to_decimal(self, american_odds):
        """Convert American odds to decimal format."""
        if american_odds is None:
            return 1.91  # Default
        
        try:
            american_odds = float(american_odds)
            if american_odds > 0:
                return 1 + (american_odds / 100)
            else:
                return 1 + (100 / abs(american_odds))
        except:
            return 1.91  # Default if conversion fails
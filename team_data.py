"""
Team and player data collection module for MLB Weather Model.
"""

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from config import DATA_DIR, TEAM_NAME_MAP

def fetch_team_stats(year=None, cache=True):
    """
    Fetch MLB team batting and pitching stats.
    
    Args:
        year (int): Year to fetch stats for. If None, uses current year.
        cache (bool): Whether to use cached data if available
        
    Returns:
        tuple: (batting_df, pitching_df) - DataFrames containing team stats
    """
    # If no year specified, use current year
    if year is None:
        year = datetime.now().year
    
    cache_file_batting = f"{DATA_DIR}/team_batting_{year}.csv"
    cache_file_pitching = f"{DATA_DIR}/team_pitching_{year}.csv"
    
    # Check if cache exists and should be used
    if cache and os.path.exists(cache_file_batting) and os.path.exists(cache_file_pitching):
        print(f"Loading cached team stats from {year}")
        batting_df = pd.read_csv(cache_file_batting)
        pitching_df = pd.read_csv(cache_file_pitching)
        return batting_df, pitching_df
    
    try:
        import pybaseball
        # Use pybaseball to fetch team stats
        print(f"Fetching team batting stats for {year}...")
        team_batting = pybaseball.team_batting(year)
        
        print(f"Fetching team pitching stats for {year}...")
        team_pitching = pybaseball.team_pitching(year)
        
        # Save to cache if requested
        if cache:
            team_batting.to_csv(cache_file_batting)
            team_pitching.to_csv(cache_file_pitching)
            print(f"Saved team stats to cache files")
        
        return team_batting, team_pitching
    
    except Exception as e:
        print(f"Error fetching team stats: {e}")
        # Return empty DataFrames
        return pd.DataFrame(), pd.DataFrame()

def fetch_starting_pitchers(start_date=None, end_date=None, cache=True):
    """
    Fetch MLB starting pitchers for date range.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        cache (bool): Whether to use cached data if available
        
    Returns:
        DataFrame: Containing starting pitcher data for each game
    """
    # If no dates specified, use recent range
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    cache_file = f"{DATA_DIR}/starting_pitchers_{start_date}_to_{end_date}.csv"
    
    # Check if cache exists and should be used
    if cache and os.path.exists(cache_file):
        print(f"Loading cached starting pitcher data")
        return pd.read_csv(cache_file)
    
    try:
        import pybaseball
        # Use pybaseball to fetch probable pitchers
        print(f"Fetching starting pitchers from {start_date} to {end_date}...")
        
        # Convert dates to datetime for processing
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Get data for each day in range
        all_starters = []
        current_dt = start_dt
        
        while current_dt <= end_dt:
            date_str = current_dt.strftime('%Y-%m-%d')
            try:
                print(f"Fetching pitchers for {date_str}...")
                # Get probable starters for this date
                day_starters = pybaseball.probable_pitchers(date_str)
                
                if day_starters is not None and not day_starters.empty:
                    # Add date column
                    day_starters['game_date'] = date_str
                    all_starters.append(day_starters)
                
                # Respect rate limits
                time.sleep(1)
            
            except Exception as e:
                print(f"Error fetching starters for {date_str}: {e}")
            
            # Move to next day
            current_dt += timedelta(days=1)
        
        # Combine all data
        if all_starters:
            starters_df = pd.concat(all_starters, ignore_index=True)
            
            # Save to cache if requested
            if cache:
                starters_df.to_csv(cache_file, index=False)
                print(f"Saved starting pitcher data to {cache_file}")
            
            return starters_df
        else:
            print("No starting pitcher data found")
            return pd.DataFrame()
    
    except Exception as e:
        print(f"Error fetching starting pitchers: {e}")
        # Return empty DataFrame
        return pd.DataFrame()

def fetch_pitcher_stats(year=None, cache=True):
    """
    Fetch MLB pitcher stats.
    
    Args:
        year (int): Year to fetch stats for. If None, uses current year.
        cache (bool): Whether to use cached data if available
        
    Returns:
        DataFrame: Containing pitcher stats
    """
    # If no year specified, use current year
    if year is None:
        year = datetime.now().year
    
    cache_file = f"{DATA_DIR}/pitcher_stats_{year}.csv"
    
    # Check if cache exists and should be used
    if cache and os.path.exists(cache_file):
        print(f"Loading cached pitcher stats from {year}")
        return pd.read_csv(cache_file)
    
    try:
        import pybaseball
        # Use pybaseball to fetch pitcher stats
        print(f"Fetching pitcher stats for {year}...")
        
        # Get qualified pitcher stats
        pitcher_stats = pybaseball.pitching_stats(year, qual=1)
        
        # Save to cache if requested
        if cache:
            pitcher_stats.to_csv(cache_file, index=False)
            print(f"Saved pitcher stats to {cache_file}")
        
        return pitcher_stats
    
    except Exception as e:
        print(f"Error fetching pitcher stats: {e}")
        # Return empty DataFrame
        return pd.DataFrame()

def get_pitcher_weather_performance(merged_data):
    """
    Analyze how pitchers perform in different weather conditions.
    
    Args:
        merged_data (DataFrame): Combined game, weather, and pitcher data
        
    Returns:
        DataFrame: Pitcher performance by weather condition
    """
    if merged_data is None or 'starting_pitcher_home' not in merged_data.columns:
        print("No pitcher data available for analysis")
        return None
    
    # Create temperature ranges
    temp_bins = [0, 50, 60, 70, 80, 90, 100]
    temp_labels = ['Cold (<50°F)', 'Cool (50-60°F)', 'Mild (60-70°F)', 
                  'Warm (70-80°F)', 'Hot (80-90°F)', 'Very Hot (>90°F)']
    
    merged_data['temp_range'] = pd.cut(merged_data['temperature'], 
                                      bins=temp_bins, 
                                      labels=temp_labels, 
                                      right=False)
    
    # Create wind speed ranges
    wind_bins = [0, 5, 10, 15, 20, 30]
    wind_labels = ['Calm (0-5 mph)', 'Light (5-10 mph)', 'Moderate (10-15 mph)', 
                   'Strong (15-20 mph)', 'Very Strong (>20 mph)']
    
    merged_data['wind_range'] = pd.cut(merged_data['wind_speed'], 
                                      bins=wind_bins, 
                                      labels=wind_labels, 
                                      right=False)
    
    # Analyze home pitcher performance by temperature range
    home_temp_perf = merged_data.groupby(['starting_pitcher_home', 'temp_range']).agg({
        'away_score': ['mean', 'count'],
    }).reset_index()
    
    # Analyze away pitcher performance by temperature range
    away_temp_perf = merged_data.groupby(['starting_pitcher_away', 'temp_range']).agg({
        'home_score': ['mean', 'count'],
    }).reset_index()
    
    # Rename columns for clarity
    home_temp_perf.columns = ['pitcher', 'temp_range', 'runs_allowed_mean', 'games']
    away_temp_perf.columns = ['pitcher', 'temp_range', 'runs_allowed_mean', 'games']
    
    # Combine home and away performance
    temp_perf = pd.concat([home_temp_perf, away_temp_perf], ignore_index=True)
    
    # Group by pitcher and temperature range
    pitcher_temp_perf = temp_perf.groupby(['pitcher', 'temp_range']).agg({
        'runs_allowed_mean': 'mean',
        'games': 'sum'
    }).reset_index()
    
    # Calculate league average for each temperature range
    league_avg = pitcher_temp_perf.groupby('temp_range')['runs_allowed_mean'].mean().reset_index()
    league_avg.columns = ['temp_range', 'league_avg_runs']
    
    # Merge with league average
    pitcher_temp_perf = pd.merge(pitcher_temp_perf, league_avg, on='temp_range')
    
    # Calculate performance vs league average
    pitcher_temp_perf['performance_vs_league'] = (
        pitcher_temp_perf['league_avg_runs'] - pitcher_temp_perf['runs_allowed_mean']
    )
    
    # Filter to pitchers with at least 3 games in a temperature range
    pitcher_temp_perf = pitcher_temp_perf[pitcher_temp_perf['games'] >= 3]
    
    return pitcher_temp_perf

def add_team_stats_features(games_df, year=None):
    """
    Add team stats features to games DataFrame.
    
    Args:
        games_df (DataFrame): Games data
        year (int): Year to fetch stats for. If None, uses current year.
        
    Returns:
        DataFrame: Games data with team stats features added
    """
    # Get team stats
    batting_df, pitching_df = fetch_team_stats(year)
    
    if batting_df.empty or pitching_df.empty:
        print("No team stats available")
        return games_df
    
    # Create copy of games_df to avoid modifying original
    enhanced_df = games_df.copy()
    
    # Ensure batting_df index is set to Team
    if 'Team' in batting_df.columns:
        batting_df = batting_df.set_index('Team')
    
    # Ensure pitching_df index is set to Team
    if 'Team' in pitching_df.columns:
        pitching_df = pitching_df.set_index('Team')
    
    # Add batting stats for home team
    home_batting_cols = {
        'R': 'home_team_runs_per_game',
        'H': 'home_team_hits_per_game',
        'HR': 'home_team_hr_per_game',
        'BB': 'home_team_bb_per_game',
        'SO': 'home_team_so_per_game',
        'BA': 'home_team_batting_avg'
    }
    
    for stat, col_name in home_batting_cols.items():
        if stat in batting_df.columns:
            try:
                # Calculate per-game stats
                if stat in ['R', 'H', 'HR', 'BB', 'SO']:
                    batting_df[f'{stat}_per_game'] = batting_df[stat] / batting_df['G']
                    enhanced_df[col_name] = enhanced_df['home_team'].map(
                        batting_df[f'{stat}_per_game'].to_dict()
                    )
                else:
                    enhanced_df[col_name] = enhanced_df['home_team'].map(
                        batting_df[stat].to_dict()
                    )
            except Exception as e:
                print(f"Error adding {stat} for home team: {e}")
    
    # Add batting stats for away team
    away_batting_cols = {
        'R': 'away_team_runs_per_game',
        'H': 'away_team_hits_per_game',
        'HR': 'away_team_hr_per_game',
        'BB': 'away_team_bb_per_game',
        'SO': 'away_team_so_per_game',
        'BA': 'away_team_batting_avg'
    }
    
    for stat, col_name in away_batting_cols.items():
        if stat in batting_df.columns:
            try:
                # Calculate per-game stats
                if stat in ['R', 'H', 'HR', 'BB', 'SO']:
                    batting_df[f'{stat}_per_game'] = batting_df[stat] / batting_df['G']
                    enhanced_df[col_name] = enhanced_df['away_team'].map(
                        batting_df[f'{stat}_per_game'].to_dict()
                    )
                else:
                    enhanced_df[col_name] = enhanced_df['away_team'].map(
                        batting_df[stat].to_dict()
                    )
            except Exception as e:
                print(f"Error adding {stat} for away team: {e}")
    
    # Add pitching stats for home team
    home_pitching_cols = {
        'ERA': 'home_team_era',
        'WHIP': 'home_team_whip',
        'H': 'home_team_hits_allowed_per_game',
        'HR': 'home_team_hr_allowed_per_game',
        'BB': 'home_team_bb_allowed_per_game',
        'SO': 'home_team_so_pitcher_per_game'
    }
    
    for stat, col_name in home_pitching_cols.items():
        if stat in pitching_df.columns:
            try:
                # Calculate per-game stats
                if stat in ['H', 'HR', 'BB', 'SO']:
                    pitching_df[f'{stat}_per_game'] = pitching_df[stat] / pitching_df['G']
                    enhanced_df[col_name] = enhanced_df['home_team'].map(
                        pitching_df[f'{stat}_per_game'].to_dict()
                    )
                else:
                    enhanced_df[col_name] = enhanced_df['home_team'].map(
                        pitching_df[stat].to_dict()
                    )
            except Exception as e:
                print(f"Error adding {stat} for home team pitching: {e}")
    
    # Add pitching stats for away team
    away_pitching_cols = {
        'ERA': 'away_team_era',
        'WHIP': 'away_team_whip',
        'H': 'away_team_hits_allowed_per_game',
        'HR': 'away_team_hr_allowed_per_game',
        'BB': 'away_team_bb_allowed_per_game',
        'SO': 'away_team_so_pitcher_per_game'
    }
    
    for stat, col_name in away_pitching_cols.items():
        if stat in pitching_df.columns:
            try:
                # Calculate per-game stats
                if stat in ['H', 'HR', 'BB', 'SO']:
                    pitching_df[f'{stat}_per_game'] = pitching_df[stat] / pitching_df['G']
                    enhanced_df[col_name] = enhanced_df['away_team'].map(
                        pitching_df[f'{stat}_per_game'].to_dict()
                    )
                else:
                    enhanced_df[col_name] = enhanced_df['away_team'].map(
                        pitching_df[stat].to_dict()
                    )
            except Exception as e:
                print(f"Error adding {stat} for away team pitching: {e}")
    
    # Calculate combined features
    try:
        # Expected total runs (team averages + opponent pitching)
        enhanced_df['expected_runs_home'] = (
            enhanced_df['home_team_runs_per_game'] + 
            enhanced_df['away_team_era'] / 9
        ) / 2
        
        enhanced_df['expected_runs_away'] = (
            enhanced_df['away_team_runs_per_game'] + 
            enhanced_df['home_team_era'] / 9
        ) / 2
        
        enhanced_df['expected_total_runs'] = (
            enhanced_df['expected_runs_home'] + 
            enhanced_df['expected_runs_away']
        )
    except Exception as e:
        print(f"Error calculating expected runs: {e}")
    
    # Fill NaN values with reasonable defaults
    numeric_cols = [col for col in enhanced_df.columns if col.startswith(('home_team_', 'away_team_', 'expected_'))]
    for col in numeric_cols:
        if col in enhanced_df.columns:
            enhanced_df[col] = enhanced_df[col].fillna(enhanced_df[col].median())
    
    return enhanced_df

def add_starting_pitcher_features(games_df, force_refresh=False):
    """
    Add starting pitcher features to games DataFrame.
    
    Args:
        games_df (DataFrame): Games data
        force_refresh (bool): Whether to force refresh from API
        
    Returns:
        DataFrame: Games data with starting pitcher features added
    """
    # Create copy of games_df to avoid modifying original
    enhanced_df = games_df.copy()
    
    # Get date range for games
    if 'date' in enhanced_df.columns:
        start_date = enhanced_df['date'].min().strftime('%Y-%m-%d')
        end_date = enhanced_df['date'].max().strftime('%Y-%m-%d')
    else:
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Get starting pitchers
    starters_df = fetch_starting_pitchers(start_date, end_date, not force_refresh)
    
    if starters_df.empty:
        print("No starting pitcher data available")
        # Add placeholder columns
        enhanced_df['starting_pitcher_home'] = np.nan
        enhanced_df['starting_pitcher_away'] = np.nan
        return enhanced_df
    
    # Process starting pitcher data to match games_df format
    # This depends on specific format of starters_df from pybaseball
    # You may need to adjust this code based on actual format
    
    starters_df['game_date'] = pd.to_datetime(starters_df['game_date'])
    
    # For each game in enhanced_df, find matching game in starters_df
    pitcher_info = {}
    
    # Map team names to abbreviations for matching
    team_name_to_abbr = {v: k for k, v in TEAM_NAME_MAP.items()}
    
    for _, game in enhanced_df.iterrows():
        game_date = pd.to_datetime(game['date']).date()
        
        # Find games on this date
        day_games = starters_df[starters_df['game_date'].dt.date == game_date]
        
        if day_games.empty:
            continue
        
        # Try to find match based on team names
        home_team = game['home_team']
        away_team = game['away_team']
        
        # This matching logic depends on starters_df format
        # Example assuming format has home_team, away_team, home_pitcher, away_pitcher
        for _, starter_game in day_games.iterrows():
            if 'home_team' in starter_game and 'away_team' in starter_game:
                # Direct match on team abbreviations
                if starter_game['home_team'] == home_team and starter_game['away_team'] == away_team:
                    pitcher_info[(game_date, home_team, away_team)] = {
                        'home': starter_game.get('home_pitcher', np.nan),
                        'away': starter_game.get('away_pitcher', np.nan)
                    }
                    break
                
                # Try matching on team names
                home_team_name = [k for k, v in TEAM_NAME_MAP.items() if v == home_team]
                away_team_name = [k for k, v in TEAM_NAME_MAP.items() if v == away_team]
                
                if home_team_name and away_team_name:
                    if starter_game['home_team'] == home_team_name[0] and starter_game['away_team'] == away_team_name[0]:
                        pitcher_info[(game_date, home_team, away_team)] = {
                            'home': starter_game.get('home_pitcher', np.nan),
                            'away': starter_game.get('away_pitcher', np.nan)
                        }
                        break
    
    # Add pitcher info to enhanced_df
    enhanced_df['starting_pitcher_home'] = np.nan
    enhanced_df['starting_pitcher_away'] = np.nan
    
    for i, game in enhanced_df.iterrows():
        game_date = pd.to_datetime(game['date']).date()
        home_team = game['home_team']
        away_team = game['away_team']
        
        if (game_date, home_team, away_team) in pitcher_info:
            enhanced_df.loc[i, 'starting_pitcher_home'] = pitcher_info[(game_date, home_team, away_team)]['home']
            enhanced_df.loc[i, 'starting_pitcher_away'] = pitcher_info[(game_date, home_team, away_team)]['away']
    
    # Add pitcher stats
    year = pd.to_datetime(start_date).year
    pitcher_stats_df = fetch_pitcher_stats(year)
    
    if not pitcher_stats_df.empty:
        # Add pitcher ERA, WHIP, K/9, etc.
        pitcher_stats = {}
        
        for _, pitcher in pitcher_stats_df.iterrows():
            if 'Name' in pitcher and 'ERA' in pitcher:
                name = pitcher['Name']
                pitcher_stats[name] = {
                    'ERA': pitcher.get('ERA', np.nan),
                    'WHIP': pitcher.get('WHIP', np.nan),
                    'K/9': pitcher.get('K/9', np.nan),
                    'BB/9': pitcher.get('BB/9', np.nan),
                    'HR/9': pitcher.get('HR/9', np.nan),
                    'FIP': pitcher.get('FIP', np.nan)
                }
        
        # Add stats to enhanced_df
        for stat in ['ERA', 'WHIP', 'K/9', 'BB/9', 'HR/9', 'FIP']:
            enhanced_df[f'home_pitcher_{stat}'] = enhanced_df['starting_pitcher_home'].map(
                {name: stats[stat] for name, stats in pitcher_stats.items()}
            )
            
            enhanced_df[f'away_pitcher_{stat}'] = enhanced_df['starting_pitcher_away'].map(
                {name: stats[stat] for name, stats in pitcher_stats.items()}
            )
    
    # Fill NaN values with reasonable defaults
    pitcher_cols = [col for col in enhanced_df.columns if col.startswith(('home_pitcher_', 'away_pitcher_'))]
    for col in pitcher_cols:
        if col in enhanced_df.columns:
            enhanced_df[col] = enhanced_df[col].fillna(enhanced_df[col].median())
    
    return enhanced_df

def analyze_weather_impact_by_team(merged_data):
    """
    Analyze how different teams perform in various weather conditions.
    
    Args:
        merged_data (DataFrame): Combined game and weather data
        
    Returns:
        dict: Team performance by weather condition
    """
    if merged_data is None:
        return None
    
    results = {}
    
    # Analyze temperature impact
    temp_bins = [0, 50, 60, 70, 80, 90, 100]
    temp_labels = ['Cold (<50°F)', 'Cool (50-60°F)', 'Mild (60-70°F)', 
                  'Warm (70-80°F)', 'Hot (80-90°F)', 'Very Hot (>90°F)']
    
    merged_data['temp_range'] = pd.cut(merged_data['temperature'], 
                                      bins=temp_bins, 
                                      labels=temp_labels, 
                                      right=False)
    
    # Team runs in different temperature ranges (home)
    home_temp_runs = merged_data.groupby(['home_team', 'temp_range']).agg({
        'home_score': ['mean', 'count'],
        'away_score': ['mean'],
        'total_runs': ['mean']
    }).reset_index()
    
    # Team runs in different temperature ranges (away)
    away_temp_runs = merged_data.groupby(['away_team', 'temp_range']).agg({
        'away_score': ['mean', 'count'],
        'home_score': ['mean'],
        'total_runs': ['mean']
    }).reset_index()
    
    # Calculate total runs in each temperature range
    temp_runs = merged_data.groupby('temp_range').agg({
        'total_runs': ['mean', 'count']
    }).reset_index()
    
    # Analyze wind impact
    wind_bins = [0, 5, 10, 15, 20, 30]
    wind_labels = ['Calm (0-5 mph)', 'Light (5-10 mph)', 'Moderate (10-15 mph)', 
                   'Strong (15-20 mph)', 'Very Strong (>20 mph)']
    
    merged_data['wind_range'] = pd.cut(merged_data['wind_speed'], 
                                      bins=wind_bins, 
                                      labels=wind_labels, 
                                      right=False)
    
    # Team runs in different wind conditions
    home_wind_runs = merged_data.groupby(['home_team', 'wind_range']).agg({
        'home_score': ['mean', 'count'],
        'total_runs': ['mean']
    }).reset_index()
    
    away_wind_runs = merged_data.groupby(['away_team', 'wind_range']).agg({
        'away_score': ['mean', 'count'],
        'total_runs': ['mean']
    }).reset_index()
    
    # Weather condition impact
    condition_runs = merged_data.groupby(['weather_condition']).agg({
        'total_runs': ['mean', 'count']
    }).reset_index()
    
    results = {
        'temperature': {
            'home': home_temp_runs,
            'away': away_temp_runs,
            'overall': temp_runs
        },
        'wind': {
            'home': home_wind_runs,
            'away': away_wind_runs
        },
        'condition': condition_runs
    }
    
    return results
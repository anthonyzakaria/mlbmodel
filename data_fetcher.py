"""
Enhanced data fetching module for MLB Weather Model.
Handles historical data collection with robust error handling.
"""

import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
import re

from config import DATA_DIR, STADIUM_MAPPING, TEAM_NAME_MAP

def fetch_historical_games(start_year=2022, end_year=2024):
    """
    Fetch historical MLB game data with improved date parsing.
    
    Args:
        start_year (int): First season to collect
        end_year (int): Last season to collect
        
    Returns:
        DataFrame: Processed game data
    """
    print(f"Fetching historical MLB game data from {start_year} to {end_year}...")
    
    # Check for cached data first
    cache_file = f"{DATA_DIR}/mlb_games_{start_year}_{end_year}.csv"
    if os.path.exists(cache_file):
        print(f"Loading cached game data from {cache_file}")
        return pd.read_csv(cache_file, parse_dates=['date'])
    
    # Current year - we shouldn't try to fetch beyond this
    current_year = datetime.now().year
    
    # Validate year range
    if end_year > current_year:
        print(f"Warning: {end_year} is in the future. Limiting to {current_year}.")
        end_year = current_year
    
    # Import pybaseball with error handling
    try:
        import pybaseball
        # Enable caching to reduce API calls
        try:
            pybaseball.cache.enable()
        except:
            print("Warning: Could not enable pybaseball caching")
    except ImportError:
        print("Installing pybaseball...")
        import subprocess
        subprocess.check_call(["pip", "install", "pybaseball"])
        import pybaseball
    
    # MLB teams
    teams = ['ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 
             'COL', 'DET', 'HOU', 'KCR', 'LAA', 'LAD', 'MIA', 'MIL', 
             'MIN', 'NYM', 'NYY', 'OAK', 'PHI', 'PIT', 'SDP', 'SEA', 
             'SFG', 'STL', 'TBR', 'TEX', 'TOR', 'WSN']
    
    all_games = []
    
    for year in range(start_year, end_year + 1):
        print(f"Processing {year} season...")
        
        # Skip future years
        if year > current_year:
            print(f"Skipping future year {year}")
            continue
            
        # For current year, use different handling if it's early in the season
        if year == current_year:
            current_month = datetime.now().month
            if current_month < 4:  # Before April
                print(f"Current year {year} hasn't started regular season yet, using previous year stats")
                continue
        
        for team in teams:
            try:
                print(f"Fetching schedule for {team} {year}...")
                
                # Get team's schedule (removed parse_dates parameter)
                try:
                    team_schedule = pybaseball.schedule_and_record(year, team)
                except Exception as e:
                    print(f"Error fetching schedule: {e}")
                    continue
                
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
                completed_games = team_schedule[team_schedule['R'].notna() & team_schedule['RA'].notna()]
                
                # Process each game
                for _, game in completed_games.iterrows():
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
                                print(f"Invalid score for {team} vs {game['Opponent']} on {game_date}")
                                continue
                            
                            # Add to games list
                            all_games.append({
                                'date': game_date,
                                'home_team': team,
                                'away_team': game['Opponent'],
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
                time.sleep(1)
                
            except Exception as e:
                print(f"Error fetching data for {team} {year}: {e}")
    
    # Create DataFrame from all games
    if all_games:
        games_df = pd.DataFrame(all_games)
        
        # Ensure date is datetime
        games_df['date'] = pd.to_datetime(games_df['date'])
        
        # Sort by date
        games_df = games_df.sort_values('date')
        
        # Save to disk
        games_df.to_csv(cache_file, index=False)
        print(f"Saved {len(games_df)} games to CSV.")
        
        # Add historical odds
        try:
            from odds_scraper import OddsScraper
            scraper = OddsScraper()
            
            # Fetch odds for each season
            for year in range(start_year, end_year + 1):
                season_start = f"{year}-04-01"  # Approximate season start
                season_end = f"{year}-10-31"    # Approximate season end
                
                odds_df = scraper.fetch_historical_odds(season_start, season_end)
                
                if not odds_df.empty:
                    # Merge odds with games
                    games_df = pd.merge(
                        games_df,
                        odds_df,
                        on=['date', 'home_team', 'away_team'],
                        how='left'
                    )
        except Exception as e:
            print(f"Error fetching historical odds: {e}")
        
        return games_df
    else:
        print("No games data collected. Generating synthetic data...")
        return generate_synthetic_games(start_year, end_year)

def parse_game_date(date_str, year):
    """
    Parse MLB game date with robust error handling.
    
    Args:
        date_str (str): Date string from pybaseball
        year (int): Year of the game
        
    Returns:
        datetime or None: Parsed date or None if parsing fails
    """
    if pd.isna(date_str):
        return None
    
    # Try multiple parsing strategies
    try:
        # Try pandas default parsing
        return pd.to_datetime(date_str)
    except:
        # Try manual parsing for common formats
        try:
            # Format: "Thursday, Apr 7"
            match = re.search(r'(\w+), (\w+) (\d+)', date_str)
            if match:
                month_str, day_str = match.group(2), match.group(3)
                month_map = {
                    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
                    'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
                    'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                }
                month_num = month_map.get(month_str, 1)
                return datetime(year=year, month=month_num, day=int(day_str))
            
            # Format: "Thursday (4/7)"
            match = re.search(r'\((\d+)/(\d+)\)', date_str)
            if match:
                month_str, day_str = match.group(1), match.group(2)
                return datetime(year=year, month=int(month_str), day=int(day_str))
            
            # Format: "04/07/2022"
            match = re.search(r'(\d+)/(\d+)/(\d+)', date_str)
            if match:
                month_str, day_str, year_str = match.group(1), match.group(2), match.group(3)
                if len(year_str) == 2:
                    year_str = f"20{year_str}"
                return datetime(year=int(year_str), month=int(month_str), day=int(day_str))
        except:
            # Fall back to creating a timestamp for the year with default month/day
            # This is better than returning None as it allows the model to still use the data
            try:
                # Extract month and day from string if possible
                month_match = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', date_str)
                day_match = re.search(r'\b(\d{1,2})\b', date_str)
                
                month = 6  # Default to mid-year
                day = 15  # Default to mid-month
                
                if month_match:
                    month_str = month_match.group(1)
                    month_map = {
                        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
                        'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
                        'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                    }
                    month = month_map.get(month_str, month)
                
                if day_match:
                    day = int(day_match.group(1))
                    # Validate day is in range 1-31
                    day = max(1, min(day, 31))
                
                return datetime(year=year, month=month, day=day)
            except:
                return None
    
    return None

def generate_synthetic_games(start_year, end_year):
    """
    Generate synthetic MLB game data when real data can't be fetched.
    
    Args:
        start_year (int): First season to simulate
        end_year (int): Last season to simulate
        
    Returns:
        DataFrame: Synthetic game data
    """
    print("Generating synthetic MLB game data...")
    
    # MLB teams
    teams = list(STADIUM_MAPPING.keys())
    
    # List to store generated games
    synthetic_games = []
    
    # Average runs per MLB game (historical average)
    avg_runs_per_team = 4.5
    
    # Generate games for each year and team
    for year in range(start_year, end_year + 1):
        # Regular season games per team (81 home games per team)
        games_per_team = 81
        
        for home_team in teams:
            # Get stadium info
            stadium_info = STADIUM_MAPPING.get(home_team, 
                                           {'name': f"{home_team} Park", 
                                            'lat': 40.0, 'lon': -75.0})
            
            # Generate games against each opponent
            other_teams = [team for team in teams if team != home_team]
            games_per_opponent = max(1, games_per_team // len(other_teams))
            
            for away_team in other_teams:
                for game_num in range(games_per_opponent):
                    # Random date during baseball season
                    month = np.random.choice([4, 5, 6, 7, 8, 9])
                    day = np.random.randint(1, 29)  # Avoid invalid dates
                    game_date = datetime(year=year, month=month, day=day)
                    
                    # Random scores using Poisson distribution
                    home_advantage = 0.25  # Home teams score ~0.25 more runs on average
                    
                    home_score = max(0, int(np.random.poisson(avg_runs_per_team + home_advantage)))
                    away_score = max(0, int(np.random.poisson(avg_runs_per_team - home_advantage)))
                    total_runs = home_score + away_score
                    
                    # Add game to list
                    synthetic_games.append({
                        'date': game_date,
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_score': home_score,
                        'away_score': away_score,
                        'total_runs': total_runs,
                        'stadium': stadium_info['name'],
                        'stadium_lat': stadium_info['lat'],
                        'stadium_lon': stadium_info['lon'],
                        'is_synthetic': True
                    })
    
    # Create DataFrame
    games_df = pd.DataFrame(synthetic_games)
    
    # Sort by date
    games_df['date'] = pd.to_datetime(games_df['date'])
    games_df = games_df.sort_values('date')
    
    # Save to disk
    games_df.to_csv(f"{DATA_DIR}/synthetic_mlb_games_{start_year}_{end_year}.csv", index=False)
    print(f"Generated {len(games_df)} synthetic games.")
    
    return games_df

def fetch_current_schedule(days=7):
    """
    Fetch upcoming MLB schedule for the next few days.
    
    Args:
        days (int): Number of days to fetch
        
    Returns:
        DataFrame: Upcoming games
    """
    print(f"Fetching MLB schedule for the next {days} days...")
    
    # Current date
    today = datetime.now().date()
    
    # Date range
    start_date = today
    end_date = today + timedelta(days=days-1)
    
    # Format dates for API
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # Try to use MLB Stats API (unofficial)
    try:
        # URL for MLB schedule API
        url = f"https://statsapi.mlb.com/api/v1/schedule"
        params = {
            "startDate": start_str,
            "endDate": end_str,
            "sportId": 1,  # MLB
            "hydrate": "team,venue"
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            schedule_data = response.json()
            
            upcoming_games = []
            
            # Process schedule data
            for date in schedule_data.get('dates', []):
                game_date = date.get('date')
                
                for game in date.get('games', []):
                    try:
                        # Extract basic game info
                        game_id = game.get('gamePk')
                        status = game.get('status', {}).get('abstractGameState')
                        
                        # Only include scheduled games
                        if status != 'Final':
                            # Get teams
                            home_team = game.get('teams', {}).get('home', {}).get('team', {})
                            away_team = game.get('teams', {}).get('away', {}).get('team', {})
                            
                            # Get venue
                            venue = game.get('venue', {})
                            
                            # Game time
                            game_time = game.get('gameDate')
                            
                            # Map team names to abbreviations
                            home_name = home_team.get('name', '')
                            away_name = away_team.get('name', '')
                            
                            # Find team abbreviations
                            home_abbr = None
                            away_abbr = None
                            
                            for abbr, name in TEAM_NAME_MAP.items():
                                if name == home_name:
                                    home_abbr = abbr
                                if name == away_name:
                                    away_abbr = abbr
                            
                            # Use teamCode if available
                            if not home_abbr and 'teamCode' in home_team:
                                home_abbr = home_team['teamCode'].upper()
                            if not away_abbr and 'teamCode' in away_team:
                                away_abbr = away_team['teamCode'].upper()
                            
                            # Get stadium info based on home team
                            stadium_info = None
                            if home_abbr in STADIUM_MAPPING:
                                stadium_info = STADIUM_MAPPING[home_abbr]
                            
                            # Fallback to venue info from API
                            if not stadium_info:
                                stadium_name = venue.get('name', f"{home_name} Stadium")
                                # Try to get coordinates if available
                                stadium_lat = venue.get('location', {}).get('latitude', 0)
                                stadium_lon = venue.get('location', {}).get('longitude', 0)
                                
                                stadium_info = {
                                    'name': stadium_name,
                                    'lat': float(stadium_lat) if stadium_lat else 0,
                                    'lon': float(stadium_lon) if stadium_lon else 0
                                }
                            
                            # Add to upcoming games
                            upcoming_games.append({
                                'date': pd.to_datetime(game_date),
                                'datetime': pd.to_datetime(game_time) if game_time else pd.to_datetime(game_date),
                                'home_team': home_abbr,
                                'away_team': away_abbr,
                                'stadium': stadium_info.get('name'),
                                'stadium_lat': stadium_info.get('lat'),
                                'stadium_lon': stadium_info.get('lon'),
                                'game_id': game_id,
                                'status': status
                            })
                    except Exception as e:
                        print(f"Error processing game: {e}")
            
            # Create DataFrame
            if upcoming_games:
                schedule_df = pd.DataFrame(upcoming_games)
                return schedule_df
            else:
                print("No upcoming games found in schedule.")
                return generate_synthetic_schedule(days)
        else:
            print(f"Error fetching schedule: {response.status_code}")
            return generate_synthetic_schedule(days)
    
    except Exception as e:
        print(f"Error accessing MLB API: {e}")
        # Fall back to synthetic schedule
        return generate_synthetic_schedule(days)

def generate_synthetic_schedule(days=7):
    """
    Generate synthetic MLB schedule when API access fails.
    
    Args:
        days (int): Number of days to generate
        
    Returns:
        DataFrame: Synthetic schedule
    """
    print("Generating synthetic MLB schedule...")
    
    # Current date
    today = datetime.now().date()
    
    # MLB teams
    teams = list(STADIUM_MAPPING.keys())
    
    # List to store generated games
    upcoming_games = []
    
    # Generate 15 games per day (all 30 teams playing)
    for day in range(days):
        game_date = today + timedelta(days=day)
        
        # Randomly pair teams
        np.random.shuffle(teams)
        
        for i in range(0, len(teams), 2):
            if i + 1 < len(teams):
                home_team = teams[i]
                away_team = teams[i+1]
                
                # Get stadium info
                stadium_info = STADIUM_MAPPING.get(home_team, 
                                               {'name': f"{home_team} Park", 
                                                'lat': 40.0, 'lon': -75.0})
                
                # Random game time (afternoon or evening)
                hour = np.random.choice([13, 16, 19])  # 1 PM, 4 PM, or 7 PM
                game_datetime = datetime.combine(game_date, datetime.min.time().replace(hour=hour))
                
                # Add game to list
                upcoming_games.append({
                    'date': game_date,
                    'datetime': game_datetime,
                    'home_team': home_team,
                    'away_team': away_team,
                    'stadium': stadium_info['name'],
                    'stadium_lat': stadium_info['lat'],
                    'stadium_lon': stadium_info['lon'],
                    'status': 'Scheduled',
                    'is_synthetic': True
                })
    
    # Create DataFrame
    schedule_df = pd.DataFrame(upcoming_games)
    
    # Convert date columns to datetime
    schedule_df['date'] = pd.to_datetime(schedule_df['date'])
    schedule_df['datetime'] = pd.to_datetime(schedule_df['datetime'])
    
    # Sort by date and time
    schedule_df = schedule_df.sort_values(['date', 'datetime'])
    
    print(f"Generated {len(schedule_df)} synthetic upcoming games.")
    
    return schedule_df

def fetch_pitcher_stats():
    """
    Fetch pitcher statistics from an external source (e.g., API or CSV).
    
    Returns:
        DataFrame: Pitcher stats
    """
    print("Fetching pitcher stats...")
    # Example: Fetch from an API or scrape a website
    # Save to a CSV for future use
    pitcher_stats = pd.DataFrame({
        'pitcher_name': ['Pitcher A', 'Pitcher B'],
        'ERA': [3.50, 4.20],
        'WHIP': [1.20, 1.35],
        'strikeouts_per_9': [9.5, 8.0]
    })
    pitcher_stats.to_csv(f"{DATA_DIR}/pitcher_stats.csv", index=False)
    return pitcher_stats

def fetch_team_stats():
    """
    Fetch team batting statistics from an external source (e.g., API or CSV).
    
    Returns:
        DataFrame: Team stats
    """
    print("Fetching team stats...")
    # Example: Fetch from an API or scrape a website
    # Save to a CSV for future use
    team_stats = pd.DataFrame({
        'team_abbr': ['NYY', 'BOS'],
        'OPS': [0.800, 0.750],
        'runs_per_game': [5.2, 4.8]
    })
    team_stats.to_csv(f"{DATA_DIR}/team_stats.csv", index=False)
    return team_stats

def fetch_bullpen_stats():
    """
    Fetch bullpen performance statistics from an external source (e.g., API or CSV).
    
    Returns:
        DataFrame: Bullpen stats
    """
    print("Fetching bullpen stats...")
    # Example: Fetch from an API or scrape a website
    # Save to a CSV for future use
    bullpen_stats = pd.DataFrame({
        'team_abbr': ['NYY', 'BOS'],
        'bullpen_ERA': [3.80, 4.10],
        'bullpen_WHIP': [1.25, 1.30]
    })
    bullpen_stats.to_csv(f"{DATA_DIR}/bullpen_stats.csv", index=False)
    return bullpen_stats

def fetch_recent_performance():
    """
    Fetch recent game performance for teams from an external source (e.g., API or CSV).
    
    Returns:
        DataFrame: Recent performance stats
    """
    print("Fetching recent performance stats...")
    # Example: Fetch from an API or scrape a website
    # Save to a CSV for future use
    recent_performance = pd.DataFrame({
        'team_abbr': ['NYY', 'BOS'],
        'last_5_games_runs': [25, 20],
        'last_5_games_allowed': [18, 22]
    })
    recent_performance.to_csv(f"{DATA_DIR}/recent_performance.csv", index=False)
    return recent_performance
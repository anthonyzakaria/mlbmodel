"""
NRFI (No Run First Inning) specialized prediction model for MLB.
This module provides first inning scoring prediction capabilities with a focus on
pitcher first inning stats, team first inning tendencies, and weather impact.
"""

import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance

# Try to import various modeling libraries with fallbacks
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
    print("Using LightGBM for NRFI modeling")
except ImportError:
    HAS_LIGHTGBM = False
    
try:
    import xgboost as xgb
    HAS_XGBOOST = True
    if not HAS_LIGHTGBM:
        print("Using XGBoost for NRFI modeling")
except ImportError:
    HAS_XGBOOST = False

if not (HAS_LIGHTGBM or HAS_XGBOOST):
    # Fall back to sklearn models
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    print("Using scikit-learn for NRFI modeling")

# Import configuration
try:
    from config import DATA_DIR, ODDS_API_KEY, STADIUM_MAPPING
except ImportError:
    # Fallbacks if config is not available
    DATA_DIR = "data"
    ODDS_API_KEY = None
    STADIUM_MAPPING = {}

class NRFIModel:
    """Specialized model for predicting No Run First Inning (NRFI) outcomes."""
    
    def __init__(self):
        """Initialize the NRFI prediction model."""
        # Model data
        self.merged_data = None
        self.model = None
        self.scaler = None
        self.features = None
        self.model_type = None
        self.timestamp = None
        
        # Pitcher first inning stats cache
        self.pitcher_first_inning_stats = {}
        
        # Team first inning tendencies cache
        self.team_first_inning_stats = {}
        
        # Set model type based on available libraries
        if HAS_LIGHTGBM:
            self.model_type = "lightgbm"
        elif HAS_XGBOOST:
            self.model_type = "xgboost"
        else:
            self.model_type = "sklearn"
    
    def fetch_first_inning_data(self, games_df, use_existing=True):
        """
        Fetch first inning scoring data for the games in games_df.
        
        Args:
            games_df (DataFrame): Base game data with dates and teams
            use_existing (bool): Whether to use existing cached data
            
        Returns:
            DataFrame: Games data with first inning runs added
        """
        print("Fetching first inning data...")
        
        # Create directory for innings data if it doesn't exist
        innings_dir = f"{DATA_DIR}/innings_data"
        if not os.path.exists(innings_dir):
            os.makedirs(innings_dir)
        
        # Copy games_df to avoid modifying original
        enhanced_df = games_df.copy()
        
        # Add inning run columns
        enhanced_df['first_inning_home_runs'] = 0
        enhanced_df['first_inning_away_runs'] = 0
        enhanced_df['first_inning_total_runs'] = 0
        enhanced_df['nrfi'] = 1  # Default to NRFI (0 runs in first inning)
        
        # Check for existing cache file by date range
        start_date = pd.to_datetime(games_df['date']).min()
        end_date = pd.to_datetime(games_df['date']).max()
        cache_file = f"{innings_dir}/first_innings_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        
        if use_existing and os.path.exists(cache_file):
            print(f"Loading cached first inning data from {cache_file}")
            first_innings_df = pd.read_csv(cache_file, parse_dates=['date'])
            
            # Merge with enhanced_df
            enhanced_df = pd.merge(
                enhanced_df,
                first_innings_df,
                on=['date', 'home_team', 'away_team'],
                how='left',
                suffixes=('', '_innings')
            )
            
            # Ensure we have NRFI column
            if 'nrfi' not in enhanced_df.columns and 'first_inning_total_runs' in enhanced_df.columns:
                enhanced_df['nrfi'] = (enhanced_df['first_inning_total_runs'] == 0).astype(int)
            
            return enhanced_df
        
        try:
            import pybaseball
            # Enable caching to reduce API calls
            try:
                pybaseball.cache.enable()
            except:
                print("Warning: Could not enable pybaseball caching")
                
            # Process each game
            for idx, game in enhanced_df.iterrows():
                try:
                    game_date = pd.to_datetime(game['date']).strftime('%Y-%m-%d')
                    home_team = game['home_team']
                    away_team = game['away_team']
                    
                    # Use date-based caching instead of per-game caching
                    # Convert game date to date-only string
                    date_str = pd.to_datetime(game_date).strftime('%Y-%m-%d')
                    date_cache_file = f"{innings_dir}/{date_str}_innings.csv"
                    
                    if os.path.exists(date_cache_file) and use_existing:
                        # Load all games for this date from the single file
                        date_innings_data = pd.read_csv(date_cache_file)
                        
                        # Filter innings data for this specific game
                        inning_data = date_innings_data[
                            (date_innings_data['home_team'] == home_team) & 
                            (date_innings_data['away_team'] == away_team)
                        ]
                        
                        if inning_data.empty:
                            # Game not found in cached data, generate synthetic data
                            inning_data = self._generate_synthetic_inning_data(game)
                            
                            # Append to the date cache file
                            if not date_innings_data.empty:
                                date_innings_data = pd.concat([date_innings_data, inning_data])
                                date_innings_data.to_csv(date_cache_file, index=False)
                    else:
                        # Generate synthetic data for this game
                        inning_data = self._generate_synthetic_inning_data(game)
                        
                        # Save to date cache file
                        if os.path.exists(date_cache_file):
                            # Append to existing file
                            try:
                                date_innings_data = pd.read_csv(date_cache_file)
                                date_innings_data = pd.concat([date_innings_data, inning_data])
                            except:
                                date_innings_data = inning_data
                        else:
                            date_innings_data = inning_data
                            
                        date_innings_data.to_csv(date_cache_file, index=False)
                    
                    # Extract first inning data
                    if not inning_data.empty and 'inning' in inning_data.columns:
                        first_inning = inning_data[inning_data['inning'] == 1]
                        
                        # Get runs scored in first inning
                        if not first_inning.empty:
                            home_runs = first_inning['home_runs'].sum() if 'home_runs' in first_inning.columns else 0
                            away_runs = first_inning['away_runs'].sum() if 'away_runs' in first_inning.columns else 0
                            
                            enhanced_df.at[idx, 'first_inning_home_runs'] = home_runs
                            enhanced_df.at[idx, 'first_inning_away_runs'] = away_runs
                            enhanced_df.at[idx, 'first_inning_total_runs'] = home_runs + away_runs
                            enhanced_df.at[idx, 'nrfi'] = 1 if (home_runs + away_runs) == 0 else 0
                            
                    # Add a small delay to avoid rate limiting
                    time.sleep(0.1)
                        
                except Exception as e:
                    print(f"Error processing inning data for {game_date} {away_team} @ {home_team}: {e}")
                    
        except ImportError:
            print("pybaseball not installed. Using synthetic inning data.")
            # Create synthetic first inning data for all games
            enhanced_df = self._add_synthetic_first_inning_data(enhanced_df)
        
        # Save compiled first inning data to main cache
        first_inning_cols = ['date', 'home_team', 'away_team', 
                           'first_inning_home_runs', 'first_inning_away_runs', 
                           'first_inning_total_runs', 'nrfi']
        
        first_innings_df = enhanced_df[first_inning_cols].copy()
        first_innings_df.to_csv(cache_file, index=False)
        print(f"Saved {len(first_innings_df)} first inning records to {cache_file}")
        
        return enhanced_df
    
    def _generate_synthetic_inning_data(self, game):
        """
        Generate synthetic inning-by-inning data for a game.
        
        Args:
            game (Series): Game data with scores
            
        Returns:
            DataFrame: Synthetic inning-by-inning data
        """
        # Extract total runs from game
        home_score = game.get('home_score', 0)
        away_score = game.get('away_score', 0)
        
        # Create a distribution of runs across 9 innings
        # More weight to early innings (1-2) and late innings (8-9)
        innings = []
        
        # Distribution weights for scoring by inning
        home_weights = [0.15, 0.12, 0.10, 0.10, 0.10, 0.10, 0.10, 0.11, 0.12]
        away_weights = [0.13, 0.11, 0.11, 0.10, 0.10, 0.10, 0.11, 0.12, 0.12]
        
        # Normalize weights in case they don't sum to 1
        home_weights = [w/sum(home_weights) for w in home_weights]
        away_weights = [w/sum(away_weights) for w in away_weights]
        
        # Check if this game actually had extra innings
        if home_score + away_score > 0:
            # Distribute runs according to weights, with randomization
            home_runs_by_inning = np.random.multinomial(home_score, home_weights)
            away_runs_by_inning = np.random.multinomial(away_score, away_weights)
        else:
            # No runs scored in the game
            home_runs_by_inning = np.zeros(9, dtype=int)
            away_runs_by_inning = np.zeros(9, dtype=int)
        
        # Create inning-by-inning data
        for inning in range(1, 10):
            idx = inning - 1  # Convert to 0-based index
            innings.append({
                'inning': inning,
                'home_runs': home_runs_by_inning[idx],
                'away_runs': away_runs_by_inning[idx],
                'total_runs': home_runs_by_inning[idx] + away_runs_by_inning[idx]
            })
        
        return pd.DataFrame(innings)
    
    def _add_synthetic_first_inning_data(self, games_df):
        """
        Add synthetic first inning data when real data can't be fetched.
        
        Args:
            games_df (DataFrame): Game data
            
        Returns:
            DataFrame: Game data with synthetic first inning metrics added
        """
        df = games_df.copy()
        
        # Create realistic first inning runs
        # MLB first innings have runs in about 52% of games (NRFI rate around 48%)
        nrfi_probability = 0.48
        
        # Generate NRFI outcomes (1=no runs, 0=runs)
        df['nrfi'] = np.random.binomial(1, nrfi_probability, size=len(df))
        
        # For games with runs in first inning, distribute runs
        # Single run in first inning is most common
        runs_distribution = [0, 1, 2, 3, 4, 5]
        runs_weights = [0, 0.5, 0.25, 0.15, 0.07, 0.03]  # Favor lower run totals
        
        # Generate first inning runs
        df['first_inning_total_runs'] = 0
        
        # For non-NRFI games, assign run totals
        non_nrfi_indices = df[df['nrfi'] == 0].index
        df.loc[non_nrfi_indices, 'first_inning_total_runs'] = np.random.choice(
            runs_distribution[1:],  # Skip 0 runs
            size=len(non_nrfi_indices),
            p=[w/sum(runs_weights[1:]) for w in runs_weights[1:]]  # Normalize p without 0
        )
        
        # Split runs between home and away teams
        df['first_inning_home_runs'] = 0
        df['first_inning_away_runs'] = 0
        
        for idx in non_nrfi_indices:
            total_runs = df.loc[idx, 'first_inning_total_runs']
            
            # Decide if home team scores (60% chance for home team to score)
            home_scores = np.random.binomial(1, 0.6)
            
            if home_scores and total_runs > 0:
                # Home team scores between 1 and total runs
                home_runs = np.random.randint(1, total_runs + 1)
                away_runs = total_runs - home_runs
            else:
                # Away team scores all runs
                home_runs = 0
                away_runs = total_runs
            
            df.loc[idx, 'first_inning_home_runs'] = home_runs
            df.loc[idx, 'first_inning_away_runs'] = away_runs
        
        return df
    
    def fetch_pitcher_first_inning_stats(self, year=None, use_existing=True):
        """
        Fetch MLB pitcher first inning statistics.
        
        Args:
            year (int): Year to fetch stats for. If None, uses current year.
            use_existing (bool): Whether to use cached data if available
            
        Returns:
            DataFrame: Pitcher first inning stats
        """
        # If no year specified, use current year
        if year is None:
            year = datetime.now().year
            
        # Check if we already have this year's data in memory
        if year in self.pitcher_first_inning_stats:
            return self.pitcher_first_inning_stats[year]
        
        cache_file = f"{DATA_DIR}/pitcher_first_inning_{year}.csv"
        
        # Check if cache exists and should be used
        if use_existing and os.path.exists(cache_file):
            print(f"Loading cached pitcher first inning stats from {year}")
            first_inning_stats = pd.read_csv(cache_file)
            self.pitcher_first_inning_stats[year] = first_inning_stats
            return first_inning_stats
        
        try:
            # Try to fetch first inning stats from specialized sources
            # This would require MLB API access or scraping specialized stats sites
            
            # If we had actual data access, we would:
            # 1. Get all pitchers who started games in the specified year
            # 2. For each one, calculate their first inning metrics:
            #    - First inning ERA: (First inning ER / First inning IP) * 9
            #    - First inning WHIP: (First inning H + First inning BB) / First inning IP
            #    - NRFI rate: % of first innings with 0 runs allowed
            
            # Since we don't have direct access, we'll generate synthetic data
            # This should be replaced with real data collection in production
            first_inning_stats = self._generate_synthetic_pitcher_stats(year)
            
            # Save to disk
            first_inning_stats.to_csv(cache_file, index=False)
            print(f"Saved {len(first_inning_stats)} pitcher first inning records")
            
            # Cache in memory
            self.pitcher_first_inning_stats[year] = first_inning_stats
            
            return first_inning_stats
            
        except Exception as e:
            print(f"Error fetching pitcher first inning stats: {e}")
            # Create synthetic data
            first_inning_stats = self._generate_synthetic_pitcher_stats(year)
            
            # Save to disk
            first_inning_stats.to_csv(cache_file, index=False)
            
            # Cache in memory
            self.pitcher_first_inning_stats[year] = first_inning_stats
            
            return first_inning_stats
    
    def _generate_synthetic_pitcher_stats(self, year):
        """
        Generate synthetic pitcher first inning stats when real data can't be fetched.
        
        Args:
            year (int): Year for synthetic data
            
        Returns:
            DataFrame: Synthetic pitcher stats
        """
        # Try to load regular pitcher stats first to get a realistic list of pitchers
        pitcher_file = f"{DATA_DIR}/pitcher_stats_{year}.csv"
        
        if os.path.exists(pitcher_file):
            try:
                pitchers_df = pd.read_csv(pitcher_file)
                pitcher_names = pitchers_df['Name'].tolist() if 'Name' in pitchers_df.columns else []
            except:
                # Create some placeholder pitcher names
                pitcher_names = [f"Pitcher {i}" for i in range(1, 101)]
        else:
            # Create some placeholder pitcher names
            pitcher_names = [f"Pitcher {i}" for i in range(1, 101)]
        
        # Create stats for each pitcher
        first_inning_stats = []
        
        for name in pitcher_names:
            # Regular stats - baseline
            regular_era = np.random.normal(4.00, 1.00)  # Mean MLB ERA ~4.00
            regular_whip = np.random.normal(1.30, 0.20)  # Mean MLB WHIP ~1.30
            
            # First inning stats - slightly higher on average
            # Many pitchers struggle more in the first inning as they "settle in"
            first_inning_era_ratio = np.random.normal(1.10, 0.25)  # 10% higher on average
            first_inning_whip_ratio = np.random.normal(1.08, 0.20)  # 8% higher on average
            
            first_inning_era = max(0, regular_era * first_inning_era_ratio)
            first_inning_whip = max(0.5, regular_whip * first_inning_whip_ratio)
            
            # Calculate NRFI rate - probability of scoreless first inning
            # Based on ERA: higher ERA → lower NRFI rate
            # First inning ERA of 0.00 → 100% NRFI rate
            # First inning ERA of 9.00 → ~35% NRFI rate
            # Using logistic function to map ERA to NRFI rate
            nrfi_rate = 1 / (1 + np.exp(0.4 * (first_inning_era - 4.5)))
            
            # Add variation around this calculated rate
            nrfi_rate = min(1.0, max(0.1, nrfi_rate + np.random.normal(0, 0.05)))
            
            # Games started - more starts = more reliable stats
            games_started = np.random.randint(5, 35)
            
            first_inning_stats.append({
                'pitcher_name': name,
                'year': year,
                'games_started': games_started,
                'ERA': regular_era,
                'WHIP': regular_whip,
                'ERA_1st': first_inning_era,
                'WHIP_1st': first_inning_whip,
                'nrfi_rate': nrfi_rate,
                'first_inning_runs_allowed': int(games_started * (1 - nrfi_rate) * 1.4)  # ~1.4 runs when they score
            })
        
        return pd.DataFrame(first_inning_stats)
    
    def fetch_team_first_inning_stats(self, year=None, use_existing=True):
        """
        Fetch MLB team first inning scoring tendencies.
        
        Args:
            year (int): Year to fetch stats for. If None, uses current year.
            use_existing (bool): Whether to use cached data if available
            
        Returns:
            DataFrame: Team first inning stats
        """
        # If no year specified, use current year
        if year is None:
            year = datetime.now().year
            
        # Check if we already have this year's data in memory
        if year in self.team_first_inning_stats:
            return self.team_first_inning_stats[year]
        
        cache_file = f"{DATA_DIR}/team_first_inning_{year}.csv"
        
        # Check if cache exists and should be used
        if use_existing and os.path.exists(cache_file):
            print(f"Loading cached team first inning stats from {year}")
            first_inning_stats = pd.read_csv(cache_file)
            self.team_first_inning_stats[year] = first_inning_stats
            return first_inning_stats
        
        try:
            # Try to fetch first inning stats from specialized sources
            # This would require MLB API access or scraping specialized stats sites
            
            # If we had actual data access, we would:
            # 1. Get all teams' first inning performances
            # 2. Calculate metrics like:
            #    - First inning runs per game
            #    - % of games with runs in first inning
            #    - First inning OPS
            
            # Since we don't have direct access, we'll generate synthetic data
            # This should be replaced with real data collection in production
            first_inning_stats = self._generate_synthetic_team_stats(year)
            
            # Save to disk
            first_inning_stats.to_csv(cache_file, index=False)
            print(f"Saved {len(first_inning_stats)} team first inning records")
            
            # Cache in memory
            self.team_first_inning_stats[year] = first_inning_stats
            
            return first_inning_stats
            
        except Exception as e:
            print(f"Error fetching team first inning stats: {e}")
            # Create synthetic data
            first_inning_stats = self._generate_synthetic_team_stats(year)
            
            # Save to disk
            first_inning_stats.to_csv(cache_file, index=False)
            
            # Cache in memory
            self.team_first_inning_stats[year] = first_inning_stats
            
            return first_inning_stats
    
    def _generate_synthetic_team_stats(self, year):
        """
        Generate synthetic team first inning stats when real data can't be fetched.
        
        Args:
            year (int): Year for synthetic data
            
        Returns:
            DataFrame: Synthetic team stats
        """
        # Get a list of MLB teams (either from STADIUM_MAPPING or hardcoded)
        if STADIUM_MAPPING:
            teams = list(STADIUM_MAPPING.keys())
        else:
            # MLB team abbreviations
            teams = ['ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 
                   'COL', 'DET', 'HOU', 'KC', 'LAA', 'LAD', 'MIA', 'MIL', 
                   'MIN', 'NYM', 'NYY', 'OAK', 'PHI', 'PIT', 'SD', 'SEA', 
                   'SF', 'STL', 'TB', 'TEX', 'TOR', 'WSH']
        
        # Create stats for each team
        first_inning_stats = []
        
        for team in teams:
            # Regular offensive stats - baseline
            runs_per_game = np.random.normal(4.5, 0.6)  # Mean MLB runs ~4.5 per game
            
            # First inning stats
            # Teams score in about 52% of first innings, averaging about 0.52-0.55 runs
            first_inning_scoring_ratio = np.random.normal(0.12, 0.02)  # ~12% of runs in first inning
            first_inning_runs = runs_per_game * first_inning_scoring_ratio
            
            # Percentage of games with runs in first inning
            # Correlates with first_inning_runs but with variation
            yrfi_rate = min(0.85, max(0.30, 0.35 + first_inning_runs * 0.30 + np.random.normal(0, 0.05)))
            
            # Home vs away splits
            home_first_inning_runs = first_inning_runs * np.random.normal(1.05, 0.10)  # Slight home advantage
            away_first_inning_runs = first_inning_runs * np.random.normal(0.95, 0.10)
            
            first_inning_stats.append({
                'team': team,
                'year': year,
                'runs_per_game': runs_per_game,
                'first_inning_runs_per_game': first_inning_runs,
                'home_first_inning_runs': home_first_inning_runs,
                'away_first_inning_runs': away_first_inning_runs,
                'yrfi_rate': yrfi_rate,
                'nrfi_rate': 1 - yrfi_rate,
                'first_inning_scoring_rank': 0  # Will be calculated after all teams processed
            })
        
        # Create DataFrame and calculate rankings
        df = pd.DataFrame(first_inning_stats)
        
        # Rank teams by first inning scoring
        df['first_inning_scoring_rank'] = df['first_inning_runs_per_game'].rank(ascending=False)
        
        return df
    
    def analyze_weather_impact_on_first_inning(self, games_with_weather):
        """
        Analyze how weather impacts first inning scoring specifically.
        
        Args:
            games_with_weather (DataFrame): Games with weather and first inning data
            
        Returns:
            dict: Analysis of weather impact on first inning scoring
        """
        if games_with_weather is None or len(games_with_weather) == 0:
            print("No game data available for weather impact analysis")
            return None
            
        if 'first_inning_total_runs' not in games_with_weather.columns:
            print("No first inning data available for weather impact analysis")
            return None
            
        if 'temperature' not in games_with_weather.columns:
            print("No weather data available for weather impact analysis")
            return None
        
        # Create temperature ranges
        temp_bins = [0, 50, 60, 70, 80, 90, 100]
        temp_labels = ['Cold (<50°F)', 'Cool (50-60°F)', 'Mild (60-70°F)', 
                      'Warm (70-80°F)', 'Hot (80-90°F)', 'Very Hot (>90°F)']
        
        games_with_weather['temp_range'] = pd.cut(games_with_weather['temperature'], 
                                                bins=temp_bins, 
                                                labels=temp_labels, 
                                                right=False)
        
        # Create wind ranges
        wind_bins = [0, 5, 10, 15, 20, 30]
        wind_labels = ['Calm (0-5 mph)', 'Light (5-10 mph)', 'Moderate (10-15 mph)', 
                      'Strong (15-20 mph)', 'Very Strong (>20 mph)']
        
        games_with_weather['wind_range'] = pd.cut(games_with_weather['wind_speed'], 
                                               bins=wind_bins, 
                                               labels=wind_labels, 
                                               right=False)
        
        # Results dictionary
        results = {}
        
        # Analyze temperature impact on first inning scoring
        temp_impact = games_with_weather.groupby('temp_range').agg({
            'first_inning_total_runs': ['mean', 'count'],
            'nrfi': ['mean', 'sum']
        })
        
        # Calculate NRFI % for each temperature range
        temp_impact['nrfi', '%'] = temp_impact['nrfi', 'mean'] * 100
        
        # Analyze wind impact on first inning scoring
        wind_impact = games_with_weather.groupby('wind_range').agg({
            'first_inning_total_runs': ['mean', 'count'],
            'nrfi': ['mean', 'sum']
        })
        
        # Calculate NRFI % for each wind range
        wind_impact['nrfi', '%'] = wind_impact['nrfi', 'mean'] * 100
        
        # Analyze wind direction impact (if available)
        if 'wind_blowing_out' in games_with_weather.columns:
            wind_direction = games_with_weather.groupby('wind_blowing_out').agg({
                'first_inning_total_runs': ['mean', 'count'],
                'nrfi': ['mean', 'sum']
            })
            
            # Map 0/1 to readable labels
            wind_direction = wind_direction.reset_index()
            wind_direction['wind_blowing_out'] = wind_direction['wind_blowing_out'].map({
                0: 'Not Blowing Out',
                1: 'Blowing Out'
            })
            wind_direction = wind_direction.set_index('wind_blowing_out')
            
            # Calculate NRFI % for each wind direction
            wind_direction['nrfi', '%'] = wind_direction['nrfi', 'mean'] * 100
            
            results['wind_direction'] = wind_direction
        
        # Analyze weather condition impact
        if 'weather_condition' in games_with_weather.columns:
            weather_impact = games_with_weather.groupby('weather_condition').agg({
                'first_inning_total_runs': ['mean', 'count'],
                'nrfi': ['mean', 'sum']
            })
            
            # Calculate NRFI % for each weather condition
            weather_impact['nrfi', '%'] = weather_impact['nrfi', 'mean'] * 100
            
            # Filter to conditions with at least 10 games
            weather_impact = weather_impact[weather_impact[('first_inning_total_runs', 'count')] >= 10]
            
            results['weather_condition'] = weather_impact
        
        # Dome vs outdoor
        if 'is_dome' in games_with_weather.columns:
            dome_impact = games_with_weather.groupby('is_dome').agg({
                'first_inning_total_runs': ['mean', 'count'],
                'nrfi': ['mean', 'sum']
            })
            
            # Map 0/1 to readable labels
            dome_impact = dome_impact.reset_index()
            dome_impact['is_dome'] = dome_impact['is_dome'].map({
                0: 'Outdoor',
                1: 'Dome'
            })
            dome_impact = dome_impact.set_index('is_dome')
            
            # Calculate NRFI % for dome vs outdoor
            dome_impact['nrfi', '%'] = dome_impact['nrfi', 'mean'] * 100
            
            results['dome'] = dome_impact
        
        # Add temperature and wind impact to results
        results['temperature'] = temp_impact
        results['wind'] = wind_impact
        
        # Calculate weighted impact coefficients for model feature engineering
        # These coefficients can be used to adjust baseline NRFI probabilities
        coefficients = {}
        
        # Temperature coefficient (relative to mild weather)
        temp_baseline = temp_impact.loc['Mild (60-70°F)', ('nrfi', 'mean')] if 'Mild (60-70°F)' in temp_impact.index else 0.5
        if temp_baseline > 0:
            coefficients['temperature'] = {}
            for temp_range in temp_impact.index:
                coefficients['temperature'][temp_range] = temp_impact.loc[temp_range, ('nrfi', 'mean')] / temp_baseline
        
        # Wind coefficient (relative to light wind)
        wind_baseline = wind_impact.loc['Light (5-10 mph)', ('nrfi', 'mean')] if 'Light (5-10 mph)' in wind_impact.index else 0.5
        if wind_baseline > 0:
            coefficients['wind'] = {}
            for wind_range in wind_impact.index:
                coefficients['wind'][wind_range] = wind_impact.loc[wind_range, ('nrfi', 'mean')] / wind_baseline
        
        # Add coefficients to results
        results['coefficients'] = coefficients
        
        return results
    
    def prepare_features(self, data):
        """
        Prepare features for NRFI model training or prediction.
        """
        # Make a copy to avoid modifying the original data
        df = data.copy()

        # First ensure all numeric columns are properly converted
        numeric_cols = [
            'temperature', 'wind_speed', 'humidity', 'pressure',
            'home_pitcher_ERA_1st', 'away_pitcher_ERA_1st',
            'home_pitcher_WHIP_1st', 'away_pitcher_WHIP_1st',
            'home_pitcher_nrfi_rate', 'away_pitcher_nrfi_rate',
            'home_team_nrfi_rate', 'away_team_nrfi_rate',
            'home_team_first_inning_runs', 'away_team_first_inning_runs'
        ]

        # Initialize missing pitcher columns
        pitcher_cols = [
            'home_pitcher_ERA_1st', 'away_pitcher_ERA_1st',
            'home_pitcher_WHIP_1st', 'away_pitcher_WHIP_1st',
            'home_pitcher_nrfi_rate', 'away_pitcher_nrfi_rate'
        ]
        
        for col in pitcher_cols:
            if col not in df.columns:
                df[col] = 0.0  # Initialize with neutral value
                
        # Convert columns to numeric and fill NaN values appropriately
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if col in pitcher_cols:
                    # For pitcher stats, use backward fill then forward fill
                    df[col] = df[col].bfill().ffill().fillna(df[col].median())
                else:
                    # For other features, use median
                    df[col] = df[col].fillna(df[col].median())

        # Add weather impact features
        if 'temperature' in df.columns:
            # Temperature ranges for NRFI
            df['temp_cold'] = (df['temperature'] < 55).astype(float)
            df['temp_hot'] = (df['temperature'] > 85).astype(float)
            
            # Temperature squared for non-linear effects
            df['temperature_squared'] = df['temperature'] ** 2

        if 'wind_speed' in df.columns:
            # High wind indicator
            df['high_wind'] = (df['wind_speed'] > 10).astype(float)
            
            # Wind-temperature interaction
            if 'temperature' in df.columns:
                df['wind_temp_interaction'] = (df['wind_speed'] * df['temperature']) / 100

        # Create wind direction features if not already present
        if 'wind_direction' in df.columns and 'wind_blowing_out' not in df.columns:
            # Convert wind_direction to numeric if needed
            df['wind_direction'] = pd.to_numeric(df['wind_direction'], errors='coerce')
            df['wind_direction'] = df['wind_direction'].fillna(0)
            
            # Simplistic wind direction impact
            df['wind_blowing_out'] = ((df['wind_direction'] > 135) & 
                                     (df['wind_direction'] < 225)).astype(float)
            df['wind_blowing_in'] = ((df['wind_direction'] < 45) | 
                                    (df['wind_direction'] > 315)).astype(float)
            df['wind_blowing_crossfield'] = (~(df['wind_blowing_out'].astype(bool)) & 
                                           ~(df['wind_blowing_in'].astype(bool))).astype(float)

        # Add dome impact if not already present
        if 'is_dome' not in df.columns and 'stadium' in df.columns:
            dome_stadiums = {
                'Tropicana Field': 1,
                'Rogers Centre': 1,
                'Minute Maid Park': 1,
                'Globe Life Field': 1,
                'American Family Field': 1
            }
            
            df['is_dome'] = df['stadium'].map(dome_stadiums).fillna(0).astype(float)

        # Feature for retractable roof
        if 'has_retractable_roof' not in df.columns and 'stadium' in df.columns:
            retractable_roof_stadiums = {
                'Rogers Centre': 1,
                'Minute Maid Park': 1,
                'Globe Life Field': 1,
                'American Family Field': 1,
                'Chase Field': 1,
                'LoanDepot Park': 1,
                'T-Mobile Park': 1
            }
            df['has_retractable_roof'] = df['stadium'].map(retractable_roof_stadiums).fillna(0).astype(float)

        # Rest of feature preparation
        # ...existing code...

        return df
    
    def train_model(self):
        """Train the NRFI prediction model."""
        if self.merged_data is None or 'nrfi' not in self.merged_data.columns:
            print("No first inning data available for training NRFI model.")
            return False
        
        print("Training NRFI prediction model...")
        
        # Prepare features specific to first inning scoring
        data = self.prepare_features(self.merged_data)
        
        # Define default features for NRFI prediction
        default_features = [
            # Pitcher stats - critical for first inning
            'home_pitcher_ERA_1st', 'away_pitcher_ERA_1st',
            'home_pitcher_WHIP_1st', 'away_pitcher_WHIP_1st',
            'home_pitcher_nrfi_rate', 'away_pitcher_nrfi_rate',
            
            # Pitcher advantages
            'home_pitcher_vs_away_team', 'away_pitcher_vs_home_team',
            
            # Team stats
            'home_team_first_inning_runs', 'away_team_first_inning_runs',
            'home_team_nrfi_rate', 'away_team_nrfi_rate',
            
            # Weather factors
            'temperature', 'wind_speed', 'wind_blowing_out', 'wind_blowing_in',
            'is_dome', 'has_retractable_roof',
            'temp_cold', 'temp_hot', 'high_wind', 'wind_temp_interaction',
            
            # Combined probabilities
            'model_nrfi_probability', 'weather_adjusted_nrfi_prob'
        ]
        
        # Filter to only available features
        self.features = [f for f in default_features if f in data.columns]
        
        # Check for highly correlated features but use a lower threshold
        if len(self.features) > 5:
            corr_matrix = data[self.features].corr().abs()
            high_corr_pairs = set()
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if corr_matrix.iloc[i, j] > 0.90:  # Reduced from 0.95
                        high_corr_pairs.add(corr_matrix.columns[i])
            
            for feature in high_corr_pairs:
                if feature in self.features:
                    self.features.remove(feature)
        
        print(f"Using {len(self.features)} features for NRFI model training")
        
        # Prepare X and y for NRFI model
        X = data[self.features].fillna(0)
        y = data['nrfi'].values
        
        # Use stratified split to maintain class balance
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize and train model based on available libraries
        if self.model_type == "lightgbm" and HAS_LIGHTGBM:
            # Updated LightGBM parameters for better performance
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'num_leaves': 63,  # Increased from 31
                'learning_rate': 0.01,  # Decreased for better generalization
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,  # Added bagging
                'bagging_freq': 5,
                'min_child_samples': 20,
                'min_split_gain': 0.01,
                'reg_alpha': 0.1,  # L1 regularization
                'reg_lambda': 0.1,  # L2 regularization
                'verbose': -1
            }

            # Handle class imbalance
            class_weights = dict(zip(
                range(2),
                [1.0 / (y_train == i).mean() for i in range(2)]
            ))
            
            train_data = lgb.Dataset(X_train_scaled, label=y_train, weight=[class_weights[y] for y in y_train])
            valid_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)
            
            callbacks = [
                lgb.early_stopping(stopping_rounds=100),  # Increased from 50
                lgb.log_evaluation(period=100)
            ]
            
            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=2000,  # Increased from 1000
                valid_sets=[valid_data],
                callbacks=callbacks
            )
            
            # Make predictions
            y_pred_prob = self.model.predict(X_test_scaled)
            y_pred = (y_pred_prob > 0.5).astype(int)
            
        elif self.model_type == "xgboost" and HAS_XGBOOST:
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8
            }
            dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
            dtest = xgb.DMatrix(X_test_scaled, label=y_test)
            
            self.model = xgb.train(
                params,
                dtrain,
                num_boost_round=1000,
                evals=[(dtest, 'test')],
                early_stopping_rounds=50,
                verbose_eval=100
            )
            
            # Make predictions
            y_pred_prob = self.model.predict(dtest)
            y_pred = (y_pred_prob > 0.5).astype(int)
            
        else:
            # Fall back to sklearn RandomForest
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
            self.model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred_prob = self.model.predict_proba(X_test_scaled)[:, 1]
            y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_prob)
        
        print(f"\nNRFI Model Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        
        # Calculate confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Print in a readable format
        print("\nConfusion Matrix:")
        print("              Predicted NRFI  Predicted YRFI")
        print(f"Actual NRFI    {cm[1][1]:14d}  {cm[1][0]:14d}")
        print(f"Actual YRFI    {cm[0][1]:14d}  {cm[0][0]:14d}")
        
        # Calculate feature importance
        if self.model_type == "lightgbm":
            importance = self.model.feature_importance('gain')
            importance_df = pd.DataFrame({
                'Feature': self.features,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            print("\nFeature Importance:")
            for i, row in importance_df.head(10).iterrows():
                print(f"{row['Feature']:30s}: {row['Importance']:.2f}")
        
        # Save timestamp
        self.timestamp = datetime.now()
        
        # Save model
        self.save_model()
        
        return True
    
    def predict(self, data):
        """
        Make NRFI predictions for games.
        """
        if self.model is None:
            print("No NRFI model available. Please train one first.")
            return None
        
        if data is None or len(data) == 0:
            print("No data provided for prediction.")
            return None
        
        # Prepare features
        prepared_data = self.prepare_features(data)
        
        if not self.features:
            print("No features defined. Model needs to be trained first.")
            return None
        
        # Make sure all features exist
        missing_features = [f for f in self.features if f not in prepared_data.columns]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            # Add missing features with zeros
            for feature in missing_features:
                prepared_data[feature] = 0
        
        # Extract features in correct order
        features = prepared_data[self.features].fillna(0)
        
        # Scale features
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        # Make predictions
        if self.model_type == "lightgbm":
            raw_probs = self.model.predict(features)
        elif self.model_type == "xgboost":
            dmatrix = xgb.DMatrix(features)
            raw_probs = self.model.predict(dmatrix)
        else:
            raw_probs = self.model.predict_proba(features)[:, 1]
        
        # Calibrate probabilities using temperature and wind
        calibrated_probs = raw_probs.copy()
        
        # Apply temperature-based calibration
        if 'temperature' in prepared_data.columns:
            temp = prepared_data['temperature'].values
            # Increase NRFI probability in cold weather
            cold_mask = temp < 50
            calibrated_probs[cold_mask] = calibrated_probs[cold_mask] * 1.05
            # Decrease NRFI probability in hot weather
            hot_mask = temp > 85
            calibrated_probs[hot_mask] = calibrated_probs[hot_mask] * 0.95
        
        # Apply wind-based calibration
        if all(col in prepared_data.columns for col in ['wind_speed', 'wind_blowing_out']):
            wind_mask = (prepared_data['wind_speed'] > 10) & (prepared_data['wind_blowing_out'] == 1)
            calibrated_probs[wind_mask] = calibrated_probs[wind_mask] * 0.95
        
        # Ensure probabilities stay in valid range
        calibrated_probs = np.clip(calibrated_probs, 0.01, 0.99)
        
        # Add predictions to output
        prepared_data['nrfi_probability'] = calibrated_probs
        prepared_data['yrfi_probability'] = 1 - calibrated_probs
        
        # Calculate edge based on true odds
        nrfi_true_odds = -110  # Standard NRFI odds
        yrfi_true_odds = -110  # Standard YRFI odds
        
        def american_to_decimal(odds):
            if odds > 0:
                return 1 + (odds / 100)
            else:
                return 1 + (100 / abs(odds))
        
        nrfi_decimal = american_to_decimal(nrfi_true_odds)
        yrfi_decimal = american_to_decimal(yrfi_true_odds)
        
        # Calculate EV for each bet type
        prepared_data['nrfi_ev'] = (prepared_data['nrfi_probability'] * (nrfi_decimal - 1)) - (1 - prepared_data['nrfi_probability'])
        prepared_data['yrfi_ev'] = (prepared_data['yrfi_probability'] * (yrfi_decimal - 1)) - (1 - prepared_data['yrfi_probability'])
        
        # Determine recommended bet with minimum edge threshold
        min_edge = 0.05  # Minimum 5% edge required
        prepared_data['recommended_bet'] = np.where(
            (prepared_data['nrfi_ev'] > prepared_data['yrfi_ev']) & (prepared_data['nrfi_ev'] > min_edge),
            'NRFI',
            np.where(
                (prepared_data['yrfi_ev'] > prepared_data['nrfi_ev']) & (prepared_data['yrfi_ev'] > min_edge),
                'YRFI',
                'NONE'
            )
        )
        
        # Set confidence based on probability and edge
        prepared_data['confidence'] = np.where(
            prepared_data['recommended_bet'] == 'NRFI',
            prepared_data['nrfi_probability'],
            np.where(
                prepared_data['recommended_bet'] == 'YRFI',
                prepared_data['yrfi_probability'],
                0
            )
        )
        
        # Set edge amount
        prepared_data['edge'] = np.where(
            prepared_data['recommended_bet'] == 'NRFI',
            prepared_data['nrfi_ev'],
            np.where(
                prepared_data['recommended_bet'] == 'YRFI',
                prepared_data['yrfi_ev'],
                0
            )
        )
        
        return prepared_data
    
    def find_betting_opportunities(self, confidence_threshold=0.6):
        """
        Find NRFI/YRFI betting opportunities in historical data.
        
        Args:
            confidence_threshold (float): Threshold for confidence to consider a bet
            
        Returns:
            DataFrame: NRFI/YRFI betting opportunities
        """
        if self.model is None:
            print("No NRFI model available. Please train one first.")
            return None
        
        if self.merged_data is None or len(self.merged_data) == 0:
            print("No data available for analysis.")
            return None
        
        if 'nrfi' not in self.merged_data.columns:
            print("No first inning data available in dataset.")
            return None
        
        # Use recent data for testing - last 20% of dataset
        test_data = self.merged_data.sort_values('date').tail(int(len(self.merged_data) * 0.2)).copy()
        
        # Make predictions
        predictions = self.predict(test_data)
        
        if predictions is None:
            return None
        
        # Add actual NRFI result
        predictions['actual_nrfi'] = test_data['nrfi'].values
        
        # Filter for opportunities with high confidence
        opportunities = predictions[
            (predictions['confidence'] >= confidence_threshold) & 
            (predictions['edge'] > 0) &
            (predictions['recommended_bet'] != 'NONE')
        ].copy()
        
        # Calculate overall win percentage
        if len(opportunities) > 0:
            wins = ((opportunities['recommended_bet'] == 'NRFI') & (opportunities['actual_nrfi'] == 1)) | \
                   ((opportunities['recommended_bet'] == 'YRFI') & (opportunities['actual_nrfi'] == 0))
            win_pct = wins.mean()
            print(f"Found {len(opportunities)} potential NRFI/YRFI betting opportunities with {win_pct:.1%} win rate")
        else:
            print("No NRFI/YRFI betting opportunities found.")
        
        return opportunities
    
    def get_todays_nrfi_recommendations(self, odds_api_key=None, confidence_threshold=0.6):
        """
        Get today's NRFI/YRFI betting recommendations.
        
        Args:
            odds_api_key (str): Optional API key for odds API
            confidence_threshold (float): Confidence threshold for recommendations
            
        Returns:
            DataFrame: Today's NRFI/YRFI betting opportunities
        """
        # Get today's games and odds
        today_games = self.fetch_todays_games_and_odds(odds_api_key)
        
        if today_games is None or len(today_games) == 0:
            print("No games found for today.")
            return None
        
        # Get pitchers
        today_games = self.add_pitcher_data(today_games)
        
        # Get weather for game locations
        today_games = self.add_weather_data(today_games)
        
        # Make predictions
        predictions = self.predict(today_games)
        
        if predictions is None:
            return None
        
        # Filter for opportunities with high confidence
        opportunities = predictions[
            (predictions['confidence'] >= confidence_threshold) & 
            (predictions['edge'] > 0) &
            (predictions['recommended_bet'] != 'NONE')
        ].copy()
        
        # Select relevant columns for display
        display_cols = [
            'date', 'home_team', 'away_team', 
            'starting_pitcher_home', 'starting_pitcher_away',
            'nrfi_probability', 'yrfi_probability', 
            'recommended_bet', 'confidence', 'edge',
            'temperature', 'wind_speed', 'weather_condition'
        ]
        
        display_cols = [col for col in display_cols if col in opportunities.columns]
        
        if len(opportunities) > 0:
            print(f"Found {len(opportunities)} NRFI/YRFI betting opportunities for today")
            opportunities = opportunities[display_cols].sort_values('confidence', ascending=False)
        else:
            print("No NRFI/YRFI betting opportunities found for today.")
        
        return opportunities
    
    def fetch_todays_games_and_odds(self, odds_api_key=None):
        """
        Fetch today's MLB games and odds.
        
        Args:
            odds_api_key (str): Optional API key for odds API
            
        Returns:
            DataFrame: Today's games with odds
        """
        print("Fetching today's MLB games and odds...")
        
        # Use passed API key or config value
        api_key = odds_api_key or ODDS_API_KEY
        
        if api_key:
            sport = 'baseball_mlb'
            regions = 'us'
            markets = 'totals'
            oddsFormat = 'american'
            date_format = 'iso'
            
            try:
                odds_response = requests.get(
                    f'https://api.the-odds-api.com/v4/sports/{sport}/odds',
                    params={
                        'api_key': api_key,
                        'regions': regions,
                        'markets': markets,
                        'oddsFormat': oddsFormat,
                        'dateFormat': date_format,
                    }
                )
                
                if odds_response.status_code != 200:
                    print(f"Failed to get odds: status_code {odds_response.status_code}")
                    return self._generate_synthetic_today_games()
                
                odds_json = odds_response.json()
                
                # Process the odds data
                games_list = []
                
                for game in odds_json:
                    try:
                        # Get game date/time
                        game_date = datetime.fromisoformat(game['commence_time'].replace('Z', '+00:00'))
                        
                        # Convert to local time
                        game_date = game_date.replace(tzinfo=None) - timedelta(hours=4)  # Rough EST conversion
                        
                        # Get teams
                        home_team = game['home_team']
                        away_team = game['away_team']
                        
                        # Get totals (over/under)
                        over_under_line = None
                        first_inning_over_under = None
                        
                        if 'bookmakers' in game and len(game['bookmakers']) > 0:
                            # Look for first inning totals market
                            for bookie in game['bookmakers']:
                                for market in bookie['markets']:
                                    if market['key'] == 'totals':
                                        # Full game total
                                        for outcome in market['outcomes']:
                                            if outcome['name'] == 'Over':
                                                over_under_line = float(outcome['point'])
                                                break
                                    
                                    if market['key'] == 'alternate_totals_innings_1':
                                        # First inning total
                                        for outcome in market['outcomes']:
                                            if outcome['name'] == 'Over 0.5':
                                                first_inning_over_under = 0.5
                                                break
                        
                        # Only add games with date, teams and first inning odds
                        if game_date and home_team and away_team:
                            games_list.append({
                                'date': game_date,
                                'home_team': home_team,
                                'away_team': away_team,
                                'over_under_line': over_under_line,
                                'first_inning_over_under': first_inning_over_under
                            })
                    except Exception as e:
                        print(f"Error processing game: {e}")
                
                # Create DataFrame
                if games_list:
                    games_df = pd.DataFrame(games_list)
                    return games_df
                
            except Exception as e:
                print(f"Error fetching odds: {e}")
        
        # Fall back to synthetic data
        return self._generate_synthetic_today_games()
    
    def _generate_synthetic_today_games(self):
        """
        Generate synthetic games data for today when API fails.
        
        Returns:
            DataFrame: Synthetic today's games
        """
        print("Generating synthetic games for today...")
        
        # MLB teams
        teams = ['ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 
                'COL', 'DET', 'HOU', 'KC', 'LAA', 'LAD', 'MIA', 'MIL', 
                'MIN', 'NYM', 'NYY', 'OAK', 'PHI', 'PIT', 'SD', 'SEA', 
                'SF', 'STL', 'TB', 'TEX', 'TOR', 'WSH']
        
        # Today's date
        today = datetime.now()
        
        # Create random matchups
        np.random.shuffle(teams)
        games_list = []
        
        for i in range(0, len(teams), 2):
            if i + 1 < len(teams):
                # Create game time (afternoon or evening)
                hour = np.random.choice([13, 16, 19])  # 1 PM, 4 PM, or 7 PM
                game_datetime = datetime.combine(today.date(), datetime.min.time().replace(hour=hour))
                
                # Create game with common betting lines
                games_list.append({
                    'date': game_datetime,
                    'home_team': teams[i],
                    'away_team': teams[i+1],
                    'over_under_line': float(np.random.choice([7.5, 8, 8.5, 9, 9.5])),
                    'first_inning_over_under': 0.5  # Standard NRFI line
                })
        
        return pd.DataFrame(games_list)
    
    def add_pitcher_data(self, games_df):
        """
        Add pitcher data to games DataFrame.
        
        Args:
            games_df (DataFrame): Games data
            
        Returns:
            DataFrame: Games data with pitcher information
        """
        # Use synthetic pitcher data for this example
        # In a real system, this would fetch actual starting pitcher info
        df = games_df.copy()
        
        # Add placeholder pitcher names
        if 'starting_pitcher_home' not in df.columns:
            df['starting_pitcher_home'] = [f"Home Pitcher {i+1}" for i in range(len(df))]
        
        if 'starting_pitcher_away' not in df.columns:
            df['starting_pitcher_away'] = [f"Away Pitcher {i+1}" for i in range(len(df))]
        
        # Add pitcher stats
        current_year = datetime.now().year
        pitcher_stats = self.fetch_pitcher_first_inning_stats(year=current_year - 1)
        
        if not pitcher_stats.empty:
            # Add placeholder values that match the real pitchers in the stats
            real_pitchers = pitcher_stats['pitcher_name'].tolist()
            if real_pitchers:
                for i in range(len(df)):
                    df.loc[i, 'starting_pitcher_home'] = np.random.choice(real_pitchers)
                    df.loc[i, 'starting_pitcher_away'] = np.random.choice(real_pitchers)
        
        return df
    
    def add_weather_data(self, games_df):
        """
        Add weather data to games DataFrame.
        
        Args:
            games_df (DataFrame): Games data
            
        Returns:
            DataFrame: Games data with weather information
        """
        try:
            # Try to import weather module
            from weather_data import get_current_weather
            
            # Add weather for each game
            df = games_df.copy()
            
            for idx, game in df.iterrows():
                try:
                    # Get weather for stadium at game time
                    weather = get_current_weather(
                        game['home_team'], 
                        game_datetime=game['date']
                    )
                    
                    if weather:
                        # Add weather data to the game
                        for key, value in weather.items():
                            df.loc[idx, key] = value
                except Exception as e:
                    print(f"Error getting weather for {game['home_team']}: {e}")
            
            return df
            
        except ImportError:
            print("Weather module not available, using synthetic weather data")
            # Generate synthetic weather data
            return self._add_synthetic_weather(games_df)
    
    def _add_synthetic_weather(self, games_df):
        """
        Add synthetic weather data to games DataFrame.
        
        Args:
            games_df (DataFrame): Games data
            
        Returns:
            DataFrame: Games data with synthetic weather
        """
        df = games_df.copy()
        
        # Current month for seasonal appropriateness
        month = datetime.now().month
        
        # Generate weather for each game
        for idx, game in df.iterrows():
            # Seasonal temperature adjustment
            if 5 <= month <= 9:  # Summer months
                base_temp = np.random.randint(70, 90)
            elif month in [4, 10]:  # Spring/Fall
                base_temp = np.random.randint(55, 75)
            else:  # Winter (rare for baseball)
                base_temp = np.random.randint(40, 60)
            
            # Wind speed (0-20 mph)
            wind_speed = np.random.randint(0, 20)
            
            # Wind direction (0-359 degrees)
            wind_direction = np.random.randint(0, 360)
            
            # Determine if wind is blowing in/out (simplified)
            wind_blowing_out = 1 if 135 <= wind_direction <= 225 else 0
            wind_blowing_in = 1 if (wind_direction <= 45 or wind_direction >= 315) else 0
            wind_blowing_crossfield = 1 if (not wind_blowing_out and not wind_blowing_in) else 0
            
            # Check if dome
            is_dome = 0
            dome_teams = ['TOR', 'TB', 'MIL', 'ARI', 'HOU', 'TEX', 'MIN', 'SEA']
            if game['home_team'] in dome_teams:
                is_dome = 1
                wind_speed = 0
                wind_blowing_out = 0
                wind_blowing_in = 0
                wind_blowing_crossfield = 0
            
            # Weather condition
            if is_dome:
                weather_condition = 'Dome'
                weather_description = 'Indoor stadium'
                precipitation = 0
            else:
                # Random weather condition weighted towards clear
                conditions = ['Clear', 'Clouds', 'Rain']
                probabilities = [0.7, 0.2, 0.1]
                weather_condition = np.random.choice(conditions, p=probabilities)
                
                # Precipitation based on condition
                if weather_condition == 'Rain':
                    precipitation = np.random.uniform(0.1, 0.5)
                    weather_description = f"Light to moderate rain"
                elif weather_condition == 'Clouds':
                    precipitation = 0
                    weather_description = f"Cloudy conditions"
                else:
                    precipitation = 0
                    weather_description = f"Clear skies"
            
            # Add to DataFrame
            df.loc[idx, 'temperature'] = base_temp
            df.loc[idx, 'wind_speed'] = wind_speed
            df.loc[idx, 'wind_direction'] = wind_direction
            df.loc[idx, 'wind_blowing_out'] = wind_blowing_out
            df.loc[idx, 'wind_blowing_in'] = wind_blowing_in
            df.loc[idx, 'wind_blowing_crossfield'] = wind_blowing_crossfield
            df.loc[idx, 'precipitation'] = precipitation
            df.loc[idx, 'weather_condition'] = weather_condition
            df.loc[idx, 'weather_description'] = weather_description
            df.loc[idx, 'is_dome'] = is_dome
            df.loc[idx, 'humidity'] = np.random.randint(30, 80)
            df.loc[idx, 'pressure'] = np.random.randint(1000, 1020)
        
        return df
    
    def save_model(self):
        """Save the trained model to disk."""
        if self.model is None:
            print("No model to save.")
            return False
        
        model_path = f"{DATA_DIR}/nrfi_model.pkl"
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'features': self.features,
            'model_type': self.model_type,
            'timestamp': self.timestamp
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"NRFI model saved to {model_path}")
        return True
    
    def load_model(self):
        """Load a trained model from disk."""
        model_path = f"{DATA_DIR}/nrfi_model.pkl"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"NRFI model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data.get('scaler')
        self.features = model_data.get('features')
        self.model_type = model_data.get('model_type')
        self.timestamp = model_data.get('timestamp')
        
        print(f"NRFI model loaded from {model_path}")
        if self.timestamp:
            print(f"Model timestamp: {self.timestamp}")
        
        return True
    
    def analyze_feature_importance(self):
        """
        Analyze and visualize feature importance for NRFI prediction.
        
        Returns:
            DataFrame: Feature importance
        """
        if self.model is None:
            print("No model available for feature importance analysis.")
            return None
        
        # Extract feature importance based on model type
        if self.model_type == "lightgbm":
            importance = self.model.feature_importance('gain')
            importance_df = pd.DataFrame({
                'Feature': self.features,
                'Importance': importance
            })
        elif self.model_type == "xgboost":
            importance = self.model.get_score(importance_type='gain')
            # Convert to DataFrame if it's a dictionary
            if isinstance(importance, dict):
                importance_df = pd.DataFrame({
                    'Feature': list(importance.keys()),
                    'Importance': list(importance.values())
                })
            else:
                importance_df = pd.DataFrame({
                    'Feature': self.features,
                    'Importance': importance
                })
        else:
            # scikit-learn models
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': self.features,
                    'Importance': importance
                })
            else:
                print("Model does not have feature importance information.")
                return None
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Visualize
        try:
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
            plt.title('NRFI Model Feature Importance')
            plt.tight_layout()
            plt.savefig(f"{DATA_DIR}/nrfi_feature_importance.png")
            plt.close()
        except Exception as e:
            print(f"Error creating visualization: {e}")
        
        return importance_df
    
    def backtest_nrfi_strategy(self, opportunities=None, nrfi_odds=-110, yrfi_odds=-110, starting_bankroll=10000):
        """
        Backtest the NRFI/YRFI betting strategy on historical opportunities.
        
        Args:
            opportunities (DataFrame): Optional pre-filtered opportunities
            nrfi_odds (int): American odds for NRFI bets (default -110)
            yrfi_odds (int): American odds for YRFI bets (default -110)
            starting_bankroll (float): Starting bankroll amount
            
        Returns:
            dict: Backtest results
        """
        # If no opportunities provided, find them
        if opportunities is None:
            opportunities = self.find_betting_opportunities()
            
        if opportunities is None or len(opportunities) == 0 or 'actual_nrfi' not in opportunities.columns:
            print("No valid opportunities provided for NRFI/YRFI backtest")
            return None
        
        # Initial bankroll
        bankroll = starting_bankroll
        bets = []
        
        print(f"Starting NRFI/YRFI backtest with ${starting_bankroll:.2f} bankroll")
        print(f"Testing {len(opportunities)} betting opportunities")
        
        # Convert odds to decimal
        def american_to_decimal(odds):
            if odds > 0:
                return 1 + (odds / 100)
            else:
                return 1 + (100 / abs(odds))
        
        nrfi_decimal = american_to_decimal(nrfi_odds)
        yrfi_decimal = american_to_decimal(yrfi_odds)
        
        # Process each bet
        for idx, bet in opportunities.iterrows():
            # Fixed bet size (can be modified for Kelly criterion)
            bet_size = bankroll * 0.01  # 1% of bankroll
            
            # Determine bet type and odds
            bet_type = bet['recommended_bet']
            
            if bet_type == 'NRFI':
                odds = nrfi_decimal
                win = bet['actual_nrfi'] == 1  # True if no runs scored in first inning
            elif bet_type == 'YRFI':
                odds = yrfi_decimal
                win = bet['actual_nrfi'] == 0  # True if runs scored in first inning
            else:
                continue  # Skip if no bet recommended
            
            # Calculate profit/loss
            if win:
                profit = bet_size * (odds - 1)
            else:
                profit = -bet_size
            
            # Update bankroll
            bankroll += profit
            
            # Record bet details
            bets.append({
                'date': bet.get('date', None),
                'matchup': f"{bet.get('away_team', '')} @ {bet.get('home_team', '')}",
                'bet_type': bet_type,
                'confidence': bet.get('confidence', np.nan),
                'odds': odds,
                'bet_size': bet_size,
                'profit': profit,
                'bankroll': bankroll,
                'win': win
            })
        
        # Create DataFrame of bet results
        results_df = pd.DataFrame(bets)
        
        # Calculate performance metrics
        if len(results_df) > 0:
            bet_count = len(results_df)
            winning_bets = results_df['win'].sum()
            win_rate = winning_bets / bet_count if bet_count > 0 else 0
            roi = (bankroll - starting_bankroll) / starting_bankroll * 100
            profit = bankroll - starting_bankroll
            
            # Calculate max drawdown
            max_drawdown = self._calculate_max_drawdown(results_df['bankroll'])
            
            # Calculate win/loss streaks
            max_win_streak, max_lose_streak = self._calculate_streaks(results_df['win'])
            
            print(f"\nNRFI/YRFI Backtest Results:")
            print(f"Total Bets: {bet_count}")
            print(f"Win Rate: {win_rate:.4f} ({winning_bets} / {bet_count})")
            print(f"ROI: {roi:.2f}%")
            print(f"Profit: ${profit:.2f}")
            print(f"Final Bankroll: ${bankroll:.2f}")
            print(f"Max Drawdown: ${max_drawdown:.2f}")
            print(f"Max Win Streak: {max_win_streak}, Max Lose Streak: {max_lose_streak}")
            
            # Calculate NRFI vs YRFI performance
            nrfi_bets = results_df[results_df['bet_type'] == 'NRFI']
            yrfi_bets = results_df[results_df['bet_type'] == 'YRFI']
            
            nrfi_win_rate = nrfi_bets['win'].mean() if len(nrfi_bets) > 0 else 0
            yrfi_win_rate = yrfi_bets['win'].mean() if len(yrfi_bets) > 0 else 0
            
            print(f"\nNRFI Bets: {len(nrfi_bets)}, Win Rate: {nrfi_win_rate:.4f}")
            print(f"YRFI Bets: {len(yrfi_bets)}, Win Rate: {yrfi_win_rate:.4f}")
            
            # Plot equity curve
            try:
                plt.figure(figsize=(12, 6))
                plt.plot(results_df['bankroll'], marker='', linewidth=2)
                plt.axhline(y=starting_bankroll, color='r', linestyle='--', alpha=0.3)
                plt.title('NRFI/YRFI Betting Strategy Equity Curve')
                plt.xlabel('Bet Number')
                plt.ylabel('Bankroll ($)')
                plt.grid(True, alpha=0.3)
                
                # Add annotations
                plt.annotate(f'Start: ${starting_bankroll:.0f}', 
                           xy=(0, starting_bankroll),
                           xytext=(10, -20),
                           textcoords='offset points',
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))
                
                plt.annotate(f'End: ${bankroll:.0f} (ROI: {roi:.1f}%)', 
                           xy=(len(results_df)-1, bankroll),
                           xytext=(-100, -40),
                           textcoords='offset points',
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))
                
                plt.tight_layout()
                plt.savefig(f"{DATA_DIR}/nrfi_equity_curve.png")
                plt.close()
            except Exception as e:
                print(f"Error creating equity curve visualization: {e}")
            
            # Return results
            return {
                'results_df': results_df,
                'bet_count': bet_count,
                'win_rate': win_rate,
                'roi': roi,
                'profit': profit,
                'final_bankroll': bankroll,
                'max_drawdown': max_drawdown,
                'max_win_streak': max_win_streak,
                'max_lose_streak': max_lose_streak,
                'nrfi_win_rate': nrfi_win_rate,
                'yrfi_win_rate': yrfi_win_rate
            }
        else:
            print("No valid bets found in backtest")
            return None
    
    def _calculate_max_drawdown(self, equity_curve):
        """
        Calculate maximum drawdown from equity curve.
        
        Args:
            equity_curve (Series): Series containing bankroll values
            
        Returns:
            float: Maximum drawdown amount
        """
        # Calculate running maximum
        running_max = equity_curve.cummax()
        
        # Calculate drawdown
        drawdown = running_max - equity_curve
        
        # Return maximum drawdown
        return drawdown.max()
    
    def _calculate_streaks(self, results):
        """
        Calculate maximum win and loss streaks.
        
        Args:
            results (Series): Series of boolean win/loss results
            
        Returns:
            tuple: (max_win_streak, max_lose_streak)
        """
        if len(results) == 0:
            return 0, 0
        
        # Convert to numpy array
        results_array = np.array(results)
        
        # Initialize counters
        current_win_streak = 0
        current_lose_streak = 0
        max_win_streak = 0
        max_lose_streak = 0
        
        # Count streaks
        for result in results_array:
            if result:
                # Win
                current_win_streak += 1
                current_lose_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            else:
                # Loss
                current_lose_streak += 1
                current_win_streak = 0
                max_lose_streak = max(max_lose_streak, current_lose_streak)
        
        return max_win_streak, max_lose_streak


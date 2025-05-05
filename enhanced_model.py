"""
Enhanced MLB Weather Model with LightGBM and automatic fallback
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import LightGBM first, fall back to GradientBoostingRegressor
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    print("Using LightGBM for modeling")
except ImportError as e:
    print(f"LightGBM import error: {e}")
    print("Falling back to GradientBoostingRegressor from scikit-learn")
    from sklearn.ensemble import GradientBoostingRegressor
    LIGHTGBM_AVAILABLE = False
except OSError as e:
    print(f"LightGBM OSError: {e}")
    print("This is likely due to missing OpenMP library (libomp) on macOS.")
    print("Install it with: brew install libomp")
    print("Falling back to GradientBoostingRegressor from scikit-learn")
    from sklearn.ensemble import GradientBoostingRegressor
    LIGHTGBM_AVAILABLE = False

from config import DATA_DIR, STADIUM_MAPPING, BALLPARK_FACTORS

class MLBAdvancedModel:
    """Advanced machine learning model for MLB run predictions based on weather."""
    
    def __init__(self, data_dir=DATA_DIR):
        """Initialize the MLB Advanced Model."""
        self.data_dir = data_dir
        self.model = None
        self.feature_names = None
        self.scaler = None
        self.merged_data = None
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    def prepare_features(self, data=None, target_col='total_runs', test_size=0.2, random_state=42):
        """
        Prepare features for modeling with enhanced feature engineering.
        
        Args:
            data (DataFrame): Merged dataset with game, weather, and team data
            target_col (str): Column to use as target variable
            test_size (float): Fraction of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, feature_names)
        """
        # Use provided data or instance data
        if data is None:
            if self.merged_data is None:
                raise ValueError("No data provided and no instance data available")
            data = self.merged_data
        
        if len(data) == 0:
            raise ValueError("Input data is empty")
        
        # FEATURE ENGINEERING
        # ==========================================
        
        # Create a copy to avoid modifying original
        df = data.copy()
        
        # Basic meteorological features
        base_weather_features = []
        for feature in ['temperature', 'humidity', 'wind_speed', 
                        'precipitation', 'cloud_cover', 'pressure']:
            if feature in df.columns:
                base_weather_features.append(feature)
        
        # Enhanced weather features
        weather_features = []
        
        # Temperature features
        if 'temperature' in df.columns:
            df['temperature_squared'] = df['temperature'] ** 2
            weather_features.append('temperature_squared')
            
            # Hot/cold indicators
            df['is_hot'] = (df['temperature'] > 85).astype(int)
            df['is_cold'] = (df['temperature'] < 50).astype(int)
            weather_features.extend(['is_hot', 'is_cold'])
        
        # Add temperature/humidity interaction
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['temp_humidity_interaction'] = df['temperature'] * df['humidity'] / 100
            weather_features.append('temp_humidity_interaction')
        
        # Wind features
        if 'wind_speed' in df.columns:
            # High wind indicator
            df['high_wind'] = (df['wind_speed'] > 10).astype(int)
            weather_features.append('high_wind')
        
        # Wind direction effects from one-hot encoded columns
        wind_cols = [col for col in df.columns if col.startswith('wind_') 
                    and col not in ['wind_speed', 'wind_direction', 'wind_cardinal']]
        weather_features.extend(wind_cols)
        
        # Stadium/ballpark features
        ballpark_features = []
        
        if 'ballpark_factor' in df.columns:
            ballpark_features.append('ballpark_factor')
        
        # Team performance features (if available)
        team_features = []
        
        for col in df.columns:
            if 'team_' in col and col not in ['home_team', 'away_team']:
                team_features.append(col)
        
        # Time features
        time_features = []
        
        if 'month' in df.columns:
            time_features.append('month')
        
        # Odds/line features
        odds_features = []
        
        if 'over_under_line' in df.columns:
            odds_features.append('over_under_line')
        
        # Combine all feature groups
        all_features = (
            base_weather_features + 
            weather_features + 
            ballpark_features + 
            team_features + 
            time_features + 
            odds_features
        )
        
        # Keep only features that exist in the dataset
        feature_cols = [col for col in all_features if col in df.columns]
        
        # Print feature selection info
        print(f"Using {len(feature_cols)} features: {feature_cols}")
        
        # Create feature matrix
        X = df[feature_cols].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Target variable
        y = df[target_col]
        
        # Split data chronologically if possible
        if 'date' in df.columns:
            # Sort by date
            df = df.sort_values('date')
            
            # Use specified percentage for training, rest for testing
            train_size = int(len(df) * (1 - test_size))
            train_idx = df.index[:train_size]
            test_idx = df.index[train_size:]
            
            X_train = X.loc[train_idx]
            X_test = X.loc[test_idx]
            y_train = y.loc[train_idx]
            y_test = y.loc[test_idx]
        else:
            # Random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        # Store feature names for later use
        self.feature_names = feature_cols
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def train_model(self, X_train=None, y_train=None):
        """
        Train the model using the best available algorithm.
        
        Args:
            X_train (DataFrame): Training features
            y_train (Series): Training target
            
        Returns:
            object: Trained model
        """
        # If features not provided, prepare them
        if X_train is None or y_train is None:
            X_train, X_test, y_train, y_test, feature_cols = self.prepare_features()
        
        # Create scaler
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Choose model based on availability
        if LIGHTGBM_AVAILABLE:
            print("Training LightGBM model...")
            model = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=31,
                min_child_samples=20,
                random_state=42
            )
        else:
            print("Training GradientBoostingRegressor model...")
            model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                random_state=42
            )
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Create pipeline for convenience
        self.model = model
        
        # Save model
        self.save_model()
        
        return model
    
    def evaluate(self, X_test=None, y_test=None):
        """
        Evaluate model performance on test data.
        
        Args:
            X_test (DataFrame): Test features
            y_test (Series): Test target
            
        Returns:
            dict: Performance metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        # If test data not provided, use feature preparation
        if X_test is None or y_test is None:
            _, X_test, _, y_test, _ = self.prepare_features()
        
        # Scale test data
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Betting-specific metrics
        if 'over_under_line' in X_test.columns:
            over_under_lines = X_test['over_under_line']
            
            # Actual over/under results
            actual_over = y_test > over_under_lines
            actual_under = y_test < over_under_lines
            actual_push = y_test == over_under_lines
            
            # Predicted over/under
            pred_over = y_pred > over_under_lines
            pred_under = y_pred < over_under_lines
            
            # Calculate accuracy (excluding pushes)
            non_push = ~actual_push
            correct_over = (pred_over & actual_over)
            correct_under = (pred_under & actual_under)
            correct_bets = correct_over | correct_under
            
            over_accuracy = np.sum(correct_over) / np.sum(actual_over) if np.sum(actual_over) > 0 else 0
            under_accuracy = np.sum(correct_under) / np.sum(actual_under) if np.sum(actual_under) > 0 else 0
            overall_accuracy = np.sum(correct_bets[non_push]) / np.sum(non_push) if np.sum(non_push) > 0 else 0
        else:
            over_accuracy = np.nan
            under_accuracy = np.nan
            overall_accuracy = np.nan
        
        print(f"Model performance:")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        if not np.isnan(overall_accuracy):
            print(f"Betting accuracy: {overall_accuracy:.4f}")
            print(f"  - Over accuracy: {over_accuracy:.4f}")
            print(f"  - Under accuracy: {under_accuracy:.4f}")
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'over_accuracy': over_accuracy,
            'under_accuracy': under_accuracy,
            'overall_accuracy': overall_accuracy
        }
    
    def predict(self, features):
        """
        Make predictions with the trained model.
        
        Args:
            features (DataFrame): Features for prediction
            
        Returns:
            ndarray: Predicted values
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        # Ensure features match training features
        if self.feature_names is not None:
            # Get intersection of available features
            common_features = [f for f in self.feature_names if f in features.columns]
            
            if len(common_features) < len(self.feature_names):
                print(f"Warning: {len(self.feature_names) - len(common_features)} features are missing.")
            
            # Use only available features
            features = features[common_features].copy()
            
            # Fill missing values
            features = features.fillna(features.median())
        
        # Scale features
        if hasattr(self, 'scaler') and self.scaler is not None:
            scaled_features = self.scaler.transform(features)
        else:
            # If no scaler available, use features as-is
            scaled_features = features
        
        # Make predictions
        predictions = self.model.predict(scaled_features)
        
        return predictions
    
    def analyze_feature_importance(self):
        """
        Analyze which features have the biggest impact on predictions.
        
        Returns:
            DataFrame: Feature importance rankings
        """
        if self.model is None:
            raise ValueError("Model must be trained before analyzing feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            
            # Create DataFrame of feature importances
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Plot feature importances
            plt.figure(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', data=importance_df)
            plt.title('MLB Run Scoring Feature Importance')
            plt.tight_layout()
            plt.savefig(f"{self.data_dir}/feature_importance.png")
            
            return importance_df
        else:
            print("Feature importance not available for this model type")
            return None
    
    def find_betting_opportunities(self, data=None, confidence_threshold=0.5):
        """
        Find potential betting opportunities based on model predictions.
        
        Args:
            data (DataFrame): Game data with features (None to use test data)
            confidence_threshold (float): Minimum difference between prediction and line
            
        Returns:
            DataFrame: Recommended bets
        """
        if self.model is None:
            raise ValueError("Model must be trained before finding opportunities")
        
        # If no data provided, use test data from feature preparation
        if data is None:
            _, X_test, _, y_test, _ = self.prepare_features()
            test_idx = X_test.index
            test_data = self.merged_data.loc[test_idx].copy()
        else:
            test_data = data.copy()
        
        # Make predictions
        if self.feature_names is not None:
            # Get intersection of available features
            common_features = [f for f in self.feature_names if f in test_data.columns]
            
            if len(common_features) < len(self.feature_names):
                print(f"Warning: Using {len(common_features)} of {len(self.feature_names)} model features")
            
            # Use only available features
            test_features = test_data[common_features].copy()
            
            # Fill missing values
            test_features = test_features.fillna(test_features.median())
        else:
            # Use all numeric features if feature names not stored
            test_features = test_data.select_dtypes(include=['number'])
        
        # Add predictions
        test_data['predicted_runs'] = self.predict(test_features)
        
        # Calculate difference from betting line
        if 'over_under_line' in test_data.columns:
            test_data['pred_diff'] = test_data['predicted_runs'] - test_data['over_under_line']
            
            # Find opportunities where prediction differs from line by more than threshold
            opportunities = test_data[abs(test_data['pred_diff']) > confidence_threshold].copy()
            
            # Recommend bet type
            opportunities['recommended_bet'] = opportunities['pred_diff'].apply(
                lambda x: 'OVER' if x > 0 else 'UNDER'
            )
            
            # Add confidence column
            opportunities['confidence'] = abs(opportunities['pred_diff'])
            
            # Add result column for historical data
            if 'total_runs' in opportunities.columns:
                opportunities['bet_correct'] = (
                    ((opportunities['recommended_bet'] == 'OVER') & 
                    (opportunities['total_runs'] > opportunities['over_under_line'])) |
                    ((opportunities['recommended_bet'] == 'UNDER') & 
                    (opportunities['total_runs'] < opportunities['over_under_line']))
                ).astype(int)
                
                # Calculate accuracy if we have results
                accuracy = opportunities['bet_correct'].mean()
                print(f"Betting model accuracy: {accuracy:.4f}")
            
            # Sort by confidence
            opportunities = opportunities.sort_values('confidence', ascending=False)
            
            # Save to disk if historical data
            if 'total_runs' in opportunities.columns:
                opportunities.to_csv(f"{self.data_dir}/betting_opportunities.csv", index=False)
            
            return opportunities
        else:
            print("No over/under lines found in data")
            return test_data[['date', 'home_team', 'away_team', 'predicted_runs']]
    
    def backtest_strategy(self, bet_size=100.0, kelly=False):
        """
        Backtest the betting strategy with realistic parameters.
        
        Args:
            bet_size (float): Standard bet size in dollars
            kelly (bool): Whether to use Kelly criterion for bet sizing
            
        Returns:
            DataFrame: Bet results and performance metrics
        """
        # Find betting opportunities
        opportunities = self.find_betting_opportunities()
        
        if len(opportunities) == 0:
            print("No betting opportunities found.")
            return None
        
        # Start with initial bankroll
        initial_bankroll = 10000.0
        bankroll = initial_bankroll
        bets = []
        
        # Process each bet
        for idx, bet in opportunities.iterrows():
            # Get odds based on bet type
            if bet['recommended_bet'] == 'OVER' and 'over_odds' in bet:
                american_odds = bet['over_odds']
            elif bet['recommended_bet'] == 'UNDER' and 'under_odds' in bet:
                american_odds = bet['under_odds']
            else:
                american_odds = -110  # Default
            
            decimal_odds = self._american_to_decimal(american_odds)
            
            # True edge (predicted vs line)
            true_edge = abs(bet['predicted_runs'] - bet['over_under_line'])
            
            # Determine bet size
            if kelly:
                # Kelly criterion: bet size = bankroll * edge / odds
                edge = true_edge / bet['over_under_line']  # Estimated edge
                bet_amount = bankroll * edge / (decimal_odds - 1)
                # Limit Kelly to 5% of bankroll for safety
                bet_amount = min(bet_amount, bankroll * 0.05)
            else:
                bet_amount = bet_size
            
            # Ensure bet doesn't exceed bankroll
            bet_amount = min(bet_amount, bankroll)
            
            # Calculate outcome with vig
            if bet['bet_correct'] == 1:
                profit = bet_amount * (decimal_odds - 1)
            else:
                profit = -bet_amount
            
            # Add transaction costs (typically 2-5%)
            transaction_cost = bet_amount * 0.02
            profit -= transaction_cost
            
            # Update bankroll
            bankroll += profit
            
            # Record bet details
            bets.append({
                'date': bet['date'],
                'game': f"{bet['away_team']} @ {bet['home_team']}",
                'bet_type': bet['recommended_bet'],
                'over_under_line': bet['over_under_line'],
                'predicted_runs': bet['predicted_runs'],
                'actual_runs': bet['total_runs'],
                'odds': decimal_odds,
                'edge': true_edge,
                'bet_amount': bet_amount,
                'profit': profit,
                'bankroll': bankroll,
                'correct': bet['bet_correct']
            })
            
            # Check for bankruptcy
            if bankroll <= 0:
                print("Bankrupt! Stopping backtest.")
                break
        
        # Create DataFrame of bet results
        results_df = pd.DataFrame(bets)
        
        # Calculate performance metrics
        bet_count = len(results_df)
        winning_bets = len(results_df[results_df['profit'] > 0])
        win_rate = winning_bets / bet_count if bet_count > 0 else 0
        roi = (bankroll - initial_bankroll) / initial_bankroll * 100
        
        print(f"Backtest Results:")
        print(f"Total Bets: {bet_count}")
        print(f"Win Rate: {win_rate:.4f}")
        print(f"ROI: {roi:.2f}%")
        print(f"Final Bankroll: ${bankroll:.2f}")
        
        # Plot equity curve
        plt.figure(figsize=(10, 6))
        plt.plot(results_df['bankroll'])
        plt.axhline(y=initial_bankroll, color='r', linestyle='--')
        plt.title('Betting Strategy Equity Curve')
        plt.xlabel('Bet Number')
        plt.ylabel('Bankroll ($)')
        plt.grid(True)
        plt.savefig(f"{self.data_dir}/equity_curve.png")
        
        # Save results to disk
        results_df.to_csv(f"{self.data_dir}/backtest_results.csv", index=False)
        
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
        import requests
        from config import ODDS_API_KEY
        
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

                        # Check if we have mappings for these teams
                        from config import TEAM_NAME_MAP
                        home_team_abbr = None
                        away_team_abbr = None
                        
                        # Try to find team abbreviations
                        for full_name, abbr in TEAM_NAME_MAP.items():
                            if full_name == home_team:
                                home_team_abbr = abbr
                            if full_name == away_team:
                                away_team_abbr = abbr
                        
                        # Skip if we can't map the teams
                        if not home_team_abbr or not away_team_abbr:
                            print(f"Could not map team names: {home_team} vs {away_team}")
                            continue

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
                            stadium_info = STADIUM_MAPPING.get(home_team_abbr, None)

                            if stadium_info:
                                # Fetch weather for the stadium
                                # Try to use weather_data module if available
                                try:
                                    from weather_data import get_current_weather
                                    weather = get_current_weather(home_team_abbr, game_date)
                                except ImportError:
                                    # Fallback to basic weather fetching
                                    weather = self._fetch_current_weather(
                                        stadium_info['lat'], 
                                        stadium_info['lon'],
                                        stadium_info['name']
                                    )

                                if weather:
                                    # Create game features
                                    features = {
                                        'date': game_date,
                                        'home_team': home_team_abbr,
                                        'away_team': away_team_abbr,
                                        'over_under_line': over_under_line,
                                        'over_odds': over_odds,
                                        'under_odds': under_odds,
                                        'stadium': stadium_info['name'],
                                        'month': game_date.month
                                    }
                                    
                                    # Add weather features
                                    features.update(weather)
                                    
                                    # Add ballpark factor
                                    features['ballpark_factor'] = BALLPARK_FACTORS.get(
                                        stadium_info['name'], {'base': 1.0}
                                    )['base']

                                    # Append to games list
                                    games_odds.append(features)
                    except Exception as e:
                        print(f"Error processing game: {e}")

                if games_odds:
                    # Convert to DataFrame
                    games_df = pd.DataFrame(games_odds)

                    # Find betting opportunities
                    opportunities = self.find_betting_opportunities(
                        games_df, 
                        confidence_threshold=confidence_threshold
                    )
                    
                    if opportunities is not None and len(opportunities) > 0:
                        print(f"Found {len(opportunities)} betting opportunities for today")
                        return opportunities
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
    
    def _fetch_current_weather(self, lat, lon, stadium_name):
        """
        Fetch current weather for a location using Open-Meteo API.
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            stadium_name (str): Stadium name for reference
            
        Returns:
            dict: Weather data
        """
        try:
            import requests
            
            # Build Open-Meteo API URL for current weather
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relativehumidity_2m,precipitation,cloudcover,pressure_msl,windspeed_10m,winddirection_10m",
                "timezone": "America/New_York"
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract current weather
                current = data.get("current", {})
                
                # Determine weather condition based on cloud cover and precipitation
                cloud_cover = current.get("cloudcover", 0)
                precip = current.get("precipitation", 0)
                
                if precip > 0.5:
                    weather_condition = 'Rain'
                elif precip > 0.1:
                    weather_condition = 'Drizzle'
                elif cloud_cover > 80:
                    weather_condition = 'Clouds'
                else:
                    weather_condition = 'Clear'
                
                # Create weather dictionary
                weather = {
                    'temperature': current.get("temperature_2m", 70) * 9/5 + 32,  # Convert C to F
                    'humidity': current.get("relativehumidity_2m", 50),
                    'pressure': current.get("pressure_msl", 1013) / 100,  # Convert Pa to hPa
                    'wind_speed': current.get("windspeed_10m", 5) * 2.237,  # Convert m/s to mph
                    'wind_direction': current.get("winddirection_10m", 0),
                    'cloud_cover': cloud_cover,
                    'precipitation': precip,
                    'weather_condition': weather_condition,
                    'weather_description': f"{weather_condition} with {cloud_cover}% cloud cover"
                }
                
                # Add derived features
                weather['temperature_squared'] = weather['temperature'] ** 2
                weather['temp_humidity_interaction'] = weather['temperature'] * weather['humidity'] / 100
                
                # Add wind direction features
                def degrees_to_cardinal(deg):
                    dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
                    ix = round(deg / 45) % 8
                    return dirs[ix]
                
                cardinal = degrees_to_cardinal(weather['wind_direction'])
                
                # Create one-hot encoded wind direction
                for direction in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']:
                    weather[f'wind_{direction}'] = 1 if cardinal == direction else 0
                
                return weather
            else:
                print(f"Error fetching weather: {response.status_code}")
                return self._generate_synthetic_weather(stadium_name)
        except Exception as e:
            print(f"Error fetching current weather: {e}")
            return self._generate_synthetic_weather(stadium_name)
    
    def _generate_synthetic_weather(self, stadium_name):
        """Generate synthetic weather when API fails."""
        # Current month for seasonal adjustment
        month = datetime.now().month
        
        # Seasonal temperature
        if 5 <= month <= 9:  # Summer
            temp = np.random.randint(70, 90)
        elif month in [4, 10]:  # Spring/Fall
            temp = np.random.randint(55, 75)
        else:  # Winter
            temp = np.random.randint(40, 60)
        
        # Generate realistic values
        humidity = np.random.randint(30, 90)
        wind_speed = np.random.randint(0, 20)
        wind_direction = np.random.randint(0, 360)
        cloud_cover = np.random.randint(0, 100)
        precipitation = 0.0 if cloud_cover < 70 else np.random.uniform(0, 0.5)
        
        # Determine weather condition
        if precipitation > 0.5:
            weather_condition = 'Rain'
        elif precipitation > 0.1:
            weather_condition = 'Drizzle'
        elif cloud_cover > 80:
            weather_condition = 'Clouds'
        else:
            weather_condition = 'Clear'
        
        # Create basic weather dict
        weather = {
            'temperature': temp,
            'humidity': humidity,
            'pressure': np.random.randint(990, 1030),
            'wind_speed': wind_speed,
            'wind_direction': wind_direction,
            'cloud_cover': cloud_cover,
            'precipitation': precipitation,
            'weather_condition': weather_condition,
            'weather_description': f"Synthetic {weather_condition}"
        }
        
        # Add derived features
        weather['temperature_squared'] = weather['temperature'] ** 2
        weather['temp_humidity_interaction'] = weather['temperature'] * weather['humidity'] / 100
        
        # Add wind direction features
        def degrees_to_cardinal(deg):
            dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
            ix = round(deg / 45) % 8
            return dirs[ix]
        
        cardinal = degrees_to_cardinal(weather['wind_direction'])
        
        # Create one-hot encoded wind direction
        for direction in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']:
            weather[f'wind_{direction}'] = 1 if cardinal == direction else 0
        
        return weather
    
    def save_model(self, filename='advanced_mlb_model.pkl'):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Please train a model first.")
        
        # Create model object with all necessary components
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save to disk
        model_path = f"{self.data_dir}/{filename}"
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {model_path}")
        return model_path
    
    def load_model(self, filename='advanced_mlb_model.pkl'):
        """Load a trained model from disk."""
        model_path = f"{self.data_dir}/{filename}"
        if not os.path.exists(model_path):
            # Try loading legacy model
            legacy_path = f"{self.data_dir}/weather_model.pkl"
            if os.path.exists(legacy_path):
                print(f"Advanced model not found, loading legacy model from {legacy_path}")
                model_path = legacy_path
            else:
                raise FileNotFoundError(f"No model file found at {model_path} or {legacy_path}")
        
        # Load from disk
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Handle different model formats
        if isinstance(model_data, dict) and 'model' in model_data:
            # New format with additional data
            self.model = model_data['model']
            self.feature_names = model_data.get('feature_names')
            self.scaler = model_data.get('scaler')
            timestamp = model_data.get('timestamp', 'unknown')
        else:
            # Legacy format (just the model)
            self.model = model_data
            self.feature_names = None
            self.scaler = None
            timestamp = 'unknown'
        
        print(f"Model loaded from {model_path}")
        if timestamp != 'unknown':
            print(f"Model timestamp: {timestamp}")
        
        return self.model
    
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
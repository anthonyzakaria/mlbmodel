"""
Enhanced MLB Weather Model implementation.
This file provides advanced modeling capabilities for MLB games with weather factors.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import warnings
from config import STADIUM_MAPPING

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

# Try to import various modeling libraries with fallbacks
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
    print("Using LightGBM for modeling")
except ImportError:
    HAS_LIGHTGBM = False
    
try:
    import xgboost as xgb
    HAS_XGBOOST = True
    if not HAS_LIGHTGBM:
        print("Using XGBoost for modeling")
except ImportError:
    HAS_XGBOOST = False

if not (HAS_LIGHTGBM or HAS_XGBOOST):
    # Fall back to sklearn models
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    print("Using scikit-learn for modeling")

# Import configuration
from config import DATA_DIR, ODDS_API_KEY, BALLPARK_FACTORS, WEATHER_THRESHOLDS

class MLBAdvancedModel:
    """Enhanced MLB Weather Model that incorporates advanced weather and ballpark factors."""
    
    def __init__(self):
        """Initialize the MLB Weather Model with advanced features."""
        # Model data
        self.merged_data = None
        self.model = None
        self.scaler = None
        self.features = None
        self.model_type = None
        self.timestamp = None
        
        # Set model type based on available libraries
        if HAS_LIGHTGBM:
            self.model_type = "lightgbm"
        elif HAS_XGBOOST:
            self.model_type = "xgboost"
        else:
            self.model_type = "sklearn"
    
    def prepare_features(self, data):
        """
        Prepare features for model training or prediction, ensuring all derived features are consistently created.
        
        Args:
            data (DataFrame): Raw data for feature preparation
            
        Returns:
            DataFrame: Data with all features prepared
        """
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Ensure all expected columns exist
        if 'temperature' in df.columns:
            # Add temperature based features
            df['temperature_squared'] = df['temperature'] ** 2
            df['is_hot'] = (df['temperature'] > 85).astype(int)
            df['is_cold'] = (df['temperature'] < 55).astype(int)
        else:
            # Add missing columns with default values
            df['temperature_squared'] = 0
            df['is_hot'] = 0
            df['is_cold'] = 0
        
        # Add humidity based features
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['temp_humidity_interaction'] = df['temperature'] * df['humidity'] / 100
        else:
            df['temp_humidity_interaction'] = 0
        
        # Add wind based features
        if 'wind_speed' in df.columns:
            df['high_wind'] = (df['wind_speed'] > 10).astype(int)
        else:
            df['high_wind'] = 0
        
        # Ensure all wind direction features exist
        wind_directions = ['wind_N', 'wind_NE', 'wind_E', 'wind_SE', 
                          'wind_S', 'wind_SW', 'wind_W', 'wind_NW']
        
        for wind_dir in wind_directions:
            if wind_dir not in df.columns:
                df[wind_dir] = 0
        
        # Add wind effect features if they don't exist
        if 'wind_effect' not in df.columns:
            df['wind_effect'] = 0
        if 'wind_blowing_out' not in df.columns:
            df['wind_blowing_out'] = 0
        if 'wind_blowing_in' not in df.columns:
            df['wind_blowing_in'] = 0
        if 'wind_blowing_crossfield' not in df.columns:
            df['wind_blowing_crossfield'] = 0
        
        # Add month if it doesn't exist
        if 'month' not in df.columns and 'date' in df.columns:
            df['month'] = pd.to_datetime(df['date']).dt.month
        elif 'month' not in df.columns:
            df['month'] = datetime.now().month  # Use current month as default
        
        # Add ballpark factor if missing
        if 'ballpark_factor' not in df.columns and 'stadium' in df.columns:
            df['ballpark_factor'] = df['stadium'].map(
                {name: factors['base'] for name, factors in BALLPARK_FACTORS.items()}
            ).fillna(1.0)
        elif 'ballpark_factor' not in df.columns:
            df['ballpark_factor'] = 1.0  # Default neutral park factor
        
        return df
    
    def add_interaction_features(self, data):
        """Add interaction features to capture complex relationships."""
        data['temp_humidity_interaction'] = data['temperature'] * data['humidity'] / 100
        data['wind_ballpark_interaction'] = data['wind_speed'] * data['ballpark_factor']
        return data

    def add_baseball_specific_features(self, data):
        """
        Add baseball-specific features such as pitcher stats, team batting stats, and bullpen performance.
        
        Args:
            data (DataFrame): Input data with game details
            
        Returns:
            DataFrame: Data with additional baseball-specific features
        """
        # Add pitcher stats (ERA, WHIP, strikeouts, etc.)
        if 'starting_pitcher_home' in data.columns and 'starting_pitcher_away' in data.columns:
            pitcher_stats = self.fetch_pitcher_stats()
            data = data.merge(pitcher_stats, left_on='starting_pitcher_home', right_on='pitcher_name', how='left')
            data = data.rename(columns=lambda col: f"home_{col}" if col not in ['pitcher_name'] else col)
            data = data.merge(pitcher_stats, left_on='starting_pitcher_away', right_on='pitcher_name', how='left')
            data = data.rename(columns=lambda col: f"away_{col}" if col not in ['pitcher_name'] else col)

        # Add team batting stats (OPS, runs per game, etc.)
        if 'home_team' in data.columns and 'away_team' in data.columns:
            team_stats = self.fetch_team_stats()
            data = data.merge(team_stats, left_on='home_team', right_on='team_abbr', how='left')
            data = data.rename(columns=lambda col: f"home_{col}" if col not in ['team_abbr'] else col)
            data = data.merge(team_stats, left_on='away_team', right_on='team_abbr', how='left')
            data = data.rename(columns=lambda col: f"away_{col}" if col not in ['team_abbr'] else col)

        # Add bullpen performance (ERA, WHIP, etc.)
        if 'home_team' in data.columns and 'away_team' in data.columns:
            bullpen_stats = self.fetch_bullpen_stats()
            data = data.merge(bullpen_stats, left_on='home_team', right_on='team_abbr', how='left')
            data = data.rename(columns=lambda col: f"home_{col}" if col not in ['team_abbr'] else col)
            data = data.merge(bullpen_stats, left_on='away_team', right_on='team_abbr', how='left')
            data = data.rename(columns=lambda col: f"away_{col}" if col not in ['team_abbr'] else col)

        # Add recent game performance (last 5 games)
        if 'home_team' in data.columns and 'away_team' in data.columns:
            recent_performance = self.fetch_recent_performance()
            data = data.merge(recent_performance, left_on='home_team', right_on='team_abbr', how='left')
            data = data.rename(columns=lambda col: f"home_{col}" if col not in ['team_abbr'] else col)
            data = data.merge(recent_performance, left_on='away_team', right_on='team_abbr', how='left')
            data = data.rename(columns=lambda col: f"away_{col}" if col not in ['team_abbr'] else col)

        return data

    def fetch_pitcher_stats(self):
        """
        Fetch pitcher statistics (e.g., ERA, WHIP, strikeouts).
        
        Returns:
            DataFrame: Pitcher stats
        """
        # Example: Load from a CSV or API
        pitcher_stats_file = f"{DATA_DIR}/pitcher_stats.csv"
        if os.path.exists(pitcher_stats_file):
            return pd.read_csv(pitcher_stats_file)
        else:
            print("Pitcher stats file not found. Please fetch and save pitcher stats.")
            return pd.DataFrame()

    def fetch_team_stats(self):
        """
        Fetch team batting statistics (e.g., OPS, runs per game).
        
        Returns:
            DataFrame: Team stats
        """
        # Example: Load from a CSV or API
        team_stats_file = f"{DATA_DIR}/team_stats.csv"
        if os.path.exists(team_stats_file):
            return pd.read_csv(team_stats_file)
        else:
            print("Team stats file not found. Please fetch and save team stats.")
            return pd.DataFrame()

    def fetch_bullpen_stats(self):
        """
        Fetch bullpen performance statistics (e.g., ERA, WHIP).
        
        Returns:
            DataFrame: Bullpen stats
        """
        # Example: Load from a CSV or API
        bullpen_stats_file = f"{DATA_DIR}/bullpen_stats.csv"
        if os.path.exists(bullpen_stats_file):
            return pd.read_csv(bullpen_stats_file)
        else:
            print("Bullpen stats file not found. Please fetch and save bullpen stats.")
            return pd.DataFrame()

    def fetch_recent_performance(self):
        """
        Fetch recent game performance for teams (e.g., last 5 games).
        
        Returns:
            DataFrame: Recent performance stats
        """
        # Example: Load from a CSV or API
        recent_performance_file = f"{DATA_DIR}/recent_performance.csv"
        if os.path.exists(recent_performance_file):
            return pd.read_csv(recent_performance_file)
        else:
            print("Recent performance file not found. Please fetch and save recent performance stats.")
            return pd.DataFrame()

    def train_model(self):
        """Train the MLB Weather Model using prepared features."""
        if self.merged_data is None:
            print("No data available for training.")
            return False

        # Prepare features
        data = self.prepare_features(self.merged_data)
        
        # Define base features - maintain consistent order
        self.features = [
            'temperature', 'humidity', 'wind_speed', 'pressure',
            'cloud_cover', 'precipitation', 'temperature_squared',
            'temp_humidity_interaction', 'ballpark_factor',
            'wind_N', 'wind_NE', 'wind_E', 'wind_SE',
            'wind_S', 'wind_SW', 'wind_W', 'wind_NW',
            'is_hot', 'is_cold', 'high_wind', 'month'
        ]
        
        # Filter to only available features
        self.features = [f for f in self.features if f in data.columns]
        
        print(f"Using {len(self.features)} features for training")
        
        # Prepare X and y
        X = data[self.features].fillna(0)
        y = data['total_runs'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize and train model based on available libraries
        if self.model_type == "lightgbm" and HAS_LIGHTGBM:
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'verbose': -1
            }
            train_data = lgb.Dataset(X_train_scaled, label=y_train)
            valid_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)
            
            callbacks = [
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
            
            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[valid_data],
                callbacks=callbacks
            )
            
            # Make predictions
            y_pred = self.model.predict(X_test_scaled)

        elif self.model_type == "xgboost" and HAS_XGBOOST:
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
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
            y_pred = self.model.predict(dtest)
            
        else:
            # Fall back to sklearn GradientBoostingRegressor
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=6,
                random_state=42
            )
            self.model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate over/under accuracy
        over_under_line = data['over_under_line'].values[-len(y_test):]  # Align with test set
        actual_over = y_test > over_under_line
        predicted_over = y_pred > over_under_line
        over_accuracy = np.mean(actual_over == predicted_over)

        print(f"\nModel Evaluation:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R-squared: {r2:.4f}")
        print(f"Over/Under Accuracy: {over_accuracy:.4f}")

        # Update timestamp and save
        self.timestamp = datetime.now()
        self.save_model()

        return True

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model performance.
        
        Args:
            X_test (DataFrame): Test features
            y_test (Series): Test target values
        """
        if self.model is None:
            print("Model not trained or loaded.")
            return None
        
        # Make predictions
        if self.model_type.lower() == 'xgboost':
            dtest = xgb.DMatrix(X_test)
            y_pred = self.model.predict(dtest)
        elif self.model_type.lower() == 'lightgbm':
            y_pred = self.model.predict(X_test)
        else:
            X_test_scaled = self.scaler.transform(X_test)
            y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Evaluation:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R-squared: {r2:.4f}")
        
        # Calculate over/under accuracy
        avg_line = X_test['over_under_line'].mean()
        
        over_indices = y_test > avg_line
        under_indices = y_test < avg_line
        
        if sum(over_indices) > 0:
            over_accuracy = sum((y_pred > avg_line) & over_indices) / sum(over_indices)
            print(f"Over Prediction Accuracy: {over_accuracy:.4f}")
            
        if sum(under_indices) > 0:
            under_accuracy = sum((y_pred < avg_line) & under_indices) / sum(under_indices)
            print(f"Under Prediction Accuracy: {under_accuracy:.4f}")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def save_model(self):
        """Save the trained model to disk."""
        if self.model is None:
            print("No model to save.")
            return False
        
        model_path = f"{DATA_DIR}/advanced_mlb_model.pkl"
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'features': self.features,
            'model_type': self.model_type,
            'timestamp': self.timestamp
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {model_path}")
        return True
    
    def load_model(self):
        """Load a trained model from disk."""
        model_path = f"{DATA_DIR}/advanced_mlb_model.pkl"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data.get('scaler')
        self.features = model_data.get('features')
        self.model_type = model_data.get('model_type')
        self.timestamp = model_data.get('timestamp')
        
        print(f"Model loaded from {model_path}")
        if self.timestamp:
            print(f"Model timestamp: {self.timestamp}")
        
        return True
    
    def recalibrate_predictions(self, X, y_actual):
        """Recalibrate predictions to correct for systematic bias."""
        y_predicted = self.model.predict(X)

        # Fit a linear regression to adjust predictions
        from sklearn.linear_model import LinearRegression
        recalibrator = LinearRegression()
        recalibrator.fit(y_predicted.reshape(-1, 1), y_actual)

        # Apply recalibration
        y_recalibrated = recalibrator.predict(y_predicted.reshape(-1, 1))
        return y_recalibrated

    def predict(self, features, recalibrate_to_vegas_mean=True, vegas_mean=8.59):
        """
        Make predictions using the trained model, with optional recalibration.
        """
        if self.model is None:
            print("Model not trained or loaded.")
            return None
        
        try:
            # Prepare features
            prepared_features = self.prepare_features(features)
            
            # Select only the features used in the model in the correct order
            model_features = prepared_features[self.features].copy()
            
            # Convert to numpy array if needed
            if isinstance(model_features, pd.DataFrame):
                model_features = model_features.values
                
            # Ensure we have a 2D array
            if len(model_features.shape) == 1:
                model_features = model_features.reshape(1, -1)
                
            # Scale features if a scaler exists
            if hasattr(self, 'scaler') and self.scaler is not None:
                model_features = self.scaler.transform(model_features)
            
            # Make predictions based on model type
            if self.model_type.lower() == 'lightgbm':
                predictions = self.model.predict(model_features)
            elif self.model_type.lower() == 'xgboost':
                dmatrix = xgb.DMatrix(model_features)
                predictions = self.model.predict(dmatrix)
            else:
                predictions = self.model.predict(model_features)
            
            # Recalibrate predictions if needed
            if recalibrate_to_vegas_mean and len(predictions) > 0:
                predictions = self._recalibrate_predictions(predictions, target_mean=vegas_mean)
            
            return predictions
            
        except Exception as e:
            print(f"Error making predictions: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _recalibrate_predictions(self, predictions, target_mean):
        """
        Internal method to recalibrate predictions to align with a target mean.
        
        Args:
            predictions (array): Original model predictions
            target_mean (float): Target mean to align predictions with
            
        Returns:
            array: Recalibrated predictions
        """
        if len(predictions) == 0:
            return predictions
            
        # Ensure predictions is a 1D array
        predictions = np.asarray(predictions).ravel()
        
        # Calculate adjustment
        original_mean = predictions.mean()
        if original_mean > 0:  # Avoid division by zero
            adjustment_factor = target_mean / original_mean
            recalibrated = predictions * adjustment_factor
        else:
            recalibrated = predictions
            
        print(f"Recalibrating predictions: Original Mean = {original_mean:.2f}, " 
              f"Target Mean = {target_mean:.2f}, "
              f"Adjustment Factor = {adjustment_factor:.4f}")
        
        return recalibrated

    def analyze_feature_importance(self):
        """
        Analyze and visualize feature importance.
        
        Returns:
            DataFrame: Feature importance
        """
        if self.model is None:
            print("Model not trained or loaded.")
            return None
        
        # Extract feature importance based on model type
        if self.model_type.lower() == 'lightgbm':
            importance = self.model.feature_importance()
            importance_df = pd.DataFrame({
                'Feature': self.features,
                'Importance': importance
            })
        elif self.model_type.lower() == 'xgboost':
            importance = self.model.get_score(importance_type='gain')
            importance_df = pd.DataFrame({
                'Feature': list(importance.keys()),
                'Importance': list(importance.values())
            })
        else:  # sklearn models
            importance = self.model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': self.features,
                'Importance': importance
            })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Visualize
        try:
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.savefig(f"{DATA_DIR}/feature_importance.png")
            plt.close()
        except Exception as e:
            print(f"Error creating visualization: {e}")
        
        return importance_df
    
    def find_betting_opportunities(self, confidence_threshold=0.6):
        """
        Find betting opportunities in historical data.
        
        Args:
            confidence_threshold (float): Threshold for confidence to consider a bet
            
        Returns:
            DataFrame: Betting opportunities
        """
        if self.model is None:
            print("Model not trained or loaded.")
            return None
        
        if self.merged_data is None or len(self.merged_data) == 0:
            print("No data available for analysis.")
            return None
        
        # Prepare the test data - use a portion not used in training
        # Use games from the most recent 20% of the dataset
        test_data = self.merged_data.sort_values('date').tail(int(len(self.merged_data) * 0.2)).copy()
        
        # Apply feature preparation
        test_data = self.prepare_features(test_data)
        
        # Make sure all features exist
        available_features = [f for f in self.features if f in test_data.columns]
        missing_features = [f for f in self.features if f not in test_data.columns]
        
        if len(missing_features) > 0:
            print(f"Warning: Using {len(available_features)} of {len(self.features)} model features")
            print(f"Warning: {len(missing_features)} features are missing.")
            
            # Add missing features with zeros
            for feature in missing_features:
                test_data[feature] = 0
        
        # Make sure all model features are present
        test_features = test_data[self.features]
        
        # Make predictions
        test_data['predicted_runs'] = self.predict(test_features)
        
        # Calculate edge and confidence
        test_data['predicted_vs_line'] = test_data['predicted_runs'] - test_data['over_under_line']
        
        # Calculate standard deviation of errors
        std_error = np.sqrt(mean_squared_error(
            test_data['total_runs'], 
            test_data['predicted_runs']
        ))
        
        # Calculate z-score of the edge
        test_data['edge_zscore'] = abs(test_data['predicted_vs_line']) / std_error
        
        # Calculate confidence (probability of correct side)
        from scipy.stats import norm
        test_data['confidence'] = test_data['edge_zscore'].apply(
            lambda z: norm.cdf(z)
        )
        
        # Determine bet direction
        test_data['bet_type'] = np.where(
            test_data['predicted_vs_line'] > 0, 
            'Over', 
            'Under'
        )
        
        # Determine if bet was correct
        test_data['bet_result'] = np.where(
            (test_data['bet_type'] == 'Over') & (test_data['total_runs'] > test_data['over_under_line']),
            1,  # Over bet wins
            np.where(
                (test_data['bet_type'] == 'Under') & (test_data['total_runs'] < test_data['over_under_line']),
                1,  # Under bet wins
                np.where(
                    test_data['total_runs'] == test_data['over_under_line'],
                    0,  # Push
                    -1  # Loss
                )
            )
        )
        
        # Filter for opportunities with high confidence
        opportunities = test_data[test_data['confidence'] >= confidence_threshold].copy()
        
        # Calculate overall win percentage
        if len(opportunities) > 0:
            win_pct = sum(opportunities['bet_result'] == 1) / len(opportunities)
            print(f"Found {len(opportunities)} potential betting opportunities with {win_pct:.1%} win rate")
        else:
            print("No betting opportunities found.")
        
        # Add weather impact assessment
        if 'temperature' in opportunities.columns:
            # Mark extreme weather conditions
            opportunities['extreme_temp'] = (
                (opportunities['temperature'] > 90) | 
                (opportunities['temperature'] < 50)
            )
            
            # Check if high winds
            if 'wind_speed' in opportunities.columns:
                opportunities['high_winds'] = opportunities['wind_speed'] > 15

        # Add more conservative filters
        opportunities = opportunities[
            (opportunities['confidence'] >= confidence_threshold) &
            (abs(opportunities['predicted_vs_line']) >= 1.0) &  # Minimum edge
            (opportunities['predicted_runs'] >= 5) &  # Realistic run totals
            (opportunities['predicted_runs'] <= 15)
        ].copy()

        # Add uncertainty estimates
        opportunities['prediction_std'] = self.estimate_prediction_uncertainty(opportunities)
        
        # Filter based on uncertainty
        opportunities = opportunities[
            opportunities['predicted_vs_line'].abs() > 2 * opportunities['prediction_std']
        ]

        # Calculate Kelly bet sizes
        if len(opportunities) > 0:
            opportunities['kelly_fraction'] = opportunities.apply(
                lambda x: self.calculate_kelly_fraction(
                    win_prob=x['confidence'],
                    odds=-110  # Standard odds
                ),
                axis=1
            )

            # Cap Kelly fractions
            opportunities['kelly_fraction'] = opportunities['kelly_fraction'].clip(0, 0.05)

        return opportunities

    def calculate_kelly_fraction(self, win_prob, odds):
        """Calculate optimal Kelly fraction."""
        # Convert American odds to decimal
        if odds > 0:
            decimal_odds = 1 + (odds / 100)
        else:
            decimal_odds = 1 + (100 / abs(odds))

        # Kelly formula
        q = 1 - win_prob
        return max(0, (win_prob * decimal_odds - q) / decimal_odds)

    def estimate_prediction_uncertainty(self, data):
        """Estimate uncertainty in predictions."""
        if not hasattr(self.model, 'feature_importance'):
            return pd.Series([1.0] * len(data))  # Default uncertainty

        # Use feature importances to weight feature contributions
        importance_sum = sum(self.model.feature_importance())
        normalized_importance = [imp / importance_sum for imp in self.model.feature_importance()]

        # Calculate weighted standard deviation of features
        weighted_std = np.zeros(len(data))
        for feat, weight in zip(self.features, normalized_importance):
            if feat in data.columns:
                weighted_std += weight * data[feat].std()

        return weighted_std

    def backtest_strategy(self, start_date=None, end_date=None, kelly=False):
        """
        Backtest the betting strategy.
        
        Args:
            start_date (str): Start date for backtest
            end_date (str): End date for backtest
            kelly (bool): Whether to use Kelly criterion for bet sizing
            
        Returns:
            DataFrame: Backtest results
        """
        # Find opportunities
        opportunities = self.find_betting_opportunities()
        
        if opportunities is None or len(opportunities) == 0:
            print("No opportunities found for backtesting.")
            return None
        
        # Filter by date range if provided
        if start_date:
            opportunities = opportunities[opportunities['date'] >= pd.to_datetime(start_date)]
        
        if end_date:
            opportunities = opportunities[opportunities['date'] <= pd.to_datetime(end_date)]
        
        # Sort chronologically
        opportunities = opportunities.sort_values('date')
        
        # Initialize bankroll
        starting_bankroll = 1000
        current_bankroll = starting_bankroll
        
        # Bet sizing
        if kelly:
            # Use Kelly criterion for bet sizing
            print("Using Kelly criterion for bet sizing...")
            
            # Calculate win probability and edge
            opportunities['win_prob'] = opportunities['confidence']
            opportunities['edge'] = (opportunities['win_prob'] * 1.9) - 1  # Assume -110 odds (1.91)
            
            # Calculate Kelly percentage (cap at 5%)
            opportunities['kelly_pct'] = np.clip(
                opportunities['win_prob'] - ((1 - opportunities['win_prob']) / 1.9),
                0,
                0.05
            )
            
            # Calculate bet amount
            opportunities['bet_amount'] = opportunities.apply(
                lambda row: row['kelly_pct'] * current_bankroll,
                axis=1
            )
        else:
            # Use flat betting (1% of bankroll)
            opportunities['bet_amount'] = starting_bankroll * 0.01
        
        # Calculate returns
        # Assume -110 odds (1.91 decimal)
        opportunities['bet_return'] = np.where(
            opportunities['bet_result'] == 1,  # Win
            opportunities['bet_amount'] * 0.91,  # Return = 1.91 - 1.0 (initial stake)
            np.where(
                opportunities['bet_result'] == 0,  # Push
                0,
                -opportunities['bet_amount']  # Loss
            )
        )
        
        # Calculate cumulative bankroll
        opportunities['bankroll'] = starting_bankroll + opportunities['bet_return'].cumsum()
        
        # Calculate performance metrics
        total_bets = len(opportunities)
        winning_bets = sum(opportunities['bet_result'] == 1)
        losing_bets = sum(opportunities['bet_result'] == -1)
        push_bets = sum(opportunities['bet_result'] == 0)
        
        win_pct = winning_bets / (winning_bets + losing_bets) if (winning_bets + losing_bets) > 0 else 0
        
        final_bankroll = opportunities['bankroll'].iloc[-1] if len(opportunities) > 0 else starting_bankroll
        roi = (final_bankroll - starting_bankroll) / starting_bankroll
        
        print("\nBacktest Results:")
        print(f"Period: {opportunities['date'].min()} to {opportunities['date'].max()}")
        print(f"Total Bets: {total_bets}")
        print(f"Wins: {winning_bets} ({win_pct:.1%})")
        print(f"Losses: {losing_bets}")
        print(f"Pushes: {push_bets}")
        print(f"Starting Bankroll: ${starting_bankroll}")
        print(f"Final Bankroll: ${final_bankroll:.2f}")
        print(f"ROI: {roi:.2%}")
        
        # Create chart of bankroll over time
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(opportunities['date'], opportunities['bankroll'])
            plt.axhline(y=starting_bankroll, color='r', linestyle='--')
            plt.title('Bankroll Over Time')
            plt.xlabel('Date')
            plt.ylabel('Bankroll ($)')
            plt.grid(True)
            plt.savefig(f"{DATA_DIR}/backtest_results.png")
            plt.close()
            
            # Create win rate by weather condition chart
            if 'extreme_temp' in opportunities.columns and 'high_winds' in opportunities.columns:
                # Group by weather conditions
                weather_groups = [
                    ('All Bets', opportunities),
                    ('Extreme Temp', opportunities[opportunities['extreme_temp']]),
                    ('Normal Temp', opportunities[~opportunities['extreme_temp']]),
                    ('High Winds', opportunities[opportunities['high_winds']]),
                    ('Low Winds', opportunities[~opportunities['high_winds']])
                ]
                
                # Calculate win rates
                win_rates = []
                for name, group in weather_groups:
                    if len(group) > 0:
                        win_rate = sum(group['bet_result'] == 1) / len(group)
                        win_rates.append({
                            'Condition': name,
                            'Win Rate': win_rate,
                            'Count': len(group)
                        })
                
                # Create chart
                win_rate_df = pd.DataFrame(win_rates)
                
                plt.figure(figsize=(10, 6))
                ax = sns.barplot(x='Condition', y='Win Rate', data=win_rate_df)
                
                # Add count labels
                for i, row in enumerate(win_rate_df.itertuples()):
                    ax.text(i, row._2 + 0.01, f"n={row.Count}", ha='center')
                
                plt.title('Win Rate by Weather Condition')
                plt.ylabel('Win Rate')
                plt.ylim(0, 1)
                plt.grid(True, axis='y')
                plt.savefig(f"{DATA_DIR}/weather_win_rates.png")
                plt.close()
        except Exception as e:
            print(f"Error creating visualization: {e}")
        
        return opportunities
    
    def get_todays_betting_recommendations(self, confidence_threshold=0.6):
        """
        Get today's MLB betting recommendations.
        
        Args:
            confidence_threshold (float): Threshold for confidence to consider a bet
            
        Returns:
            DataFrame: Today's betting opportunities
        """
        if self.model is None:
            print("Model not trained or loaded.")
            return None
        
        # Get today's games and odds
        today_games = self.fetch_todays_odds()
        
        if today_games is None or len(today_games) == 0:
            print("No games found for today.")
            return None
        
        # Get weather for game locations
        try:
            today_games = self.add_weather_to_games(today_games)
        except Exception as e:
            print(f"Error adding weather data: {e}")
        
        # Prepare the data for prediction
        today_features = self.prepare_features(today_games)
        
        # Make sure all features exist
        available_features = [f for f in self.features if f in today_features.columns]
        missing_features = [f for f in self.features if f not in today_features.columns]
        
        if len(missing_features) > 0:
            print(f"Warning: Using {len(available_features)} of {len(self.features)} model features")
            if len(missing_features) > 5:
                print(f"Warning: {len(missing_features)} features are missing.")
            else:
                print(f"Warning: {len(missing_features)} features are missing: {missing_features}")
            
            # Add missing features with zeros
            for feature in missing_features:
                today_features[feature] = 0
        
        # Make sure all model features are present
        today_model_features = today_features[self.features]
        
        # Make predictions
        try:
            today_games['predicted_runs'] = self.predict(today_features)
            
            # Calculate edge and confidence
            today_games['predicted_vs_line'] = today_games['predicted_runs'] - today_games['over_under_line']
            
            # Calculate standard deviation of errors (use a reasonable estimate)
            std_error = 2.5  # Typical RMSE for run prediction
            
            # Calculate z-score of the edge
            today_games['edge_zscore'] = abs(today_games['predicted_vs_line']) / std_error
            
            # Calculate confidence (probability of correct side)
            from scipy.stats import norm
            today_games['confidence'] = today_games['edge_zscore'].apply(
                lambda z: norm.cdf(z)
            )
            
            # Determine bet direction
            today_games['bet_type'] = np.where(
                today_games['predicted_vs_line'] > 0, 
                'Over', 
                'Under'
            )
            
            # Format for better readability
            today_games['confidence_pct'] = (today_games['confidence'] * 100).round(1)
            today_games['prediction'] = today_games['predicted_runs'].round(1)
            
            # Filter for opportunities with high confidence
            opportunities = today_games[today_games['confidence'] >= confidence_threshold].copy()
            
            if len(opportunities) > 0:
                print(f"Found {len(opportunities)} potential betting opportunities for today")
                
                # Sort by confidence
                opportunities = opportunities.sort_values('confidence', ascending=False)
                
                # Add a rating system (5★ = highest confidence)
                opportunities['stars'] = pd.cut(
                    opportunities['confidence'],
                    bins=[confidence_threshold, 0.7, 0.8, 0.9, 1.0],
                    labels=['2★', '3★', '4★', '5★'],
                    include_lowest=True
                )
                
                # Print recommendations
                print("\nToday's Betting Recommendations:")
                for _, game in opportunities.iterrows():
                    stars = game.get('stars', '')
                    print(f"{game['away_team']} @ {game['home_team']} - {game['bet_type']} {game['over_under_line']} ({game['confidence_pct']}% confidence) {stars}")
                    print(f"  Predicted: {game['prediction']} runs | Weather: {game.get('weather_description', 'N/A')}")
                    print()
            else:
                print("No betting opportunities found for today that meet the confidence threshold.")
            
            return opportunities
            
        except Exception as e:
            print(f"Error fetching odds: {e}")
            return None
    
    def fetch_todays_odds(self):
        """
        Fetch today's MLB games and betting odds.
        
        Returns:
            DataFrame: Today's games with odds
        """
        print("Fetching today's MLB odds...")
        
        # API parameters
        api_key = ODDS_API_KEY
        sport = 'baseball_mlb'
        regions = 'us'
        markets = 'totals'
        oddsFormat = 'american'
        date_format = 'iso'
        
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
            print(f"Failed to get odds: status_code {odds_response.status_code}, response body {odds_response.text}")
            return None
        
        odds_json = odds_response.json()
        print(f"API Response: {odds_response.status_code}")
        
        # Process the odds data
        games_list = []
        
        for game in odds_json:
            try:
                # Get teams
                home_team = None
                away_team = None
                
                for team in game['home_team'], game['away_team']:
                    # Map team name to abbreviation
                    abbr = self.get_team_abbreviation(team)
                    
                    if team == game['home_team']:
                        home_team = abbr
                    else:
                        away_team = abbr
                
                # Get totals (over/under)
                over_under_line = None
                over_odds = None
                under_odds = None
                
                if 'bookmakers' in game and len(game['bookmakers']) > 0:
                    # Use first bookmaker with totals market
                    for bookie in game['bookmakers']:
                        for market in bookie['markets']:
                            if (market['key'] == 'totals'):
                                for outcome in market['outcomes']:
                                    if outcome['name'] == 'Over':
                                        over_under_line = float(outcome['point'])
                                        over_odds = int(outcome['price'])
                                    elif outcome['name'] == 'Under':
                                        under_odds = int(outcome['price'])
                                break
                        if over_under_line is not None:
                            break
                
                # Get game date/time
                game_date = datetime.fromisoformat(game['commence_time'].replace('Z', '+00:00'))
                
                # Convert to local time
                game_date = game_date.replace(tzinfo=None) - timedelta(hours=4)  # Rough EST conversion
                
                # Get stadium info
                stadium_info = self.get_stadium_info(home_team)
                
                # Create game object
                if home_team and away_team and over_under_line:
                    games_list.append({
                        'date': game_date,
                        'home_team': home_team,
                        'away_team': away_team,
                        'over_under_line': over_under_line,
                        'over_odds': over_odds,
                        'under_odds': under_odds,
                        'stadium': stadium_info.get('name', f"{home_team} Stadium"),
                        'stadium_lat': stadium_info.get('lat'),
                        'stadium_lon': stadium_info.get('lon'),
                        'game_id': game['id']
                    })
            except Exception as e:
                print(f"Error processing game: {e}")
        
        # Create DataFrame
        if games_list:
            games_df = pd.DataFrame(games_list)
            print(f"Found {len(games_df)} MLB games with odds")
            return games_df
        else:
            print("No games with odds found")
            return None
    
    def add_weather_to_games(self, games_df):
        """
        Add weather data to games DataFrame.
        
        Args:
            games_df (DataFrame): Games data
            
        Returns:
            DataFrame: Games data with weather information
        """
        if games_df is None or len(games_df) == 0:
            return games_df
        
        # Add weather data
        try:
            # Import weather module dynamically
            from weather_data import get_current_weather
            
            # Add weather for each game
            weather_list = []
            
            for idx, game in games_df.iterrows():
                try:
                    # Get weather for stadium at game time
                    weather = get_current_weather(
                        game['home_team'], 
                        game_datetime=game['date']
                    )
                    
                    if (weather):
                        weather_list.append(weather)
                    else:
                        # Create empty weather data
                        weather_list.append({
                            'temperature': 70,
                            'humidity': 50,
                            'wind_speed': 5,
                            'wind_direction': 0,
                            'pressure': 1010,
                            'precipitation': 0,
                            'cloud_cover': 30,
                            'weather_condition': 'Unknown',
                            'weather_description': 'Weather data unavailable'
                        })
                except Exception as e:
                    print(f"Error getting weather for {game['home_team']}: {e}")
                    # Create empty weather data
                    weather_list.append({
                        'temperature': 70,
                        'humidity': 50,
                        'wind_speed': 5,
                        'wind_direction': 0,
                        'pressure': 1010,
                        'precipitation': 0,
                        'cloud_cover': 30,
                        'weather_condition': 'Unknown',
                        'weather_description': 'Weather data unavailable'
                    })
            
            # Convert to DataFrame
            weather_df = pd.DataFrame(weather_list)
            
            # Join with games data
            for col in weather_df.columns:
                if col not in games_df.columns:
                    games_df[col] = weather_df[col].values
            
            # Add wind cardinal direction
            if 'wind_direction' in games_df.columns:
                def degrees_to_cardinal(deg):
                    dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
                    ix = round(float(deg) / 45) % 8
                    return dirs[ix]
                
                games_df['wind_cardinal'] = games_df['wind_direction'].apply(
                    lambda x: degrees_to_cardinal(x) if pd.notnull(x) else 'Unknown'
                )
                
                # Create dummy variables for wind direction
                wind_dummies = pd.get_dummies(games_df['wind_cardinal'], prefix='wind')
                
                # Add columns to games_df
                for col in wind_dummies.columns:
                    games_df[col] = wind_dummies[col]
        
        except Exception as e:
            print(f"Error adding weather data: {e}")
        
        return games_df
    
    def get_team_abbreviation(self, team_name):
        """
        Get team abbreviation from full team name.
        
        Args:
            team_name (str): Full team name
            
        Returns:
            str: Team abbreviation
        """
        # Dictionary mapping full team names to abbreviations
        team_map = {
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
        
        # Handle common variations
        variations = {
            'Cleveland Indians': 'CLE',
            'Cleveland Guardians': 'CLE',
            'LA Angels': 'LAA',
            'LA Dodgers': 'LAD',
            'St Louis Cardinals': 'STL',
            'St.Louis Cardinals': 'STL',
            'Tampa Bay': 'TBR'
        }
        
        # Check direct match
        if team_name in team_map:
            return team_map[team_name]
        
        # Check variations
        if team_name in variations:
            return variations[team_name]
        
        # Try to match partial names
        for abbr in TEAM_NAME_MAP.keys():
            if abbr in team_name:
                return abbr
        
        # If all else fails, try fuzzy matching
        from difflib import get_close_matches
        matches = get_close_matches(team_name, list(TEAM_NAME_MAP.keys()) + list(TEAM_NAME_MAP.values()))
        if matches:
            match = matches[0]
            if match in TEAM_NAME_MAP:
                return match
            for abbr, name in TEAM_NAME_MAP.items():
                if name == match:
                    return abbr
        
        return team_name[:3].upper()  # Last resort

    def get_stadium_info(self, home_team):
        """Get stadium information for a team."""
        if home_team in STADIUM_MAPPING:
            return STADIUM_MAPPING[home_team]

        # Try to infer team from stadium name
        for team, info in STADIUM_MAPPING.items():
            if team in home_team or home_team in team:
                return info

        # Default values if team not found
        return {
            'name': f"{home_team} Stadium",
            'lat': 40.0,  # Default latitude
            'lon': -75.0,  # Default longitude
            'dome': False,
            'retractable_roof': False,
            'orientation': 0
        }

    def __str__(self):
        """String representation of model state."""
        status = []
        status.append(f"Model Type: {self.model_type}")
        status.append(f"Features: {len(self.features) if self.features else 0}")

        status.append(f"Data Records: {len(self.merged_data) if self.merged_data is not None else 0}")
        status.append(f"Model Trained: {self.model is not None}")
        if self.timestamp:
            status.append(f"Last Updated: {self.timestamp}")
        return "\n".join(status)
        status.append(f"Features: {len(self.features) if self.features else 0}")
        status.append(f"Data Records: {len(self.merged_data) if self.merged_data is not None else 0}")
        status.append(f"Model Trained: {self.model is not None}")
        if self.timestamp:
            status.append(f"Last Updated: {self.timestamp}")
        return "\n".join(status)
        # Calculate bias
        bias = y_predicted - y_actual
        print(f"Mean Bias: {bias.mean():.2f}")
        print(f"Median Bias: {np.median(bias):.2f}")

        # Visualize
        plt.figure(figsize=(10, 6))
        sns.histplot(bias, kde=True, bins=30)
        plt.title("Prediction Bias (Predicted - Actual Runs)")
        plt.xlabel("Bias")
        plt.ylabel("Frequency")
        if self.timestamp:
            status.append(f"Last Updated: {self.timestamp}")
        return "\n".join(status)
        plt.grid(True)
        status.append(f"Features: {len(self.features) if self.features else 0}")
        status.append(f"Data Records: {len(self.merged_data) if self.merged_data is not None else 0}")
        status.append(f"Model Trained: {self.model is not None}")
        plt.show()

    def __str__(self):
        """String representation of model state."""
        status = []
        status.append(f"Model Type: {self.model_type}")
    def recalibrate_predictions(self, X, y_actual):
        """Recalibrate predictions to correct for systematic bias."""
        y_predicted = self.model.predict(X)

        # Fit a linear regression to adjust predictions
        from sklearn.linear_model import LinearRegression
        recalibrator = LinearRegression()
        recalibrator.fit(y_predicted.reshape(-1, 1), y_actual)

        # Apply recalibration
        y_recalibrated = recalibrator.predict(y_predicted.reshape(-1, 1))
        return y_recalibrated
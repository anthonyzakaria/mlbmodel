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
from datetime import datetime

from mlb_weather_model import MLBWeatherModel
from visualization import visualize_mlb_bets, visualize_weather_impact
from config import DATA_DIR

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MLB Weather Model for betting predictions')
    
    parser.add_argument('--fetch-data', action='store_true', 
                        help='Fetch new MLB game data')
    parser.add_argument('--start-year', type=int, default=2024, 
                        help='Starting year for data collection')
    parser.add_argument('--end-year', type=int, default=2024, 
                        help='Ending year for data collection')
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

def main():
    """Main function to run the MLB Weather Model."""
    args = parse_args()
    
    print("MLB Weather Model")
    print("-" * 30)
    
    # Initialize the model
    mlb_model = MLBWeatherModel()
    print("Model initialized successfully!")
    
    # Fetch data if requested
    if args.fetch_data:
        print(f"\nFetching MLB game data from {args.start_year} to {args.end_year}...")
        mlb_model.fetch_game_data(args.start_year, args.end_year)
        
        print("\nFetching weather data...")
        mlb_model.fetch_weather_data()
        
        print("\nFetching betting odds...")
        mlb_model.fetch_odds_data()
        
        print("\nMerging datasets...")
        mlb_model.merge_data()
    
    # Train model if requested
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
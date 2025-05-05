"""
Visualization utilities for the MLB Weather Model
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def visualize_mlb_bets(betting_df):
    """Create a visualization of MLB betting recommendations.
    
    Args:
        betting_df (DataFrame): DataFrame containing betting recommendations
    """
    if betting_df is None or len(betting_df) == 0:
        print("No betting data available.")
        return
    
    # Sort by confidence for better visualization
    betting_df = betting_df.sort_values('confidence', ascending=False).reset_index(drop=True)
    
    # Set up the style
    plt.style.use('fivethirtyeight')
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('MLB Weather Model - Betting Recommendations', fontsize=20)
    
    # 1. Bar chart of predicted runs vs. Vegas line
    betting_df['matchup'] = betting_df['away_team'] + ' @ ' + betting_df['home_team']
    
    # Set colors based on recommendation
    colors = ['#1e88e5' if bet == 'OVER' else '#e53935' for bet in betting_df['recommended_bet']]
    
    # Create the bar chart
    ax1.bar(betting_df['matchup'], betting_df['predicted_runs'], color=colors, alpha=0.7, label='Predicted')
    ax1.plot(betting_df['matchup'], betting_df['over_under_line'], 'ko--', linewidth=2, markersize=8, label='Vegas Line')
    
    # Add labels
    for i, (idx, row) in enumerate(betting_df.iterrows()):
        diff = row['predicted_runs'] - row['over_under_line']
        y_pos = max(row['predicted_runs'], row['over_under_line']) + 0.3
        ax1.text(i, y_pos, f"{diff:.1f}", ha='center', fontweight='bold')
    
    # Format the first plot
    ax1.set_title('Predicted Runs vs. Vegas Line')
    ax1.set_ylabel('Total Runs')
    ax1.set_xticklabels(betting_df['matchup'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Horizontal bar chart for confidence levels
    ax2.barh(betting_df['matchup'], betting_df['confidence'], color=colors, alpha=0.7)
    ax2.set_title('Confidence Level (Difference in Runs)')
    ax2.set_xlabel('Confidence')
    ax2.grid(True, alpha=0.3)
    
    # Add confidence values as text
    for i, v in enumerate(betting_df['confidence']):
        ax2.text(v + 0.05, i, f"{v:.2f}", va='center')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Display the best bet recommendation in text
    if len(betting_df) > 0:
        best_bet = betting_df.iloc[0]
        print("\nBEST BET OF THE DAY:")
        print(f"{best_bet['away_team']} @ {best_bet['home_team']}")
        print(f"Line: {best_bet['over_under_line']}")
        print(f"Model Prediction: {best_bet['predicted_runs']:.2f} runs")
        print(f"Recommendation: {best_bet['recommended_bet']}")
        print(f"Confidence: {best_bet['confidence']:.2f} runs difference")
    
    # Show the plot
    plt.show()
    
    # Create a simple table of all bets
    display_df = betting_df[['matchup', 'over_under_line', 'predicted_runs', 
                           'recommended_bet', 'confidence']].copy()
    
    # Calculate the difference column
    display_df['difference'] = display_df['predicted_runs'] - display_df['over_under_line']
    display_df['difference'] = display_df['difference'].round(2)
    display_df['predicted_runs'] = display_df['predicted_runs'].round(2)
    
    # Display the table
    print("\nAll MLB Betting Recommendations:")
    print(display_df.to_string(index=False))

def visualize_weather_impact(opportunities):
    """
    Visualize the impact of weather conditions on model performance.
    
    Args:
        opportunities (DataFrame): DataFrame containing betting opportunities with weather data
    """
    if opportunities is None or len(opportunities) == 0:
        print("No opportunities data available.")
        return
    
    # Analyze temperature impact
    plt.figure(figsize=(12, 6))
    
    # Temperature vs Run Scoring
    plt.subplot(1, 2, 1)
    plt.scatter(opportunities['temperature'], opportunities['total_runs'], alpha=0.6)
    plt.title('Temperature vs Actual Runs')
    plt.xlabel('Temperature (°F)')
    plt.ylabel('Total Runs')
    plt.grid(True)
    
    # Temperature vs Prediction Accuracy
    plt.subplot(1, 2, 2)
    # Calculate prediction error
    opportunities['pred_error'] = abs(opportunities['predicted_runs'] - opportunities['total_runs'])
    
    # Bin temperature into ranges
    temp_bins = [50, 60, 70, 80, 90, 100]
    opportunities['temp_bin'] = pd.cut(opportunities['temperature'], temp_bins)
    
    # Calculate average error by temperature bin
    error_by_temp = opportunities.groupby('temp_bin')['pred_error'].mean().reset_index()
    
    # Plot
    sns.barplot(x='temp_bin', y='pred_error', data=error_by_temp)
    plt.title('Prediction Error by Temperature Range')
    plt.xlabel('Temperature Range (°F)')
    plt.ylabel('Average Prediction Error (Runs)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Wind conditions impact
    if 'wind_speed' in opportunities.columns:
        plt.figure(figsize=(10, 6))
        wind_bins = [0, 5, 10, 15, 20, 30]
        opportunities['wind_bin'] = pd.cut(opportunities['wind_speed'], wind_bins)
        win_rate_by_wind = opportunities.groupby('wind_bin')['bet_correct'].mean().reset_index()
        
        sns.barplot(x='wind_bin', y='bet_correct', data=win_rate_by_wind)
        plt.title('Betting Success Rate by Wind Speed')
        plt.xlabel('Wind Speed (mph)')
        plt.ylabel('Win Rate')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()
    
    # Weather condition impact
    if 'weather_condition' in opportunities.columns:
        plt.figure(figsize=(10, 6))
        condition_counts = opportunities['weather_condition'].value_counts()
        weather_to_include = condition_counts[condition_counts >= 5].index.tolist()
        
        weather_impact = opportunities[opportunities['weather_condition'].isin(weather_to_include)]
        if len(weather_impact) > 0:
            weather_success = weather_impact.groupby('weather_condition')['bet_correct'].mean().reset_index()
            weather_success = weather_success.sort_values('bet_correct', ascending=False)
            
            sns.barplot(x='weather_condition', y='bet_correct', data=weather_success)
            plt.title('Betting Success Rate by Weather Condition')
            plt.xlabel('Weather Condition')
            plt.ylabel('Win Rate')
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
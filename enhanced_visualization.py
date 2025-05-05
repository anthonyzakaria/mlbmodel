"""
Enhanced data visualizations for MLB Weather Model.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

from config import DATA_DIR, STADIUM_MAPPING

# Set default style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Custom color palette
COLORS = {
    'over': '#1e88e5',  # Blue
    'under': '#e53935',  # Red
    'push': '#9e9e9e',   # Gray
    'win': '#4caf50',    # Green
    'loss': '#f44336',   # Red
    'text': '#212121',   # Dark gray
    'grid': '#e0e0e0',   # Light gray
    'background': '#f5f5f5'  # Very light gray
}

def visualize_betting_opportunities(betting_df, results=None, title=None, save=False):
    """
    Create a comprehensive visualization of MLB betting recommendations.
    
    Args:
        betting_df (DataFrame): DataFrame containing betting recommendations
        results (dict): Optional backtest results
        title (str): Custom title for the plot
        save (bool): Whether to save the plot to file
    """
    if betting_df is None or len(betting_df) == 0:
        print("No betting data available to visualize.")
        return
    
    # Sort by confidence for better visualization
    betting_df = betting_df.sort_values('confidence', ascending=False).reset_index(drop=True)
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1.5])
    
    # Create matchup labels
    betting_df['matchup'] = betting_df['away_team'] + ' @ ' + betting_df['home_team']
    
    # Set colors based on recommendation
    colors = [COLORS['over'] if bet == 'OVER' else COLORS['under'] for bet in betting_df['recommended_bet']]
    
    # 1. Bar chart of predicted runs vs. Vegas line
    ax1 = plt.subplot(gs[0, :])
    
    # Create the bar chart
    bars = ax1.bar(betting_df['matchup'], betting_df['predicted_runs'], color=colors, alpha=0.7)
    ax1.plot(betting_df['matchup'], betting_df['over_under_line'], 'ko--', linewidth=2, markersize=8, label='Vegas Line')
    
    # Add labels
    for i, (idx, row) in enumerate(betting_df.iterrows()):
        diff = row['predicted_runs'] - row['over_under_line']
        y_pos = max(row['predicted_runs'], row['over_under_line']) + 0.3
        ax1.text(i, y_pos, f"{diff:.1f}", ha='center', fontweight='bold', color=COLORS['text'])
    
    # Format the plot
    ax1.set_title('Predicted Runs vs. Vegas Line', fontsize=14)
    ax1.set_ylabel('Total Runs', fontsize=12)
    ax1.set_xticklabels(betting_df['matchup'], rotation=45, ha='right')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, color=COLORS['grid'])
    
    # 2. Horizontal bar chart for confidence levels
    ax2 = plt.subplot(gs[1, :])
    bars2 = ax2.barh(betting_df['matchup'], betting_df['confidence'], color=colors, alpha=0.7)
    
    # Add confidence values as text
    for i, v in enumerate(betting_df['confidence']):
        ax2.text(v + 0.05, i, f"{v:.2f}", va='center', color=COLORS['text'])
    
    ax2.set_title('Confidence Level (Difference in Runs)', fontsize=14)
    ax2.set_xlabel('Confidence', fontsize=12)
    ax2.grid(True, alpha=0.3, color=COLORS['grid'])
    
    # 3. Additional charts
    # Temperature vs. prediction
    ax3 = plt.subplot(gs[2, 0])
    
    if 'temperature' in betting_df.columns:
        ax3.scatter(betting_df['temperature'], betting_df['predicted_runs'], 
                  c=colors, s=100, alpha=0.7, edgecolor='white')
        
        # Add linear trend
        if len(betting_df) > 1:
            z = np.polyfit(betting_df['temperature'], betting_df['predicted_runs'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(betting_df['temperature'].min(), betting_df['temperature'].max(), 100)
            ax3.plot(x_trend, p(x_trend), 'k--', alpha=0.6)
            
            # Add correlation coefficient
            corr = betting_df['temperature'].corr(betting_df['predicted_runs'])
            ax3.text(0.05, 0.95, f"Correlation: {corr:.2f}", transform=ax3.transAxes,
                   fontsize=10, va='top', ha='left', bbox=dict(boxstyle='round', alpha=0.1))
        
        ax3.set_title('Temperature Impact', fontsize=14)
        ax3.set_xlabel('Temperature (°F)', fontsize=12)
        ax3.set_ylabel('Predicted Runs', fontsize=12)
        ax3.grid(True, alpha=0.3, color=COLORS['grid'])
    else:
        ax3.text(0.5, 0.5, "Temperature data not available", 
               ha='center', va='center', transform=ax3.transAxes, fontsize=12)
    
    # Wind speed vs. prediction
    ax4 = plt.subplot(gs[2, 1])
    
    if 'wind_speed' in betting_df.columns:
        ax4.scatter(betting_df['wind_speed'], betting_df['predicted_runs'], 
                  c=colors, s=100, alpha=0.7, edgecolor='white')
        
        # Add linear trend
        if len(betting_df) > 1:
            z = np.polyfit(betting_df['wind_speed'], betting_df['predicted_runs'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(betting_df['wind_speed'].min(), betting_df['wind_speed'].max(), 100)
            ax4.plot(x_trend, p(x_trend), 'k--', alpha=0.6)
            
            # Add correlation coefficient
            corr = betting_df['wind_speed'].corr(betting_df['predicted_runs'])
            ax4.text(0.05, 0.95, f"Correlation: {corr:.2f}", transform=ax4.transAxes,
                   fontsize=10, va='top', ha='left', bbox=dict(boxstyle='round', alpha=0.1))
        
        ax4.set_title('Wind Speed Impact', fontsize=14)
        ax4.set_xlabel('Wind Speed (mph)', fontsize=12)
        ax4.set_ylabel('Predicted Runs', fontsize=12)
        ax4.grid(True, alpha=0.3, color=COLORS['grid'])
    else:
        ax4.text(0.5, 0.5, "Wind data not available", 
               ha='center', va='center', transform=ax4.transAxes, fontsize=12)
    
    # Prediction distribution
    ax5 = plt.subplot(gs[2, 2])
    
    sns.histplot(betting_df['predicted_runs'], bins=10, kde=True, color='gray', ax=ax5)
    
    # Add mean line
    mean_pred = betting_df['predicted_runs'].mean()
    ax5.axvline(mean_pred, color='k', linestyle='--', alpha=0.6, 
              label=f'Mean: {mean_pred:.2f}')
    
    # Add Vegas mean line if available
    if 'over_under_line' in betting_df.columns:
        mean_line = betting_df['over_under_line'].mean()
        ax5.axvline(mean_line, color='r', linestyle=':', alpha=0.6,
                  label=f'Vegas Mean: {mean_line:.2f}')
    
    ax5.set_title('Run Prediction Distribution', fontsize=14)
    ax5.set_xlabel('Predicted Runs', fontsize=12)
    ax5.set_ylabel('Frequency', fontsize=12)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, color=COLORS['grid'])
    
    # Set custom title or default
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    else:
        fig.suptitle('MLB Weather Model - Betting Recommendations', fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Display the best bet recommendation
    if len(betting_df) > 0:
        best_bet = betting_df.iloc[0]
        print("\nBEST BET:")
        print(f"{best_bet['away_team']} @ {best_bet['home_team']}")
        print(f"Line: {best_bet['over_under_line']}")
        print(f"Model Prediction: {best_bet['predicted_runs']:.2f} runs")
        print(f"Recommendation: {best_bet['recommended_bet']}")
        print(f"Confidence: {best_bet['confidence']:.2f} runs difference")
    
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
    
    # Save if requested
    if save:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{DATA_DIR}/betting_recommendations_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {filename}")
    
    # Show the plot
    plt.show()

def visualize_weather_impact(opportunities, results=None, save=False):
    """
    Visualize the impact of weather conditions on betting outcomes.
    
    Args:
        opportunities (DataFrame): DataFrame with betting opportunities and results
        results (dict): Optional results from weather pattern analysis
        save (bool): Whether to save the plot to file
    """
    if opportunities is None or len(opportunities) == 0:
        print("No opportunities data available to visualize.")
        return
    
    # Check if we have the necessary columns to show results
    has_results = 'total_runs' in opportunities.columns and 'over_under_line' in opportunities.columns
    
    if has_results:
        # Calculate if bet was correct
        opportunities['over_result'] = (opportunities['total_runs'] > opportunities['over_under_line']).astype(int)
        opportunities['under_result'] = (opportunities['total_runs'] < opportunities['over_under_line']).astype(int)
        opportunities['push_result'] = (opportunities['total_runs'] == opportunities['over_under_line']).astype(int)
        
        opportunities['bet_correct'] = (
            ((opportunities['recommended_bet'] == 'OVER') & (opportunities['over_result'] == 1)) |
            ((opportunities['recommended_bet'] == 'UNDER') & (opportunities['under_result'] == 1))
        ).astype(int)
        
        opportunities['bet_incorrect'] = (
            ((opportunities['recommended_bet'] == 'OVER') & (opportunities['under_result'] == 1)) |
            ((opportunities['recommended_bet'] == 'UNDER') & (opportunities['over_result'] == 1))
        ).astype(int)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3)
    
    # 1. Temperature vs. Run Scoring
    ax1 = plt.subplot(gs[0, 0])
    
    if 'temperature' in opportunities.columns and 'total_runs' in opportunities.columns:
        scatter = ax1.scatter(
            opportunities['temperature'], 
            opportunities['total_runs'],
            c=opportunities['bet_correct'] if has_results else 'blue',
            alpha=0.7,
            cmap='RdYlGn',
            s=70,
            edgecolor='white'
        )
        
        # Add trend line
        z = np.polyfit(opportunities['temperature'], opportunities['total_runs'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(opportunities['temperature'].min(), opportunities['temperature'].max(), 100)
        ax1.plot(x_trend, p(x_trend), 'k--', alpha=0.6)
        
        # Show correlation
        corr = opportunities['temperature'].corr(opportunities['total_runs'])
        ax1.text(0.05, 0.95, f"Correlation: {corr:.2f}", transform=ax1.transAxes,
               fontsize=10, va='top', ha='left', bbox=dict(boxstyle='round', alpha=0.1))
        
        ax1.set_title('Temperature vs. Actual Runs', fontsize=14)
        ax1.set_xlabel('Temperature (°F)', fontsize=12)
        ax1.set_ylabel('Total Runs', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        if has_results:
            # Add legend for correct/incorrect bets
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, alpha=0.7, label='Correct Bet'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, alpha=0.7, label='Incorrect Bet')
            ]
            ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)
    else:
        ax1.text(0.5, 0.5, "Temperature or run data not available", 
               ha='center', va='center', transform=ax1.transAxes, fontsize=12)
    
    # 2. Wind Speed vs. Run Scoring
    ax2 = plt.subplot(gs[0, 1])
    
    if 'wind_speed' in opportunities.columns and 'total_runs' in opportunities.columns:
        scatter = ax2.scatter(
            opportunities['wind_speed'], 
            opportunities['total_runs'],
            c=opportunities['bet_correct'] if has_results else 'blue',
            alpha=0.7,
            cmap='RdYlGn',
            s=70,
            edgecolor='white'
        )
        
        # Add trend line
        z = np.polyfit(opportunities['wind_speed'], opportunities['total_runs'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(opportunities['wind_speed'].min(), opportunities['wind_speed'].max(), 100)
        ax2.plot(x_trend, p(x_trend), 'k--', alpha=0.6)
        
        # Show correlation
        corr = opportunities['wind_speed'].corr(opportunities['total_runs'])
        ax2.text(0.05, 0.95, f"Correlation: {corr:.2f}", transform=ax2.transAxes,
               fontsize=10, va='top', ha='left', bbox=dict(boxstyle='round', alpha=0.1))
        
        ax2.set_title('Wind Speed vs. Actual Runs', fontsize=14)
        ax2.set_xlabel('Wind Speed (mph)', fontsize=12)
        ax2.set_ylabel('Total Runs', fontsize=12)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Wind speed or run data not available", 
               ha='center', va='center', transform=ax2.transAxes, fontsize=12)
    
    # 3. Temperature bins vs. Win Rate
    ax3 = plt.subplot(gs[0, 2])
    
    if 'temperature' in opportunities.columns and has_results:
        # Create temperature bins
        temp_bins = [0, 50, 60, 70, 80, 90, 100]
        temp_labels = ['<50°F', '50-60°F', '60-70°F', '70-80°F', '80-90°F', '>90°F']
        
        opportunities['temp_bin'] = pd.cut(
            opportunities['temperature'], 
            bins=temp_bins, 
            labels=temp_labels, 
            right=False
        )
        
        # Calculate win rate by temperature bin
        temp_win_rates = opportunities.groupby('temp_bin')['bet_correct'].mean().reset_index()
        temp_win_rates['count'] = opportunities.groupby('temp_bin')['bet_correct'].count().values
        
        # Filter bins with sufficient data
        temp_win_rates = temp_win_rates[temp_win_rates['count'] >= 5]
        
        # Plot win rates
        bars = ax3.bar(
            temp_win_rates['temp_bin'], 
            temp_win_rates['bet_correct'],
            color=[plt.cm.RdYlGn(rate) for rate in temp_win_rates['bet_correct']],
            alpha=0.7
        )
        
        # Add count labels
        for i, (_, row) in enumerate(temp_win_rates.iterrows()):
            ax3.text(
                i, row['bet_correct'] + 0.02, 
                f"n={row['count']}", 
                ha='center', fontsize=9
            )
        
        # Add reference line at 50%
        ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
        
        ax3.set_title('Betting Success Rate by Temperature', fontsize=14)
        ax3.set_xlabel('Temperature Range', fontsize=12)
        ax3.set_ylabel('Win Rate', fontsize=12)
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "Temperature or result data not available", 
               ha='center', va='center', transform=ax3.transAxes, fontsize=12)
    
    # 4. Wind bins vs. Win Rate
    ax4 = plt.subplot(gs[1, 0])
    
    if 'wind_speed' in opportunities.columns and has_results:
        # Create wind bins
        wind_bins = [0, 5, 10, 15, 20, 30]
        wind_labels = ['0-5', '5-10', '10-15', '15-20', '>20']
        
        opportunities['wind_bin'] = pd.cut(
            opportunities['wind_speed'], 
            bins=wind_bins, 
            labels=wind_labels, 
            right=False
        )
        
        # Calculate win rate by wind bin
        wind_win_rates = opportunities.groupby('wind_bin')['bet_correct'].mean().reset_index()
        wind_win_rates['count'] = opportunities.groupby('wind_bin')['bet_correct'].count().values
        
        # Filter bins with sufficient data
        wind_win_rates = wind_win_rates[wind_win_rates['count'] >= 5]
        
        # Plot win rates
        bars = ax4.bar(
            wind_win_rates['wind_bin'], 
            wind_win_rates['bet_correct'],
            color=[plt.cm.RdYlGn(rate) for rate in wind_win_rates['bet_correct']],
            alpha=0.7
        )
        
        # Add count labels
        for i, (_, row) in enumerate(wind_win_rates.iterrows()):
            ax4.text(
                i, row['bet_correct'] + 0.02, 
                f"n={row['count']}", 
                ha='center', fontsize=9
            )
        
        # Add reference line at 50%
        ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
        
        ax4.set_title('Betting Success Rate by Wind Speed', fontsize=14)
        ax4.set_xlabel('Wind Speed Range (mph)', fontsize=12)
        ax4.set_ylabel('Win Rate', fontsize=12)
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, "Wind or result data not available", 
               ha='center', va='center', transform=ax4.transAxes, fontsize=12)
    
    # 5. Weather condition vs. Win Rate
    ax5 = plt.subplot(gs[1, 1])
    
    if 'weather_condition' in opportunities.columns and has_results:
        # Count by weather condition
        condition_counts = opportunities['weather_condition'].value_counts()
        
        # Filter conditions with sufficient samples
        valid_conditions = condition_counts[condition_counts >= 5].index.tolist()
        
        if valid_conditions:
            # Filter opportunities
            condition_data = opportunities[opportunities['weather_condition'].isin(valid_conditions)]
            
            # Calculate win rate by condition
            condition_win_rates = condition_data.groupby('weather_condition')['bet_correct'].mean().reset_index()
            condition_win_rates['count'] = condition_data.groupby('weather_condition')['bet_correct'].count().values
            
            # Sort by win rate
            condition_win_rates = condition_win_rates.sort_values('bet_correct', ascending=False)
            
            # Plot win rates
            bars = ax5.bar(
                condition_win_rates['weather_condition'], 
                condition_win_rates['bet_correct'],
                color=[plt.cm.RdYlGn(rate) for rate in condition_win_rates['bet_correct']],
                alpha=0.7
            )
            
            # Add count labels
            for i, (_, row) in enumerate(condition_win_rates.iterrows()):
                ax5.text(
                    i, row['bet_correct'] + 0.02, 
                    f"n={row['count']}", 
                    ha='center', fontsize=9
                )
            
            # Add reference line at 50%
            ax5.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
            
            ax5.set_title('Betting Success by Weather Condition', fontsize=14)
            ax5.set_xlabel('Weather Condition', fontsize=12)
            ax5.set_ylabel('Win Rate', fontsize=12)
            ax5.set_ylim(0, 1)
            ax5.grid(True, alpha=0.3)
            plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax5.text(0.5, 0.5, "Insufficient weather condition data", 
                   ha='center', va='center', transform=ax5.transAxes, fontsize=12)
    else:
        ax5.text(0.5, 0.5, "Weather condition or result data not available", 
               ha='center', va='center', transform=ax5.transAxes, fontsize=12)
    
    # 6. Stadium Altitude vs. Run Scoring
    ax6 = plt.subplot(gs[1, 2])
    
    # Create a mapping of stadium altitudes
    stadium_altitudes = {stadium_info['name']: stadium_info.get('altitude', 0) 
                         for team, stadium_info in STADIUM_MAPPING.items()}
    
    if 'stadium' in opportunities.columns and 'total_runs' in opportunities.columns:
        # Add altitude to opportunities
        opportunities['altitude'] = opportunities['stadium'].map(stadium_altitudes)
        
        # Get stadiums with at least 5 games
        stadium_counts = opportunities['stadium'].value_counts()
        valid_stadiums = stadium_counts[stadium_counts >= 5].index.tolist()
        
        # Calculate average runs by stadium
        stadium_runs = opportunities[opportunities['stadium'].isin(valid_stadiums)]
        stadium_avg_runs = stadium_runs.groupby('stadium').agg({
            'total_runs': ['mean', 'count'],
            'altitude': 'first'
        }).reset_index()
        
        stadium_avg_runs.columns = ['stadium', 'avg_runs', 'count', 'altitude']
        
        # Sort by altitude
        stadium_avg_runs = stadium_avg_runs.sort_values('altitude')
        
        # Scatter plot
        scatter = ax6.scatter(
            stadium_avg_runs['altitude'],
            stadium_avg_runs['avg_runs'],
            s=stadium_avg_runs['count'] * 3,  # Size by game count
            alpha=0.7,
            c=stadium_avg_runs['avg_runs'],
            cmap='viridis',
            edgecolor='k'
        )
        
        # Add stadium labels to notable points
        for i, row in stadium_avg_runs.iterrows():
            if row['avg_runs'] > stadium_avg_runs['avg_runs'].mean() + 0.5 or \
               row['altitude'] > 1000 or row['count'] > 20:
                ax6.annotate(
                    row['stadium'],
                    xy=(row['altitude'], row['avg_runs']),
                    xytext=(5, 0),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.8
                )
        
        # Add trend line
        z = np.polyfit(stadium_avg_runs['altitude'], stadium_avg_runs['avg_runs'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(stadium_avg_runs['altitude'].min(), stadium_avg_runs['altitude'].max(), 100)
        ax6.plot(x_trend, p(x_trend), 'k--', alpha=0.6)
        
        # Show correlation
        corr = stadium_avg_runs['altitude'].corr(stadium_avg_runs['avg_runs'])
        ax6.text(0.05, 0.95, f"Correlation: {corr:.2f}", transform=ax6.transAxes,
               fontsize=10, va='top', ha='left', bbox=dict(boxstyle='round', alpha=0.1))
        
        ax6.set_title('Stadium Altitude vs. Average Runs', fontsize=14)
        ax6.set_xlabel('Altitude (feet)', fontsize=12)
        ax6.set_ylabel('Average Runs', fontsize=12)
        ax6.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax6)
        cbar.set_label('Average Runs', fontsize=10)
    else:
        ax6.text(0.5, 0.5, "Stadium or run data not available", 
               ha='center', va='center', transform=ax6.transAxes, fontsize=12)
    
    # Set overall title
    plt.suptitle('Weather Impact on MLB Run Scoring and Betting Success', fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save if requested
    if save:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{DATA_DIR}/weather_impact_analysis_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Weather impact visualization saved to {filename}")
    
    # Show the plot
    plt.show()

def visualize_model_performance(model, test_data, save=False):
    """
    Visualize model performance and diagnostics.
    
    Args:
        model (object): Trained model object
        test_data (DataFrame): Test data for evaluation
        save (bool): Whether to save the plot to file
    """
    if model is None or not hasattr(model, 'predict'):
        print("No valid model provided for visualization.")
        return
    
    if test_data is None or len(test_data) == 0:
        print("No test data provided for visualization.")
        return
    
    # Create figure
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])
    
    # Prepare data for visualization
    # Use a subset of test data if it's very large
    if len(test_data) > 5000:
        test_sample = test_data.sample(5000, random_state=42)
    else:
        test_sample = test_data
    
    # Make predictions
    if hasattr(model, 'feature_names') and model.feature_names:
        # Advanced model with stored feature names
        features = model.feature_names
        X_test = test_sample[features].fillna(test_sample[features].median())
        y_pred = model.predict(X_test)
    else:
        # Generic model - try basic features
        try:
            basic_features = [
                'temperature', 'humidity', 'wind_speed',
                'precipitation', 'cloud_cover', 'pressure'
            ]
            available_features = [f for f in basic_features if f in test_sample.columns]
            X_test = test_sample[available_features].fillna(test_sample[available_features].median())
            
            # Try to transform data if model has a scaler
            if hasattr(model, 'scaler') and model.scaler:
                X_test_scaled = model.scaler.transform(X_test)
                y_pred = model.model.predict(X_test_scaled)
            else:
                y_pred = model.predict(X_test)
        except Exception as e:
            print(f"Error making predictions: {e}")
            return
    
    # Get actual values
    if 'total_runs' in test_sample.columns:
        y_true = test_sample['total_runs']
        
        # Calculate errors
        errors = y_pred - y_true
        abs_errors = np.abs(errors)
        
        # 1. Actual vs. Predicted scatter plot
        ax1 = plt.subplot(gs[0, 0])
        scatter = ax1.scatter(y_true, y_pred, alpha=0.5, s=30, c=abs_errors, cmap='viridis')
        
        # Add identity line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
        
        # Add metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        ax1.text(0.05, 0.95, 
               f"RMSE: {rmse:.3f}\nMAE: {mae:.3f}\nR²: {r2:.3f}", 
               transform=ax1.transAxes, fontsize=10, va='top',
               bbox=dict(boxstyle='round', alpha=0.1))
        
        ax1.set_title('Actual vs. Predicted Runs', fontsize=14)
        ax1.set_xlabel('Actual Runs', fontsize=12)
        ax1.set_ylabel('Predicted Runs', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Absolute Error', fontsize=10)
        
        # 2. Error histogram
        ax2 = plt.subplot(gs[0, 1])
        ax2.hist(errors, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        ax2.axvline(x=errors.mean(), color='g', linestyle='-', alpha=0.7,
                   label=f'Mean: {errors.mean():.3f}')
        
        ax2.set_title('Prediction Error Distribution', fontsize=14)
        ax2.set_xlabel('Prediction Error (Predicted - Actual)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. Error vs. Temperature
        ax3 = plt.subplot(gs[1, 0])
        
        if 'temperature' in test_sample.columns:
            scatter = ax3.scatter(test_sample['temperature'], errors, 
                               alpha=0.5, s=30, c=abs_errors, cmap='viridis')
            
            # Add horizontal line at 0
            ax3.axhline(y=0, color='r', linestyle='--', alpha=0.7)
            
            # Add trend line
            z = np.polyfit(test_sample['temperature'], errors, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(test_sample['temperature'].min(), test_sample['temperature'].max(), 100)
            ax3.plot(x_trend, p(x_trend), 'k--', alpha=0.6)
            
            # Show correlation
            corr = pd.Series(test_sample['temperature'].values).corr(pd.Series(errors.values))
            ax3.text(0.05, 0.95, f"Correlation: {corr:.2f}", transform=ax3.transAxes,
                   fontsize=10, va='top', ha='left', bbox=dict(boxstyle='round', alpha=0.1))
            
            ax3.set_title('Prediction Error vs. Temperature', fontsize=14)
            ax3.set_xlabel('Temperature (°F)', fontsize=12)
            ax3.set_ylabel('Prediction Error', fontsize=12)
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, "Temperature data not available", 
                   ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        
        # 4. Error vs. Over/Under Line
        ax4 = plt.subplot(gs[1, 1])
        
        if 'over_under_line' in test_sample.columns:
            scatter = ax4.scatter(test_sample['over_under_line'], errors, 
                                alpha=0.5, s=30, c=abs_errors, cmap='viridis')
            
            # Add horizontal line at 0
            ax4.axhline(y=0, color='r', linestyle='--', alpha=0.7)
            
            # Add trend line
            z = np.polyfit(test_sample['over_under_line'], errors, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(test_sample['over_under_line'].min(), test_sample['over_under_line'].max(), 100)
            ax4.plot(x_trend, p(x_trend), 'k--', alpha=0.6)
            
            # Show correlation
            corr = pd.Series(test_sample['over_under_line'].values).corr(pd.Series(errors.values))
            ax4.text(0.05, 0.95, f"Correlation: {corr:.2f}", transform=ax4.transAxes,
                   fontsize=10, va='top', ha='left', bbox=dict(boxstyle='round', alpha=0.1))
            
            ax4.set_title('Prediction Error vs. Vegas Line', fontsize=14)
            ax4.set_xlabel('Over/Under Line', fontsize=12)
            ax4.set_ylabel('Prediction Error', fontsize=12)
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, "Over/Under line data not available", 
                   ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        
        # 5. Feature importance
        ax5 = plt.subplot(gs[2, :])
        
        # Try to get feature importance
        feature_importance = None
        
        if hasattr(model, 'analyze_feature_importance'):
            try:
                feature_importance = model.analyze_feature_importance()
            except:
                pass
        
        if hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
            # Model with feature importances attribute
            try:
                importances = model.model.feature_importances_
                feature_names = model.feature_names
                
                feature_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
            except:
                pass
        
        if feature_importance is not None and len(feature_importance) > 0:
            # Plot top 15 features
            top_features = feature_importance.head(15)
            bars = ax5.barh(
                top_features['Feature'],
                top_features['Importance'],
                alpha=0.7,
                color='teal'
            )
            
            ax5.set_title('Top 15 Feature Importance', fontsize=14)
            ax5.set_xlabel('Importance', fontsize=12)
            ax5.set_ylabel('Feature', fontsize=12)
            ax5.grid(True, alpha=0.3)
            ax5.invert_yaxis()  # Highest importance at the top
        else:
            ax5.text(0.5, 0.5, "Feature importance not available for this model", 
                   ha='center', va='center', transform=ax5.transAxes, fontsize=12)
    else:
        # No actual run data - can't evaluate performance
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "Cannot evaluate model: No 'total_runs' column in test data", 
              ha='center', va='center', transform=ax.transAxes, fontsize=14)
    
    # Set overall title
    plt.suptitle('MLB Weather Model Performance Analysis', fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save if requested
    if save:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{DATA_DIR}/model_performance_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Model performance visualization saved to {filename}")
    
    # Show the plot
    plt.show()
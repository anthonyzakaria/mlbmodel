"""
NRFI/YRFI backtesting script for the MLB Weather Model.
This script demonstrates how to use the NRFI model for analysis and backtesting.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

from config import DATA_DIR
from nrfi_model import NRFIModel

# Try to import visualization module with fallbacks
try:
    from enhanced_visualization import visualize_nrfi_betting_opportunities
except ImportError:
    # Simple fallback visualization
    def visualize_nrfi_betting_opportunities(nrfi_df):
        """Simple visualization fallback when the enhanced module isn't available."""
        print(f"Found {len(nrfi_df)} NRFI/YRFI betting opportunities.")
        if len(nrfi_df) > 0:
            for _, bet in nrfi_df.iterrows():
                print(f"* {bet['date'].strftime('%Y-%m-%d')}: {bet['away_team']} @ {bet['home_team']} - " 
                     f"Bet: {bet['recommended_bet']}, " 
                     f"NRFI Prob: {bet['nrfi_probability']:.2f}, " 
                     f"Confidence: {bet['confidence']:.2f}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='NRFI/YRFI Backtesting for MLB Weather Model')
    
    parser.add_argument('--start-year', type=int, default=2022, 
                        help='Starting year for backtest')
    parser.add_argument('--end-year', type=int, default=2023, 
                        help='Ending year for backtest')
    parser.add_argument('--confidence', type=float, default=0.6,
                        help='Confidence threshold for NRFI/YRFI bets')
    parser.add_argument('--bankroll', type=float, default=10000,
                        help='Starting bankroll for backtest')
    parser.add_argument('--analyze-weather', action='store_true',
                        help='Analyze weather impact on first inning scoring')
    parser.add_argument('--save-figs', action='store_true',
                        help='Save all figures to disk')
    
    return parser.parse_args()

def run_nrfi_backtest(args):
    """Run NRFI/YRFI backtest with specified parameters."""
    print(f"Running NRFI/YRFI backtest for {args.start_year}-{args.end_year}")
    
    # Initialize NRFI model
    nrfi_model = NRFIModel()
    
    # Load merged data
    merged_file = f"{DATA_DIR}/merged_data.csv"
    if os.path.exists(merged_file):
        print(f"Loading data from {merged_file}")
        merged_data = pd.read_csv(merged_file, parse_dates=['date'])
        
        # Filter to requested years
        merged_data['year'] = pd.to_datetime(merged_data['date']).dt.year
        merged_data = merged_data[
            (merged_data['year'] >= args.start_year) & 
            (merged_data['year'] <= args.end_year)
        ]
        
        # Check if first inning data exists
        if 'first_inning_total_runs' not in merged_data.columns or 'nrfi' not in merged_data.columns:
            print("No first inning data found. Adding first inning data...")
            merged_data = nrfi_model.fetch_first_inning_data(merged_data, use_existing=True)
        
        # Set data in model
        nrfi_model.merged_data = merged_data
        
        # Try to load pre-trained model
        try:
            print("Loading pre-trained NRFI model...")
            nrfi_model.load_model()
        except FileNotFoundError:
            print("No pre-trained model found. Training new model...")
            nrfi_model.train_model()
        
        # Analyze weather impact if requested
        if args.analyze_weather:
            print("\nAnalyzing weather impact on first inning scoring...")
            weather_impact = nrfi_model.analyze_weather_impact_on_first_inning(merged_data)
            
            if weather_impact:
                # Plot temperature impact
                if 'temperature' in weather_impact and len(weather_impact['temperature']) > 0:
                    temp_data = weather_impact['temperature'].reset_index()
                    
                    plt.figure(figsize=(10, 6))
                    plt.bar(temp_data['temp_range'], temp_data[('first_inning_total_runs', 'mean')], alpha=0.7)
                    plt.title('Average First Inning Runs by Temperature', fontsize=14)
                    plt.xlabel('Temperature Range', fontsize=12)
                    plt.ylabel('Average First Inning Runs', fontsize=12)
                    plt.grid(axis='y', alpha=0.3)
                    plt.xticks(rotation=45)
                    
                    # Add NRFI rate as a line on secondary axis
                    ax2 = plt.gca().twinx()
                    ax2.plot(temp_data['temp_range'], temp_data[('nrfi', '%')], 'ro-', linewidth=2)
                    ax2.set_ylabel('NRFI %', color='r', fontsize=12)
                    ax2.tick_params(axis='y', colors='r')
                    
                    plt.tight_layout()
                    
                    if args.save_figs:
                        plt.savefig(f"{DATA_DIR}/nrfi_temperature_impact.png", dpi=300, bbox_inches='tight')
                    
                    plt.show()
                
                # Plot wind impact
                if 'wind' in weather_impact and len(weather_impact['wind']) > 0:
                    wind_data = weather_impact['wind'].reset_index()
                    
                    plt.figure(figsize=(10, 6))
                    plt.bar(wind_data['wind_range'], wind_data[('first_inning_total_runs', 'mean')], alpha=0.7)
                    plt.title('Average First Inning Runs by Wind Speed', fontsize=14)
                    plt.xlabel('Wind Speed Range', fontsize=12)
                    plt.ylabel('Average First Inning Runs', fontsize=12)
                    plt.grid(axis='y', alpha=0.3)
                    plt.xticks(rotation=45)
                    
                    # Add NRFI rate as a line on secondary axis
                    ax2 = plt.gca().twinx()
                    ax2.plot(wind_data['wind_range'], wind_data[('nrfi', '%')], 'ro-', linewidth=2)
                    ax2.set_ylabel('NRFI %', color='r', fontsize=12)
                    ax2.tick_params(axis='y', colors='r')
                    
                    plt.tight_layout()
                    
                    if args.save_figs:
                        plt.savefig(f"{DATA_DIR}/nrfi_wind_impact.png", dpi=300, bbox_inches='tight')
                    
                    plt.show()
                
                # Plot dome impact
                if 'dome' in weather_impact and len(weather_impact['dome']) > 0:
                    dome_data = weather_impact['dome'].reset_index()
                    
                    plt.figure(figsize=(8, 6))
                    plt.bar(dome_data['is_dome'], dome_data[('first_inning_total_runs', 'mean')], alpha=0.7)
                    plt.title('Average First Inning Runs: Dome vs Outdoor', fontsize=14)
                    plt.xlabel('Stadium Type', fontsize=12)
                    plt.ylabel('Average First Inning Runs', fontsize=12)
                    plt.grid(axis='y', alpha=0.3)
                    
                    # Add NRFI rate as text annotations
                    for i, row in dome_data.iterrows():
                        plt.text(i, row[('first_inning_total_runs', 'mean')] + 0.03, 
                               f"NRFI: {row[('nrfi', '%')]:.1f}%", 
                               ha='center', fontsize=10, fontweight='bold')
                    
                    plt.tight_layout()
                    
                    if args.save_figs:
                        plt.savefig(f"{DATA_DIR}/nrfi_dome_impact.png", dpi=300, bbox_inches='tight')
                    
                    plt.show()
        
        # Find betting opportunities
        print(f"\nFinding NRFI/YRFI betting opportunities with {args.confidence:.1%} confidence threshold...")
        opportunities = nrfi_model.find_betting_opportunities(confidence_threshold=args.confidence)
        
        if opportunities is not None and len(opportunities) > 0:
            # Run backtest
            print(f"\nRunning backtest with ${args.bankroll:,.2f} starting bankroll...")
            backtest_results = nrfi_model.backtest_nrfi_strategy(
                opportunities=opportunities, 
                starting_bankroll=args.bankroll
            )
            
            # Visualize opportunities
            visualize_nrfi_betting_opportunities(
                opportunities, 
                results=backtest_results,
                title=f"NRFI/YRFI Betting Opportunities ({args.start_year}-{args.end_year})",
                save=args.save_figs
            )
            
            # Show profit by month
            if backtest_results and 'results_df' in backtest_results:
                results_df = backtest_results['results_df']
                if 'date' in results_df.columns:
                    # Add month column
                    results_df['month'] = pd.to_datetime(results_df['date']).dt.strftime('%Y-%m')
                    
                    # Group by month
                    monthly = results_df.groupby('month').agg({
                        'profit': 'sum',
                        'win': ['mean', 'count']
                    })
                    
                    # Plot monthly profits
                    plt.figure(figsize=(12, 6))
                    plt.bar(monthly.index, monthly[('profit', 'sum')], alpha=0.7)
                    plt.title('NRFI/YRFI Monthly Profit', fontsize=14)
                    plt.xlabel('Month', fontsize=12)
                    plt.ylabel('Profit ($)', fontsize=12)
                    plt.grid(axis='y', alpha=0.3)
                    plt.xticks(rotation=45)
                    
                    # Add win rate as a line on secondary axis
                    ax2 = plt.gca().twinx()
                    ax2.plot(monthly.index, monthly[('win', 'mean')] * 100, 'ro-', linewidth=2)
                    ax2.set_ylabel('Win Rate (%)', color='r', fontsize=12)
                    ax2.tick_params(axis='y', colors='r')
                    
                    # Add bet count annotations
                    for i, idx in enumerate(monthly.index):
                        plt.text(i, monthly.loc[idx, ('profit', 'sum')] + 10, 
                               f"n={int(monthly.loc[idx, ('win', 'count')])}", 
                               ha='center', fontsize=9)
                    
                    plt.tight_layout()
                    
                    if args.save_figs:
                        plt.savefig(f"{DATA_DIR}/nrfi_monthly_profit.png", dpi=300, bbox_inches='tight')
                    
                    plt.show()
                    
                    # Show performance split by NRFI vs YRFI
                    nrfi_bets = results_df[results_df['bet_type'] == 'NRFI']
                    yrfi_bets = results_df[results_df['bet_type'] == 'YRFI']
                    
                    nrfi_win_rate = nrfi_bets['win'].mean() if len(nrfi_bets) > 0 else 0
                    yrfi_win_rate = yrfi_bets['win'].mean() if len(yrfi_bets) > 0 else 0
                    
                    nrfi_profit = nrfi_bets['profit'].sum() if len(nrfi_bets) > 0 else 0
                    yrfi_profit = yrfi_bets['profit'].sum() if len(yrfi_bets) > 0 else 0
                    
                    print("\nPerformance by Bet Type:")
                    print(f"NRFI: {len(nrfi_bets)} bets, {nrfi_win_rate:.1%} win rate, ${nrfi_profit:.2f} profit")
                    print(f"YRFI: {len(yrfi_bets)} bets, {yrfi_win_rate:.1%} win rate, ${yrfi_profit:.2f} profit")
                    
                    # Create bet type comparison chart
                    plt.figure(figsize=(8, 6))
                    bet_types = ['NRFI', 'YRFI']
                    counts = [len(nrfi_bets), len(yrfi_bets)]
                    win_rates = [nrfi_win_rate * 100, yrfi_win_rate * 100]
                    profits = [nrfi_profit, yrfi_profit]
                    
                    # Plot bet counts
                    ax1 = plt.subplot(1, 3, 1)
                    ax1.bar(bet_types, counts, alpha=0.7)
                    ax1.set_title('Bet Count')
                    ax1.set_ylabel('Number of Bets')
                    
                    # Plot win rates
                    ax2 = plt.subplot(1, 3, 2)
                    ax2.bar(bet_types, win_rates, alpha=0.7, color='green')
                    ax2.set_title('Win Rate')
                    ax2.set_ylabel('Win Rate (%)')
                    ax2.set_ylim(0, 100)
                    
                    # Plot profits
                    ax3 = plt.subplot(1, 3, 3)
                    ax3.bar(bet_types, profits, alpha=0.7, color='purple')
                    ax3.set_title('Total Profit')
                    ax3.set_ylabel('Profit ($)')
                    
                    plt.tight_layout()
                    
                    if args.save_figs:
                        plt.savefig(f"{DATA_DIR}/nrfi_bet_type_comparison.png", dpi=300, bbox_inches='tight')
                    
                    plt.show()
            
            return backtest_results
        else:
            print("No NRFI/YRFI betting opportunities found for backtest.")
            return None
    else:
        print(f"Error: Cannot find merged data file {merged_file}")
        print("Please run main.py with --fetch-data and --fetch-innings flags first")
        return None

if __name__ == "__main__":
    args = parse_args()
    run_nrfi_backtest(args)
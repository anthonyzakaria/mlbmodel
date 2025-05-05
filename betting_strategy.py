"""
Advanced betting strategy optimization module for MLB Weather Model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from config import DATA_DIR, MAX_KELLY_FRACTION, DEFAULT_BET_SIZE

def convert_american_to_decimal(american_odds):
    """
    Convert American odds to decimal format.
    
    Args:
        american_odds (float): Odds in American format
        
    Returns:
        float: Odds in decimal format
    """
    if pd.isna(american_odds):
        return 1.91  # Default
    
    try:
        american_odds = float(american_odds)
        if american_odds > 0:
            return 1 + (american_odds / 100)
        else:
            return 1 + (100 / abs(american_odds))
    except:
        return 1.91  # Default if conversion fails

def calculate_kelly_size(bankroll, edge, odds):
    """
    Calculate Kelly Criterion bet size.
    
    Args:
        bankroll (float): Current bankroll
        edge (float): Estimated edge in decimal (e.g., 0.05 for 5% edge)
        odds (float): Decimal odds
        
    Returns:
        float: Recommended bet size
    """
    # Calculate Kelly bet size
    win_prob = 1 / odds  # Convert odds to probability
    
    # Adjust probability based on edge
    adjusted_prob = win_prob * (1 + edge)
    
    # Kelly formula: f* = (p * b - q) / b
    # where p = probability of winning, q = probability of losing, b = odds - 1
    b = odds - 1  # Net odds
    q = 1 - adjusted_prob  # Probability of losing
    
    # Limit to avoid extreme values
    kelly_fraction = max(0, min((adjusted_prob * b - q) / b, MAX_KELLY_FRACTION))
    
    # Calculate actual bet size
    bet_size = bankroll * kelly_fraction
    
    return bet_size

def optimize_betting_strategy(opportunities, start_bankroll=10000.0, detailed=False):
    """
    Optimize the betting strategy parameters.
    
    Args:
        opportunities (DataFrame): Betting opportunities with results
        start_bankroll (float): Starting bankroll
        detailed (bool): Whether to run detailed optimization
        
    Returns:
        dict: Optimized strategy parameters
    """
    if opportunities is None or len(opportunities) == 0:
        print("No opportunities provided for optimization")
        return None
    
    # Results container
    results = {}
    
    # 1. Optimize confidence threshold
    print("Optimizing confidence threshold...")
    
    thresholds = np.linspace(0.2, 2.0, 19)  # 0.2 to 2.0 in 0.1 increments
    threshold_results = []
    
    for threshold in thresholds:
        # Filter opportunities by threshold
        filtered = opportunities[opportunities['confidence'] >= threshold].copy()
        
        if len(filtered) < 20:
            # Not enough data for reliable statistics
            threshold_results.append({
                'threshold': threshold,
                'bet_count': len(filtered),
                'win_rate': np.nan,
                'roi': np.nan,
                'final_bankroll': np.nan
            })
            continue
        
        # Calculate basic metrics
        actual_over = filtered['total_runs'] > filtered['over_under_line']
        actual_under = filtered['total_runs'] < filtered['over_under_line']
        
        # Check if bet was correct
        filtered['correct'] = (
            ((filtered['recommended_bet'] == 'OVER') & actual_over) |
            ((filtered['recommended_bet'] == 'UNDER') & actual_under)
        )
        
        # Calculate win rate
        win_rate = filtered['correct'].mean()
        
        # Simple backtest with fixed bet size
        bet_size = DEFAULT_BET_SIZE
        bankroll = start_bankroll
        
        for _, bet in filtered.iterrows():
            # Get odds based on bet type
            if bet['recommended_bet'] == 'OVER' and 'over_odds' in bet:
                odds = convert_american_to_decimal(bet['over_odds'])
            elif bet['recommended_bet'] == 'UNDER' and 'under_odds' in bet:
                odds = convert_american_to_decimal(bet['under_odds'])
            else:
                odds = 1.91  # Default -110 in decimal
            
            # Calculate outcome
            if bet['correct']:
                bankroll += bet_size * (odds - 1)
            else:
                bankroll -= bet_size
            
            # Check for bankruptcy
            if bankroll <= 0:
                break
        
        # Calculate ROI
        roi = (bankroll - start_bankroll) / start_bankroll * 100
        
        threshold_results.append({
            'threshold': threshold,
            'bet_count': len(filtered),
            'win_rate': win_rate,
            'roi': roi,
            'final_bankroll': bankroll
        })
    
    # Convert to DataFrame
    threshold_df = pd.DataFrame(threshold_results)
    
    # Find optimal threshold
    if not threshold_df.empty and not threshold_df['roi'].isna().all():
        optimal_idx = threshold_df['roi'].idxmax()
        optimal_threshold = threshold_df.loc[optimal_idx, 'threshold']
        max_roi = threshold_df.loc[optimal_idx, 'roi']
        
        print(f"Optimal confidence threshold: {optimal_threshold:.2f}")
        print(f"Expected ROI at optimal threshold: {max_roi:.2f}%")
        
        results['optimal_threshold'] = optimal_threshold
        results['threshold_results'] = threshold_df
    else:
        results['optimal_threshold'] = DEFAULT_CONFIDENCE_THRESHOLD
        results['threshold_results'] = threshold_df
    
    # Optionally run more detailed optimizations
    if detailed:
        # 2. Optimize bet sizing
        print("Optimizing bet sizing strategy...")
        
        # Use optimal threshold from previous step
        optimal_threshold = results['optimal_threshold']
        filtered = opportunities[opportunities['confidence'] >= optimal_threshold].copy()
        
        # Test different sizing approaches
        sizing_methods = ['fixed', 'confidence', 'kelly', 'kelly_half']
        sizing_results = []
        
        for method in sizing_methods:
            bankroll = start_bankroll
            bets = []
            
            for _, bet in filtered.iterrows():
                # Get odds based on bet type
                if bet['recommended_bet'] == 'OVER' and 'over_odds' in bet:
                    odds = convert_american_to_decimal(bet['over_odds'])
                elif bet['recommended_bet'] == 'UNDER' and 'under_odds' in bet:
                    odds = convert_american_to_decimal(bet['under_odds'])
                else:
                    odds = 1.91  # Default -110 in decimal
                
                # Calculate bet size based on method
                if method == 'fixed':
                    bet_size = DEFAULT_BET_SIZE
                elif method == 'confidence':
                    # Scale by confidence (diff between prediction and line)
                    bet_size = DEFAULT_BET_SIZE * (bet['confidence'] / optimal_threshold)
                    bet_size = min(bet_size, DEFAULT_BET_SIZE * 2)  # Cap at 2x default
                elif method == 'kelly':
                    # Full Kelly criterion
                    edge = bet['confidence'] / bet['over_under_line']  # Normalize by line
                    bet_size = calculate_kelly_size(bankroll, edge, odds)
                elif method == 'kelly_half':
                    # Half Kelly for reduced variance
                    edge = bet['confidence'] / bet['over_under_line']
                    bet_size = calculate_kelly_size(bankroll, edge, odds) / 2
                
                # Cap bet size relative to bankroll
                bet_size = min(bet_size, bankroll * 0.1)
                
                # Calculate outcome
                correct = (
                    (bet['recommended_bet'] == 'OVER' and bet['total_runs'] > bet['over_under_line']) or
                    (bet['recommended_bet'] == 'UNDER' and bet['total_runs'] < bet['over_under_line'])
                )
                
                if correct:
                    profit = bet_size * (odds - 1)
                else:
                    profit = -bet_size
                
                # Update bankroll
                bankroll += profit
                
                # Record bet details
                bets.append({
                    'method': method,
                    'bet_size': bet_size,
                    'correct': correct,
                    'profit': profit,
                    'bankroll': bankroll
                })
                
                # Check for bankruptcy
                if bankroll <= 0:
                    break
            
            # Calculate metrics
            bets_df = pd.DataFrame(bets)
            
            if not bets_df.empty:
                win_rate = bets_df['correct'].mean()
                max_drawdown = calculate_max_drawdown(bets_df['bankroll'])
                roi = (bankroll - start_bankroll) / start_bankroll * 100
                bet_count = len(bets_df)
                
                sizing_results.append({
                    'method': method,
                    'win_rate': win_rate,
                    'max_drawdown': max_drawdown,
                    'roi': roi,
                    'final_bankroll': bankroll,
                    'bet_count': bet_count
                })
        
        # Convert to DataFrame
        sizing_df = pd.DataFrame(sizing_results)
        
        if not sizing_df.empty:
            # Find optimal method
            optimal_method = sizing_df.loc[sizing_df['roi'].idxmax(), 'method']
            max_roi = sizing_df.loc[sizing_df['roi'].idxmax(), 'roi']
            
            print(f"Optimal bet sizing method: {optimal_method}")
            print(f"Expected ROI with optimal sizing: {max_roi:.2f}%")
            
            results['optimal_sizing'] = optimal_method
            results['sizing_results'] = sizing_df
    
    return results

def backtest_strategy(opportunities, kelly=False, detailed=False):
    """
    Backtest the betting strategy on historical opportunities.
    
    Args:
        opportunities (DataFrame): Betting opportunities with results
        kelly (bool): Whether to use Kelly criterion for bet sizing
        detailed (bool): Whether to show detailed results
        
    Returns:
        dict: Backtest results
    """
    if opportunities is None or len(opportunities) == 0:
        print("No opportunities provided for backtest")
        return None
    
    # Initial bankroll
    initial_bankroll = 10000.0
    bankroll = initial_bankroll
    bets = []
    
    print(f"Starting backtest with ${initial_bankroll:.2f} bankroll")
    print(f"Testing {len(opportunities)} betting opportunities")
    
    # Process each bet
    for _, bet in opportunities.iterrows():
        # Get odds based on bet type
        if bet['recommended_bet'] == 'OVER' and 'over_odds' in bet:
            odds = convert_american_to_decimal(bet['over_odds'])
        elif bet['recommended_bet'] == 'UNDER' and 'under_odds' in bet:
            odds = convert_american_to_decimal(bet['under_odds'])
        else:
            odds = 1.91  # Default -110 in decimal
        
        # Calculate bet size
        if kelly:
            # Use Kelly criterion
            edge = bet['confidence'] / bet['over_under_line']  # Normalize by line
            bet_size = calculate_kelly_size(bankroll, edge, odds)
        else:
            # Fixed bet size
            bet_size = DEFAULT_BET_SIZE
        
        # Ensure bet size doesn't exceed bankroll
        bet_size = min(bet_size, bankroll)
        
        # Determine outcome
        if 'total_runs' in bet:
            # Historical data with known outcome
            correct = (
                (bet['recommended_bet'] == 'OVER' and bet['total_runs'] > bet['over_under_line']) or
                (bet['recommended_bet'] == 'UNDER' and bet['total_runs'] < bet['over_under_line'])
            )
            
            # Calculate profit/loss
            if correct:
                profit = bet_size * (odds - 1)
            else:
                profit = -bet_size
            
            # Update bankroll
            bankroll += profit
            
            # Record bet details
            bets.append({
                'date': bet.get('date', None),
                'matchup': f"{bet.get('away_team', '')} @ {bet.get('home_team', '')}",
                'bet_type': bet['recommended_bet'],
                'line': bet['over_under_line'],
                'prediction': bet.get('predicted_runs', np.nan),
                'actual': bet.get('total_runs', np.nan),
                'confidence': bet.get('confidence', np.nan),
                'odds': odds,
                'bet_size': bet_size,
                'profit': profit,
                'bankroll': bankroll,
                'correct': correct
            })
        else:
            # Future prediction without outcome
            bets.append({
                'date': bet.get('date', None),
                'matchup': f"{bet.get('away_team', '')} @ {bet.get('home_team', '')}",
                'bet_type': bet['recommended_bet'],
                'line': bet['over_under_line'],
                'prediction': bet.get('predicted_runs', np.nan),
                'confidence': bet.get('confidence', np.nan),
                'odds': odds,
                'bet_size': bet_size
            })
    
    # Create DataFrame of bet results
    results_df = pd.DataFrame(bets)
    
    # Calculate performance metrics
    if 'correct' in results_df.columns:
        bet_count = len(results_df)
        winning_bets = results_df['correct'].sum()
        win_rate = winning_bets / bet_count if bet_count > 0 else 0
        roi = (bankroll - initial_bankroll) / initial_bankroll * 100
        profit = bankroll - initial_bankroll
        
        # Calculate max drawdown
        max_drawdown = calculate_max_drawdown(results_df['bankroll'])
        
        # Calculate streaks
        max_win_streak, max_lose_streak = calculate_streaks(results_df['correct'])
        
        # Calculate profit by bet type
        over_profit = results_df[results_df['bet_type'] == 'OVER']['profit'].sum()
        under_profit = results_df[results_df['bet_type'] == 'UNDER']['profit'].sum()
        
        print(f"\nBacktest Results:")
        print(f"Total Bets: {bet_count}")
        print(f"Win Rate: {win_rate:.4f} ({winning_bets} / {bet_count})")
        print(f"ROI: {roi:.2f}%")
        print(f"Profit: ${profit:.2f}")
        print(f"Final Bankroll: ${bankroll:.2f}")
        print(f"Max Drawdown: ${max_drawdown:.2f}")
        print(f"Max Win Streak: {max_win_streak}, Max Lose Streak: {max_lose_streak}")
        print(f"OVER Profit: ${over_profit:.2f}, UNDER Profit: ${under_profit:.2f}")
        
        if detailed:
            # Group by month to see performance trend
            if 'date' in results_df.columns:
                results_df['month'] = pd.to_datetime(results_df['date']).dt.strftime('%Y-%m')
                monthly = results_df.groupby('month').agg({
                    'profit': 'sum',
                    'correct': ['count', 'sum', 'mean']
                })
                monthly.columns = ['profit', 'bets', 'wins', 'win_rate']
                
                print("\nMonthly Performance:")
                print(monthly)
        
        # Plot equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(results_df['bankroll'], marker='', linewidth=2)
        plt.axhline(y=initial_bankroll, color='r', linestyle='--', alpha=0.3)
        plt.title('Betting Strategy Equity Curve')
        plt.xlabel('Bet Number')
        plt.ylabel('Bankroll ($)')
        plt.grid(True, alpha=0.3)
        
        # Add annotations
        plt.annotate(f'Start: ${initial_bankroll:.0f}', 
                    xy=(0, initial_bankroll),
                    xytext=(10, -20),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))
        
        plt.annotate(f'End: ${bankroll:.0f} (ROI: {roi:.1f}%)', 
                    xy=(len(results_df)-1, bankroll),
                    xytext=(-100, -40),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))
        
        plt.tight_layout()
        plt.savefig(f"{DATA_DIR}/equity_curve.png")
        
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
            'max_lose_streak': max_lose_streak
        }
    else:
        # For future predictions, just return the bets DataFrame
        return {'results_df': results_df}

def calculate_max_drawdown(equity_curve):
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

def calculate_streaks(results):
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

def simulate_long_term_results(win_rate, odds, bet_count=1000, start_bankroll=10000.0, bet_size=100.0):
    """
    Simulate long-term results of a betting strategy.
    
    Args:
        win_rate (float): Expected win rate
        odds (float): Decimal odds
        bet_count (int): Number of bets to simulate
        start_bankroll (float): Starting bankroll
        bet_size (float): Fixed bet size
        
    Returns:
        dict: Simulation results
    """
    np.random.seed(42)  # For reproducibility
    
    print(f"Simulating {bet_count} bets with {win_rate:.4f} win rate at {odds:.2f} odds")
    
    # Generate random outcomes based on win rate
    outcomes = np.random.random(bet_count) < win_rate
    
    # Calculate profits
    profits = np.where(outcomes, bet_size * (odds - 1), -bet_size)
    
    # Calculate cumulative bankroll
    bankroll = np.cumsum(profits) + start_bankroll
    
    # Calculate metrics
    final_bankroll = bankroll[-1]
    roi = (final_bankroll - start_bankroll) / start_bankroll * 100
    
    # Calculate max drawdown
    running_max = np.maximum.accumulate(bankroll)
    drawdowns = running_max - bankroll
    max_drawdown = np.max(drawdowns)
    
    # Calculate win and loss streaks
    win_streak, lose_streak = calculate_streaks(outcomes)
    
    # Calculate probability of bankruptcy
    bankruptcy = np.any(bankroll <= 0)
    
    print(f"Simulation Results:")
    print(f"Final Bankroll: ${final_bankroll:.2f}")
    print(f"ROI: {roi:.2f}%")
    print(f"Max Drawdown: ${max_drawdown:.2f}")
    print(f"Probability of Bankruptcy: {int(bankruptcy)}")
    print(f"Max Win Streak: {win_streak}, Max Lose Streak: {lose_streak}")
    
    # Plot simulated equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(bankroll, linewidth=1)
    plt.axhline(y=start_bankroll, color='r', linestyle='--', alpha=0.3)
    plt.title('Simulated Long-Term Equity Curve')
    plt.xlabel('Bet Number')
    plt.ylabel('Bankroll ($)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/simulation_equity_curve.png")
    
    return {
        'bankroll': bankroll,
        'final_bankroll': final_bankroll,
        'roi': roi,
        'max_drawdown': max_drawdown,
        'bankruptcy': bankruptcy,
        'win_streak': win_streak,
        'lose_streak': lose_streak
    }

def analyze_weather_patterns(opportunities):
    """
    Analyze how different weather patterns affect betting success.
    
    Args:
        opportunities (DataFrame): Betting opportunities with results
        
    Returns:
        dict: Analysis results by weather condition
    """
    if opportunities is None or len(opportunities) == 0 or 'correct' not in opportunities.columns:
        print("No usable opportunities for weather pattern analysis")
        return None
    
    results = {}
    
    # Analyze by temperature
    if 'temperature' in opportunities.columns:
        # Create temperature bins
        temp_bins = [0, 50, 60, 70, 80, 90, 100]
        temp_labels = ['Cold (<50°F)', 'Cool (50-60°F)', 'Mild (60-70°F)', 
                      'Warm (70-80°F)', 'Hot (80-90°F)', 'Very Hot (>90°F)']
        
        opportunities['temp_range'] = pd.cut(opportunities['temperature'], 
                                            bins=temp_bins, 
                                            labels=temp_labels, 
                                            right=False)
        
        # Group by temperature range
        temp_results = opportunities.groupby('temp_range').agg({
            'correct': ['count', 'sum', 'mean'],
            'profit': ['sum', 'mean']
        })
        
        temp_results.columns = ['bets', 'wins', 'win_rate', 'total_profit', 'avg_profit']
        
        results['temperature'] = temp_results
        
    # Analyze by wind speed
    if 'wind_speed' in opportunities.columns:
        # Create wind bins
        wind_bins = [0, 5, 10, 15, 20, 30]
        wind_labels = ['Calm (0-5 mph)', 'Light (5-10 mph)', 'Moderate (10-15 mph)', 
                      'Strong (15-20 mph)', 'Very Strong (>20 mph)']
        
        opportunities['wind_range'] = pd.cut(opportunities['wind_speed'], 
                                           bins=wind_bins, 
                                           labels=wind_labels, 
                                           right=False)
        
        # Group by wind range
        wind_results = opportunities.groupby('wind_range').agg({
            'correct': ['count', 'sum', 'mean'],
            'profit': ['sum', 'mean']
        })
        
        wind_results.columns = ['bets', 'wins', 'win_rate', 'total_profit', 'avg_profit']
        
        results['wind'] = wind_results
    
    # Analyze by weather condition
    if 'weather_condition' in opportunities.columns:
        # Group by weather condition
        condition_results = opportunities.groupby('weather_condition').agg({
            'correct': ['count', 'sum', 'mean'],
            'profit': ['sum', 'mean']
        })
        
        condition_results.columns = ['bets', 'wins', 'win_rate', 'total_profit', 'avg_profit']
        
        # Filter to conditions with sufficient samples
        condition_results = condition_results[condition_results['bets'] >= 5]
        
        results['condition'] = condition_results
    
    # Analyze dome vs outdoor
    if 'is_dome' in opportunities.columns:
        # Group by dome status
        dome_results = opportunities.groupby('is_dome').agg({
            'correct': ['count', 'sum', 'mean'],
            'profit': ['sum', 'mean']
        })
        
        dome_results.columns = ['bets', 'wins', 'win_rate', 'total_profit', 'avg_profit']
        
        # Replace binary values with labels
        dome_results = dome_results.reset_index()
        dome_results['is_dome'] = dome_results['is_dome'].map({0: 'Outdoor', 1: 'Dome'})
        dome_results = dome_results.set_index('is_dome')
        
        results['dome'] = dome_results
    
    # Analyze by over/under
    over_results = opportunities[opportunities['recommended_bet'] == 'OVER'].agg({
        'correct': ['count', 'sum', 'mean'],
        'profit': ['sum', 'mean']
    })
    
    under_results = opportunities[opportunities['recommended_bet'] == 'UNDER'].agg({
        'correct': ['count', 'sum', 'mean'],
        'profit': ['sum', 'mean']
    })
    
    over_results.columns = ['bets', 'wins', 'win_rate', 'total_profit', 'avg_profit']
    under_results.columns = ['bets', 'wins', 'win_rate', 'total_profit', 'avg_profit']
    
    results['over'] = over_results
    results['under'] = under_results
    
    # Print summary
    print("\nWeather Pattern Analysis:")
    for category, data in results.items():
        print(f"\n{category.capitalize()} Analysis:")
        print(data[['bets', 'win_rate', 'total_profit']])
    
    return results

import pandas as pd

def process_mlb_data(odds_df, weather_df):
    """
    Processes MLB odds and weather data to generate betting recommendations.

    Args:
        odds_df (pd.DataFrame): DataFrame containing MLB odds data.
        weather_df (pd.DataFrame): DataFrame containing MLB weather data.

    Returns:
        pd.DataFrame: DataFrame with unique betting recommendations.
    """

    # Merge odds and weather data
    merged_df = pd.merge(odds_df, weather_df, on='matchup', how='inner')

    # Calculate predicted runs (example calculation, adjust as needed)
    merged_df['predicted_runs'] = merged_df['temperature'] * 0.75 + merged_df['wind_speed'] * 0.25

    # Generate betting recommendations
    merged_df['recommended_bet'] = merged_df.apply(
        lambda row: 'OVER' if row['predicted_runs'] > row['over_under_line'] else 'UNDER',
        axis=1
    )

    # Calculate confidence (example calculation)
    merged_df['confidence_difference'] = abs(merged_df['predicted_runs'] - merged_df['over_under_line'])

    # Remove duplicate entries based on the 'matchup' column
    final_df = merged_df.drop_duplicates(subset=['matchup'], keep='first').reset_index(drop=True)

    return final_df
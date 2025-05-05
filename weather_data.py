"""
Enhanced weather data module for MLB Weather Model using Open-Meteo API.
Provides historical and forecast weather data with better reliability.
"""

import os
import time
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta

from config import DATA_DIR, STADIUM_MAPPING, WEATHER_THRESHOLDS

def fetch_historical_weather(games_df, cache=True):
    """
    Fetch historical weather data using Open-Meteo API for game locations and dates.
    
    Args:
        games_df (DataFrame): DataFrame containing game information with stadium and date
        cache (bool): Whether to use cached data if available
        
    Returns:
        DataFrame: Weather data for all games
    """
    print("Fetching historical weather data using Open-Meteo API...")
    
    cache_file = f"{DATA_DIR}/open_meteo_weather_cache.csv"
    
    # Check if cache exists and should be used
    if cache and os.path.exists(cache_file):
        print(f"Loading cached weather data from {cache_file}")
        return pd.read_csv(cache_file, parse_dates=['date'])
    
    if games_df is None or len(games_df) == 0:
        raise ValueError("Game data must be provided before fetching weather data.")
    
    weather_list = []
    
    # Group by stadium to reduce API calls
    if 'stadium' in games_df.columns and 'stadium_lat' in games_df.columns and 'stadium_lon' in games_df.columns:
        stadium_groups = games_df.groupby(['stadium', 'stadium_lat', 'stadium_lon'])
    else:
        # Create groups based on home team using stadium mapping
        grouped_games = []
        for _, game in games_df.iterrows():
            if 'home_team' in game:
                home_team = game['home_team']
                if home_team in STADIUM_MAPPING:
                    stadium_info = STADIUM_MAPPING[home_team]
                    grouped_games.append({
                        'stadium': stadium_info['name'],
                        'stadium_lat': stadium_info['lat'],
                        'stadium_lon': stadium_info['lon'],
                        'date': game['date'],
                        'original_idx': _
                    })
        
        if not grouped_games:
            raise ValueError("Cannot group games by stadium. Missing required columns.")
            
        temp_df = pd.DataFrame(grouped_games)
        stadium_groups = temp_df.groupby(['stadium', 'stadium_lat', 'stadium_lon'])
    
    # Process each stadium's games
    for (stadium, lat, lon), stadium_games in stadium_groups:
        print(f"Processing weather for {stadium} ({len(stadium_games)} games)")
        
        # Get unique dates for this stadium
        if 'date' in stadium_games.columns:
            dates = pd.to_datetime(stadium_games['date']).dt.strftime('%Y-%m-%d').unique()
        else:
            print(f"No date column found for {stadium}")
            continue
        
        # Get stadium info for context
        stadium_info = None
        for team, info in STADIUM_MAPPING.items():
            if info['name'] == stadium:
                stadium_info = info
                break
        
        is_domed = stadium_info and stadium_info.get('dome', False)
        has_retractable_roof = stadium_info and stadium_info.get('retractable_roof', False)
        
        # Process dates in chunks to avoid too long URLs
        # Open-Meteo supports up to 365 days in a single API call
        chunk_size = 365  # Maximum number of dates per API call
        
        for i in range(0, len(dates), chunk_size):
            date_chunk = dates[i:i+chunk_size]
            start_date = min(date_chunk)
            end_date = max(date_chunk)
            
            print(f"Fetching weather from {start_date} to {end_date}")
            
            try:
                # Build URL for Open-Meteo Historical Weather API
                url = f"https://archive-api.open-meteo.com/v1/archive"
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": start_date,
                    "end_date": end_date,
                    "hourly": [
                        "temperature_2m", 
                        "relativehumidity_2m", 
                        "precipitation", 
                        "cloudcover", 
                        "pressure_msl", 
                        "windspeed_10m", 
                        "winddirection_10m"
                    ],
                    "timezone": "America/New_York"  # Most MLB games are in US timezones
                }
                
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract hourly data
                    hourly_times = data.get("hourly", {}).get("time", [])
                    hourly_data = {
                        "temp": data.get("hourly", {}).get("temperature_2m", []),
                        "humidity": data.get("hourly", {}).get("relativehumidity_2m", []),
                        "precip": data.get("hourly", {}).get("precipitation", []),
                        "cloud_cover": data.get("hourly", {}).get("cloudcover", []),
                        "pressure": data.get("hourly", {}).get("pressure_msl", []),
                        "wind_speed": data.get("hourly", {}).get("windspeed_10m", []),
                        "wind_direction": data.get("hourly", {}).get("winddirection_10m", [])
                    }
                    
                    # Process hourly data into daily weather for game days
                    for date_str in date_chunk:
                        # Find indices for this date (at typical game time window)
                        # Convert to datetime for comparison
                        game_date = pd.to_datetime(date_str).date()
                        
                        # Find indices for this date between 12:00 and 19:00 (typical game times)
                        game_indices = []
                        for idx, time_str in enumerate(hourly_times):
                            dt = pd.to_datetime(time_str)
                            if dt.date() == game_date and 12 <= dt.hour <= 19:
                                game_indices.append(idx)
                        
                        if game_indices:
                            # Calculate average values during game time
                            avg_temp = np.mean([hourly_data["temp"][i] for i in game_indices if i < len(hourly_data["temp"])])
                            avg_humidity = np.mean([hourly_data["humidity"][i] for i in game_indices if i < len(hourly_data["humidity"])])
                            total_precip = np.sum([hourly_data["precip"][i] for i in game_indices if i < len(hourly_data["precip"])])
                            avg_cloud = np.mean([hourly_data["cloud_cover"][i] for i in game_indices if i < len(hourly_data["cloud_cover"])])
                            avg_pressure = np.mean([hourly_data["pressure"][i] for i in game_indices if i < len(hourly_data["pressure"])])
                            avg_wind = np.mean([hourly_data["wind_speed"][i] for i in game_indices if i < len(hourly_data["wind_speed"])])
                            
                            # For wind direction, we need to use vector averaging
                            sin_sum = np.sum([np.sin(np.radians(hourly_data["wind_direction"][i])) for i in game_indices if i < len(hourly_data["wind_direction"])])
                            cos_sum = np.sum([np.cos(np.radians(hourly_data["wind_direction"][i])) for i in game_indices if i < len(hourly_data["wind_direction"])])
                            avg_direction = np.degrees(np.arctan2(sin_sum, cos_sum)) % 360
                            
                            # Determine weather condition based on cloud cover and precipitation
                            weather_condition = 'Clear'
                            if avg_cloud > 80:
                                weather_condition = 'Clouds'
                            if total_precip > 0.5:
                                weather_condition = 'Rain'
                            elif total_precip > 0.1:
                                weather_condition = 'Drizzle'
                            
                            # Calculate "feels like" temperature
                            feels_like = calculate_feels_like(avg_temp, avg_humidity, avg_wind)
                            
                            # Get stadium orientation to calculate effective wind
                            stadium_orientation = stadium_info.get('orientation', 0) if stadium_info else 0
                            
                            # Calculate wind effect relative to stadium orientation
                            # 0 degrees = wind blowing in from outfield
                            # 180 degrees = wind blowing out to outfield
                            wind_angle_to_field = (avg_direction - stadium_orientation) % 360
                            wind_blowing_out = 135 <= wind_angle_to_field <= 225
                            wind_blowing_in = wind_angle_to_field <= 45 or wind_angle_to_field >= 315
                            wind_blowing_crossfield = not (wind_blowing_in or wind_blowing_out)
                            
                            # Calculate wind effect strength (dot product of wind vector and stadium vector)
                            # 1.0 = directly out, -1.0 = directly in, 0 = perpendicular
                            wind_effect = np.cos(np.radians(wind_angle_to_field - 180))
                            
                            # Apply dome adjustments
                            if is_domed and not has_retractable_roof:
                                # Fully domed stadium - no weather effect
                                avg_wind = 0
                                total_precip = 0
                                wind_effect = 0
                                weather_condition = "Dome"
                                weather_description = "Indoor stadium"
                            elif has_retractable_roof:
                                # Retractable roof likely closed in bad weather
                                if total_precip > WEATHER_THRESHOLDS.get('rain_precip', 0.1) or avg_temp < WEATHER_THRESHOLDS.get('cold_temp', 50):
                                    avg_wind = 0
                                    total_precip = 0
                                    wind_effect = 0
                                    weather_condition = "Dome"
                                    weather_description = "Retractable roof likely closed"
                                else:
                                    weather_description = f"{weather_condition}, {avg_cloud:.0f}% cloud cover"
                            else:
                                weather_description = f"{weather_condition}, {avg_cloud:.0f}% cloud cover"
                            
                            # Add to weather list
                            weather_list.append({
                                'date': date_str,
                                'stadium': stadium,
                                'temperature': avg_temp * 9/5 + 32,  # Convert from C to F
                                'feels_like': feels_like * 9/5 + 32,  # Convert from C to F
                                'humidity': avg_humidity,
                                'pressure': avg_pressure / 100,  # Convert from Pa to hPa
                                'wind_speed': avg_wind * 2.237,  # Convert from m/s to mph
                                'wind_direction': avg_direction,
                                'wind_effect': wind_effect,
                                'wind_blowing_out': int(wind_blowing_out),
                                'wind_blowing_in': int(wind_blowing_in),
                                'wind_blowing_crossfield': int(wind_blowing_crossfield),
                                'cloud_cover': avg_cloud,
                                'weather_condition': weather_condition,
                                'weather_description': weather_description,
                                'precipitation': total_precip,
                                'is_dome': int(is_domed),
                                'has_retractable_roof': int(has_retractable_roof),
                                'data_source': 'open-meteo'
                            })
                        else:
                            print(f"No weather data found for {date_str} at {stadium}")
                            
                            # Add missing data placeholder with reasonable values
                            weather_list.append(generate_synthetic_weather(date_str, stadium, is_domed, has_retractable_roof))
                else:
                    print(f"Error fetching weather: {response.status_code}, {response.text}")
                    
                    # Generate synthetic data for this date range
                    for date_str in date_chunk:
                        weather_list.append(generate_synthetic_weather(date_str, stadium, is_domed, has_retractable_roof))
                
                # Add a small delay between API calls
                time.sleep(1)
                
            except Exception as e:
                print(f"Exception fetching weather for {stadium} ({start_date} to {end_date}): {e}")
                
                # Generate synthetic data for this date range
                for date_str in date_chunk:
                    weather_list.append(generate_synthetic_weather(date_str, stadium, is_domed, has_retractable_roof))
    
    # Create DataFrame from weather data
    weather_df = pd.DataFrame(weather_list)
    
    # Fill missing values with reasonable estimates
    numeric_cols = ['temperature', 'feels_like', 'humidity', 'pressure', 
                   'wind_speed', 'wind_direction', 'cloud_cover', 
                   'precipitation', 'wind_effect']
    
    for col in numeric_cols:
        if col in weather_df.columns:
            weather_df[col] = pd.to_numeric(weather_df[col], errors='coerce')
            # Fill missing values with column median
            weather_df[col].fillna(weather_df[col].median(), inplace=True)
    
    # Save to cache
    if cache:
        weather_df.to_csv(cache_file, index=False)
        print(f"Cached weather data to {cache_file}")
    
    # Calculate percentage of real vs synthetic data
    if 'data_source' in weather_df.columns:
        real_count = len(weather_df[weather_df['data_source'] == 'open-meteo'])
        synthetic_count = len(weather_df[weather_df['data_source'] == 'synthetic'])
        total_count = len(weather_df)
        print(f"Weather data sources: {real_count} real ({real_count/total_count*100:.1f}%), "
              f"{synthetic_count} synthetic ({synthetic_count/total_count*100:.1f}%)")
    
    return weather_df

def get_current_weather(stadium_code, game_datetime=None):
    """
    Get current or forecast weather for a stadium using Open-Meteo API.
    
    Args:
        stadium_code (str): Stadium code (e.g., 'NYY')
        game_datetime (datetime): Game date and time
        
    Returns:
        dict: Weather data for the stadium at game time
    """
    stadium_info = STADIUM_MAPPING.get(stadium_code)
    if not stadium_info:
        print(f"Stadium code {stadium_code} not found")
        return None
    
    lat = stadium_info.get('lat')
    lon = stadium_info.get('lon')
    stadium_name = stadium_info.get('name')
    
    # Default to current time if no game time provided
    if game_datetime is None:
        game_datetime = datetime.now()
    
    # Determine if we need current, forecast, or historical weather
    now = datetime.now()
    
    # Convert to date for comparison
    game_date = game_datetime.date()
    now_date = now.date()
    
    # Get weather condition
    try:
        # For future dates, use forecast API
        if game_date > now_date:
            return get_forecast_weather(lat, lon, game_datetime, stadium_info)
        # For past dates, use historical API
        elif game_date < now_date:
            return get_historical_weather(lat, lon, game_datetime, stadium_info)
        # For today, use current weather API
        else:
            return get_forecast_weather(lat, lon, game_datetime, stadium_info)
    except Exception as e:
        print(f"Error getting weather for {stadium_name}: {e}")
        return generate_synthetic_weather(game_datetime.strftime('%Y-%m-%d'), stadium_name, 
                                         stadium_info.get('dome', False), 
                                         stadium_info.get('retractable_roof', False))

def get_forecast_weather(lat, lon, game_datetime, stadium_info):
    """
    Get weather forecast for a specific location and time using Open-Meteo API.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        game_datetime (datetime): Game date and time
        stadium_info (dict): Stadium information
        
    Returns:
        dict: Forecast weather data
    """
    # Format dates for API
    start_date = game_datetime.strftime('%Y-%m-%d')
    
    # Build URL for Open-Meteo Forecast API
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m", 
            "relativehumidity_2m", 
            "precipitation", 
            "cloudcover", 
            "pressure_msl", 
            "windspeed_10m", 
            "winddirection_10m"
        ],
        "start_date": start_date,
        "end_date": start_date,
        "timezone": "America/New_York"
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        
        # Extract hourly data
        hourly_times = data.get("hourly", {}).get("time", [])
        hourly_data = {
            "temp": data.get("hourly", {}).get("temperature_2m", []),
            "humidity": data.get("hourly", {}).get("relativehumidity_2m", []),
            "precip": data.get("hourly", {}).get("precipitation", []),
            "cloud_cover": data.get("hourly", {}).get("cloudcover", []),
            "pressure": data.get("hourly", {}).get("pressure_msl", []),
            "wind_speed": data.get("hourly", {}).get("windspeed_10m", []),
            "wind_direction": data.get("hourly", {}).get("winddirection_10m", [])
        }
        
        # Find the closest hour to game time
        game_hour = game_datetime.hour
        target_hour_idx = None
        
        for idx, time_str in enumerate(hourly_times):
            dt = pd.to_datetime(time_str)
            if dt.hour == game_hour:
                target_hour_idx = idx
                break
        
        # If exact hour not found, find the closest
        if target_hour_idx is None:
            for idx, time_str in enumerate(hourly_times):
                dt = pd.to_datetime(time_str)
                if dt.date() == game_datetime.date():
                    if target_hour_idx is None or abs(dt.hour - game_hour) < abs(pd.to_datetime(hourly_times[target_hour_idx]).hour - game_hour):
                        target_hour_idx = idx
        
        # If still no match, use midday
        if target_hour_idx is None:
            print(f"No forecast found for game time, using midday forecast")
            for idx, time_str in enumerate(hourly_times):
                dt = pd.to_datetime(time_str)
                if dt.date() == game_datetime.date() and dt.hour in [12, 13, 14]:
                    target_hour_idx = idx
                    break
        
        # If still no match, use first hour of the day
        if target_hour_idx is None:
            for idx, time_str in enumerate(hourly_times):
                dt = pd.to_datetime(time_str)
                if dt.date() == game_datetime.date():
                    target_hour_idx = idx
                    break
        
        # If no hours found for the date, return synthetic data
        if target_hour_idx is None:
            print(f"No forecast data found for {game_datetime.strftime('%Y-%m-%d')}")
            return generate_synthetic_weather(
                game_datetime.strftime('%Y-%m-%d'),
                stadium_info['name'],
                stadium_info.get('dome', False),
                stadium_info.get('retractable_roof', False)
            )
        
        # Extract weather at game time
        try:
            temp = hourly_data["temp"][target_hour_idx]
            humidity = hourly_data["humidity"][target_hour_idx]
            precip = hourly_data["precip"][target_hour_idx]
            cloud_cover = hourly_data["cloud_cover"][target_hour_idx]
            pressure = hourly_data["pressure"][target_hour_idx]
            wind_speed = hourly_data["wind_speed"][target_hour_idx]
            wind_direction = hourly_data["wind_direction"][target_hour_idx]
            
            # Determine weather condition
            weather_condition = 'Clear'
            if cloud_cover > 80:
                weather_condition = 'Clouds'
            if precip > 0.5:
                weather_condition = 'Rain'
            elif precip > 0.1:
                weather_condition = 'Drizzle'
            
            # Calculate feels like temperature
            feels_like = calculate_feels_like(temp, humidity, wind_speed)
            
            # Get dome status
            is_domed = stadium_info.get('dome', False)
            has_retractable_roof = stadium_info.get('retractable_roof', False)
            
            # Get stadium orientation for wind effect
            stadium_orientation = stadium_info.get('orientation', 0)
            
            # Calculate wind effect
            wind_angle_to_field = (wind_direction - stadium_orientation) % 360
            wind_blowing_out = 135 <= wind_angle_to_field <= 225
            wind_blowing_in = wind_angle_to_field <= 45 or wind_angle_to_field >= 315
            wind_blowing_crossfield = not (wind_blowing_in or wind_blowing_out)
            
            # Calculate wind effect strength
            wind_effect = np.cos(np.radians(wind_angle_to_field - 180))
            
            # Apply dome adjustments
            if is_domed and not has_retractable_roof:
                # Fully domed stadium - no weather effect
                wind_speed = 0
                precip = 0
                wind_effect = 0
                weather_condition = "Dome"
                weather_description = "Indoor stadium"
            elif has_retractable_roof:
                # Retractable roof likely closed in bad weather
                if precip > WEATHER_THRESHOLDS.get('rain_precip', 0.1) or temp < WEATHER_THRESHOLDS.get('cold_temp', 50):
                    wind_speed = 0
                    precip = 0
                    wind_effect = 0
                    weather_condition = "Dome"
                    weather_description = "Retractable roof likely closed"
                else:
                    weather_description = f"{weather_condition}, {cloud_cover:.0f}% cloud cover"
            else:
                weather_description = f"{weather_condition}, {cloud_cover:.0f}% cloud cover"
            
            # Return weather data
            return {
                'stadium': stadium_info['name'],
                'temperature': temp * 9/5 + 32,  # Convert from C to F
                'feels_like': feels_like * 9/5 + 32,  # Convert from C to F
                'humidity': humidity,
                'pressure': pressure / 100,  # Convert from Pa to hPa
                'wind_speed': wind_speed * 2.237,  # Convert from m/s to mph
                'wind_direction': wind_direction,
                'wind_effect': wind_effect,
                'wind_blowing_out': int(wind_blowing_out),
                'wind_blowing_in': int(wind_blowing_in),
                'wind_blowing_crossfield': int(wind_blowing_crossfield),
                'cloud_cover': cloud_cover,
                'weather_condition': weather_condition,
                'weather_description': weather_description,
                'precipitation': precip,
                'is_dome': int(is_domed),
                'has_retractable_roof': int(has_retractable_roof),
                'data_source': 'open-meteo-forecast'
            }
        except Exception as e:
            print(f"Error extracting forecast data: {e}")
            return generate_synthetic_weather(
                game_datetime.strftime('%Y-%m-%d'),
                stadium_info['name'],
                stadium_info.get('dome', False),
                stadium_info.get('retractable_roof', False)
            )
    else:
        print(f"Error fetching forecast: {response.status_code}, {response.text}")
        return generate_synthetic_weather(
            game_datetime.strftime('%Y-%m-%d'),
            stadium_info['name'],
            stadium_info.get('dome', False),
            stadium_info.get('retractable_roof', False)
        )

def get_historical_weather(lat, lon, game_datetime, stadium_info):
    """
    Get historical weather for a specific location and time using Open-Meteo API.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        game_datetime (datetime): Game date and time
        stadium_info (dict): Stadium information
        
    Returns:
        dict: Historical weather data
    """
    # Format dates for API
    date_str = game_datetime.strftime('%Y-%m-%d')
    
    # Build URL for Open-Meteo Historical API
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": [
            "temperature_2m", 
            "relativehumidity_2m", 
            "precipitation", 
            "cloudcover", 
            "pressure_msl", 
            "windspeed_10m", 
            "winddirection_10m"
        ],
        "timezone": "America/New_York"
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        
        # Process same as forecast data
        return process_weather_data(data, game_datetime, stadium_info)
    else:
        print(f"Error fetching historical weather: {response.status_code}, {response.text}")
        return generate_synthetic_weather(
            date_str,
            stadium_info['name'],
            stadium_info.get('dome', False),
            stadium_info.get('retractable_roof', False)
        )

def process_weather_data(data, game_datetime, stadium_info):
    """
    Process Open-Meteo API response into weather data.
    
    Args:
        data (dict): API response data
        game_datetime (datetime): Game date and time
        stadium_info (dict): Stadium information
        
    Returns:
        dict: Processed weather data
    """
    # Extract hourly data
    hourly_times = data.get("hourly", {}).get("time", [])
    hourly_data = {
        "temp": data.get("hourly", {}).get("temperature_2m", []),
        "humidity": data.get("hourly", {}).get("relativehumidity_2m", []),
        "precip": data.get("hourly", {}).get("precipitation", []),
        "cloud_cover": data.get("hourly", {}).get("cloudcover", []),
        "pressure": data.get("hourly", {}).get("pressure_msl", []),
        "wind_speed": data.get("hourly", {}).get("windspeed_10m", []),
        "wind_direction": data.get("hourly", {}).get("winddirection_10m", [])
    }
    
    # Find the closest hour to game time
    game_hour = game_datetime.hour
    target_hour_idx = None
    
    for idx, time_str in enumerate(hourly_times):
        dt = pd.to_datetime(time_str)
        if dt.hour == game_hour:
            target_hour_idx = idx
            break
    
    # If exact hour not found, find the closest
    if target_hour_idx is None:
        for idx, time_str in enumerate(hourly_times):
            dt = pd.to_datetime(time_str)
            if dt.date() == game_datetime.date():
                if target_hour_idx is None or abs(dt.hour - game_hour) < abs(pd.to_datetime(hourly_times[target_hour_idx]).hour - game_hour):
                    target_hour_idx = idx
    
    # If still no match, use midday
    if target_hour_idx is None:
        for idx, time_str in enumerate(hourly_times):
            dt = pd.to_datetime(time_str)
            if dt.date() == game_datetime.date() and dt.hour in [12, 13, 14]:
                target_hour_idx = idx
                break
    
    # If still no match, use first hour of the day
    if target_hour_idx is None:
        for idx, time_str in enumerate(hourly_times):
            dt = pd.to_datetime(time_str)
            if dt.date() == game_datetime.date():
                target_hour_idx = idx
                break
    
    # If no hours found for the date, return synthetic data
    if target_hour_idx is None or not hourly_times:
        print(f"No weather data found for {game_datetime.strftime('%Y-%m-%d')}")
        return generate_synthetic_weather(
            game_datetime.strftime('%Y-%m-%d'),
            stadium_info['name'],
            stadium_info.get('dome', False),
            stadium_info.get('retractable_roof', False)
        )
    
    # Extract weather at game time
    try:
        temp = hourly_data["temp"][target_hour_idx]
        humidity = hourly_data["humidity"][target_hour_idx]
        precip = hourly_data["precip"][target_hour_idx]
        cloud_cover = hourly_data["cloud_cover"][target_hour_idx]
        pressure = hourly_data["pressure"][target_hour_idx]
        wind_speed = hourly_data["wind_speed"][target_hour_idx]
        wind_direction = hourly_data["wind_direction"][target_hour_idx]
        
        # Determine weather condition
        weather_condition = 'Clear'
        if cloud_cover > 80:
            weather_condition = 'Clouds'
        if precip > 0.5:
            weather_condition = 'Rain'
        elif precip > 0.1:
            weather_condition = 'Drizzle'
        
        # Calculate feels like temperature
        feels_like = calculate_feels_like(temp, humidity, wind_speed)
        
        # Get dome status
        is_domed = stadium_info.get('dome', False)
        has_retractable_roof = stadium_info.get('retractable_roof', False)
        
        # Get stadium orientation for wind effect
        stadium_orientation = stadium_info.get('orientation', 0)
        
        # Calculate wind effect
        wind_angle_to_field = (wind_direction - stadium_orientation) % 360
        wind_blowing_out = 135 <= wind_angle_to_field <= 225
        wind_blowing_in = wind_angle_to_field <= 45 or wind_angle_to_field >= 315
        wind_blowing_crossfield = not (wind_blowing_in or wind_blowing_out)
        
        # Calculate wind effect strength
        wind_effect = np.cos(np.radians(wind_angle_to_field - 180))
        
        # Apply dome adjustments
        if is_domed and not has_retractable_roof:
            # Fully domed stadium - no weather effect
            wind_speed = 0
            precip = 0
            wind_effect = 0
            weather_condition = "Dome"
            weather_description = "Indoor stadium"
        elif has_retractable_roof:
            # Retractable roof likely closed in bad weather
            if precip > WEATHER_THRESHOLDS.get('rain_precip', 0.1) or temp < WEATHER_THRESHOLDS.get('cold_temp', 50):
                wind_speed = 0
                precip = 0
                wind_effect = 0
                weather_condition = "Dome"
                weather_description = "Retractable roof likely closed"
            else:
                weather_description = f"{weather_condition}, {cloud_cover:.0f}% cloud cover"
        else:
            weather_description = f"{weather_condition}, {cloud_cover:.0f}% cloud cover"
        
        # Return weather data
        return {
            'stadium': stadium_info['name'],
            'temperature': temp * 9/5 + 32,  # Convert from C to F
            'feels_like': feels_like * 9/5 + 32,  # Convert from C to F
            'humidity': humidity,
            'pressure': pressure / 100,  # Convert from Pa to hPa
            'wind_speed': wind_speed * 2.237,  # Convert from m/s to mph
            'wind_direction': wind_direction,
            'wind_effect': wind_effect,
            'wind_blowing_out': int(wind_blowing_out),
            'wind_blowing_in': int(wind_blowing_in),
            'wind_blowing_crossfield': int(wind_blowing_crossfield),
            'cloud_cover': cloud_cover,
            'weather_condition': weather_condition,
            'weather_description': weather_description,
            'precipitation': precip,
            'is_dome': int(is_domed),
            'has_retractable_roof': int(has_retractable_roof),
            'data_source': 'open-meteo'
        }
    except Exception as e:
        print(f"Error extracting weather data: {e}")
        return generate_synthetic_weather(
            game_datetime.strftime('%Y-%m-%d'),
            stadium_info['name'],
            stadium_info.get('dome', False),
            stadium_info.get('retractable_roof', False)
        )

def calculate_feels_like(temp_c, humidity, wind_speed):
    """
    Calculate "feels like" temperature using both heat index and wind chill.
    
    Args:
        temp_c (float): Temperature in Celsius
        humidity (float): Relative humidity (%)
        wind_speed (float): Wind speed in m/s
        
    Returns:
        float: Feels like temperature in Celsius
    """
    # Convert to Fahrenheit for standard formulas
    temp_f = temp_c * 9/5 + 32
    wind_mph = wind_speed * 2.237
    
    # Heat index - used when temperature is above 80°F
    if temp_f >= 80:
        # Simplified heat index formula
        hi = 0.5 * (temp_f + 61.0 + ((temp_f - 68.0) * 1.2) + (humidity * 0.094))
        
        # If the result is less than 80°F, use a more accurate formula
        if hi >= 80:
            hi = -42.379 + (2.04901523 * temp_f) + (10.14333127 * humidity) - \
                (0.22475541 * temp_f * humidity) - (6.83783e-3 * temp_f**2) - \
                (5.481717e-2 * humidity**2) + (1.22874e-3 * temp_f**2 * humidity) + \
                (8.5282e-4 * temp_f * humidity**2) - (1.99e-6 * temp_f**2 * humidity**2)
            
        # Convert back to Celsius
        return (hi - 32) * 5/9
    
    # Wind chill - used when temperature is below 50°F and wind speed is above 3 mph
    elif temp_f <= 50 and wind_mph > 3:
        wc = 35.74 + (0.6215 * temp_f) - (35.75 * wind_mph**0.16) + (0.4275 * temp_f * wind_mph**0.16)
        
        # Convert back to Celsius
        return (wc - 32) * 5/9
    
    # If neither heat index nor wind chill apply, return the actual temperature
    else:
        return temp_c

def generate_synthetic_weather(date_str, stadium, is_domed=False, has_retractable_roof=False):
    """
    Generate synthetic weather data when API calls fail.
    
    Args:
        date_str (str): Date string in YYYY-MM-DD format
        stadium (str): Stadium name
        is_domed (bool): Whether stadium has a dome
        has_retractable_roof (bool): Whether stadium has a retractable roof
        
    Returns:
        dict: Synthetic weather data
    """
    # Get season-appropriate temperature based on date
    date_obj = pd.to_datetime(date_str).date()
    month = date_obj.month
    
    # Seasonal temperature adjustment
    if 5 <= month <= 9:  # Summer months
        base_temp = np.random.randint(70, 90)
    elif month in [4, 10]:  # Spring/Fall
        base_temp = np.random.randint(55, 75)
    else:  # Winter (rare for baseball)
        base_temp = np.random.randint(40, 60)
    
    # Wind and precipitation adjustments for domed stadiums
    if is_domed and not has_retractable_roof:
        # Fully enclosed dome
        wind_speed = 0
        precipitation = 0
        weather_condition = "Dome"
        weather_description = "Indoor stadium"
        wind_effect = 0
        wind_blowing_out = 0
        wind_blowing_in = 0
        wind_blowing_crossfield = 0
    elif has_retractable_roof:
        # Retractable roof - might be open in good weather
        wind_speed = np.random.randint(0, 10)
        precipitation = np.random.uniform(0, 0.1)
        
        if precipitation > 0.05 or base_temp < 50:
            # Bad weather - roof likely closed
            wind_speed = 0
            precipitation = 0
            weather_condition = "Dome"
            weather_description = "Retractable roof likely closed"
            wind_effect = 0
            wind_blowing_out = 0
            wind_blowing_in = 0
            wind_blowing_crossfield = 0
        else:
            # Good weather - roof likely open
            weather_condition = np.random.choice(['Clear', 'Clouds', 'Partly Cloudy'])
            weather_description = f"Synthetic {weather_condition.lower()} data"
            wind_direction = np.random.randint(0, 360)
            wind_effect = np.random.uniform(-1, 1)
            wind_blowing_out = int(wind_effect > 0.3)
            wind_blowing_in = int(wind_effect < -0.3)
            wind_blowing_crossfield = int(not (wind_blowing_out or wind_blowing_in))
    else:
        # Outdoor stadium
        wind_speed = np.random.randint(0, 20)
        precipitation = np.random.uniform(0, 0.5) if np.random.random() < 0.2 else 0
        
        if precipitation > 0.1:
            weather_condition = 'Rain'
        elif precipitation > 0:
            weather_condition = 'Drizzle'
        else:
            cloud_cover = np.random.randint(0, 100)
            if cloud_cover > 80:
                weather_condition = 'Clouds'
            elif cloud_cover > 30:
                weather_condition = 'Partly Cloudy'
            else:
                weather_condition = 'Clear'
        
        weather_description = f"Synthetic {weather_condition.lower()} data"
        wind_direction = np.random.randint(0, 360)
        wind_effect = np.random.uniform(-1, 1)
        wind_blowing_out = int(wind_effect > 0.3)
        wind_blowing_in = int(wind_effect < -0.3)
        wind_blowing_crossfield = int(not (wind_blowing_out or wind_blowing_in))
    
    return {
        'date': date_str,
        'stadium': stadium,
        'temperature': base_temp,
        'feels_like': base_temp + np.random.randint(-5, 5),
        'humidity': np.random.randint(30, 90),
        'pressure': np.random.randint(990, 1030),
        'wind_speed': wind_speed,
        'wind_direction': np.random.randint(0, 360) if not is_domed else 0,
        'wind_effect': wind_effect if 'wind_effect' in locals() else 0,
        'wind_blowing_out': wind_blowing_out if 'wind_blowing_out' in locals() else 0,
        'wind_blowing_in': wind_blowing_in if 'wind_blowing_in' in locals() else 0,
        'wind_blowing_crossfield': wind_blowing_crossfield if 'wind_blowing_crossfield' in locals() else 0,
        'cloud_cover': np.random.randint(0, 100) if not is_domed else 0,
        'weather_condition': weather_condition,
        'weather_description': weather_description,
        'precipitation': precipitation,
        'is_dome': int(is_domed),
        'has_retractable_roof': int(has_retractable_roof),
        'data_source': 'synthetic'
    }
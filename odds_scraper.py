
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime, timedelta
import os
from config import DATA_DIR, TEAM_NAME_MAP

class OddsScraper:
    def __init__(self, cache=True):
        self.base_url = "https://www.sportsbookreview.com/betting-odds/mlb-baseball/"
        self.cache_dir = f"{DATA_DIR}/odds_cache"
        self.use_cache = cache
        
        # Create cache directory if needed
        if self.use_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
    def fetch_historical_odds(self, start_date, end_date):
        """Fetch historical MLB odds for a date range."""
        all_odds = []
        current_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            print(f"Fetching odds for {date_str}")
            
            # Check cache first
            cache_file = f"{self.cache_dir}/odds_{date_str}.csv"
            if self.use_cache and os.path.exists(cache_file):
                daily_odds = pd.read_csv(cache_file)
                all_odds.append(daily_odds)
            else:
                try:
                    # Construct URL
                    url = f"{self.base_url}?date={date_str}"
                    
                    # Add delay to avoid rate limiting
                    time.sleep(2)
                    
                    # Make request
                    response = requests.get(url, headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    })
                    
                    if response.status_code == 200:
                        # Parse odds data
                        daily_odds = self._parse_odds_page(response.text, date_str)
                        
                        # Cache the results
                        if self.use_cache and len(daily_odds) > 0:
                            daily_odds.to_csv(cache_file, index=False)
                            
                        all_odds.append(daily_odds)
                    else:
                        print(f"Error fetching odds for {date_str}: {response.status_code}")
                        
                except Exception as e:
                    print(f"Error processing {date_str}: {e}")
            
            current_date += timedelta(days=1)
            
        # Combine all odds
        if all_odds:
            combined_odds = pd.concat(all_odds, ignore_index=True)
            return combined_odds
        return pd.DataFrame()
    
    def _parse_odds_page(self, html, date_str):
        """Parse the odds page HTML."""
        soup = BeautifulSoup(html, 'html.parser')
        games = []
        
        # Find game containers
        for game in soup.find_all('div', class_='game'):
            try:
                # Extract teams
                teams = game.find_all('div', class_='team-name')
                away_team = self._map_team_name(teams[0].text.strip())
                home_team = self._map_team_name(teams[1].text.strip())
                
                # Extract odds
                odds = game.find_all('div', class_='odds')
                totals = []
                over_odds = []
                under_odds = []
                
                for book_odds in odds:
                    if 'total' in book_odds.get('class', []):
                        total = float(book_odds.text.strip().split()[0])
                        over = int(book_odds.find('span', class_='over').text)
                        under = int(book_odds.find('span', class_='under').text)
                        
                        totals.append(total)
                        over_odds.append(over)
                        under_odds.append(under)
                
                # Use consensus line if available, otherwise average
                if totals:
                    games.append({
                        'date': date_str,
                        'home_team': home_team,
                        'away_team': away_team,
                        'over_under_line': totals[0],  # Consensus line
                        'over_odds': over_odds[0],
                        'under_odds': under_odds[0],
                        'avg_total': sum(totals) / len(totals)
                    })
                    
            except Exception as e:
                print(f"Error parsing game: {e}")
                
        return pd.DataFrame(games)
    
    def _map_team_name(self, name):
        """Map team name to standard abbreviation."""
        if name in TEAM_NAME_MAP:
            return TEAM_NAME_MAP[name]
        # Try to match partial names
        for abbr, full_name in TEAM_NAME_MAP.items():
            if name in full_name:
                return abbr
        return name
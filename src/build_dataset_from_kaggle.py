"""
Alternative data pipeline: Build the game dataset from the Kaggle historical NBA dataset.
Source: https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores/

Since we can't use APIs, this script can also generate synthetic-complete game data
from Basketball Reference HTML pages, or load from pre-downloaded CSVs.

This module provides a FALLBACK dataset builder that constructs the game table
from multiple publicly available CSV sources commonly found on Kaggle.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime


DATA_DIR = "/home/user/Basketballbet/data/raw"
PROCESSED_DIR = "/home/user/Basketballbet/data/processed"


# ============================================================
# Team name normalization - critical for joining across sources
# ============================================================
TEAM_NAME_MAP = {
    # Current teams
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Charlotte Bobcats": "CHA",
    "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET", "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "LA Clippers": "LAC",
    "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
    "New Orleans Hornets": "NOP", "New Orleans/Oklahoma City Hornets": "NOP",
    "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Seattle SuperSonics": "OKC",  # Franchise continuity
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
    # Abbreviation variants
    "ATL": "ATL", "BOS": "BOS", "BKN": "BKN", "BRK": "BKN",
    "CHA": "CHA", "CHO": "CHA", "CHH": "CHA", "CHI": "CHI",
    "CLE": "CLE", "DAL": "DAL", "DEN": "DEN", "DET": "DET",
    "GSW": "GSW", "GS": "GSW", "HOU": "HOU", "IND": "IND",
    "LAC": "LAC", "LAL": "LAL", "MEM": "MEM", "MIA": "MIA",
    "MIL": "MIL", "MIN": "MIN", "NOP": "NOP", "NO": "NOP",
    "NOH": "NOP", "NOK": "NOP", "NYK": "NYK", "NY": "NYK",
    "OKC": "OKC", "SEA": "OKC", "ORL": "ORL", "PHI": "PHI",
    "PHX": "PHX", "PHO": "PHX", "POR": "POR", "SAC": "SAC",
    "SAS": "SAS", "SA": "SAS", "TOR": "TOR", "UTA": "UTA",
    "UTAH": "UTA", "WAS": "WAS", "WSH": "WAS",
    "New Jersey Nets": "BKN", "NJN": "BKN",
    "Vancouver Grizzlies": "MEM", "VAN": "MEM",
}


def normalize_team(name):
    """Normalize team name to 3-letter abbreviation."""
    if name is None:
        return None
    name = str(name).strip().replace("*", "")
    if name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[name]
    # Try partial matching
    for full_name, abbrev in TEAM_NAME_MAP.items():
        if name.lower() in full_name.lower() or full_name.lower() in name.lower():
            return abbrev
    return name


# ============================================================
# Arena locations for travel/timezone computation
# ============================================================
ARENA_LOCATIONS = {
    "ATL": {"city": "Atlanta", "state": "GA", "lat": 33.757, "lon": -84.396, "tz": "US/Eastern", "altitude_ft": 1050},
    "BOS": {"city": "Boston", "state": "MA", "lat": 42.366, "lon": -71.062, "tz": "US/Eastern", "altitude_ft": 20},
    "BKN": {"city": "Brooklyn", "state": "NY", "lat": 40.682, "lon": -73.975, "tz": "US/Eastern", "altitude_ft": 30},
    "CHA": {"city": "Charlotte", "state": "NC", "lat": 35.225, "lon": -80.839, "tz": "US/Eastern", "altitude_ft": 751},
    "CHI": {"city": "Chicago", "state": "IL", "lat": 41.881, "lon": -87.674, "tz": "US/Central", "altitude_ft": 594},
    "CLE": {"city": "Cleveland", "state": "OH", "lat": 41.496, "lon": -81.688, "tz": "US/Eastern", "altitude_ft": 653},
    "DAL": {"city": "Dallas", "state": "TX", "lat": 32.790, "lon": -96.810, "tz": "US/Central", "altitude_ft": 430},
    "DEN": {"city": "Denver", "state": "CO", "lat": 39.749, "lon": -104.999, "tz": "US/Mountain", "altitude_ft": 5280},
    "DET": {"city": "Detroit", "state": "MI", "lat": 42.341, "lon": -83.055, "tz": "US/Eastern", "altitude_ft": 600},
    "GSW": {"city": "San Francisco", "state": "CA", "lat": 37.768, "lon": -122.388, "tz": "US/Pacific", "altitude_ft": 16},
    "HOU": {"city": "Houston", "state": "TX", "lat": 29.751, "lon": -95.362, "tz": "US/Central", "altitude_ft": 50},
    "IND": {"city": "Indianapolis", "state": "IN", "lat": 39.764, "lon": -86.155, "tz": "US/Eastern", "altitude_ft": 717},
    "LAC": {"city": "Los Angeles", "state": "CA", "lat": 34.043, "lon": -118.267, "tz": "US/Pacific", "altitude_ft": 305},
    "LAL": {"city": "Los Angeles", "state": "CA", "lat": 34.043, "lon": -118.267, "tz": "US/Pacific", "altitude_ft": 305},
    "MEM": {"city": "Memphis", "state": "TN", "lat": 35.138, "lon": -90.051, "tz": "US/Central", "altitude_ft": 337},
    "MIA": {"city": "Miami", "state": "FL", "lat": 25.781, "lon": -80.187, "tz": "US/Eastern", "altitude_ft": 6},
    "MIL": {"city": "Milwaukee", "state": "WI", "lat": 43.045, "lon": -87.917, "tz": "US/Central", "altitude_ft": 617},
    "MIN": {"city": "Minneapolis", "state": "MN", "lat": 44.979, "lon": -93.276, "tz": "US/Central", "altitude_ft": 830},
    "NOP": {"city": "New Orleans", "state": "LA", "lat": 29.949, "lon": -90.082, "tz": "US/Central", "altitude_ft": 3},
    "NYK": {"city": "New York", "state": "NY", "lat": 40.751, "lon": -73.994, "tz": "US/Eastern", "altitude_ft": 33},
    "OKC": {"city": "Oklahoma City", "state": "OK", "lat": 35.463, "lon": -97.515, "tz": "US/Central", "altitude_ft": 1201},
    "ORL": {"city": "Orlando", "state": "FL", "lat": 28.539, "lon": -81.384, "tz": "US/Eastern", "altitude_ft": 82},
    "PHI": {"city": "Philadelphia", "state": "PA", "lat": 39.901, "lon": -75.172, "tz": "US/Eastern", "altitude_ft": 39},
    "PHX": {"city": "Phoenix", "state": "AZ", "lat": 33.446, "lon": -112.071, "tz": "US/Arizona", "altitude_ft": 1086},
    "POR": {"city": "Portland", "state": "OR", "lat": 45.532, "lon": -122.667, "tz": "US/Pacific", "altitude_ft": 50},
    "SAC": {"city": "Sacramento", "state": "CA", "lat": 38.580, "lon": -121.500, "tz": "US/Pacific", "altitude_ft": 30},
    "SAS": {"city": "San Antonio", "state": "TX", "lat": 29.427, "lon": -98.438, "tz": "US/Central", "altitude_ft": 650},
    "TOR": {"city": "Toronto", "state": "ON", "lat": 43.643, "lon": -79.379, "tz": "US/Eastern", "altitude_ft": 249},
    "UTA": {"city": "Salt Lake City", "state": "UT", "lat": 40.768, "lon": -111.901, "tz": "US/Mountain", "altitude_ft": 4327},
    "WAS": {"city": "Washington", "state": "DC", "lat": 38.898, "lon": -77.021, "tz": "US/Eastern", "altitude_ft": 25},
}

# Timezone offsets from Eastern (for simple computation without pytz)
TZ_OFFSET = {
    "US/Eastern": 0, "US/Central": -1, "US/Mountain": -2,
    "US/Pacific": -3, "US/Arizona": -2,
}


def haversine_miles(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in miles."""
    R = 3959  # Earth's radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def build_game_dataset_from_scraped(games_csv_path):
    """
    Build a clean game dataset from scraped Basketball Reference data.
    Normalizes team names and adds basic computed fields.
    """
    print(f"Loading games from {games_csv_path}...")
    df = pd.read_csv(games_csv_path)
    print(f"  Loaded {len(df)} games")

    # Normalize team names
    df["home_team_id"] = df["home_team"].apply(normalize_team)
    df["visitor_team_id"] = df["visitor_team"].apply(normalize_team)

    # Ensure date is datetime
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Add basic derived columns
    df["home_win"] = (df["home_pts"] > df["visitor_pts"]).astype(int)
    df["margin"] = df["home_pts"] - df["visitor_pts"]
    df["total_points"] = df["home_pts"] + df["visitor_pts"]
    df["is_overtime"] = df["overtime"].apply(lambda x: 1 if pd.notna(x) and str(x).strip() != "" else 0)

    # Game ID
    df["game_id"] = df.apply(
        lambda r: f"{r['date'].strftime('%Y%m%d')}_{r['visitor_team_id']}_{r['home_team_id']}",
        axis=1
    )

    output_path = os.path.join(PROCESSED_DIR, "games_clean.csv")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Saved clean games to {output_path}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Seasons: {sorted(df['season'].unique())}")

    return df


if __name__ == "__main__":
    games_csv = os.path.join(DATA_DIR, "all_nba_games_2003_2026.csv")
    if os.path.exists(games_csv):
        build_game_dataset_from_scraped(games_csv)
    else:
        print("No scraped games file found. Run scrape_games.py first.")

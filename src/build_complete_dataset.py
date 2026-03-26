"""
Complete NBA Game Dataset Builder
====================================
Combines:
1. FiveThirtyEight historical data (real games, 2003-04 season only)
2. Basketball Reference season CSVs (real games, 2004-05 through 2024-25)

Every game from 2003-04 through 2024-25 with: date, teams, scores, home/away.
All data is from REAL games — no synthetic/generated data.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

DATA_DIR = "/home/user/Basketballbet/data/raw"
BBALL_REF_DIR = os.path.join(DATA_DIR, "bball_ref_seasons")
PROCESSED_DIR = "/home/user/Basketballbet/data/processed"

# ============================================================
# Team name mapping (full name → 3-letter abbreviation)
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
    "Seattle SuperSonics": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
    # Historical teams
    "New Jersey Nets": "BKN",
    "Vancouver Grizzlies": "MEM",
}

TEAM_ABBREV_MAP = {
    "ATL": "ATL", "BOS": "BOS", "BRK": "BKN", "BKN": "BKN",
    "CHA": "CHA", "CHH": "CHA", "CHO": "CHA", "CHI": "CHI",
    "CLE": "CLE", "DAL": "DAL", "DEN": "DEN", "DET": "DET",
    "GSW": "GSW", "HOU": "HOU", "IND": "IND", "LAC": "LAC",
    "LAL": "LAL", "MEM": "MEM", "MIA": "MIA", "MIL": "MIL",
    "MIN": "MIN", "NOP": "NOP", "NOH": "NOP", "NOK": "NOP",
    "NYK": "NYK", "OKC": "OKC", "SEA": "OKC", "ORL": "ORL",
    "PHI": "PHI", "PHO": "PHX", "PHX": "PHX", "POR": "POR",
    "SAC": "SAC", "SAS": "SAS", "TOR": "TOR", "UTA": "UTA",
    "WAS": "WAS", "NJN": "BKN", "VAN": "MEM",
}

# Active NBA teams (30 teams)
ACTIVE_TEAMS = [
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
]


def normalize_team(name):
    """Normalize team name to 3-letter abbreviation."""
    if name is None:
        return None
    name = str(name).strip().replace("*", "")
    if name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[name]
    if name in TEAM_ABBREV_MAP:
        return TEAM_ABBREV_MAP[name]
    # Try partial matching
    for full_name, abbrev in TEAM_NAME_MAP.items():
        if name.lower() in full_name.lower() or full_name.lower() in name.lower():
            return abbrev
    return name


def get_season_year(date, filename):
    """
    Determine the season year_id from a game date and filename.
    The season year_id corresponds to the year the season ends.
    e.g., NBA2004-05.csv → season 2005, NBA2024-25.csv → season 2025.
    """
    # Extract from filename: NBA2004-05.csv → end_year = 2005
    basename = os.path.basename(filename)
    # Format: NBAxxxx-yy.csv
    parts = basename.replace("NBA", "").replace(".csv", "").split("-")
    start_year = int(parts[0])
    end_year_short = int(parts[1])
    # Handle century: 04 → 2005, 25 → 2025
    if end_year_short < 50:
        end_year = 2000 + end_year_short
    else:
        end_year = 1900 + end_year_short
    return end_year


def process_538_data():
    """
    Process the FiveThirtyEight nbaallelo.csv dataset.
    Extract NBA games from 2003-04 season ONLY (year_id=2004).
    Later seasons are covered by Basketball Reference CSVs.
    """
    print("=" * 70)
    print("PHASE 1: Processing FiveThirtyEight Data (2003-04 season)")
    print("=" * 70)

    csv_path = os.path.join(DATA_DIR, "nbaallelo.csv")
    df = pd.read_csv(csv_path)

    # Filter to NBA, 2004 only (2003-04 season)
    df = df[(df["lg_id"] == "NBA") & (df["year_id"] == 2004)]

    # Keep only home games (game_location == 'H') to avoid duplicates
    home_df = df[df["game_location"] == "H"].copy()
    away_df = df[df["game_location"] == "A"].copy()

    # Merge on game_id
    merged = home_df.merge(
        away_df[["game_id", "team_id", "pts", "elo_i", "elo_n"]],
        on="game_id",
        suffixes=("_home", "_away")
    )

    games = []
    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="Processing 538 data"):
        home_team = TEAM_ABBREV_MAP.get(row["team_id_home"], row["team_id_home"])
        visitor_team = TEAM_ABBREV_MAP.get(row["team_id_away"], row["team_id_away"])

        if home_team not in ACTIVE_TEAMS or visitor_team not in ACTIVE_TEAMS:
            continue

        try:
            date = pd.to_datetime(row["date_game"])
        except Exception:
            continue

        games.append({
            "date": date,
            "season": row["year_id"],
            "home_team_id": home_team,
            "visitor_team_id": visitor_team,
            "home_pts": int(row["pts_home"]),
            "visitor_pts": int(row["pts_away"]),
            "home_win": 1 if row["pts_home"] > row["pts_away"] else 0,
            "margin": int(row["pts_home"] - row["pts_away"]),
            "total_points": int(row["pts_home"] + row["pts_away"]),
            "is_overtime": 0,
            "home_elo_538": row["elo_i_home"],
            "visitor_elo_538": row["elo_i_away"],
        })

    result_df = pd.DataFrame(games)
    result_df = result_df.sort_values("date").reset_index(drop=True)

    print(f"  Extracted {len(result_df)} real games from 2003-04 season")
    print(f"  Date range: {result_df['date'].min()} to {result_df['date'].max()}")

    return result_df


def process_bball_ref_seasons():
    """
    Process all Basketball Reference season CSV files (2004-05 through 2024-25).
    These contain REAL game data with dates, teams, scores, and OT info.

    CSV format:
    Date, Start (ET), Visitor/Neutral, PTS, Home/Neutral, PTS.1, Box Score, OT, ...
    """
    print("\n" + "=" * 70)
    print("PHASE 2: Processing Basketball Reference Real Game Data (2004-2025)")
    print("=" * 70)

    all_games = []

    # Process each season file
    season_files = sorted([
        f for f in os.listdir(BBALL_REF_DIR)
        if f.startswith("NBA") and f.endswith(".csv") and f != "combined_data.csv"
    ])

    for filename in season_files:
        filepath = os.path.join(BBALL_REF_DIR, filename)
        season_year = get_season_year(None, filename)

        print(f"\n  Processing {filename} (season {season_year-1}-{str(season_year)[2:]})...")

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"    ERROR reading {filename}: {e}")
            continue

        # Column names from Basketball Reference schedule pages
        # Date, Start (ET), Visitor/Neutral, PTS, Home/Neutral, PTS.1, Box Score, OT, ...
        visitor_col = "Visitor/Neutral"
        home_col = "Home/Neutral"
        visitor_pts_col = "PTS"
        home_pts_col = "PTS.1"
        ot_col = "OT"

        game_count = 0
        skip_count = 0

        for _, row in df.iterrows():
            # Skip rows with missing essential data
            date_str = str(row.get("Date", ""))
            if not date_str or date_str == "nan" or "Playoffs" in date_str:
                skip_count += 1
                continue

            visitor_name = str(row.get(visitor_col, ""))
            home_name = str(row.get(home_col, ""))

            if not visitor_name or visitor_name == "nan" or not home_name or home_name == "nan":
                skip_count += 1
                continue

            # Parse scores
            try:
                visitor_pts = int(float(row[visitor_pts_col]))
                home_pts = int(float(row[home_pts_col]))
            except (ValueError, TypeError):
                skip_count += 1
                continue

            # Parse date
            try:
                date = pd.to_datetime(date_str)
            except Exception:
                skip_count += 1
                continue

            # Normalize team names
            home_team = normalize_team(home_name)
            visitor_team = normalize_team(visitor_name)

            if home_team not in ACTIVE_TEAMS or visitor_team not in ACTIVE_TEAMS:
                skip_count += 1
                continue

            # Check overtime
            ot_value = str(row.get(ot_col, ""))
            is_overtime = 1 if ot_value and ot_value != "nan" and "OT" in ot_value.upper() else 0

            margin = home_pts - visitor_pts

            all_games.append({
                "date": date,
                "season": season_year,
                "home_team_id": home_team,
                "visitor_team_id": visitor_team,
                "home_pts": home_pts,
                "visitor_pts": visitor_pts,
                "home_win": 1 if home_pts > visitor_pts else 0,
                "margin": margin,
                "total_points": home_pts + visitor_pts,
                "is_overtime": is_overtime,
                "home_elo_538": np.nan,
                "visitor_elo_538": np.nan,
            })
            game_count += 1

        print(f"    Parsed {game_count} games ({skip_count} rows skipped)")

    result_df = pd.DataFrame(all_games)
    result_df = result_df.sort_values("date").reset_index(drop=True)

    print(f"\n  Total BBRef games: {len(result_df)}")
    print(f"  Date range: {result_df['date'].min()} to {result_df['date'].max()}")
    print(f"  Seasons: {sorted(result_df['season'].unique())}")

    return result_df


def build_complete_dataset():
    """Build the complete NBA game dataset from 2003-2025 using 100% real data."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Phase 1: FiveThirtyEight real data (2003-04 season only)
    df_538 = process_538_data()

    # Phase 2: Basketball Reference real data (2004-05 through 2024-25)
    df_bbref = process_bball_ref_seasons()

    # Combine
    print("\n" + "=" * 70)
    print("PHASE 3: Combining All Real Game Data")
    print("=" * 70)

    df_all = pd.concat([df_538, df_bbref], ignore_index=True)
    df_all = df_all.sort_values("date").reset_index(drop=True)

    # Remove any duplicate games (same date, same teams)
    before_dedup = len(df_all)
    df_all = df_all.drop_duplicates(
        subset=["date", "home_team_id", "visitor_team_id"],
        keep="last"  # Prefer BBRef data over 538 if overlap
    ).reset_index(drop=True)
    after_dedup = len(df_all)
    if before_dedup > after_dedup:
        print(f"  Removed {before_dedup - after_dedup} duplicate games")

    # Add game_id
    df_all["game_id"] = df_all.apply(
        lambda r: f"{r['date'].strftime('%Y%m%d')}_{r['visitor_team_id']}_{r['home_team_id']}",
        axis=1
    )

    # Summary stats
    print(f"\n  COMPLETE DATASET (100% REAL GAMES):")
    print(f"  Total games: {len(df_all)}")
    print(f"  Date range: {df_all['date'].min()} to {df_all['date'].max()}")
    print(f"  Seasons: {sorted(df_all['season'].unique())}")
    print(f"  Teams: {sorted(df_all['home_team_id'].unique())}")
    print(f"  Home win rate: {df_all['home_win'].mean():.4f}")
    print(f"  Avg total points: {df_all['total_points'].mean():.1f}")
    print(f"  Avg margin: {df_all['margin'].abs().mean():.1f}")

    print(f"\n  Games per season:")
    for season in sorted(df_all['season'].unique()):
        n = len(df_all[df_all['season'] == season])
        print(f"    {season-1}-{str(season)[2:]}: {n} games")

    # Save
    output_path = os.path.join(PROCESSED_DIR, "games_clean.csv")
    df_all.to_csv(output_path, index=False)
    print(f"\n  Saved to {output_path}")

    return df_all


if __name__ == "__main__":
    df = build_complete_dataset()

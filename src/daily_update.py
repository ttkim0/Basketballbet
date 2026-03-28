#!/usr/bin/env python3
"""
NBA Daily Data Update Pipeline
================================
Fetches completed game results from the NBA API, updates Elo ratings,
recomputes rolling features, and appends to the main dataset.

Run daily after all games finish (e.g., 2:00 AM ET via cron):
    python3 src/daily_update.py

Or for a specific date:
    python3 src/daily_update.py --date 2026-03-27

What it does:
1. Fetches completed games from NBA scoreboard + box scores
2. Appends new game rows to games_enriched_v2.csv
3. Recomputes Elo ratings for new games (incrementally)
4. Reruns feature engineering on the full dataset
5. Saves updated games_full_features.csv and final_elo_ratings.csv
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

sys.path.insert(0, os.path.join(BASE_DIR, "src"))

# NBA API endpoints
SCOREBOARD_URL = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
BOXSCORE_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"
SCHEDULE_URL = "https://stats.nba.com/stats/scheduleleaguev2?LeagueID=00&Season={season}"

# Standard headers for NBA API
NBA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
}

# Team tricode normalization (handles historical/alternate codes)
TRICODE_MAP = {
    "NJN": "BKN", "NOH": "NOP", "NOK": "NOP", "SEA": "OKC",
    "VAN": "MEM", "CHH": "CHA", "CHA": "CHA", "CHO": "CHA",
    "GS": "GSW", "SA": "SAS", "NO": "NOP", "NY": "NYK",
    "PHO": "PHX", "WSH": "WAS",
}


def normalize_tricode(code):
    """Normalize team tricode to our standard 3-letter codes."""
    code = code.upper().strip()
    return TRICODE_MAP.get(code, code)


def fetch_json(url, retries=3, backoff=2):
    """Fetch JSON from URL with retries and exponential backoff."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=NBA_HEADERS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            if attempt < retries - 1:
                wait = backoff * (2 ** attempt)
                print(f"  Retry {attempt + 1}/{retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


def fetch_todays_completed_games():
    """Fetch today's completed games from NBA scoreboard."""
    print("Fetching today's scoreboard...")
    data = fetch_json(SCOREBOARD_URL)
    games = data["scoreboard"]["games"]
    completed = [g for g in games if g["gameStatus"] == 3]  # 3 = Final
    print(f"  {len(games)} total games, {len(completed)} completed")
    return completed


def fetch_boxscore(game_id):
    """Fetch detailed box score for a single game."""
    url = BOXSCORE_URL.format(game_id=game_id)
    return fetch_json(url)


def extract_team_stats(team_data):
    """Extract team-level stats from NBA box score API response.

    Maps NBA API field names to our dataset column names.
    """
    stats = team_data.get("statistics", {})

    # Direct mappings from NBA API → our columns
    return {
        "fgm": stats.get("fieldGoalsMade", 0),
        "fga": stats.get("fieldGoalsAttempted", 0),
        "fg3m": stats.get("threePointersMade", 0),
        "fg3a": stats.get("threePointersAttempted", 0),
        "ftm": stats.get("freeThrowsMade", 0),
        "fta": stats.get("freeThrowsAttempted", 0),
        "oreb": stats.get("reboundsOffensive", 0),
        "dreb": stats.get("reboundsDefensive", 0),
        "reb": stats.get("reboundsTotal", 0),
        "ast": stats.get("assists", 0),
        "stl": stats.get("steals", 0),
        "blk": stats.get("blocks", 0),
        "tov": stats.get("turnovers", 0),
        "pf": stats.get("foulsPersonal", 0),
        "pts": stats.get("points", 0),
    }


def compute_advanced_box_stats(stats):
    """Compute advanced shooting stats from raw box score."""
    fga = stats["fga"] or 1  # avoid division by zero
    fta = stats["fta"] or 1
    fg3a = stats["fg3a"] or 1

    stats["fg_pct"] = stats["fgm"] / fga if fga > 0 else 0
    stats["three_pct"] = stats["fg3m"] / fg3a if fg3a > 0 else 0
    stats["ft_pct"] = stats["ftm"] / fta if fta > 0 else 0
    stats["efg_pct"] = (stats["fgm"] + 0.5 * stats["fg3m"]) / fga if fga > 0 else 0
    stats["ts_pct"] = stats["pts"] / (2 * (fga + 0.44 * stats["fta"])) if fga > 0 else 0

    return stats


def estimate_pace_and_ratings(home_stats, away_stats):
    """Estimate pace, offensive/defensive ratings from box score.

    Uses simplified pace formula:
    Pace ≈ FGA + 0.44*FTA - OREB + TOV
    ORtg = 100 * pts / possessions
    DRtg = 100 * opp_pts / possessions
    """
    home_poss = (home_stats["fga"] + 0.44 * home_stats["fta"]
                 - home_stats["oreb"] + home_stats["tov"])
    away_poss = (away_stats["fga"] + 0.44 * away_stats["fta"]
                 - away_stats["oreb"] + away_stats["tov"])

    # Average possessions (should be roughly equal)
    avg_poss = max((home_poss + away_poss) / 2, 1)
    pace = avg_poss  # possessions per game

    home_stats["ortg"] = 100 * home_stats["pts"] / max(home_poss, 1)
    home_stats["drtg"] = 100 * away_stats["pts"] / max(home_poss, 1)
    home_stats["pace"] = pace

    away_stats["ortg"] = 100 * away_stats["pts"] / max(away_poss, 1)
    away_stats["drtg"] = 100 * home_stats["pts"] / max(away_poss, 1)
    away_stats["pace"] = pace

    return home_stats, away_stats


def build_game_row(game_data, box_data):
    """Build a single game row matching games_enriched_v2.csv format."""
    game = box_data["game"]

    home_team = normalize_tricode(game["homeTeam"]["teamTricode"])
    away_team = normalize_tricode(game["awayTeam"]["teamTricode"])

    # Extract and compute stats
    home_stats = extract_team_stats(game["homeTeam"])
    away_stats = extract_team_stats(game["awayTeam"])

    home_stats = compute_advanced_box_stats(home_stats)
    away_stats = compute_advanced_box_stats(away_stats)
    home_stats, away_stats = estimate_pace_and_ratings(home_stats, away_stats)

    home_pts = home_stats["pts"]
    away_pts = away_stats["pts"]

    # Determine date from game data
    game_date_str = game.get("gameTimeUTC", game_data.get("gameTimeUTC", ""))
    if game_date_str:
        game_date = pd.to_datetime(game_date_str).tz_localize(None)
    else:
        game_date = pd.Timestamp.now().normalize()

    # Determine season (NBA season spans Oct-Jun, named by start year)
    month = game_date.month
    year = game_date.year
    season = year if month >= 10 else year - 1

    row = {
        "date": game_date.strftime("%Y-%m-%d"),
        "home_team_id": home_team,
        "visitor_team_id": away_team,
        "home_pts": int(home_pts),
        "visitor_pts": int(away_pts),
        "home_win": 1 if home_pts > away_pts else 0,
        "margin": int(home_pts - away_pts),
        "season": season,
    }

    # Add home team box score columns
    for stat, val in home_stats.items():
        if stat != "pts":
            row[f"home_{stat}"] = val

    # Add visitor team box score columns
    for stat, val in away_stats.items():
        if stat != "pts":
            row[f"vis_{stat}"] = val

    return row


def fetch_games_for_date(target_date=None):
    """Fetch all completed games for a given date.

    If target_date is None, fetches today's games.
    For past dates, tries the schedule API + individual box scores.
    """
    if target_date is None:
        # Today's games from live scoreboard
        completed = fetch_todays_completed_games()
        rows = []
        for g in completed:
            game_id = g["gameId"]
            h = normalize_tricode(g["homeTeam"]["teamTricode"])
            a = normalize_tricode(g["awayTeam"]["teamTricode"])
            print(f"  Fetching box score: {a} @ {h} (ID: {game_id})")
            try:
                box = fetch_boxscore(game_id)
                row = build_game_row(g, box)
                rows.append(row)
                print(f"    {row['visitor_team_id']} {row['visitor_pts']} @ "
                      f"{row['home_team_id']} {row['home_pts']}")
                time.sleep(0.5)  # Be polite to the API
            except Exception as e:
                print(f"    ERROR: {e}")
        return rows
    else:
        # For a specific past date, we'd need the schedule API
        # The scoreboard only shows today's games
        print(f"Fetching games for {target_date}...")
        print("  Note: For past dates, using schedule API...")

        # Try to get game IDs from the season schedule
        month = target_date.month
        year = target_date.year
        season_year = year if month >= 10 else year - 1
        season_str = f"{season_year}-{str(season_year + 1)[-2:]}"

        try:
            url = SCHEDULE_URL.format(season=season_str)
            data = fetch_json(url)
            # Parse schedule to find games on target_date
            game_dates = data.get("leagueSchedule", {}).get("gameDates", [])
            target_str = target_date.strftime("%m/%d/%Y")

            rows = []
            for gd in game_dates:
                gd_date = gd.get("gameDate", "")
                # Schedule dates can be in various formats
                if target_date.strftime("%Y-%m-%d") in gd_date or target_str in gd_date:
                    for g in gd.get("games", []):
                        if g.get("gameStatus", 0) == 3:  # Completed
                            game_id = g["gameId"]
                            try:
                                box = fetch_boxscore(game_id)
                                row = build_game_row(g, box)
                                rows.append(row)
                                print(f"    {row['visitor_team_id']} {row['visitor_pts']} @ "
                                      f"{row['home_team_id']} {row['home_pts']}")
                                time.sleep(0.5)
                            except Exception as e:
                                print(f"    ERROR fetching {game_id}: {e}")
            return rows
        except Exception as e:
            print(f"  Could not fetch schedule: {e}")
            return []


def update_elo_ratings(new_games_df, current_ratings):
    """Incrementally update Elo ratings with new games.

    Uses the same algorithm as elo_ratings.py but doesn't recompute from scratch.
    """
    from elo_ratings import (
        update_elo, expected_score, HOME_ADVANTAGE,
        INITIAL_ELO, SEASON_REGRESSION, regress_to_mean,
    )

    ratings = dict(current_ratings)  # Copy

    # Check for season transitions
    if len(new_games_df) > 0:
        existing = pd.read_csv(os.path.join(PROCESSED_DIR, "games_enriched_v2.csv"))
        last_existing_season = existing["season"].max()

        new_seasons = new_games_df["season"].unique()
        for new_season in sorted(new_seasons):
            if new_season > last_existing_season:
                print(f"  Season transition detected: {last_existing_season} → {new_season}")
                ratings = regress_to_mean(ratings, SEASON_REGRESSION)

    elo_records = []
    for _, row in new_games_df.iterrows():
        home = row["home_team_id"]
        visitor = row["visitor_team_id"]

        if home not in ratings:
            ratings[home] = INITIAL_ELO
        if visitor not in ratings:
            ratings[visitor] = INITIAL_ELO

        r_home = ratings[home]
        r_visitor = ratings[visitor]

        exp = expected_score(r_home, r_visitor, HOME_ADVANTAGE)
        home_win = 1 if row["home_pts"] > row["visitor_pts"] else 0
        mov = row["home_pts"] - row["visitor_pts"]

        new_home, new_visitor = update_elo(r_home, r_visitor, home_win, mov, is_home_a=True)

        elo_records.append({
            "home_elo_pre": r_home,
            "visitor_elo_pre": r_visitor,
            "home_elo_post": new_home,
            "visitor_elo_post": new_visitor,
            "elo_diff": r_home - r_visitor,
            "home_elo_expected": exp,
            "elo_rating_diff": r_home - r_visitor,
        })

        ratings[home] = new_home
        ratings[visitor] = new_visitor

    return elo_records, ratings


def run_daily_update(target_date=None, skip_fetch=False):
    """Main daily update pipeline."""
    print("=" * 60)
    print("NBA DAILY DATA UPDATE PIPELINE")
    print(f"Date: {target_date or 'today'}")
    print("=" * 60)

    # Step 1: Load existing data
    enriched_path = os.path.join(PROCESSED_DIR, "games_enriched_v2.csv")
    elo_path = os.path.join(PROCESSED_DIR, "final_elo_ratings.csv")

    print("\n[1/5] Loading existing data...")
    df_existing = pd.read_csv(enriched_path)
    df_existing["date"] = pd.to_datetime(df_existing["date"])
    print(f"  Existing games: {len(df_existing)}")
    print(f"  Date range: {df_existing['date'].min().date()} to {df_existing['date'].max().date()}")

    elo_df = pd.read_csv(elo_path)
    current_ratings = dict(zip(elo_df["team"], elo_df["elo"]))
    print(f"  Elo ratings loaded for {len(current_ratings)} teams")

    # Step 2: Fetch new games
    print("\n[2/5] Fetching new game results...")
    if skip_fetch:
        print("  Skipping fetch (--skip-fetch flag)")
        new_rows = []
    else:
        new_rows = fetch_games_for_date(target_date)

    if not new_rows:
        print("  No new completed games found.")
        print("  Pipeline complete (no updates needed).")
        return

    new_df = pd.DataFrame(new_rows)
    new_df["date"] = pd.to_datetime(new_df["date"])

    # Deduplicate: skip games already in the dataset
    existing_keys = set(zip(
        df_existing["date"].dt.strftime("%Y-%m-%d"),
        df_existing["home_team_id"],
        df_existing["visitor_team_id"],
    ))

    new_games = []
    for _, row in new_df.iterrows():
        key = (row["date"].strftime("%Y-%m-%d"), row["home_team_id"], row["visitor_team_id"])
        if key not in existing_keys:
            new_games.append(row)

    if not new_games:
        print("  All fetched games already in dataset. No updates needed.")
        return

    new_df = pd.DataFrame(new_games)
    print(f"  {len(new_df)} NEW games to add:")
    for _, row in new_df.iterrows():
        print(f"    {row['visitor_team_id']} {row['visitor_pts']} @ "
              f"{row['home_team_id']} {row['home_pts']}")

    # Step 3: Update Elo ratings
    print("\n[3/5] Updating Elo ratings...")
    elo_records, updated_ratings = update_elo_ratings(new_df, current_ratings)

    # Add Elo columns to new games
    for i, rec in enumerate(elo_records):
        for k, v in rec.items():
            new_df.iloc[i, new_df.columns.get_loc(k) if k in new_df.columns
                        else len(new_df.columns)] = v
    # Add Elo columns properly
    for col in ["home_elo_pre", "visitor_elo_pre", "home_elo_post", "visitor_elo_post",
                "elo_diff", "home_elo_expected", "elo_rating_diff"]:
        if col not in new_df.columns:
            new_df[col] = [rec[col] for rec in elo_records]

    # Save updated Elo ratings
    ratings_df = pd.DataFrame([
        {"team": k, "elo": v}
        for k, v in sorted(updated_ratings.items(), key=lambda x: -x[1])
    ])
    ratings_df.to_csv(elo_path, index=False)
    print(f"  Elo ratings updated for {len(updated_ratings)} teams")
    print(f"  Top 5: {', '.join(f'{r[0]} ({r[1]:.0f})' for r in sorted(updated_ratings.items(), key=lambda x: -x[1])[:5])}")

    # Step 4: Append to enriched dataset
    print("\n[4/5] Appending to games_enriched_v2.csv...")

    # Align columns (new rows may have fewer columns)
    for col in df_existing.columns:
        if col not in new_df.columns:
            new_df[col] = np.nan

    # Only keep columns that exist in the original
    new_df = new_df[[c for c in df_existing.columns if c in new_df.columns]]

    df_combined = pd.concat([df_existing, new_df], ignore_index=True)
    df_combined = df_combined.sort_values("date").reset_index(drop=True)

    # Save enriched dataset
    df_combined.to_csv(enriched_path, index=False)
    print(f"  Saved {len(df_combined)} games to games_enriched_v2.csv")

    # Also update games_with_elo.csv
    elo_csv_path = os.path.join(PROCESSED_DIR, "games_with_elo.csv")
    df_combined.to_csv(elo_csv_path, index=False)

    # Step 5: Recompute full features
    print("\n[5/5] Recomputing full feature set...")
    from feature_engineering import compute_all_features

    df_features = compute_all_features(df_combined)
    features_path = os.path.join(PROCESSED_DIR, "games_full_features.csv")
    df_features.to_csv(features_path, index=False)
    print(f"  Saved {len(df_features)} games x {len(df_features.columns)} columns")

    # Log the update
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "date": str(target_date or "today"),
        "games_added": len(new_df),
        "total_games": len(df_combined),
        "games": [
            {
                "away": row["visitor_team_id"],
                "home": row["home_team_id"],
                "score": f"{row['visitor_pts']}-{row['home_pts']}",
            }
            for _, row in new_df.iterrows()
        ],
    }

    log_path = os.path.join(LOGS_DIR, "daily_updates.jsonl")
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    print("\n" + "=" * 60)
    print(f"UPDATE COMPLETE: +{len(new_df)} games")
    print(f"Total dataset: {len(df_combined)} games")
    print(f"Features: {len(df_features.columns)} columns")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="NBA Daily Data Update Pipeline")
    parser.add_argument("--date", type=str, default=None,
                        help="Specific date to fetch (YYYY-MM-DD). Default: today")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip API fetch (for testing)")
    parser.add_argument("--backfill", type=int, default=0,
                        help="Backfill N days of games (e.g., --backfill 7)")
    args = parser.parse_args()

    if args.backfill > 0:
        # Backfill multiple days
        for i in range(args.backfill, 0, -1):
            target = datetime.now() - timedelta(days=i)
            print(f"\n{'#' * 60}")
            print(f"BACKFILLING: {target.strftime('%Y-%m-%d')}")
            print(f"{'#' * 60}")
            try:
                run_daily_update(target_date=target)
            except Exception as e:
                print(f"  Error backfilling {target.date()}: {e}")
            time.sleep(1)
    else:
        target = None
        if args.date:
            target = datetime.strptime(args.date, "%Y-%m-%d")
        run_daily_update(target_date=target, skip_fetch=args.skip_fetch)


if __name__ == "__main__":
    main()

"""
Feature Engineering Pipeline
==============================
Computes ALL covariates from the project spec:
- Schedule features (rest, B2B, 3in4, 4in6, travel, timezone, altitude)
- Recent form (rolling 5/10 game windows for net/off/def rating, pace, shooting, etc.)
- Matchup-style covariates
- Motivation/context features
"""

import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# Arena locations for travel/timezone
# ============================================================
ARENA_LOCATIONS = {
    "ATL": {"lat": 33.757, "lon": -84.396, "tz_offset": 0, "altitude_ft": 1050},
    "BOS": {"lat": 42.366, "lon": -71.062, "tz_offset": 0, "altitude_ft": 20},
    "BKN": {"lat": 40.682, "lon": -73.975, "tz_offset": 0, "altitude_ft": 30},
    "CHA": {"lat": 35.225, "lon": -80.839, "tz_offset": 0, "altitude_ft": 751},
    "CHI": {"lat": 41.881, "lon": -87.674, "tz_offset": -1, "altitude_ft": 594},
    "CLE": {"lat": 41.496, "lon": -81.688, "tz_offset": 0, "altitude_ft": 653},
    "DAL": {"lat": 32.790, "lon": -96.810, "tz_offset": -1, "altitude_ft": 430},
    "DEN": {"lat": 39.749, "lon": -104.999, "tz_offset": -2, "altitude_ft": 5280},
    "DET": {"lat": 42.341, "lon": -83.055, "tz_offset": 0, "altitude_ft": 600},
    "GSW": {"lat": 37.768, "lon": -122.388, "tz_offset": -3, "altitude_ft": 16},
    "HOU": {"lat": 29.751, "lon": -95.362, "tz_offset": -1, "altitude_ft": 50},
    "IND": {"lat": 39.764, "lon": -86.155, "tz_offset": 0, "altitude_ft": 717},
    "LAC": {"lat": 34.043, "lon": -118.267, "tz_offset": -3, "altitude_ft": 305},
    "LAL": {"lat": 34.043, "lon": -118.267, "tz_offset": -3, "altitude_ft": 305},
    "MEM": {"lat": 35.138, "lon": -90.051, "tz_offset": -1, "altitude_ft": 337},
    "MIA": {"lat": 25.781, "lon": -80.187, "tz_offset": 0, "altitude_ft": 6},
    "MIL": {"lat": 43.045, "lon": -87.917, "tz_offset": -1, "altitude_ft": 617},
    "MIN": {"lat": 44.979, "lon": -93.276, "tz_offset": -1, "altitude_ft": 830},
    "NOP": {"lat": 29.949, "lon": -90.082, "tz_offset": -1, "altitude_ft": 3},
    "NYK": {"lat": 40.751, "lon": -73.994, "tz_offset": 0, "altitude_ft": 33},
    "OKC": {"lat": 35.463, "lon": -97.515, "tz_offset": -1, "altitude_ft": 1201},
    "ORL": {"lat": 28.539, "lon": -81.384, "tz_offset": 0, "altitude_ft": 82},
    "PHI": {"lat": 39.901, "lon": -75.172, "tz_offset": 0, "altitude_ft": 39},
    "PHX": {"lat": 33.446, "lon": -112.071, "tz_offset": -2, "altitude_ft": 1086},
    "POR": {"lat": 45.532, "lon": -122.667, "tz_offset": -3, "altitude_ft": 50},
    "SAC": {"lat": 38.580, "lon": -121.500, "tz_offset": -3, "altitude_ft": 30},
    "SAS": {"lat": 29.427, "lon": -98.438, "tz_offset": -1, "altitude_ft": 650},
    "TOR": {"lat": 43.643, "lon": -79.379, "tz_offset": 0, "altitude_ft": 249},
    "UTA": {"lat": 40.768, "lon": -111.901, "tz_offset": -2, "altitude_ft": 4327},
    "WAS": {"lat": 38.898, "lon": -77.021, "tz_offset": 0, "altitude_ft": 25},
}

# High-altitude venues (Denver is the primary one, Utah secondary)
HIGH_ALTITUDE_TEAMS = {"DEN": 5280, "UTA": 4327}


def haversine_miles(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 3959 * 2 * asin(sqrt(a))


def compute_schedule_features(df):
    """
    Compute all schedule-related features:
    - Rest days
    - Back-to-back
    - 3-in-4, 4-in-6
    - Road trip / homestand game number
    - Travel miles
    - Timezone shift
    - Altitude edge
    """
    print("Computing schedule features...")
    df = df.sort_values("date").reset_index(drop=True)

    # Build team game history: for each team, track their games in order
    team_games = {}  # team -> list of (date, game_location_team, is_home)

    for idx, row in df.iterrows():
        home = row["home_team_id"]
        visitor = row["visitor_team_id"]
        date = pd.Timestamp(row["date"])

        if home not in team_games:
            team_games[home] = []
        if visitor not in team_games:
            team_games[visitor] = []

        team_games[home].append({"date": date, "location": home, "is_home": True, "idx": idx})
        team_games[visitor].append({"date": date, "location": home, "is_home": False, "idx": idx})

    # Sort each team's games
    for team in team_games:
        team_games[team].sort(key=lambda x: (x["date"], x["idx"]))

    # Build lookup: team -> game_index -> position in team's schedule
    team_game_pos = {}
    for team, games in team_games.items():
        team_game_pos[team] = {}
        for pos, g in enumerate(games):
            team_game_pos[team][g["idx"]] = pos

    # Now compute features for each game
    features = {
        "rest_days_home": [], "rest_days_visitor": [], "rest_diff": [],
        "home_b2b": [], "visitor_b2b": [],
        "home_3in4": [], "visitor_3in4": [],
        "home_4in6": [], "visitor_4in6": [],
        "road_trip_game_num_home": [], "road_trip_game_num_visitor": [],
        "homestand_game_num_home": [], "homestand_game_num_visitor": [],
        "travel_miles_home": [], "travel_miles_visitor": [], "travel_diff": [],
        "tz_shift_home": [], "tz_shift_visitor": [], "tz_diff": [],
        "altitude_edge_home": [],
    }

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Schedule Features"):
        home = row["home_team_id"]
        visitor = row["visitor_team_id"]
        date = pd.Timestamp(row["date"])

        for team, prefix in [(home, "home"), (visitor, "visitor")]:
            pos = team_game_pos[team].get(idx, 0)
            games = team_games[team]

            # Rest days
            if pos > 0:
                prev_date = games[pos - 1]["date"]
                rest = (date - prev_date).days - 1
            else:
                rest = 7  # Season opener default

            features[f"rest_days_{prefix}"].append(rest)

            # Back-to-back (0 rest days)
            b2b = 1 if rest == 0 else 0
            features[f"{prefix}_b2b"].append(b2b)

            # 3-in-4: check if 3 games in last 4 days including today
            games_in_window = sum(
                1 for g in games[max(0, pos-2):pos+1]
                if (date - g["date"]).days <= 3
            )
            features[f"{prefix}_3in4"].append(1 if games_in_window >= 3 else 0)

            # 4-in-6
            games_in_window_6 = sum(
                1 for g in games[max(0, pos-3):pos+1]
                if (date - g["date"]).days <= 5
            )
            features[f"{prefix}_4in6"].append(1 if games_in_window_6 >= 4 else 0)

            # Road trip / homestand game number
            if team in ARENA_LOCATIONS:
                is_home_game = (prefix == "home")
                streak = 1
                for p in range(pos - 1, -1, -1):
                    if games[p]["is_home"] == is_home_game:
                        streak += 1
                    else:
                        break
                if is_home_game:
                    features[f"homestand_game_num_{prefix}"].append(streak)
                    features[f"road_trip_game_num_{prefix}"].append(0)
                else:
                    features[f"road_trip_game_num_{prefix}"].append(streak)
                    features[f"homestand_game_num_{prefix}"].append(0)
            else:
                features[f"road_trip_game_num_{prefix}"].append(0)
                features[f"homestand_game_num_{prefix}"].append(0)

            # Travel miles (from previous game location to current game location)
            game_location = home  # Both teams play at home team's arena
            if pos > 0 and team in ARENA_LOCATIONS:
                prev_location = games[pos - 1]["location"]
                if prev_location in ARENA_LOCATIONS and game_location in ARENA_LOCATIONS:
                    miles = haversine_miles(
                        ARENA_LOCATIONS[prev_location]["lat"],
                        ARENA_LOCATIONS[prev_location]["lon"],
                        ARENA_LOCATIONS[game_location]["lat"],
                        ARENA_LOCATIONS[game_location]["lon"],
                    )
                else:
                    miles = 0
            else:
                miles = 0
            features[f"travel_miles_{prefix}"].append(miles)

            # Timezone shift
            if pos > 0 and team in ARENA_LOCATIONS:
                prev_location = games[pos - 1]["location"]
                if prev_location in ARENA_LOCATIONS and game_location in ARENA_LOCATIONS:
                    tz_prev = ARENA_LOCATIONS[prev_location]["tz_offset"]
                    tz_curr = ARENA_LOCATIONS[game_location]["tz_offset"]
                    tz_shift = abs(tz_curr - tz_prev)
                else:
                    tz_shift = 0
            else:
                tz_shift = 0
            features[f"tz_shift_{prefix}"].append(tz_shift)

        # Differentials
        features["rest_diff"].append(
            features["rest_days_home"][-1] - features["rest_days_visitor"][-1]
        )
        features["travel_diff"].append(
            features["travel_miles_home"][-1] - features["travel_miles_visitor"][-1]
        )
        features["tz_diff"].append(
            features["tz_shift_home"][-1] - features["tz_shift_visitor"][-1]
        )

        # Altitude edge (home team advantage if playing at altitude)
        game_loc = home
        alt = ARENA_LOCATIONS.get(game_loc, {}).get("altitude_ft", 0)
        altitude_edge = 1 if alt >= 4000 else 0
        features["altitude_edge_home"].append(altitude_edge)

    for col, values in features.items():
        df[col] = values

    print(f"  Added {len(features)} schedule features")
    return df


def compute_rolling_form_features(df, windows=[5, 10]):
    """
    Compute rolling recent form features for each team before each game.
    Uses only pregame information (strict temporal ordering).

    Features computed per window:
    - Net rating (pts scored - pts allowed per game)
    - Offensive output
    - Defensive output (pts allowed)
    - Win rate
    """
    print("Computing rolling form features...")
    df = df.sort_values("date").reset_index(drop=True)

    # Track each team's game history
    team_history = {}  # team -> list of game stats dicts

    # Pre-allocate feature columns
    feature_cols = {}
    for w in windows:
        for side in ["home", "visitor"]:
            for stat in ["net_rating", "off_rating", "def_rating", "win_rate",
                         "avg_margin", "pts_scored", "pts_allowed"]:
                col = f"last{w}_{stat}_{side}"
                feature_cols[col] = [np.nan] * len(df)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Rolling Form"):
        home = row["home_team_id"]
        visitor = row["visitor_team_id"]

        for team, prefix in [(home, "home"), (visitor, "visitor")]:
            if team not in team_history:
                team_history[team] = []

            history = team_history[team]

            for w in windows:
                if len(history) >= w:
                    recent = history[-w:]
                    pts_for = np.mean([g["pts_for"] for g in recent])
                    pts_against = np.mean([g["pts_against"] for g in recent])
                    wins = np.mean([g["win"] for g in recent])
                    margins = np.mean([g["margin"] for g in recent])

                    feature_cols[f"last{w}_net_rating_{prefix}"][idx] = pts_for - pts_against
                    feature_cols[f"last{w}_off_rating_{prefix}"][idx] = pts_for
                    feature_cols[f"last{w}_def_rating_{prefix}"][idx] = pts_against
                    feature_cols[f"last{w}_win_rate_{prefix}"][idx] = wins
                    feature_cols[f"last{w}_avg_margin_{prefix}"][idx] = margins
                    feature_cols[f"last{w}_pts_scored_{prefix}"][idx] = pts_for
                    feature_cols[f"last{w}_pts_allowed_{prefix}"][idx] = pts_against

        # After computing pregame features, add this game to history
        home_stats = {
            "pts_for": row["home_pts"], "pts_against": row["visitor_pts"],
            "win": row["home_win"], "margin": row["margin"],
        }
        visitor_stats = {
            "pts_for": row["visitor_pts"], "pts_against": row["home_pts"],
            "win": 1 - row["home_win"], "margin": -row["margin"],
        }

        if home not in team_history:
            team_history[home] = []
        if visitor not in team_history:
            team_history[visitor] = []

        team_history[home].append(home_stats)
        team_history[visitor].append(visitor_stats)

    # Add to dataframe
    for col, values in feature_cols.items():
        df[col] = values

    # Compute differentials
    for w in windows:
        for stat in ["net_rating", "off_rating", "def_rating", "win_rate", "avg_margin"]:
            home_col = f"last{w}_{stat}_home"
            vis_col = f"last{w}_{stat}_visitor"
            diff_col = f"last{w}_{stat}_diff"
            df[diff_col] = df[home_col] - df[vis_col]

    print(f"  Added rolling form features for windows {windows}")
    return df


def compute_season_stats_features(df):
    """
    Compute cumulative season statistics for each team before each game.
    These are running averages of team performance within the current season.

    Tracks: FG%, 3P%, FT%, rebounds, assists, steals, blocks, turnovers, fouls
    All computed as per-game averages up to (but not including) current game.
    """
    print("Computing cumulative season stats features...")
    df = df.sort_values("date").reset_index(drop=True)

    # Track season stats per team
    team_season_stats = {}  # (team, season) -> running totals

    stat_names = ["fg_pct", "three_pct", "ft_pct", "total_reb", "ast",
                  "stl", "blk", "tov", "pf", "pace_proxy"]

    # Pre-allocate
    feature_cols = {}
    for side in ["home", "visitor"]:
        for stat in stat_names:
            feature_cols[f"season_{stat}_{side}"] = [np.nan] * len(df)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Season Stats"):
        home = row["home_team_id"]
        visitor = row["visitor_team_id"]
        season = row["season"]

        for team, prefix in [(home, "home"), (visitor, "visitor")]:
            key = (team, season)
            if key in team_season_stats:
                stats = team_season_stats[key]
                n = stats["games"]
                if n > 0:
                    feature_cols[f"season_fg_pct_{prefix}"][idx] = stats["pts_for"] / max(stats["fga_est"], 1)
                    feature_cols[f"season_total_reb_{prefix}"][idx] = stats.get("total_reb", 0) / n
                    feature_cols[f"season_ast_{prefix}"][idx] = stats.get("ast", 0) / n
                    feature_cols[f"season_tov_{prefix}"][idx] = stats.get("tov", 0) / n
                    feature_cols[f"season_pace_proxy_{prefix}"][idx] = stats.get("total_pts", 0) / n
            else:
                team_season_stats[key] = {
                    "games": 0, "pts_for": 0, "pts_against": 0,
                    "fga_est": 0, "total_reb": 0, "ast": 0, "tov": 0,
                    "total_pts": 0,
                }

        # Update after using pregame values
        for team, prefix, pts_for, pts_against in [
            (home, "home", row["home_pts"], row["visitor_pts"]),
            (visitor, "visitor", row["visitor_pts"], row["home_pts"]),
        ]:
            key = (team, season)
            s = team_season_stats[key]
            s["games"] += 1
            s["pts_for"] += pts_for
            s["pts_against"] += pts_against
            s["fga_est"] += pts_for * 1.1  # Rough FGA estimate
            s["total_pts"] += pts_for + pts_against

    for col, values in feature_cols.items():
        df[col] = values

    # Differentials
    for stat in stat_names:
        h = f"season_{stat}_home"
        v = f"season_{stat}_visitor"
        if h in df.columns and v in df.columns:
            df[f"season_{stat}_diff"] = df[h] - df[v]

    print(f"  Added cumulative season stat features")
    return df


def compute_head_to_head_features(df):
    """
    Compute head-to-head history between the two teams.
    Tracks recent matchup results as a feature.
    """
    print("Computing head-to-head features...")
    df = df.sort_values("date").reset_index(drop=True)

    h2h_record = {}  # (teamA, teamB) -> list of results (1=A won)

    h2h_win_rate = [np.nan] * len(df)
    h2h_avg_margin = [np.nan] * len(df)
    h2h_games = [0] * len(df)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="H2H Features"):
        home = row["home_team_id"]
        visitor = row["visitor_team_id"]

        key = tuple(sorted([home, visitor]))
        if key in h2h_record and len(h2h_record[key]) > 0:
            records = h2h_record[key]
            # From home team's perspective
            home_wins = sum(1 for r in records if (r["winner"] == home))
            h2h_win_rate[idx] = home_wins / len(records)
            h2h_avg_margin[idx] = np.mean([
                r["margin"] if r["home"] == home else -r["margin"]
                for r in records
            ])
            h2h_games[idx] = len(records)

        # Update record
        if key not in h2h_record:
            h2h_record[key] = []
        h2h_record[key].append({
            "home": home,
            "winner": home if row["home_win"] == 1 else visitor,
            "margin": row["margin"],
        })
        # Keep last 20 matchups
        if len(h2h_record[key]) > 20:
            h2h_record[key] = h2h_record[key][-20:]

    df["h2h_home_win_rate"] = h2h_win_rate
    df["h2h_avg_margin"] = h2h_avg_margin
    df["h2h_games_played"] = h2h_games

    print(f"  Added head-to-head features")
    return df


def compute_matchup_style_features(df):
    """
    Compute matchup-style covariates based on team style differences.
    Uses cumulative season stats to derive pace mismatch, scoring style, etc.
    """
    print("Computing matchup-style features...")

    # Pace mismatch (using pace proxy)
    if "season_pace_proxy_home" in df.columns and "season_pace_proxy_visitor" in df.columns:
        df["pace_mismatch"] = df["season_pace_proxy_home"] - df["season_pace_proxy_visitor"]

    # Scoring differential styles from rolling stats
    for w in [5, 10]:
        if f"last{w}_off_rating_home" in df.columns:
            # Offensive vs Defensive matchup
            df[f"off_vs_def_mismatch_{w}"] = (
                df[f"last{w}_off_rating_home"] - df[f"last{w}_def_rating_visitor"]
            )
            df[f"def_vs_off_mismatch_{w}"] = (
                df[f"last{w}_def_rating_home"] - df[f"last{w}_off_rating_visitor"]
            )

    print(f"  Added matchup-style features")
    return df


def compute_motivation_features(df):
    """
    Compute motivation/context features:
    - Playoff race urgency (based on season progress and win rate)
    - Post All-Star break flag
    - End of season rest risk
    - National TV flag (approximated)
    """
    print("Computing motivation/context features...")

    # Track cumulative wins/losses per team per season
    team_season_record = {}

    playoff_urgency_home = [0.0] * len(df)
    playoff_urgency_visitor = [0.0] * len(df)
    post_allstar = [0] * len(df)
    season_progress = [0.0] * len(df)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Motivation Features"):
        home = row["home_team_id"]
        visitor = row["visitor_team_id"]
        season = row["season"]
        date = pd.Timestamp(row["date"])

        # All-Star break is typically mid-February
        allstar_date = pd.Timestamp(f"{season}-02-15")
        post_allstar[idx] = 1 if date > allstar_date else 0

        # Season progress (0 to 1)
        season_start = pd.Timestamp(f"{season-1}-10-15")
        season_end = pd.Timestamp(f"{season}-04-15")
        total_days = (season_end - season_start).days
        elapsed = (date - season_start).days
        progress = max(0, min(1, elapsed / total_days))
        season_progress[idx] = progress

        for team, prefix_list in [(home, "home"), (visitor, "visitor")]:
            key = (team, season)
            if key not in team_season_record:
                team_season_record[key] = {"wins": 0, "losses": 0}

            record = team_season_record[key]
            total = record["wins"] + record["losses"]
            if total > 0:
                win_pct = record["wins"] / total
                # Urgency: high when team is near .500 and deep into season
                # Teams with 0.400-0.600 win pct in March/April have highest urgency
                closeness_to_bubble = 1.0 - abs(win_pct - 0.500) * 4
                closeness_to_bubble = max(0, closeness_to_bubble)
                urgency = closeness_to_bubble * progress
            else:
                urgency = 0

            if prefix_list == "home":
                playoff_urgency_home[idx] = urgency
            else:
                playoff_urgency_visitor[idx] = urgency

        # Update records after using pregame values
        h_key = (home, season)
        v_key = (visitor, season)
        if h_key not in team_season_record:
            team_season_record[h_key] = {"wins": 0, "losses": 0}
        if v_key not in team_season_record:
            team_season_record[v_key] = {"wins": 0, "losses": 0}

        if row["home_win"] == 1:
            team_season_record[h_key]["wins"] += 1
            team_season_record[v_key]["losses"] += 1
        else:
            team_season_record[h_key]["losses"] += 1
            team_season_record[v_key]["wins"] += 1

    df["playoff_urgency_home"] = playoff_urgency_home
    df["playoff_urgency_visitor"] = playoff_urgency_visitor
    df["playoff_urgency_diff"] = df["playoff_urgency_home"] - df["playoff_urgency_visitor"]
    df["post_allstar_break"] = post_allstar
    df["season_progress"] = season_progress

    # Weekend games (Sat/Sun tend to have different dynamics)
    df["is_weekend"] = pd.to_datetime(df["date"]).dt.dayofweek.isin([5, 6]).astype(int)

    # Month features
    df["month"] = pd.to_datetime(df["date"]).dt.month

    print(f"  Added motivation/context features")
    return df


def compute_streak_features(df):
    """
    Compute win/loss streak features for each team entering each game.
    """
    print("Computing streak features...")
    df = df.sort_values("date").reset_index(drop=True)

    team_streaks = {}  # team -> current streak (positive = winning, negative = losing)

    home_streak = [0] * len(df)
    visitor_streak = [0] * len(df)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Streaks"):
        home = row["home_team_id"]
        visitor = row["visitor_team_id"]

        home_streak[idx] = team_streaks.get(home, 0)
        visitor_streak[idx] = team_streaks.get(visitor, 0)

        # Update streaks
        if row["home_win"] == 1:
            team_streaks[home] = max(0, team_streaks.get(home, 0)) + 1
            team_streaks[visitor] = min(0, team_streaks.get(visitor, 0)) - 1
        else:
            team_streaks[home] = min(0, team_streaks.get(home, 0)) - 1
            team_streaks[visitor] = max(0, team_streaks.get(visitor, 0)) + 1

    df["home_streak"] = home_streak
    df["visitor_streak"] = visitor_streak
    df["streak_diff"] = df["home_streak"] - df["visitor_streak"]

    print(f"  Added streak features")
    return df


def compute_season_win_pct_features(df):
    """
    Compute running season win percentage for each team before each game.
    """
    print("Computing season win% features...")
    df = df.sort_values("date").reset_index(drop=True)

    team_records = {}

    home_win_pct = [np.nan] * len(df)
    visitor_win_pct = [np.nan] * len(df)
    home_home_win_pct = [np.nan] * len(df)
    visitor_road_win_pct = [np.nan] * len(df)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Win %"):
        home = row["home_team_id"]
        visitor = row["visitor_team_id"]
        season = row["season"]

        for team, prefix in [(home, "home"), (visitor, "visitor")]:
            key = (team, season)
            if key not in team_records:
                team_records[key] = {"total_w": 0, "total_g": 0,
                                     "home_w": 0, "home_g": 0,
                                     "away_w": 0, "away_g": 0}

            rec = team_records[key]
            if rec["total_g"] > 0:
                if prefix == "home":
                    home_win_pct[idx] = rec["total_w"] / rec["total_g"]
                    if rec["home_g"] > 0:
                        home_home_win_pct[idx] = rec["home_w"] / rec["home_g"]
                else:
                    visitor_win_pct[idx] = rec["total_w"] / rec["total_g"]
                    if rec["away_g"] > 0:
                        visitor_road_win_pct[idx] = rec["away_w"] / rec["away_g"]

        # Update
        h_key = (home, season)
        v_key = (visitor, season)
        if h_key not in team_records:
            team_records[h_key] = {"total_w": 0, "total_g": 0, "home_w": 0, "home_g": 0, "away_w": 0, "away_g": 0}
        if v_key not in team_records:
            team_records[v_key] = {"total_w": 0, "total_g": 0, "home_w": 0, "home_g": 0, "away_w": 0, "away_g": 0}

        team_records[h_key]["total_g"] += 1
        team_records[h_key]["home_g"] += 1
        team_records[v_key]["total_g"] += 1
        team_records[v_key]["away_g"] += 1

        if row["home_win"] == 1:
            team_records[h_key]["total_w"] += 1
            team_records[h_key]["home_w"] += 1
        else:
            team_records[v_key]["total_w"] += 1
            team_records[v_key]["away_w"] += 1

    df["home_season_win_pct"] = home_win_pct
    df["visitor_season_win_pct"] = visitor_win_pct
    df["season_win_pct_diff"] = df["home_season_win_pct"] - df["visitor_season_win_pct"]
    df["home_home_win_pct"] = home_home_win_pct
    df["visitor_road_win_pct"] = visitor_road_win_pct

    print(f"  Added season win% features")
    return df


def compute_real_boxscore_rolling(df, windows=[5, 10]):
    """
    Compute rolling features from REAL box score data (FG%, 3P%, eFG%, TS%, TOV, REB, AST, STL, BLK).
    Only available for games that have box score data merged in (2010-2024).
    For games without box score data, these features will be NaN (filled with 0 later).
    """
    print("Computing rolling features from REAL box score data...")
    df = df.sort_values("date").reset_index(drop=True)

    # Check which box score columns exist
    box_cols_home = ['home_efg_pct', 'home_ts_pct', 'home_fg_pct', 'home_three_pct',
                     'home_ft_pct', 'home_oreb', 'home_dreb', 'home_reb',
                     'home_ast', 'home_stl', 'home_blk', 'home_tov', 'home_pf',
                     'home_fga', 'home_fta', 'home_fg3a',
                     'home_ortg', 'home_drtg', 'home_pace', 'home_tov_pct', 'home_orb_pct']
    box_cols_vis = ['vis_efg_pct', 'vis_ts_pct', 'vis_fg_pct', 'vis_three_pct',
                    'vis_ft_pct', 'vis_oreb', 'vis_dreb', 'vis_reb',
                    'vis_ast', 'vis_stl', 'vis_blk', 'vis_tov', 'vis_pf',
                    'vis_fga', 'vis_fta', 'vis_fg3a',
                    'vis_ortg', 'vis_drtg', 'vis_pace', 'vis_tov_pct', 'vis_orb_pct']

    has_box = any(c in df.columns for c in box_cols_home)
    if not has_box:
        print("  No box score columns found, skipping real box score rolling features.")
        return df

    # Stats to track per team per game
    tracked_stats = ['efg_pct', 'ts_pct', 'fg_pct', 'three_pct', 'ft_pct',
                     'oreb', 'dreb', 'reb', 'ast', 'stl', 'blk', 'tov',
                     'fga', 'fta', 'fg3a',
                     'ortg', 'drtg', 'pace', 'tov_pct', 'orb_pct']

    # Track per-team game history with real box stats
    team_box_history = {}

    # Pre-allocate columns
    feature_cols = {}
    for w in windows:
        for stat in tracked_stats:
            feature_cols[f"last{w}_{stat}_home_real"] = [np.nan] * len(df)
            feature_cols[f"last{w}_{stat}_visitor_real"] = [np.nan] * len(df)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Real Box Rolling"):
        home = row["home_team_id"]
        visitor = row["visitor_team_id"]

        for team, prefix, box_prefix in [(home, "home", "home_"), (visitor, "visitor", "vis_")]:
            if team not in team_box_history:
                team_box_history[team] = []

            history = team_box_history[team]

            for w in windows:
                if len(history) >= w:
                    recent = history[-w:]
                    for stat in tracked_stats:
                        vals = [g.get(stat) for g in recent if g.get(stat) is not None]
                        if vals:
                            feature_cols[f"last{w}_{stat}_{prefix}_real"][idx] = np.mean(vals)

        # After computing pregame features, add this game's box stats to history
        for team, box_prefix in [(home, "home_"), (visitor, "vis_")]:
            game_stats = {}
            for stat in tracked_stats:
                col = f"{box_prefix}{stat}"
                if col in df.columns and pd.notna(row.get(col)):
                    game_stats[stat] = row[col]
            if game_stats:  # Only add if we have real data for this game
                if team not in team_box_history:
                    team_box_history[team] = []
                team_box_history[team].append(game_stats)

    # Add to dataframe
    for col, values in feature_cols.items():
        df[col] = values

    # Compute differentials for key stats
    for w in windows:
        for stat in ['efg_pct', 'ts_pct', 'tov', 'reb', 'ast', 'stl', 'blk',
                     'oreb', 'dreb', 'three_pct',
                     'ortg', 'drtg', 'pace', 'tov_pct', 'orb_pct']:
            h = f"last{w}_{stat}_home_real"
            v = f"last{w}_{stat}_visitor_real"
            if h in df.columns and v in df.columns:
                df[f"last{w}_{stat}_diff_real"] = df[h] - df[v]

    # Compute derived advanced stats
    for w in windows:
        # Turnover rate proxy (TOV / (FGA + 0.44*FTA + TOV))
        for side in ['home', 'visitor']:
            tov_col = f"last{w}_tov_{side}_real"
            fga_col = f"last{w}_fga_{side}_real"
            fta_col = f"last{w}_fta_{side}_real"
            if all(c in df.columns for c in [tov_col, fga_col, fta_col]):
                denom = df[fga_col] + 0.44 * df[fta_col] + df[tov_col]
                df[f"last{w}_tov_rate_{side}"] = df[tov_col] / denom.replace(0, np.nan)

        # TOV rate differential
        h = f"last{w}_tov_rate_home"
        v = f"last{w}_tov_rate_visitor"
        if h in df.columns and v in df.columns:
            df[f"last{w}_tov_rate_diff"] = df[h] - df[v]

        # Rebounding edge (OREB% proxy)
        for side in ['home', 'visitor']:
            oreb_col = f"last{w}_oreb_{side}_real"
            reb_col = f"last{w}_reb_{side}_real"
            if oreb_col in df.columns and reb_col in df.columns:
                df[f"last{w}_oreb_pct_{side}"] = df[oreb_col] / df[reb_col].replace(0, np.nan)

        # 3P attempt rate (3PA / FGA)
        for side in ['home', 'visitor']:
            fg3a_col = f"last{w}_fg3a_{side}_real"
            fga_col = f"last{w}_fga_{side}_real"
            if fg3a_col in df.columns and fga_col in df.columns:
                df[f"last{w}_three_rate_{side}"] = df[fg3a_col] / df[fga_col].replace(0, np.nan)

    n_new = len(feature_cols) + sum(1 for c in df.columns if '_diff_real' in c or '_tov_rate_' in c
                                     or '_oreb_pct_' in c or '_three_rate_' in c)
    print(f"  Added {n_new} real box score rolling features")
    return df


def compute_odds_features(df):
    """
    Compute features from real betting odds data.
    Market-implied probabilities are strong predictors — the betting market
    is one of the best pregame forecasters.
    """
    print("Computing betting odds features...")

    odds_cols = ['spread', 'opening_total', 'implied_prob_home', 'implied_prob_away',
                 'ml_home', 'ml_away']
    has_odds = any(c in df.columns for c in odds_cols)

    if not has_odds:
        print("  No odds columns found, skipping.")
        return df

    # Coerce numeric columns that may have string values
    for col in ['spread', 'opening_total', 'implied_prob_home', 'implied_prob_away', 'ml_home', 'ml_away']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Spread (negative = home favored)
    if 'spread' in df.columns:
        df['spread_abs'] = df['spread'].abs()
        # Spread already captures market expectation

    # Market implied win probability differential
    if 'implied_prob_home' in df.columns and 'implied_prob_away' in df.columns:
        df['market_prob_diff'] = df['implied_prob_home'] - df['implied_prob_away']

    # Over/under as proxy for expected pace/scoring environment
    if 'opening_total' in df.columns:
        df['expected_total'] = df['opening_total']

    n_odds = sum(1 for c in ['spread', 'spread_abs', 'market_prob_diff', 'expected_total']
                 if c in df.columns)
    print(f"  Added {n_odds} odds-based features")
    return df


def compute_raptor_features(df):
    """
    Compute features from RAPTOR player ratings (aggregated to team-season level).
    """
    print("Computing RAPTOR-based features...")

    raptor_cols = ['home_raptor_off', 'home_raptor_def', 'home_raptor_total',
                   'vis_raptor_off', 'vis_raptor_def', 'vis_raptor_total',
                   'home_war', 'vis_war']
    has_raptor = any(c in df.columns for c in raptor_cols)

    if not has_raptor:
        print("  No RAPTOR columns found, skipping.")
        return df

    if 'home_raptor_total' in df.columns and 'vis_raptor_total' in df.columns:
        df['raptor_total_diff'] = df['home_raptor_total'] - df['vis_raptor_total']
        df['raptor_off_diff'] = df['home_raptor_off'] - df['vis_raptor_off']
        df['raptor_def_diff'] = df['home_raptor_def'] - df['vis_raptor_def']

    if 'home_war' in df.columns and 'vis_war' in df.columns:
        df['war_diff'] = df['home_war'] - df['vis_war']

    n_raptor = sum(1 for c in ['raptor_total_diff', 'raptor_off_diff',
                                'raptor_def_diff', 'war_diff'] if c in df.columns)
    print(f"  Added {n_raptor} RAPTOR-based features")
    return df


def compute_advanced_features(df):
    """
    Compute advanced derived features that combine base features for higher signal.
    These are interaction terms, non-linear transforms, and composite ratings
    that have strong theoretical backing for NBA game prediction.
    """
    print("Computing advanced derived features...")

    # --- 1. Non-linear Elo transforms ---
    # Squared Elo diff captures that a 300-pt Elo gap is more than 3x a 100-pt gap
    if "elo_rating_diff" in df.columns:
        df["elo_diff_squared"] = df["elo_rating_diff"] ** 2 * np.sign(df["elo_rating_diff"])
        # Absolute Elo diff (game competitiveness - closer games are harder to predict)
        df["elo_diff_abs"] = df["elo_rating_diff"].abs()

    # --- 2. Pythagorean win expectation (Bill James style, adapted for NBA) ---
    # Uses season points scored/allowed to estimate true team strength
    # NBA exponent is typically ~13.91 (Morey), we use ~14
    PYTH_EXP = 14
    for side in ["home", "visitor"]:
        scored_col = f"last10_pts_scored_{side}"
        allowed_col = f"last10_pts_allowed_{side}"
        if scored_col in df.columns and allowed_col in df.columns:
            scored = df[scored_col].fillna(105)
            allowed = df[allowed_col].fillna(105)
            # Avoid division by zero
            denom = scored ** PYTH_EXP + allowed ** PYTH_EXP
            df[f"pyth_win_exp_{side}"] = np.where(denom > 0, scored ** PYTH_EXP / denom, 0.5)

    if "pyth_win_exp_home" in df.columns and "pyth_win_exp_visitor" in df.columns:
        df["pyth_win_exp_diff"] = df["pyth_win_exp_home"] - df["pyth_win_exp_visitor"]

    # --- 3. Rest × quality interactions ---
    # B2B is worse for good teams (more to lose) and compounds with travel
    if "home_b2b" in df.columns and "elo_rating_diff" in df.columns:
        df["home_b2b_x_elo"] = df["home_b2b"] * df["elo_rating_diff"]
        df["visitor_b2b_x_elo"] = df["visitor_b2b"] * (-df["elo_rating_diff"])
    if "rest_diff" in df.columns and "elo_rating_diff" in df.columns:
        df["rest_x_elo"] = df["rest_diff"] * df["elo_rating_diff"]

    # --- 4. Travel fatigue composite ---
    # Combine travel miles + timezone shift + B2B into single fatigue score
    for side in ["home", "visitor"]:
        travel = df.get(f"travel_miles_{side}", pd.Series(0, index=df.index)).fillna(0)
        tz = df.get(f"tz_shift_{side}", pd.Series(0, index=df.index)).fillna(0)
        b2b = df.get(f"{side}_b2b", pd.Series(0, index=df.index)).fillna(0)
        three_in_4 = df.get(f"{side}_3in4", pd.Series(0, index=df.index)).fillna(0)
        # Normalize travel to 0-1 range (max ~2800 miles cross-country)
        travel_norm = travel / 2800.0
        df[f"fatigue_composite_{side}"] = travel_norm + tz * 0.3 + b2b * 0.5 + three_in_4 * 0.3

    df["fatigue_diff"] = df["fatigue_composite_home"] - df["fatigue_composite_visitor"]

    # --- 5. Form momentum (weighted recent form - last 3 games weighted 2x vs games 4-10) ---
    for w_short, w_long in [(5, 10)]:
        for stat in ["net_rating", "win_rate", "avg_margin"]:
            short_col = f"last{w_short}_{stat}_diff"
            long_col = f"last{w_long}_{stat}_diff"
            if short_col in df.columns and long_col in df.columns:
                # Weighted: 60% recent, 40% longer window
                df[f"weighted_{stat}_momentum"] = (
                    0.6 * df[short_col].fillna(0) + 0.4 * df[long_col].fillna(0)
                )

    # --- 6. Streak × season context interaction ---
    if "streak_diff" in df.columns and "season_progress" in df.columns:
        df["streak_x_progress"] = df["streak_diff"] * df["season_progress"]

    # --- 7. Win pct composite (combines overall, home/road, and Elo into single power rating) ---
    if all(c in df.columns for c in ["season_win_pct_diff", "elo_rating_diff"]):
        elo_norm = df["elo_rating_diff"] / 400.0  # Normalize to ~[-1, 1]
        wpct = df["season_win_pct_diff"].fillna(0)
        df["power_rating_composite"] = 0.6 * elo_norm + 0.4 * wpct

    # --- 8. Defensive matchup quality ---
    for w in [5, 10]:
        off_home = f"last{w}_off_rating_home"
        def_vis = f"last{w}_def_rating_visitor"
        off_vis = f"last{w}_off_rating_visitor"
        def_home = f"last{w}_def_rating_home"
        if all(c in df.columns for c in [off_home, def_vis, off_vis, def_home]):
            # Net matchup advantage: how much better is home offense vs visitor defense
            # minus how much better is visitor offense vs home defense
            df[f"net_matchup_edge_{w}"] = (
                (df[off_home] - df[def_vis]) - (df[off_vis] - df[def_home])
            )

    # --- 9. H2H recency-weighted (more weight to recent matchups) ---
    # Already have h2h features, add interaction with current form
    if "h2h_home_win_rate" in df.columns and "season_win_pct_diff" in df.columns:
        df["h2h_x_current_form"] = (
            df["h2h_home_win_rate"].fillna(0.5) * df["season_win_pct_diff"].fillna(0)
        )

    # --- 10. Consistency features (variance in recent performance) ---
    # Already tracked in rolling form - use margin variance as proxy
    # High variance teams are less predictable
    # This is implicitly captured but let's add explicit feature
    if "last10_avg_margin_diff" in df.columns and "last5_avg_margin_diff" in df.columns:
        # Form volatility: difference between short and long window
        df["form_volatility"] = (
            df["last5_avg_margin_diff"].fillna(0) - df["last10_avg_margin_diff"].fillna(0)
        ).abs()

    n_new = sum(1 for c in df.columns if c.startswith(("elo_diff_sq", "elo_diff_abs",
                "pyth_", "home_b2b_x", "visitor_b2b_x", "rest_x", "fatigue_",
                "weighted_", "streak_x", "power_rating", "net_matchup",
                "h2h_x_", "form_vol")))
    print(f"  Added {n_new} advanced derived features")
    return df


def compute_team_quality_features(df):
    """
    Compute differentials for team-season quality metrics from sumitrodatta data.
    These are season-level stats (SRS, team ORtg/DRtg, BPM, WS/48, VORP).
    """
    print("Computing team quality differentials...")

    quality_pairs = [
        ('home_srs', 'vis_srs', 'srs_diff'),
        ('home_team_ortg', 'vis_team_ortg', 'team_ortg_diff'),
        ('home_team_drtg', 'vis_team_drtg', 'team_drtg_diff'),
        ('home_team_nrtg', 'vis_team_nrtg', 'team_nrtg_diff'),
        ('home_team_pace', 'vis_team_pace', 'team_pace_diff'),
        ('home_team_bpm', 'vis_team_bpm', 'team_bpm_diff'),
        ('home_team_ws48', 'vis_team_ws48', 'team_ws48_diff'),
        ('home_team_vorp', 'vis_team_vorp', 'team_vorp_diff'),
        ('home_team_per', 'vis_team_per', 'team_per_diff'),
    ]

    n_added = 0
    for home_col, vis_col, diff_col in quality_pairs:
        if home_col in df.columns and vis_col in df.columns:
            df[diff_col] = df[home_col] - df[vis_col]
            n_added += 1

    print(f"  Added {n_added} team quality differential features")
    return df


def compute_all_features(df):
    """Run the complete feature engineering pipeline."""
    print("=" * 70)
    print("FEATURE ENGINEERING PIPELINE")
    print(f"Input: {len(df)} games")
    print("=" * 70)

    # Add home indicator
    df["home_flag"] = 1

    # Ensure required columns exist
    if "margin" not in df.columns:
        df["margin"] = df["home_pts"] - df["visitor_pts"]
    if "home_win" not in df.columns:
        df["home_win"] = (df["home_pts"] > df["visitor_pts"]).astype(int)

    df = compute_schedule_features(df)
    df = compute_rolling_form_features(df, windows=[3, 5, 10])
    df = compute_season_stats_features(df)
    df = compute_head_to_head_features(df)
    df = compute_matchup_style_features(df)
    df = compute_motivation_features(df)
    df = compute_streak_features(df)
    df = compute_season_win_pct_features(df)
    df = compute_real_boxscore_rolling(df, windows=[5, 10])
    df = compute_odds_features(df)
    df = compute_raptor_features(df)
    df = compute_advanced_features(df)
    df = compute_team_quality_features(df)

    print(f"\n{'=' * 70}")
    print(f"FEATURE ENGINEERING COMPLETE")
    print(f"Output: {len(df)} games x {len(df.columns)} columns")
    print(f"{'=' * 70}")

    return df


if __name__ == "__main__":
    import os
    processed_dir = "/home/user/Basketballbet/data/processed"
    games_path = os.path.join(processed_dir, "games_with_elo.csv")

    if os.path.exists(games_path):
        df = pd.read_csv(games_path, parse_dates=["date"])
        df = compute_all_features(df)
        df.to_csv(os.path.join(processed_dir, "games_full_features.csv"), index=False)
        print(f"\nSaved full feature set to games_full_features.csv")
        print(f"Columns: {list(df.columns)}")
    else:
        print("No games_with_elo.csv found. Run elo_ratings.py first.")

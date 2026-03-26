"""
Real Data Integration Pipeline
=================================
Aggregates player-level box scores into team-level per-game stats,
joins with betting odds and RAPTOR player ratings to create a
comprehensive feature dataset.

Data sources:
1. Player box scores (2010-2024): FGM, FGA, 3PM, 3PA, FTM, FTA, REB, AST, STL, BLK, TOV, PF
2. Betting odds (2007-2026): Spread, OU, Moneyline
3. RAPTOR player ratings (2014-2022): Offensive/Defensive/Total ratings, WAR
4. Team advanced stats (1996-2023): OFF_RATING, DEF_RATING, NET_RATING, eFG%, TS%, PACE
"""

import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

DATA_DIR = "/home/user/Basketballbet/data/raw"
PROCESSED_DIR = "/home/user/Basketballbet/data/processed"

# Team name normalization (full names to 3-letter abbreviations)
TEAM_NAME_MAP = {
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
    "New Jersey Nets": "BKN", "Vancouver Grizzlies": "MEM",
}

TRICODE_MAP = {
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
    "WAS": "WAS", "NJN": "BKN", "VAN": "MEM",
}


def normalize_team(name):
    if name is None:
        return None
    name = str(name).strip()
    if name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[name]
    if name in TRICODE_MAP:
        return TRICODE_MAP[name]
    for full, abbr in TEAM_NAME_MAP.items():
        if name.lower() in full.lower() or full.lower() in name.lower():
            return abbr
    return name


def build_team_game_boxscores():
    """
    Aggregate player-level box scores into team-level per-game stats.
    Produces one row per team per game with: FGM, FGA, FG%, 3PM, 3PA, 3P%,
    FTM, FTA, FT%, OREB, DREB, REB, AST, STL, BLK, TOV, PF, PTS, +/-.
    """
    print("=" * 70)
    print("AGGREGATING PLAYER BOX SCORES → TEAM GAME STATS")
    print("=" * 70)

    box_dir = os.path.join(DATA_DIR, "box_scores")
    parts = sorted(glob.glob(os.path.join(box_dir, "regular_season_box_scores_*.csv")))

    if not parts:
        print("  No box score files found!")
        return None

    dfs = []
    for p in parts:
        print(f"  Loading {os.path.basename(p)}...")
        df = pd.read_csv(p, low_memory=False)
        dfs.append(df)
    raw = pd.concat(dfs, ignore_index=True)
    print(f"  Total player rows: {len(raw)}")

    # Parse minutes to float
    def parse_minutes(m):
        if pd.isna(m) or m == '' or m is None:
            return 0.0
        m = str(m)
        if ':' in m:
            parts = m.split(':')
            return float(parts[0]) + float(parts[1]) / 60.0
        try:
            return float(m)
        except:
            return 0.0

    raw['minutes_float'] = raw['minutes'].apply(parse_minutes)

    # Numeric columns to aggregate
    stat_cols = [
        'fieldGoalsMade', 'fieldGoalsAttempted',
        'threePointersMade', 'threePointersAttempted',
        'freeThrowsMade', 'freeThrowsAttempted',
        'reboundsOffensive', 'reboundsDefensive', 'reboundsTotal',
        'assists', 'steals', 'blocks', 'turnovers', 'foulsPersonal',
        'points', 'plusMinusPoints'
    ]

    for col in stat_cols:
        raw[col] = pd.to_numeric(raw[col], errors='coerce').fillna(0)

    raw['minutes_float'] = raw['minutes_float'].fillna(0)

    # Aggregate to team-game level
    print("  Aggregating to team-game level...")
    team_games = raw.groupby(['season_year', 'game_date', 'gameId', 'teamTricode']).agg(
        **{col: (col, 'sum') for col in stat_cols},
        total_minutes=('minutes_float', 'sum'),
        players_used=('personId', 'nunique'),
    ).reset_index()

    # Compute shooting percentages
    team_games['fg_pct'] = team_games['fieldGoalsMade'] / team_games['fieldGoalsAttempted'].replace(0, np.nan)
    team_games['three_pct'] = team_games['threePointersMade'] / team_games['threePointersAttempted'].replace(0, np.nan)
    team_games['ft_pct'] = team_games['freeThrowsMade'] / team_games['freeThrowsAttempted'].replace(0, np.nan)

    # Advanced shooting: eFG% and TS%
    team_games['efg_pct'] = (team_games['fieldGoalsMade'] + 0.5 * team_games['threePointersMade']) / \
                            team_games['fieldGoalsAttempted'].replace(0, np.nan)
    team_games['ts_pct'] = team_games['points'] / \
                           (2 * (team_games['fieldGoalsAttempted'] + 0.44 * team_games['freeThrowsAttempted'])).replace(0, np.nan)

    # Normalize team tricode
    team_games['team_id'] = team_games['teamTricode'].apply(lambda x: TRICODE_MAP.get(x, x))

    # Parse date
    team_games['date'] = pd.to_datetime(team_games['game_date'])

    # Rename columns for clarity
    team_games = team_games.rename(columns={
        'fieldGoalsMade': 'fgm', 'fieldGoalsAttempted': 'fga',
        'threePointersMade': 'fg3m', 'threePointersAttempted': 'fg3a',
        'freeThrowsMade': 'ftm', 'freeThrowsAttempted': 'fta',
        'reboundsOffensive': 'oreb', 'reboundsDefensive': 'dreb',
        'reboundsTotal': 'reb', 'assists': 'ast', 'steals': 'stl',
        'blocks': 'blk', 'turnovers': 'tov', 'foulsPersonal': 'pf',
        'points': 'pts', 'plusMinusPoints': 'plus_minus',
    })

    print(f"  Team-game rows: {len(team_games)}")
    print(f"  Date range: {team_games['date'].min()} to {team_games['date'].max()}")
    print(f"  Seasons: {sorted(team_games['season_year'].unique())}")

    # Save
    out_path = os.path.join(PROCESSED_DIR, "team_game_boxscores.csv")
    team_games.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")

    return team_games


def build_odds_dataset():
    """Load and clean the betting odds data."""
    print("\n" + "=" * 70)
    print("LOADING BETTING ODDS DATA")
    print("=" * 70)

    odds_path = os.path.join(DATA_DIR, "odds", "nba_odds_all.csv")
    df = pd.read_csv(odds_path)

    # Drop duplicates — prefer '_new' tables (more recent data)
    # Keep the row with the most non-null values
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['home_id'] = df['Home'].apply(normalize_team)
    df['away_id'] = df['Away'].apply(normalize_team)

    # Drop exact duplicates (same date, same matchup)
    df = df.sort_values('season_id').drop_duplicates(
        subset=['Date', 'home_id', 'away_id'], keep='last'
    ).reset_index(drop=True)

    # Keep essential columns
    odds_clean = df[['Date', 'home_id', 'away_id', 'OU', 'Spread', 'ML_Home', 'ML_Away', 'season_id']].copy()
    odds_clean = odds_clean.rename(columns={
        'Date': 'date', 'OU': 'opening_total', 'Spread': 'spread',
        'ML_Home': 'ml_home', 'ML_Away': 'ml_away',
    })

    # Convert moneyline to implied probability
    def ml_to_prob(ml):
        if pd.isna(ml) or ml == 0:
            return 0.5
        try:
            ml = float(ml)
        except (ValueError, TypeError):
            return 0.5
        if ml > 0:
            return 100.0 / (ml + 100.0)
        else:
            return abs(ml) / (abs(ml) + 100.0)

    odds_clean['implied_prob_home'] = odds_clean['ml_home'].apply(ml_to_prob)
    odds_clean['implied_prob_away'] = odds_clean['ml_away'].apply(ml_to_prob)

    print(f"  Odds rows: {len(odds_clean)}")
    print(f"  Date range: {odds_clean['date'].min()} to {odds_clean['date'].max()}")

    out_path = os.path.join(PROCESSED_DIR, "odds_clean.csv")
    odds_clean.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")

    return odds_clean


def build_raptor_season_ratings():
    """
    Aggregate RAPTOR player ratings to team-season level.
    Creates a team strength proxy from player-level RAPTOR data.
    """
    print("\n" + "=" * 70)
    print("LOADING RAPTOR PLAYER RATINGS")
    print("=" * 70)

    raptor_path = os.path.join(DATA_DIR, "raptor_by_team.csv")
    df = pd.read_csv(raptor_path)

    # Filter to regular season
    df = df[df['season_type'] == 'RS'].copy()

    # Normalize team names
    df['team_id'] = df['team'].apply(lambda x: TRICODE_MAP.get(x, x))

    # Weight by minutes played
    df['mp'] = pd.to_numeric(df['mp'], errors='coerce').fillna(0)

    # Aggregate to team-season: minutes-weighted RAPTOR
    def weighted_avg(group, col, weight_col='mp'):
        weights = group[weight_col]
        values = group[col]
        total_weight = weights.sum()
        if total_weight == 0:
            return 0
        return (values * weights).sum() / total_weight

    team_season = df.groupby(['team_id', 'season']).apply(
        lambda g: pd.Series({
            'raptor_offense': weighted_avg(g, 'raptor_offense'),
            'raptor_defense': weighted_avg(g, 'raptor_defense'),
            'raptor_total': weighted_avg(g, 'raptor_total'),
            'war_total': g['war_total'].sum(),
            'total_minutes': g['mp'].sum(),
            'n_players': len(g),
        })
    ).reset_index()

    print(f"  Team-season rows: {len(team_season)}")
    print(f"  Seasons: {sorted(team_season['season'].unique())}")

    out_path = os.path.join(PROCESSED_DIR, "raptor_team_season.csv")
    team_season.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")

    return team_season


def build_team_season_advanced():
    """
    Load team-level advanced stats per season from NBA.com data.
    Includes: OFF_RATING, DEF_RATING, NET_RATING, eFG%, TS%, PACE, etc.
    """
    print("\n" + "=" * 70)
    print("LOADING TEAM ADVANCED STATS (PER SEASON)")
    print("=" * 70)

    stats_dir = os.path.join(DATA_DIR, "team_stats")

    # Load advanced stats
    adv_path = os.path.join(stats_dir, "team_stats_advanced_rs.csv")
    adv = pd.read_csv(adv_path)

    # Load traditional stats
    trad_path = os.path.join(stats_dir, "team_stats_traditional_rs.csv")
    trad = pd.read_csv(trad_path)

    # Load four factors
    ff_path = os.path.join(stats_dir, "team_stats_four_factors_rs.csv")
    ff = pd.read_csv(ff_path)

    # Normalize team names from TEAM_ID
    # We'll use TEAM_NAME → abbreviation
    for df in [adv, trad, ff]:
        df['team_id'] = df['TEAM_NAME'].apply(normalize_team)

    # Merge advanced + traditional + four factors
    merged = adv[['team_id', 'SEASON', 'OFF_RATING', 'DEF_RATING', 'NET_RATING',
                   'EFG_PCT', 'TS_PCT', 'PACE', 'AST_PCT', 'OREB_PCT', 'DREB_PCT',
                   'REB_PCT', 'TM_TOV_PCT']].copy()

    trad_cols = trad[['team_id', 'SEASON', 'FG_PCT', 'FG3_PCT', 'FT_PCT',
                       'REB', 'AST', 'TOV', 'STL', 'BLK', 'PTS', 'PLUS_MINUS']].copy()
    merged = merged.merge(trad_cols, on=['team_id', 'SEASON'], how='left')

    ff_cols = ff[['team_id', 'SEASON', 'FTA_RATE', 'OPP_EFG_PCT',
                   'OPP_FTA_RATE', 'OPP_TOV_PCT', 'OPP_OREB_PCT']].copy()
    merged = merged.merge(ff_cols, on=['team_id', 'SEASON'], how='left')

    # Rename for clarity
    merged = merged.rename(columns={'SEASON': 'season'})

    print(f"  Team-season rows: {len(merged)}")
    print(f"  Seasons: {sorted(merged['season'].unique())[:5]}...{sorted(merged['season'].unique())[-5:]}")
    print(f"  Columns: {list(merged.columns)}")

    out_path = os.path.join(PROCESSED_DIR, "team_season_advanced.csv")
    merged.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")

    return merged


def merge_all_into_games(games_df=None):
    """
    Merge team-game box scores, odds, RAPTOR, and advanced stats
    into the main games dataset. Optionally accepts a pre-built DataFrame.
    """
    print("\n" + "=" * 70)
    print("MERGING ALL DATA INTO GAMES DATASET")
    print("=" * 70)

    # Load main games or use provided DataFrame
    if games_df is not None:
        games = games_df.copy()
        games['date'] = pd.to_datetime(games['date'])
    else:
        games = pd.read_csv(os.path.join(PROCESSED_DIR, "games_clean.csv"), parse_dates=['date'])
    print(f"  Main games: {len(games)}")

    # Load team game box scores
    box_path = os.path.join(PROCESSED_DIR, "team_game_boxscores.csv")
    if os.path.exists(box_path):
        box = pd.read_csv(box_path, parse_dates=['date'])
        # Create a date key for matching (normalize to date only)
        box['date_key'] = box['date'].dt.date
        games['date_key'] = games['date'].dt.date

        # Merge home team box scores
        home_box = box.rename(columns={c: f'home_{c}' for c in box.columns
                                        if c not in ['date', 'date_key', 'team_id', 'gameId', 'season_year', 'game_date', 'teamTricode']})
        home_box['date_key'] = box['date_key']
        home_box['team_id'] = box['team_id']

        games = games.merge(
            home_box[['date_key', 'team_id', 'home_fgm', 'home_fga', 'home_fg_pct',
                       'home_fg3m', 'home_fg3a', 'home_three_pct',
                       'home_ftm', 'home_fta', 'home_ft_pct',
                       'home_oreb', 'home_dreb', 'home_reb',
                       'home_ast', 'home_stl', 'home_blk', 'home_tov', 'home_pf',
                       'home_efg_pct', 'home_ts_pct', 'home_plus_minus']],
            left_on=['date_key', 'home_team_id'],
            right_on=['date_key', 'team_id'],
            how='left'
        ).drop(columns=['team_id'], errors='ignore')

        # Merge visitor team box scores
        vis_box = box.rename(columns={c: f'vis_{c}' for c in box.columns
                                       if c not in ['date', 'date_key', 'team_id', 'gameId', 'season_year', 'game_date', 'teamTricode']})
        vis_box['date_key'] = box['date_key']
        vis_box['team_id'] = box['team_id']

        games = games.merge(
            vis_box[['date_key', 'team_id', 'vis_fgm', 'vis_fga', 'vis_fg_pct',
                      'vis_fg3m', 'vis_fg3a', 'vis_three_pct',
                      'vis_ftm', 'vis_fta', 'vis_ft_pct',
                      'vis_oreb', 'vis_dreb', 'vis_reb',
                      'vis_ast', 'vis_stl', 'vis_blk', 'vis_tov', 'vis_pf',
                      'vis_efg_pct', 'vis_ts_pct', 'vis_plus_minus']],
            left_on=['date_key', 'visitor_team_id'],
            right_on=['date_key', 'team_id'],
            how='left'
        ).drop(columns=['team_id'], errors='ignore')

        box_matched = games['home_fgm'].notna().sum()
        print(f"  Box scores matched: {box_matched}/{len(games)} games ({100*box_matched/len(games):.1f}%)")

    # Load and merge odds
    odds_path = os.path.join(PROCESSED_DIR, "odds_clean.csv")
    if os.path.exists(odds_path):
        odds = pd.read_csv(odds_path, parse_dates=['date'])
        odds['date_key'] = odds['date'].dt.date

        games = games.merge(
            odds[['date_key', 'home_id', 'opening_total', 'spread',
                  'ml_home', 'ml_away', 'implied_prob_home', 'implied_prob_away']],
            left_on=['date_key', 'home_team_id'],
            right_on=['date_key', 'home_id'],
            how='left'
        ).drop(columns=['home_id'], errors='ignore')

        odds_matched = games['spread'].notna().sum()
        print(f"  Odds matched: {odds_matched}/{len(games)} games ({100*odds_matched/len(games):.1f}%)")

    # Load RAPTOR team-season ratings
    raptor_path = os.path.join(PROCESSED_DIR, "raptor_team_season.csv")
    if os.path.exists(raptor_path):
        raptor = pd.read_csv(raptor_path)
        # RAPTOR season is like 2022 for 2021-22 season
        # Our season column: 2022 = 2021-22

        games = games.merge(
            raptor[['team_id', 'season', 'raptor_offense', 'raptor_defense', 'raptor_total', 'war_total']].rename(
                columns={'raptor_offense': 'home_raptor_off', 'raptor_defense': 'home_raptor_def',
                         'raptor_total': 'home_raptor_total', 'war_total': 'home_war'}
            ),
            left_on=['home_team_id', 'season'],
            right_on=['team_id', 'season'],
            how='left'
        ).drop(columns=['team_id'], errors='ignore')

        games = games.merge(
            raptor[['team_id', 'season', 'raptor_offense', 'raptor_defense', 'raptor_total', 'war_total']].rename(
                columns={'raptor_offense': 'vis_raptor_off', 'raptor_defense': 'vis_raptor_def',
                         'raptor_total': 'vis_raptor_total', 'war_total': 'vis_war'}
            ),
            left_on=['visitor_team_id', 'season'],
            right_on=['team_id', 'season'],
            how='left'
        ).drop(columns=['team_id'], errors='ignore')

        raptor_matched = games['home_raptor_total'].notna().sum()
        print(f"  RAPTOR matched: {raptor_matched}/{len(games)} games ({100*raptor_matched/len(games):.1f}%)")

    games = games.drop(columns=['date_key'], errors='ignore')

    # Save enriched games
    out_path = os.path.join(PROCESSED_DIR, "games_enriched.csv")
    games.to_csv(out_path, index=False)
    print(f"\n  Enriched games: {len(games)} rows x {len(games.columns)} columns")
    print(f"  Saved to {out_path}")

    return games


if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Step 1: Aggregate player box scores → team game stats
    team_box = build_team_game_boxscores()

    # Step 2: Clean odds data
    odds = build_odds_dataset()

    # Step 3: Aggregate RAPTOR to team-season
    raptor = build_raptor_season_ratings()

    # Step 4: Load team advanced stats
    team_adv = build_team_season_advanced()

    # Step 5: Merge everything into main games dataset
    enriched = merge_all_into_games()

"""
Comprehensive Real Data Integration v3
==========================================
Integrates data from multiple real sources to cover ALL covariates
for 2003-2026 (23 seasons):

Sources:
1. deepshot/schedule.csv (2000-2026): Home/away assignments for all games
2. deepshot/gamelogs.csv (2000-2026): 38K+ team-game logs with REAL integer scores
   + advanced stats (ORtg, DRtg, Pace, eFG%, TOV%, ORB%, TS%, AST%, STL%, BLK%)
3. deepshot/results.csv (2000-2025): Game results with winning_team flag
4. fivethirtyeight/nba-data-historical.csv: RAPTOR player ratings (1977-2020)
5. Existing raptor_by_team.csv: RAPTOR (2014-2022)
6. Existing odds data: Betting lines (2007-2026)
7. Existing BBRef season CSVs: Game results for COVID seasons (2020-2021)

NOTE: deepshot/dataset.csv has ADJUSTED (non-integer) scores and is NOT used.
      All game data comes from schedule.csv + gamelogs.csv which have real scores.
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm

DATA_DIR = "/home/user/Basketballbet/data/raw"
PROCESSED_DIR = "/home/user/Basketballbet/data/processed"

# ============================================================
# Team name normalization
# ============================================================
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
    "New Orleans:Oklahoma City Hornets": "NOP",
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
    if name is None or pd.isna(name):
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


def get_season(date):
    """Get NBA season year (e.g., 2024 for 2023-24 season)."""
    if date.month >= 10:
        return date.year + 1
    return date.year


# ============================================================
# Step 1: Build complete game dataset from BBRef + gamelogs
# ============================================================
def build_complete_games():
    """
    Build complete game dataset with real scores for ALL seasons 2004-2026.

    Primary sources:
    - BBRef CSVs (2005-2025): Complete game results with real integer scores
    - Deepshot schedule + gamelogs (2004, 2026): Games not covered by BBRef

    Enrichment:
    - Deepshot gamelogs: Advanced stats (ORtg, DRtg, Pace, eFG%, etc.) for ~30-40%
      of games per season + near-complete 2026 season

    IMPORTANT: We do NOT use deepshot/dataset.csv because its pts values are
    adjusted (non-integer) scores, not real game scores.
    """
    print("=" * 70)
    print("STEP 1: BUILD COMPLETE GAMES (2004-2026)")
    print("=" * 70)

    # ---- Part A: Load ALL BBRef season CSVs (real scores) ----
    bbref_dir = os.path.join(DATA_DIR, "bball_ref_seasons")
    all_bbref_games = []

    season_files = {
        2005: "NBA2004-05.csv", 2006: "NBA2005-06.csv", 2007: "NBA2006-07.csv",
        2008: "NBA2007-08.csv", 2009: "NBA2008-09.csv", 2010: "NBA2009-10.csv",
        2011: "NBA2010-11.csv", 2012: "NBA2011-12.csv", 2013: "NBA2012-13.csv",
        2014: "NBA2013-14.csv", 2015: "NBA2014-15.csv", 2016: "NBA2015-16.csv",
        2017: "NBA2016-17.csv", 2018: "NBA2017-18.csv", 2019: "NBA2018-19.csv",
        2020: "NBA2019-20.csv", 2021: "NBA2020-21.csv", 2022: "NBA2021-22.csv",
        2023: "NBA2022-23.csv", 2024: "NBA2023-24.csv", 2025: "NBA2024-25.csv",
    }

    for season, fn in sorted(season_files.items()):
        path = os.path.join(bbref_dir, fn)
        if not os.path.exists(path):
            print(f"  WARNING: Missing {fn}")
            continue
        games = _parse_bbref_csv(path, season)
        all_bbref_games.extend(games)

    bbref_df = pd.DataFrame(all_bbref_games)
    print(f"  BBRef: {len(bbref_df)} games (seasons {bbref_df['season'].min()}-{bbref_df['season'].max()})")

    # ---- Part B: Load deepshot schedule + gamelogs for 2004 and 2026 ----
    sch = pd.read_csv(os.path.join(DATA_DIR, "deepshot", "schedule.csv"))
    sch['date'] = pd.to_datetime(sch['date'])
    sch['season'] = sch['date'].apply(get_season)
    sch['home_id'] = sch['home_team'].apply(normalize_team)
    sch['away_id'] = sch['away_team'].apply(normalize_team)

    gl = pd.read_csv(os.path.join(DATA_DIR, "deepshot", "gamelogs.csv"))
    gl['date'] = pd.to_datetime(gl['date'])
    gl['team_id'] = gl['team'].apply(normalize_team)
    gl['season'] = gl['date'].apply(get_season)
    print(f"  Gamelogs: {len(gl)} team-game rows ({gl['date'].min().date()} to {gl['date'].max().date()})")

    # Build 2004 season from schedule+gamelogs (BBRef doesn't have this season)
    sch_2004 = sch[sch['season'] == 2004].copy()
    gl_2004 = gl[gl['season'] == 2004].copy()
    games_2004 = _build_all_games_from_schedule_gamelogs(sch_2004, gl_2004)
    print(f"  2004 season from gamelogs: {len(games_2004)} games")

    # Build 2026 season from schedule+gamelogs
    sch_2026 = sch[sch['season'] == 2026].copy()
    gl_2026 = gl[gl['season'] == 2026].copy()
    games_2026 = _build_all_games_from_schedule_gamelogs(sch_2026, gl_2026)
    print(f"  2026 season from gamelogs: {len(games_2026)} games")

    # ---- Part C: Combine all sources ----
    parts = [bbref_df]
    if len(games_2004) > 0:
        parts.append(games_2004)
    if len(games_2026) > 0:
        parts.append(games_2026)
    all_games = pd.concat(parts, ignore_index=True)
    all_games = all_games.sort_values('date').reset_index(drop=True)

    # Remove exact duplicates
    before = len(all_games)
    all_games = all_games.drop_duplicates(subset=['date', 'home_team_id', 'visitor_team_id'], keep='first')
    if len(all_games) < before:
        print(f"  Removed {before - len(all_games)} duplicate games")

    # ---- Part D: Enrich with gamelog advanced stats ----
    print("  Enriching with gamelog advanced stats...")
    all_games = _enrich_with_gamelogs(all_games, gl)

    # Add derived columns
    all_games['home_win'] = (all_games['home_pts'] > all_games['visitor_pts']).astype(int)
    all_games['margin'] = all_games['home_pts'] - all_games['visitor_pts']

    # Verify data quality
    non_int = (all_games['home_pts'] != all_games['home_pts'].round(0)).sum()
    print(f"  Score integrity: {len(all_games) - non_int}/{len(all_games)} integer scores")
    print(f"  Home win rate: {all_games['home_win'].mean():.3f} (expected ~0.58-0.60)")

    # Coverage stats
    adv_coverage = all_games['home_ortg'].notna().sum() if 'home_ortg' in all_games.columns else 0
    print(f"  Advanced stats coverage: {adv_coverage}/{len(all_games)} ({100*adv_coverage/len(all_games):.1f}%)")

    print(f"\n  COMPLETE DATASET:")
    print(f"  Total games: {len(all_games)}")
    print(f"  Date range: {all_games['date'].min()} to {all_games['date'].max()}")
    print(f"  Seasons: {sorted(all_games['season'].unique())}")
    print(f"  Columns: {len(all_games.columns)}")

    return all_games


def _parse_bbref_csv(path, season):
    """Parse a Basketball Reference season CSV into our standard game format."""
    df = pd.read_csv(path)
    games = []
    for _, row in df.iterrows():
        try:
            date = pd.to_datetime(row.get('Date', ''), errors='coerce')
            if pd.isna(date):
                continue
            visitor = normalize_team(str(row.get('Visitor/Neutral', '')))
            home = normalize_team(str(row.get('Home/Neutral', '')))
            vis_pts = pd.to_numeric(row.get('PTS', np.nan), errors='coerce')
            home_pts = pd.to_numeric(row.get('PTS.1', np.nan), errors='coerce')
            if pd.isna(vis_pts) or pd.isna(home_pts) or vis_pts == 0 or home_pts == 0:
                continue
            if visitor is None or home is None:
                continue
            games.append({
                'date': date,
                'home_team_id': home,
                'visitor_team_id': visitor,
                'season': season,
                'home_pts': int(home_pts),
                'visitor_pts': int(vis_pts),
            })
        except Exception:
            continue
    return games


def _enrich_with_gamelogs(games_df, gamelogs):
    """
    Enrich game rows with advanced stats from deepshot gamelogs.
    Matches by date + team_id and adds box score + advanced stats columns.
    """
    # Index gamelogs by (date, team_id) for fast lookup
    gl_indexed = gamelogs.set_index(['date', 'team_id'])
    stat_map_home = _gamelog_stat_map('home')
    stat_map_vis = _gamelog_stat_map('vis')

    # Pre-allocate columns
    for col in list(stat_map_home.keys()) + list(stat_map_vis.keys()):
        if col not in games_df.columns:
            games_df[col] = np.nan

    matched = 0
    for idx, row in tqdm(games_df.iterrows(), total=len(games_df), desc="Enriching"):
        date = row['date']
        home_id = row['home_team_id']
        vis_id = row['visitor_team_id']

        # Try to find home team gamelog
        try:
            home_gl = gl_indexed.loc[(date, home_id)]
            if isinstance(home_gl, pd.DataFrame):
                home_gl = home_gl.iloc[0]
            for our_col, gl_col in stat_map_home.items():
                val = home_gl.get(gl_col)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    games_df.at[idx, our_col] = val
            matched += 1
        except KeyError:
            pass

        # Try to find visitor team gamelog
        try:
            vis_gl = gl_indexed.loc[(date, vis_id)]
            if isinstance(vis_gl, pd.DataFrame):
                vis_gl = vis_gl.iloc[0]
            for our_col, gl_col in stat_map_vis.items():
                val = vis_gl.get(gl_col)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    games_df.at[idx, our_col] = val
        except KeyError:
            pass

    print(f"  Enriched {matched}/{len(games_df)} games with gamelog stats ({100*matched/len(games_df):.1f}%)")
    return games_df


def _build_all_games_from_schedule_gamelogs(schedule, gamelogs):
    """
    Build game rows for ALL seasons by matching schedule (home/away) with gamelogs (real stats+pts).
    This is the primary data builder - uses real integer scores from gamelogs.
    """
    games_list = []

    # Index gamelogs by (date, team_id) for fast lookup
    gl_indexed = gamelogs.set_index(['date', 'team_id'])

    # Process schedule entries with progress bar
    matched = 0
    missed = 0
    for _, s_row in tqdm(schedule.iterrows(), total=len(schedule), desc="Building games"):
        date = s_row['date']
        home_id = s_row['home_id']
        away_id = s_row['away_id']
        season = s_row['season']

        # Look up home team gamelog
        try:
            home_gl = gl_indexed.loc[(date, home_id)]
        except KeyError:
            missed += 1
            continue
        # Look up away team gamelog
        try:
            away_gl = gl_indexed.loc[(date, away_id)]
        except KeyError:
            missed += 1
            continue

        # If multiple rows (rare), take first
        if isinstance(home_gl, pd.DataFrame):
            home_gl = home_gl.iloc[0]
        if isinstance(away_gl, pd.DataFrame):
            away_gl = away_gl.iloc[0]

        game = {
            'date': date,
            'home_team_id': home_id,
            'visitor_team_id': away_id,
            'season': season,
            'home_pts': home_gl.get('pts', np.nan),
            'visitor_pts': away_gl.get('pts', np.nan),
        }

        # Map all box score + advanced stats
        for stat, col in _gamelog_stat_map('home').items():
            val = home_gl.get(col)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                game[stat] = val
        for stat, col in _gamelog_stat_map('vis').items():
            val = away_gl.get(col)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                game[stat] = val

        games_list.append(game)
        matched += 1

    print(f"  Matched: {matched}, Missed: {missed}")
    return pd.DataFrame(games_list) if games_list else pd.DataFrame()




def _gamelog_stat_map(prefix):
    """Map our standard column names to gamelog column names."""
    return {
        f'{prefix}_fgm': 'fg', f'{prefix}_fga': 'fga', f'{prefix}_fg_pct': 'fg_pct',
        f'{prefix}_fg3m': 'fg3', f'{prefix}_fg3a': 'fg3a', f'{prefix}_three_pct': 'fg3_pct',
        f'{prefix}_fg2m': 'fg2', f'{prefix}_fg2a': 'fg2a',
        f'{prefix}_ftm': 'ft', f'{prefix}_fta': 'fta', f'{prefix}_ft_pct': 'ft_pct',
        f'{prefix}_oreb': 'orb', f'{prefix}_dreb': 'drb', f'{prefix}_reb': 'trb',
        f'{prefix}_ast': 'ast', f'{prefix}_stl': 'stl', f'{prefix}_blk': 'blk',
        f'{prefix}_tov': 'tov', f'{prefix}_pf': 'pf',
        f'{prefix}_ortg': 'ortg', f'{prefix}_drtg': 'drtg', f'{prefix}_pace': 'pace',
        f'{prefix}_efg_pct': 'efg_pct', f'{prefix}_ts_pct': 'ts',
        f'{prefix}_tov_pct': 'tov_pct', f'{prefix}_orb_pct': 'orb_pct',
        f'{prefix}_ftr': 'ftr', f'{prefix}_3ptar': '3ptar',
        f'{prefix}_trb_pct': 'trb_pct', f'{prefix}_ast_pct': 'ast_pct',
        f'{prefix}_stl_pct': 'stl_pct', f'{prefix}_blk_pct': 'blk_pct',
    }


# ============================================================
# Step 2: Build expanded RAPTOR ratings (1977-2022)
# ============================================================
def build_expanded_raptor():
    """
    Combine FiveThirtyEight historical RAPTOR (1977-2020) with
    existing raptor_by_team.csv (2014-2022) for maximum coverage.
    """
    print("\n" + "=" * 70)
    print("STEP 2: BUILD EXPANDED RAPTOR RATINGS")
    print("=" * 70)

    team_season_raptor = {}

    # Source 1: FiveThirtyEight historical (1977-2020)
    hist_path = os.path.join(DATA_DIR, "fivethirtyeight", "nba-data-historical.csv")
    if os.path.exists(hist_path):
        hist = pd.read_csv(hist_path)
        hist = hist[hist['type'] == 'RS'].copy()

        # Convert to numeric
        for col in ['Raptor O', 'Raptor D', 'Raptor+/-', 'Raptor WAR', 'Min']:
            hist[col] = pd.to_numeric(hist[col], errors='coerce')

        hist['team_id'] = hist['team_id'].apply(lambda x: TRICODE_MAP.get(x, x))
        hist = hist.dropna(subset=['Raptor O', 'Min'])
        hist = hist[hist['Min'] > 0]

        # Aggregate to team-season (minutes-weighted)
        for (team, year), group in hist.groupby(['team_id', 'year_id']):
            weights = group['Min'].values
            total_w = weights.sum()
            if total_w > 0:
                team_season_raptor[(team, int(year))] = {
                    'raptor_offense': np.average(group['Raptor O'].values, weights=weights),
                    'raptor_defense': np.average(group['Raptor D'].values, weights=weights),
                    'raptor_total': np.average(group['Raptor+/-'].values, weights=weights),
                    'war_total': group['Raptor WAR'].sum(),
                }

        print(f"  FiveThirtyEight RAPTOR: {len(team_season_raptor)} team-seasons (years {int(hist['year_id'].min())}-{int(hist['year_id'].max())})")

    # Source 2: Existing raptor_by_team.csv (2014-2022) - override where available
    raptor_path = os.path.join(DATA_DIR, "raptor_by_team.csv")
    if os.path.exists(raptor_path):
        rap = pd.read_csv(raptor_path)
        rap = rap[rap['season_type'] == 'RS'].copy()
        rap['team_id'] = rap['team'].apply(lambda x: TRICODE_MAP.get(x, x))

        for col in ['raptor_offense', 'raptor_defense', 'raptor_total', 'war_total', 'mp']:
            rap[col] = pd.to_numeric(rap[col], errors='coerce')
        rap = rap.dropna(subset=['raptor_offense', 'mp'])
        rap = rap[rap['mp'] > 0]

        count_updated = 0
        for (team, year), group in rap.groupby(['team_id', 'season']):
            weights = group['mp'].values
            total_w = weights.sum()
            if total_w > 0:
                team_season_raptor[(team, int(year))] = {
                    'raptor_offense': np.average(group['raptor_offense'].values, weights=weights),
                    'raptor_defense': np.average(group['raptor_defense'].values, weights=weights),
                    'raptor_total': np.average(group['raptor_total'].values, weights=weights),
                    'war_total': group['war_total'].sum(),
                }
                count_updated += 1
        print(f"  raptor_by_team.csv: updated/added {count_updated} team-seasons")

    # Convert to DataFrame
    rows = []
    for (team, year), vals in team_season_raptor.items():
        rows.append({'team_id': team, 'season': year, **vals})
    raptor_df = pd.DataFrame(rows).sort_values(['season', 'team_id']).reset_index(drop=True)

    # Filter to 2004+ (our range)
    raptor_df = raptor_df[raptor_df['season'] >= 2004]
    print(f"  Total RAPTOR team-seasons (2004+): {len(raptor_df)}")
    print(f"  Season range: {raptor_df['season'].min()}-{raptor_df['season'].max()}")

    out_path = os.path.join(PROCESSED_DIR, "raptor_expanded.csv")
    raptor_df.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")

    return raptor_df


# ============================================================
# Step 3: Load and clean odds data
# ============================================================
def load_odds():
    """Load existing cleaned odds data."""
    print("\n" + "=" * 70)
    print("STEP 3: LOAD BETTING ODDS")
    print("=" * 70)

    odds_path = os.path.join(PROCESSED_DIR, "odds_clean.csv")
    if os.path.exists(odds_path):
        odds = pd.read_csv(odds_path, parse_dates=['date'])
        print(f"  Odds: {len(odds)} rows ({odds['date'].min().date()} to {odds['date'].max().date()})")
        return odds

    # Build from raw if not available
    from integrate_real_data import build_odds_dataset
    return build_odds_dataset()


# ============================================================
# Step 4: Merge everything into enriched games
# ============================================================
def merge_all(games, raptor, odds):
    """Merge RAPTOR and odds into the games dataset."""
    print("\n" + "=" * 70)
    print("STEP 4: MERGE ALL DATA")
    print("=" * 70)
    print(f"  Input games: {len(games)}")

    # Merge RAPTOR (by team-season)
    if raptor is not None and len(raptor) > 0:
        games = games.merge(
            raptor.rename(columns={
                'raptor_offense': 'home_raptor_off',
                'raptor_defense': 'home_raptor_def',
                'raptor_total': 'home_raptor_total',
                'war_total': 'home_war',
            }),
            left_on=['home_team_id', 'season'],
            right_on=['team_id', 'season'],
            how='left'
        ).drop(columns=['team_id'], errors='ignore')

        games = games.merge(
            raptor.rename(columns={
                'raptor_offense': 'vis_raptor_off',
                'raptor_defense': 'vis_raptor_def',
                'raptor_total': 'vis_raptor_total',
                'war_total': 'vis_war',
            }),
            left_on=['visitor_team_id', 'season'],
            right_on=['team_id', 'season'],
            how='left'
        ).drop(columns=['team_id'], errors='ignore')

        raptor_matched = games['home_raptor_total'].notna().sum()
        print(f"  RAPTOR matched: {raptor_matched}/{len(games)} ({100*raptor_matched/len(games):.1f}%)")

    # Merge odds (by date + home team)
    if odds is not None and len(odds) > 0:
        odds['date_key'] = odds['date'].dt.date
        games['date_key'] = games['date'].dt.date

        games = games.merge(
            odds[['date_key', 'home_id', 'opening_total', 'spread',
                  'ml_home', 'ml_away', 'implied_prob_home', 'implied_prob_away']],
            left_on=['date_key', 'home_team_id'],
            right_on=['date_key', 'home_id'],
            how='left'
        ).drop(columns=['home_id', 'date_key'], errors='ignore')

        odds_matched = games['spread'].notna().sum()
        print(f"  Odds matched: {odds_matched}/{len(games)} ({100*odds_matched/len(games):.1f}%)")

    print(f"  Final enriched dataset: {len(games)} games x {len(games.columns)} columns")

    return games


# ============================================================
# Main pipeline
# ============================================================
def run_full_integration():
    """Run the complete data integration pipeline."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Step 1: Build complete game dataset with box scores + advanced stats
    games = build_complete_games()

    # Step 2: Build expanded RAPTOR
    raptor = build_expanded_raptor()

    # Step 3: Load odds
    odds = load_odds()

    # Step 4: Merge everything
    games = merge_all(games, raptor, odds)

    # Save
    out_path = os.path.join(PROCESSED_DIR, "games_enriched_v2.csv")
    games.to_csv(out_path, index=False)
    print(f"\n  Saved enriched dataset to {out_path}")

    # Summary
    print("\n" + "=" * 70)
    print("DATA INTEGRATION COMPLETE")
    print("=" * 70)
    print(f"  Games: {len(games)}")
    print(f"  Seasons: {sorted(games['season'].unique())}")
    print(f"  Columns: {len(games.columns)}")

    # Check coverage of key columns
    key_cols = ['home_efg_pct', 'home_ortg', 'home_drtg', 'home_pace', 'home_tov_pct',
                'home_orb_pct', 'home_ts_pct', 'spread', 'home_raptor_total']
    print("\n  Column coverage:")
    for col in key_cols:
        if col in games.columns:
            n = games[col].notna().sum()
            print(f"    {col}: {n}/{len(games)} ({100*n/len(games):.1f}%)")

    return games


if __name__ == "__main__":
    run_full_integration()

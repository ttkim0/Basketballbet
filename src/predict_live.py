"""
NBA Live Game Prediction
========================
Fetches live NBA data and predicts game winners using our trained ensemble.

Usage:
  python3 src/predict_live.py --home NOP --away DET
  python3 src/predict_live.py --home NOP --away DET --fetch-live
"""

import argparse
import json
import os
import pickle
import sys
import urllib.request
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import lightgbm as lgbm

BASE_DIR = "/home/user/Basketballbet"
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# The 7 raw features used by the extended meta-learner
EXT_META_RAW_FEATURES = [
    "elo_rating_diff", "season_win_pct_diff", "pyth_win_exp_diff",
    "weighted_net_rating_momentum", "fatigue_diff",
    "last3_net_rating_diff", "elo_diff_squared",
]


def load_models():
    """Load all trained models and config."""
    lr_model = joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "feature_list.pkl"))
    ext_meta_model = joblib.load(os.path.join(MODEL_DIR, "ensemble_ext_meta.pkl"))
    ensemble_config = joblib.load(os.path.join(MODEL_DIR, "ensemble_config.pkl"))

    xgb_model = xgb.Booster()
    xgb_model.load_model(os.path.join(MODEL_DIR, "xgboost_model.json"))

    lgb_model = lgbm.Booster(model_file=os.path.join(MODEL_DIR, "lightgbm_model.txt"))

    return {
        "lr": lr_model,
        "xgb": xgb_model,
        "lgb": lgb_model,
        "scaler": scaler,
        "features": features,
        "ext_meta": ext_meta_model,
        "config": ensemble_config,
    }


def load_elo_ratings():
    """Load final Elo ratings."""
    df = pd.read_csv(os.path.join(DATA_DIR, "processed/final_elo_ratings.csv"))
    return dict(zip(df["team"], df["elo"]))


def get_latest_team_features(team_id, role, df_feat, features):
    """
    Extract the most recent feature values for a team from the full features dataset.
    role: 'home' or 'visitor' — determines which columns to pull.
    Returns a dict of feature_name -> value for features that start with the role prefix.
    """
    if role == "home":
        mask = df_feat["home_team_id"] == team_id
    else:
        mask = df_feat["visitor_team_id"] == team_id

    team_rows = df_feat[mask].sort_values("date")
    if len(team_rows) == 0:
        return {}

    last_row = team_rows.iloc[-1]
    return {f: last_row.get(f, np.nan) for f in features}


def build_feature_vector(home_team, away_team, df_feat, features, elo_ratings):
    """
    Build the full 107-feature vector for a home vs away matchup.
    Uses the most recent game data for each team from our dataset.
    """
    # Get the latest game row where each team played in each role
    # For rolling stats, we use the last game the team appeared in (regardless of role)
    # because rolling stats are computed per-team across all games

    home_last = df_feat[
        (df_feat["home_team_id"] == home_team) | (df_feat["visitor_team_id"] == home_team)
    ].sort_values("date").iloc[-1] if len(df_feat[
        (df_feat["home_team_id"] == home_team) | (df_feat["visitor_team_id"] == home_team)
    ]) > 0 else None

    away_last = df_feat[
        (df_feat["home_team_id"] == away_team) | (df_feat["visitor_team_id"] == away_team)
    ].sort_values("date").iloc[-1] if len(df_feat[
        (df_feat["home_team_id"] == away_team) | (df_feat["visitor_team_id"] == away_team)
    ]) > 0 else None

    # Also get last game where each team was specifically home or away
    home_as_home = df_feat[df_feat["home_team_id"] == home_team].sort_values("date")
    home_as_away = df_feat[df_feat["visitor_team_id"] == home_team].sort_values("date")
    away_as_home = df_feat[df_feat["home_team_id"] == away_team].sort_values("date")
    away_as_away = df_feat[df_feat["visitor_team_id"] == away_team].sort_values("date")

    feat_values = {}

    for feat in features:
        val = 0.0

        # --- Elo features ---
        if feat == "elo_rating_diff":
            val = elo_ratings.get(home_team, 1500) - elo_ratings.get(away_team, 1500)
        elif feat == "elo_diff_squared":
            diff = elo_ratings.get(home_team, 1500) - elo_ratings.get(away_team, 1500)
            val = diff * abs(diff)
        elif feat == "elo_diff_abs":
            val = abs(elo_ratings.get(home_team, 1500) - elo_ratings.get(away_team, 1500))

        # --- Diff features: home_X - visitor_X pattern ---
        elif feat.endswith("_diff") or feat.endswith("_mismatch") or feat.endswith("_edge_5") or feat.endswith("_edge_10"):
            # Try to get from last home-as-home row and away-as-away row
            home_val = np.nan
            away_val = np.nan

            if len(home_as_home) > 0:
                home_val = home_as_home.iloc[-1].get(feat, np.nan)
            if home_val is np.nan or (isinstance(home_val, float) and np.isnan(home_val)):
                # Try from the last game regardless of role
                if home_last is not None:
                    if home_last.get("home_team_id") == home_team:
                        home_val = home_last.get(feat, 0)
                    else:
                        # Team was away, so the diff is flipped
                        home_val = -home_last.get(feat, 0) if feat in home_last.index else 0

            if len(away_as_away) > 0:
                # When team was away, the diff features are from the away perspective already
                away_val = away_as_away.iloc[-1].get(feat, np.nan)
            if away_val is np.nan or (isinstance(away_val, float) and np.isnan(away_val)):
                if away_last is not None:
                    if away_last.get("visitor_team_id") == away_team:
                        away_val = away_last.get(feat, 0)
                    else:
                        away_val = -away_last.get(feat, 0) if feat in away_last.index else 0

            # For diff features, the value IS the diff already (home - visitor perspective)
            # We just use the last value from when this team was home
            if len(home_as_home) > 0:
                val = home_as_home.iloc[-1].get(feat, 0)
            elif home_last is not None:
                val = home_last.get(feat, 0)

        # --- Per-team features (home_X or visitor_X) ---
        elif feat.startswith("home_"):
            # Get from the home team's last game as home
            if len(home_as_home) > 0:
                val = home_as_home.iloc[-1].get(feat, 0)
            elif home_last is not None:
                # Map: if team was visitor last, use the visitor_ version
                vis_feat = feat.replace("home_", "visitor_", 1)
                if home_last.get("visitor_team_id") == home_team and vis_feat in home_last.index:
                    val = home_last.get(vis_feat, 0)
                else:
                    val = home_last.get(feat, 0)

        elif feat.startswith("visitor_"):
            if len(away_as_away) > 0:
                val = away_as_away.iloc[-1].get(feat, 0)
            elif away_last is not None:
                home_feat = feat.replace("visitor_", "home_", 1)
                if away_last.get("home_team_id") == away_team and home_feat in away_last.index:
                    val = away_last.get(home_feat, 0)
                else:
                    val = away_last.get(feat, 0)

        # --- Standalone features (spread, expected_total, etc.) ---
        else:
            # Try from the home team's last home game
            if len(home_as_home) > 0:
                val = home_as_home.iloc[-1].get(feat, 0)
            elif home_last is not None:
                val = home_last.get(feat, 0)

        # Clean NaN/inf
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            val = 0.0

        feat_values[feat] = val

    return feat_values


def fetch_nba_scoreboard():
    """Fetch today's NBA scoreboard."""
    url = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Referer": "https://www.nba.com/",
    })
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


def fetch_nba_boxscore(game_id):
    """Fetch live box score for a game."""
    url = f"https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Referer": "https://www.nba.com/",
    })
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


def predict_game(home_team, away_team, models, df_feat, elo_ratings, live_data=None):
    """Run prediction for a single game."""
    features = models["features"]

    # Build feature vector
    feat_values = build_feature_vector(home_team, away_team, df_feat, features, elo_ratings)
    X = pd.DataFrame([feat_values])[features].fillna(0).replace([np.inf, -np.inf], 0)

    # Base model predictions
    # Logistic Regression
    X_scaled = models["scaler"].transform(X)
    lr_prob = models["lr"].predict_proba(X_scaled)[0, 1]

    # XGBoost
    dmat = xgb.DMatrix(X, feature_names=features)
    xgb_prob = float(models["xgb"].predict(dmat)[0])

    # LightGBM
    lgb_prob = float(models["lgb"].predict(X)[0])

    # Extended meta-learner ensemble
    base_probs = np.array([[lr_prob, xgb_prob, lgb_prob]])
    raw_feats = X[EXT_META_RAW_FEATURES].fillna(0).values
    meta_input = np.column_stack([base_probs, raw_feats])
    ensemble_prob = models["ext_meta"].predict_proba(meta_input)[0, 1]

    # Apply threshold from training
    threshold = models["config"].get("threshold", 0.52)

    return {
        "home_team": home_team,
        "away_team": away_team,
        "lr_prob": lr_prob,
        "xgb_prob": xgb_prob,
        "lgb_prob": lgb_prob,
        "ensemble_prob": ensemble_prob,
        "threshold": threshold,
        "predicted_winner": home_team if ensemble_prob >= threshold else away_team,
        "confidence": max(ensemble_prob, 1 - ensemble_prob),
        "feature_values": feat_values,
        "elo_home": elo_ratings.get(home_team, 1500),
        "elo_away": elo_ratings.get(away_team, 1500),
    }


def print_prediction(result):
    """Print a formatted prediction."""
    home = result["home_team"]
    away = result["away_team"]
    ep = result["ensemble_prob"]

    print("\n" + "=" * 60)
    print(f"  PREDICTION: {away} @ {home}")
    print("=" * 60)

    print(f"\n  Elo Ratings:")
    print(f"    {home} (Home): {result['elo_home']:.1f}")
    print(f"    {away} (Away): {result['elo_away']:.1f}")
    print(f"    Elo Diff:     {result['elo_home'] - result['elo_away']:+.1f}")

    print(f"\n  Model Probabilities (P(home win)):")
    print(f"    Logistic Regression: {result['lr_prob']:.1%}")
    print(f"    XGBoost:             {result['xgb_prob']:.1%}")
    print(f"    LightGBM:            {result['lgb_prob']:.1%}")
    print(f"    Ensemble:            {result['ensemble_prob']:.1%}")

    print(f"\n  Threshold: {result['threshold']:.2f}")

    winner = result["predicted_winner"]
    loser = away if winner == home else home
    conf = result["confidence"]

    print(f"\n  >>> PREDICTED WINNER: {winner} ({conf:.1%} confidence) <<<")
    print(f"      {winner} over {loser}")

    # Show key feature values
    print(f"\n  Key Features:")
    key_feats = [
        ("elo_rating_diff", "Elo Diff"),
        ("season_win_pct_diff", "Season Win% Diff"),
        ("pyth_win_exp_diff", "Pythagorean Diff"),
        ("last3_net_rating_diff", "Last 3 Net Rating Diff"),
        ("last5_net_rating_diff", "Last 5 Net Rating Diff"),
        ("last10_net_rating_diff", "Last 10 Net Rating Diff"),
        ("weighted_net_rating_momentum", "Momentum"),
        ("rest_diff", "Rest Days Diff"),
        ("fatigue_diff", "Fatigue Diff"),
        ("srs_diff", "SRS Diff"),
        ("team_ortg_diff", "Team ORtg Diff"),
        ("team_drtg_diff", "Team DRtg Diff"),
        ("team_bpm_diff", "Team BPM Diff"),
        ("spread", "Spread"),
    ]
    for feat, label in key_feats:
        val = result["feature_values"].get(feat, 0)
        if val != 0:
            print(f"    {label:>30s}: {val:>+.3f}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="NBA Live Game Prediction")
    parser.add_argument("--home", required=True, help="Home team (3-letter code, e.g. NOP)")
    parser.add_argument("--away", required=True, help="Away team (3-letter code, e.g. DET)")
    parser.add_argument("--fetch-live", action="store_true", help="Try to fetch live NBA data")
    args = parser.parse_args()

    home = args.home.upper()
    away = args.away.upper()

    print(f"\nLoading models...")
    models = load_models()
    print(f"  Loaded {len(models['features'])} features")

    print(f"Loading Elo ratings...")
    elo_ratings = load_elo_ratings()

    print(f"Loading feature dataset...")
    df_feat = pd.read_csv(os.path.join(DATA_DIR, "processed/games_full_features.csv"))
    df_feat["date"] = pd.to_datetime(df_feat["date"])
    print(f"  {len(df_feat)} games loaded")

    live_data = None
    if args.fetch_live:
        try:
            print(f"\nFetching live scoreboard...")
            scoreboard = fetch_nba_scoreboard()
            games = scoreboard["scoreboard"]["games"]
            print(f"  Found {len(games)} games today")

            for g in games:
                h = g["homeTeam"]["teamTricode"]
                a = g["awayTeam"]["teamTricode"]
                print(f"    {a} @ {h} | {g['gameStatusText']} | {a} {g['awayTeam']['score']} - {h} {g['homeTeam']['score']}")

                if (h == home and a == away) or (h == away and a == home):
                    game_id = g["gameId"]
                    print(f"\n  Found matchup! GameID: {game_id}")
                    print(f"  Fetching box score...")
                    live_data = fetch_nba_boxscore(game_id)
                    print(f"  Got live box score data")

                    # If home/away is flipped from what user specified, swap
                    if h != home:
                        home, away = h, a
                        print(f"  Note: Corrected home/away to {away} @ {home}")

        except Exception as e:
            print(f"  Could not fetch live data: {e}")
            print(f"  Falling back to historical data...")

    result = predict_game(home, away, models, df_feat, elo_ratings, live_data)
    print_prediction(result)


if __name__ == "__main__":
    main()

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
    meta_model = joblib.load(os.path.join(MODEL_DIR, "ensemble_meta.pkl"))
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
        "meta": meta_model,
        "ext_meta": ext_meta_model,
        "config": ensemble_config,
    }


def load_elo_ratings():
    """Load final Elo ratings."""
    df = pd.read_csv(os.path.join(DATA_DIR, "processed/final_elo_ratings.csv"))
    return dict(zip(df["team"], df["elo"]))


def build_diff_to_sources(df_feat):
    """Build mapping of diff features to their (home_col, visitor_col) sources."""
    diff_map = {}
    cols = df_feat.columns.tolist()
    for w in ['last3', 'last5', 'last10']:
        for s in ['net_rating', 'off_rating', 'def_rating', 'win_rate', 'avg_margin']:
            d, h, v = f'{w}_{s}_diff', f'{w}_{s}_home', f'{w}_{s}_visitor'
            if all(c in cols for c in [d, h, v]):
                diff_map[d] = (h, v)
        for s in ['efg_pct', 'ts_pct', 'tov', 'reb', 'ast', 'stl', 'blk', 'oreb',
                   'three_pct', 'ortg', 'drtg', 'pace', 'tov_pct', 'orb_pct']:
            d = f'{w}_{s}_diff_real'
            h, v = f'{w}_{s}_home_real', f'{w}_{s}_visitor_real'
            if all(c in cols for c in [d, h, v]):
                diff_map[d] = (h, v)
        for s in ['tov_rate']:
            d, h, v = f'{w}_{s}_diff', f'{w}_{s}_home', f'{w}_{s}_visitor'
            if all(c in cols for c in [d, h, v]):
                diff_map[d] = (h, v)
    for s in ['srs', 'team_ortg', 'team_drtg', 'team_nrtg', 'team_pace',
              'team_bpm', 'team_ws48', 'team_vorp', 'team_per']:
        d, h, v = f'{s}_diff', f'home_{s}', f'vis_{s}'
        if all(c in cols for c in [d, h, v]):
            diff_map[d] = (h, v)
    for s in ['raptor_total', 'raptor_off', 'raptor_def', 'war']:
        d, h, v = f'{s}_diff', f'home_{s}', f'vis_{s}'
        if all(c in cols for c in [d, h, v]):
            diff_map[d] = (h, v)
    if 'season_win_pct_diff' in cols:
        diff_map['season_win_pct_diff'] = ('home_season_win_pct', 'visitor_season_win_pct')
    return diff_map


def _get_team_stat(team_id, col_name, df_feat):
    """Get a team's stat from their most recent game, handling role swaps."""
    team_games = df_feat[
        (df_feat["home_team_id"] == team_id) | (df_feat["visitor_team_id"] == team_id)
    ].sort_values("date")
    if len(team_games) == 0:
        return 0.0

    last = team_games.iloc[-1]
    was_home = last["home_team_id"] == team_id

    actual_col = col_name
    if was_home:
        if col_name.endswith("_visitor"):
            actual_col = col_name.replace("_visitor", "_home")
        elif col_name.startswith("visitor_"):
            actual_col = col_name.replace("visitor_", "home_", 1)
        elif col_name.startswith("vis_"):
            actual_col = col_name.replace("vis_", "home_", 1)
    else:
        if col_name.endswith("_home"):
            actual_col = col_name.replace("_home", "_visitor")
        elif col_name.startswith("home_"):
            candidate = col_name.replace("home_", "visitor_", 1)
            if candidate in last.index:
                actual_col = candidate
            else:
                actual_col = col_name.replace("home_", "vis_", 1)

    if actual_col in last.index:
        val = last[actual_col]
        if not pd.isna(val):
            return float(val)
    if actual_col != col_name and col_name in last.index:
        val = last[col_name]
        if not pd.isna(val):
            return float(val)
    return 0.0


def build_feature_vector(home_team, away_team, df_feat, features, elo_ratings):
    """
    Build the 107-feature vector using template-based approach.
    1. Start from home team's last home game as template
    2. Override Elo features with current ratings
    3. Override diff features with fresh per-team computation
    4. Override visitor features with away team's actual stats
    """
    diff_map = build_diff_to_sources(df_feat)

    # Step 1: Template from home team's last home game
    home_as_home = df_feat[df_feat["home_team_id"] == home_team].sort_values("date")
    if len(home_as_home) == 0:
        home_as_home = df_feat.sort_values("date").tail(1)
    template = home_as_home.iloc[-1]

    feat_values = {}
    for feat in features:
        val = template.get(feat, 0)
        if pd.isna(val) or (isinstance(val, float) and np.isinf(val)):
            val = 0.0
        feat_values[feat] = float(val)

    # Step 2: Override Elo features
    elo_diff = elo_ratings.get(home_team, 1500) - elo_ratings.get(away_team, 1500)
    feat_values["elo_rating_diff"] = elo_diff
    feat_values["elo_diff_squared"] = elo_diff * abs(elo_diff)
    if "elo_diff_abs" in feat_values:
        feat_values["elo_diff_abs"] = abs(elo_diff)

    # Step 3: Override diff features with fresh computation
    for feat, (home_col, vis_col) in diff_map.items():
        if feat in feat_values:
            hv = _get_team_stat(home_team, home_col, df_feat)
            av = _get_team_stat(away_team, vis_col, df_feat)
            feat_values[feat] = hv - av

    # Step 4: Override visitor features with away team's actual stats
    for feat in features:
        if feat.startswith("visitor_") or feat.startswith("vis_"):
            feat_values[feat] = _get_team_stat(away_team, feat, df_feat)

    # Step 5: Compute recency-adjusted derived features
    _compute_recency_features(feat_values, home_team, away_team, df_feat)

    # Clean NaN/inf
    for feat in features:
        val = feat_values.get(feat, 0.0)
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            feat_values[feat] = 0.0

    return feat_values


def _compute_recency_features(fv, home_team, away_team, df_feat):
    """Compute recency-adjusted features to prevent stale season stats from dominating."""
    h_l10_nr = _get_team_stat(home_team, "last10_net_rating_home", df_feat)
    v_l10_nr = _get_team_stat(away_team, "last10_net_rating_visitor", df_feat)
    h_nrtg = _get_team_stat(home_team, "home_team_nrtg", df_feat)
    v_nrtg = _get_team_stat(away_team, "vis_team_nrtg", df_feat)
    h_srs = _get_team_stat(home_team, "home_srs", df_feat)
    v_srs = _get_team_stat(away_team, "vis_srs", df_feat)
    h_l10_wr = _get_team_stat(home_team, "last10_win_rate_home", df_feat)
    v_l10_wr = _get_team_stat(away_team, "last10_win_rate_visitor", df_feat)
    h_l10_mg = _get_team_stat(home_team, "last10_avg_margin_home", df_feat)
    v_l10_mg = _get_team_stat(away_team, "last10_avg_margin_visitor", df_feat)
    h_streak = fv.get("home_streak", 0)
    v_streak = fv.get("visitor_streak", 0)

    h_div = h_l10_nr - h_nrtg
    v_div = v_l10_nr - v_nrtg
    fv["form_divergence_home"] = h_div
    fv["form_divergence_visitor"] = v_div
    fv["form_divergence_diff"] = h_div - v_div
    fv["srs_blended_diff"] = (0.5 * h_srs + 0.5 * h_l10_nr) - (0.5 * v_srs + 0.5 * v_l10_nr)

    h_ortg_s = _get_team_stat(home_team, "home_team_ortg", df_feat)
    v_ortg_s = _get_team_stat(away_team, "vis_team_ortg", df_feat)
    h_drtg_s = _get_team_stat(home_team, "home_team_drtg", df_feat)
    v_drtg_s = _get_team_stat(away_team, "vis_team_drtg", df_feat)
    h_l10_or = _get_team_stat(home_team, "last10_off_rating_home", df_feat)
    v_l10_or = _get_team_stat(away_team, "last10_off_rating_visitor", df_feat)
    h_l10_dr = _get_team_stat(home_team, "last10_def_rating_home", df_feat)
    v_l10_dr = _get_team_stat(away_team, "last10_def_rating_visitor", df_feat)

    fv["ortg_blended_diff"] = (0.5 * h_ortg_s + 0.5 * h_l10_or) - (0.5 * v_ortg_s + 0.5 * v_l10_or)
    fv["drtg_blended_diff"] = (0.5 * h_drtg_s + 0.5 * h_l10_dr) - (0.5 * v_drtg_s + 0.5 * v_l10_dr)

    fv["streak_severity_home"] = h_streak * abs(h_streak)
    fv["streak_severity_visitor"] = v_streak * abs(v_streak)
    fv["streak_severity_diff"] = fv["streak_severity_home"] - fv["streak_severity_visitor"]

    fv["home_collapsing"] = 1.0 if h_div < -8 else 0.0
    fv["home_surging"] = 1.0 if h_div > 8 else 0.0
    fv["visitor_collapsing"] = 1.0 if v_div < -8 else 0.0
    fv["visitor_surging"] = 1.0 if v_div > 8 else 0.0

    fv["recent_dominance_home"] = h_l10_wr * h_l10_mg
    fv["recent_dominance_visitor"] = v_l10_wr * v_l10_mg
    fv["recent_dominance_diff"] = fv["recent_dominance_home"] - fv["recent_dominance_visitor"]


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

    # Ensemble prediction (use method from training config)
    base_probs = np.array([[lr_prob, xgb_prob, lgb_prob]])
    ens_method = models["config"].get("method", "meta_learner")
    if ens_method == "extended_meta":
        raw_feats = X[EXT_META_RAW_FEATURES].fillna(0).values
        meta_input = np.column_stack([base_probs, raw_feats])
        ensemble_prob = models["ext_meta"].predict_proba(meta_input)[0, 1]
    elif ens_method == "meta_learner":
        ensemble_prob = models["meta"].predict_proba(base_probs)[0, 1]
    elif ens_method == "weighted_avg":
        weights = models["config"].get("weights", {})
        w = np.array([weights.get("logistic", 1/3), weights.get("xgboost", 1/3),
                       weights.get("lightgbm", 1/3)])
        ensemble_prob = float(np.dot(w, [lr_prob, xgb_prob, lgb_prob]))
    else:
        ensemble_prob = float(np.mean([lr_prob, xgb_prob, lgb_prob]))

    # Apply threshold from training
    threshold = models["config"].get("threshold", 0.5)

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

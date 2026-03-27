"""
NBA Live Prediction Web App
============================
Flask app that serves a UI for real-time NBA game predictions.

The browser (running on YOUR machine) fetches live NBA data from cdn.nba.com,
then sends it to this Flask backend which runs the trained ensemble model.

Usage: python3 src/web_app.py
Then open http://localhost:5050 in your browser.
"""

import os
import sys
import json
import traceback
import urllib.request
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import lightgbm as lgbm
from flask import Flask, render_template_string, jsonify, request, Response

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

sys.path.insert(0, os.path.join(BASE_DIR, "src"))

EXT_META_RAW_FEATURES = [
    "elo_rating_diff", "season_win_pct_diff", "pyth_win_exp_diff",
    "weighted_net_rating_momentum", "fatigue_diff",
    "last3_net_rating_diff", "elo_diff_squared",
]

# ============================================================
# RL Probability Adjustments (from reinforcement_learning.py)
# ============================================================
# These correct systematic biases discovered by the RL error analysis.
# Applied after the ensemble prediction to nudge probabilities.

from reinforcement_learning import categorize_game, get_team_tier

RL_ADJUSTMENTS_PATH = os.path.join(BASE_DIR, "logs/rl_results/error_patterns.json")

def load_rl_adjustments():
    """Load RL-discovered probability adjustments."""
    if not os.path.exists(RL_ADJUSTMENTS_PATH):
        return {}
    with open(RL_ADJUSTMENTS_PATH) as f:
        patterns = json.load(f)
    # Build adjustment map: pattern_name -> correction value
    # Only use univariate patterns with 50+ games and significant excess error
    adjustments = {}
    for p in patterns:
        if int(p.get("n_games", 0)) >= 50 and abs(p.get("excess_error", 0)) >= 0.03:
            adjustments[p["name"]] = p["correction"]
    return adjustments


def apply_rl_adjustment(ensemble_prob, feat_values):
    """Apply RL corrections to ensemble probability based on game context."""
    if not RL_ADJUSTMENTS:
        return ensemble_prob

    # Build a fake row for categorize_game
    row = {
        "elo_rating_diff": feat_values.get("elo_rating_diff", 0),
        "home_win": 0,  # doesn't matter for categorization
        "home_season_win_pct": feat_values.get("home_season_win_pct", 0.5),
        "visitor_season_win_pct": feat_values.get("visitor_season_win_pct", 0.5),
        "home_b2b": feat_values.get("home_b2b", 0),
        "visitor_b2b": feat_values.get("visitor_b2b", 0),
        "home_3in4": feat_values.get("home_3in4", 0),
        "visitor_3in4": feat_values.get("visitor_3in4", 0),
        "home_streak": feat_values.get("home_streak", 0),
        "visitor_streak": feat_values.get("visitor_streak", 0),
        "rest_diff": feat_values.get("rest_diff", 0),
        "spread": feat_values.get("spread", 0),
        "travel_miles_home": feat_values.get("travel_miles_home", 0),
        "travel_miles_visitor": feat_values.get("travel_miles_visitor", 0),
        "altitude_edge_home": feat_values.get("altitude_edge_home", 0),
        "date": pd.Timestamp.now(),
    }

    tags = categorize_game(row, ensemble_prob)

    total_adj = 0.0
    matched_rules = []

    for pattern_name, correction in RL_ADJUSTMENTS.items():
        # Check univariate patterns: "category=value"
        if "+" not in pattern_name and "=" in pattern_name:
            cat, val = pattern_name.split("=", 1)
            if str(tags.get(cat)) == val:
                total_adj += correction
                matched_rules.append((pattern_name, correction))
        # Check interaction patterns: "cat1=val1+cat2=val2"
        elif "+" in pattern_name:
            parts = pattern_name.split("+")
            if len(parts) == 2:
                match = True
                for part in parts:
                    if "=" in part:
                        cat, val = part.split("=", 1)
                        if str(tags.get(cat)) != val:
                            match = False
                            break
                if match:
                    total_adj += correction
                    matched_rules.append((pattern_name, correction))

    # Cap total adjustment to avoid wild swings
    total_adj = np.clip(total_adj, -0.15, 0.15)
    adjusted = np.clip(ensemble_prob + total_adj, 0.01, 0.99)

    return adjusted, total_adj, matched_rules


# ============================================================
# Load models and data at startup
# ============================================================
print("Loading models...")
MODELS = {
    "lr": joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl")),
    "scaler": joblib.load(os.path.join(MODEL_DIR, "scaler.pkl")),
    "features": joblib.load(os.path.join(MODEL_DIR, "feature_list.pkl")),
    "ext_meta": joblib.load(os.path.join(MODEL_DIR, "ensemble_ext_meta.pkl")),
    "config": joblib.load(os.path.join(MODEL_DIR, "ensemble_config.pkl")),
}

xgb_model = xgb.Booster()
xgb_model.load_model(os.path.join(MODEL_DIR, "xgboost_model.json"))
MODELS["xgb"] = xgb_model
MODELS["lgb"] = lgbm.Booster(model_file=os.path.join(MODEL_DIR, "lightgbm_model.txt"))

print("Loading RL adjustments...")
RL_ADJUSTMENTS = load_rl_adjustments()
print(f"  {len(RL_ADJUSTMENTS)} RL correction rules loaded")

print("Loading Elo ratings...")
elo_df = pd.read_csv(os.path.join(DATA_DIR, "processed/final_elo_ratings.csv"))
ELO_RATINGS = dict(zip(elo_df["team"], elo_df["elo"]))

print("Loading feature dataset...")
DF_FEAT = pd.read_csv(os.path.join(DATA_DIR, "processed/games_full_features.csv"))
DF_FEAT["date"] = pd.to_datetime(DF_FEAT["date"])
print(f"  {len(DF_FEAT)} games loaded, {len(MODELS['features'])} features")

# NBA team info
NBA_TEAMS = {
    "ATL": "Hawks", "BOS": "Celtics", "BKN": "Nets", "CHA": "Hornets",
    "CHI": "Bulls", "CLE": "Cavaliers", "DAL": "Mavericks", "DEN": "Nuggets",
    "DET": "Pistons", "GSW": "Warriors", "HOU": "Rockets", "IND": "Pacers",
    "LAC": "Clippers", "LAL": "Lakers", "MEM": "Grizzlies", "MIA": "Heat",
    "MIL": "Bucks", "MIN": "Timberwolves", "NOP": "Pelicans", "NYK": "Knicks",
    "OKC": "Thunder", "ORL": "Magic", "PHI": "76ers", "PHX": "Suns",
    "POR": "Trail Blazers", "SAC": "Kings", "SAS": "Spurs", "TOR": "Raptors",
    "UTA": "Jazz", "WAS": "Wizards",
}

app = Flask(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


app.json.encoder = NumpyEncoder  # type: ignore


def build_feature_vector(home_team, away_team):
    """Build the 107-feature vector for a matchup using latest data."""
    features = MODELS["features"]
    df_feat = DF_FEAT

    home_as_home = df_feat[df_feat["home_team_id"] == home_team].sort_values("date")
    away_as_away = df_feat[df_feat["visitor_team_id"] == away_team].sort_values("date")

    home_last = df_feat[
        (df_feat["home_team_id"] == home_team) | (df_feat["visitor_team_id"] == home_team)
    ].sort_values("date")
    home_last = home_last.iloc[-1] if len(home_last) > 0 else None

    away_last = df_feat[
        (df_feat["home_team_id"] == away_team) | (df_feat["visitor_team_id"] == away_team)
    ].sort_values("date")
    away_last = away_last.iloc[-1] if len(away_last) > 0 else None

    feat_values = {}
    for feat in features:
        val = 0.0

        if feat == "elo_rating_diff":
            val = ELO_RATINGS.get(home_team, 1500) - ELO_RATINGS.get(away_team, 1500)
        elif feat == "elo_diff_squared":
            diff = ELO_RATINGS.get(home_team, 1500) - ELO_RATINGS.get(away_team, 1500)
            val = diff * abs(diff)
        elif feat == "elo_diff_abs":
            val = abs(ELO_RATINGS.get(home_team, 1500) - ELO_RATINGS.get(away_team, 1500))
        elif feat.endswith("_diff") or feat.endswith("_mismatch") or feat.endswith("_edge_5") or feat.endswith("_edge_10"):
            if len(home_as_home) > 0:
                val = home_as_home.iloc[-1].get(feat, 0)
            elif home_last is not None:
                val = home_last.get(feat, 0)
        elif feat.startswith("home_"):
            if len(home_as_home) > 0:
                val = home_as_home.iloc[-1].get(feat, 0)
            elif home_last is not None:
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
        else:
            if len(home_as_home) > 0:
                val = home_as_home.iloc[-1].get(feat, 0)
            elif home_last is not None:
                val = home_last.get(feat, 0)

        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            val = 0.0
        feat_values[feat] = val

    return feat_values


def run_prediction(home_team, away_team):
    """Run the full ensemble prediction."""
    features = MODELS["features"]
    feat_values = build_feature_vector(home_team, away_team)
    X = pd.DataFrame([feat_values])[features].fillna(0).replace([np.inf, -np.inf], 0)

    X_scaled = MODELS["scaler"].transform(X)
    lr_prob = float(MODELS["lr"].predict_proba(X_scaled)[0, 1])

    dmat = xgb.DMatrix(X, feature_names=features)
    xgb_prob = float(MODELS["xgb"].predict(dmat)[0])

    lgb_prob = float(MODELS["lgb"].predict(X)[0])

    base_probs = np.array([[lr_prob, xgb_prob, lgb_prob]])
    raw_feats = X[EXT_META_RAW_FEATURES].fillna(0).values
    meta_input = np.column_stack([base_probs, raw_feats])
    raw_ensemble_prob = float(MODELS["ext_meta"].predict_proba(meta_input)[0, 1])

    # Apply RL corrections
    ensemble_prob, rl_adjustment, rl_rules = apply_rl_adjustment(raw_ensemble_prob, feat_values)

    threshold = MODELS["config"].get("threshold", 0.52)
    winner = home_team if ensemble_prob >= threshold else away_team
    confidence = max(ensemble_prob, 1 - ensemble_prob)

    key_features = {}
    for k in ["elo_rating_diff", "season_win_pct_diff", "pyth_win_exp_diff",
              "last3_net_rating_diff", "last5_net_rating_diff", "last10_net_rating_diff",
              "weighted_net_rating_momentum", "rest_diff", "fatigue_diff",
              "srs_diff", "team_ortg_diff", "team_drtg_diff", "team_bpm_diff", "spread"]:
        key_features[k] = round(feat_values.get(k, 0), 3)

    result = {
        "home_team": home_team,
        "away_team": away_team,
        "home_name": NBA_TEAMS.get(home_team, home_team),
        "away_name": NBA_TEAMS.get(away_team, away_team),
        "lr_prob": round(lr_prob, 4),
        "xgb_prob": round(xgb_prob, 4),
        "lgb_prob": round(lgb_prob, 4),
        "ensemble_prob": round(ensemble_prob, 4),
        "threshold": float(threshold),
        "winner": winner,
        "winner_name": NBA_TEAMS.get(winner, winner),
        "loser": away_team if winner == home_team else home_team,
        "loser_name": NBA_TEAMS.get(away_team if winner == home_team else home_team, ""),
        "confidence": round(confidence, 4),
        "elo_home": float(round(ELO_RATINGS.get(home_team, 1500), 1)),
        "elo_away": float(round(ELO_RATINGS.get(away_team, 1500), 1)),
        "key_features": {k: float(v) for k, v in key_features.items()},
        "raw_ensemble_prob": round(raw_ensemble_prob, 4),
        "rl_adjustment": round(float(rl_adjustment), 4),
        "rl_rules_matched": len(rl_rules),
        "rl_rules": [{"pattern": r[0], "correction": round(float(r[1]), 4)} for r in rl_rules[:5]],
    }
    # Ensure all values are JSON-serializable native Python types
    return json.loads(json.dumps(result, default=lambda o: float(o) if hasattr(o, '__float__') else str(o)))


# ============================================================
# Routes
# ============================================================

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, teams=NBA_TEAMS, elo=ELO_RATINGS)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.json
        home = data.get("home", "").upper()
        away = data.get("away", "").upper()
        if home not in NBA_TEAMS or away not in NBA_TEAMS:
            return jsonify({"error": "Invalid team code"}), 400
        if home == away:
            return jsonify({"error": "Home and away must be different"}), 400
        result = run_prediction(home, away)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ============================================================
# NBA CDN Proxy (avoids CORS issues in the browser)
# ============================================================
def _fetch_nba_url(url):
    """Server-side fetch from NBA CDN."""
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "application/json",
        "Referer": "https://www.nba.com/",
    })
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read()


@app.route("/api/nba/scoreboard")
def nba_scoreboard_proxy():
    """Proxy today's NBA scoreboard to avoid CORS."""
    try:
        data = _fetch_nba_url("https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json")
        return Response(data, mimetype="application/json")
    except Exception as e:
        return jsonify({"error": f"Could not fetch NBA scoreboard: {e}"}), 502


@app.route("/api/nba/boxscore/<game_id>")
def nba_boxscore_proxy(game_id):
    """Proxy a live NBA box score to avoid CORS."""
    try:
        data = _fetch_nba_url(f"https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json")
        return Response(data, mimetype="application/json")
    except Exception as e:
        return jsonify({"error": f"Could not fetch box score: {e}"}), 502


@app.route("/api/teams")
def api_teams():
    return jsonify({code: {"name": name, "elo": round(ELO_RATINGS.get(code, 1500), 1)}
                    for code, name in NBA_TEAMS.items()})


# ============================================================
# HTML Template (single-file, no external dependencies)
# ============================================================
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NBA Win Predictor</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0a0e27;
    color: #e0e0e0;
    min-height: 100vh;
  }
  .header {
    background: linear-gradient(135deg, #1a1f4e 0%, #2d1b69 100%);
    padding: 24px;
    text-align: center;
    border-bottom: 3px solid #f57c00;
  }
  .header h1 { font-size: 28px; color: #fff; letter-spacing: 1px; }
  .header p { color: #aaa; margin-top: 6px; font-size: 14px; }
  .container { max-width: 1100px; margin: 0 auto; padding: 24px; }

  /* Live Games Section */
  .live-section {
    background: #111640;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 24px;
    border: 1px solid #2a2f5e;
  }
  .live-section h2 { font-size: 18px; color: #f57c00; margin-bottom: 14px; }
  .live-games { display: flex; flex-wrap: wrap; gap: 12px; }
  .live-game-card {
    background: #1a1f4e;
    border: 1px solid #2a2f5e;
    border-radius: 8px;
    padding: 14px 18px;
    cursor: pointer;
    transition: all 0.2s;
    min-width: 220px;
    flex: 1;
  }
  .live-game-card:hover { border-color: #f57c00; transform: translateY(-2px); }
  .live-game-card.selected { border-color: #f57c00; background: #252b6e; }
  .live-game-teams { font-size: 16px; font-weight: 600; }
  .live-game-score { font-size: 22px; font-weight: 700; color: #fff; margin: 6px 0; }
  .live-game-status { font-size: 12px; color: #aaa; }
  .live-game-status.live { color: #4caf50; font-weight: 600; }
  #live-loading { color: #888; font-style: italic; }
  #live-error { color: #ff5252; display: none; }

  /* Manual Picker */
  .picker-section {
    background: #111640;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 24px;
    border: 1px solid #2a2f5e;
  }
  .picker-section h2 { font-size: 18px; color: #f57c00; margin-bottom: 14px; }
  .picker-row { display: flex; gap: 16px; align-items: end; flex-wrap: wrap; }
  .picker-group { flex: 1; min-width: 200px; }
  .picker-group label { display: block; font-size: 13px; color: #aaa; margin-bottom: 6px; }
  .picker-group select {
    width: 100%; padding: 10px 14px; font-size: 15px;
    background: #1a1f4e; color: #fff; border: 1px solid #2a2f5e;
    border-radius: 6px; cursor: pointer;
  }
  .picker-group select:focus { outline: none; border-color: #f57c00; }
  .btn-predict {
    padding: 10px 32px; font-size: 16px; font-weight: 600;
    background: linear-gradient(135deg, #f57c00, #e65100);
    color: #fff; border: none; border-radius: 6px;
    cursor: pointer; transition: all 0.2s; white-space: nowrap;
  }
  .btn-predict:hover { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(245,124,0,0.4); }
  .btn-predict:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

  /* Results */
  .result-section {
    display: none;
    background: #111640;
    border-radius: 12px;
    padding: 24px;
    border: 1px solid #2a2f5e;
    margin-bottom: 24px;
  }
  .result-section.visible { display: block; animation: fadeIn 0.3s; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

  .matchup-header {
    text-align: center;
    margin-bottom: 20px;
    padding-bottom: 16px;
    border-bottom: 1px solid #2a2f5e;
  }
  .matchup-header .vs { font-size: 28px; font-weight: 700; color: #fff; }
  .matchup-header .vs span { color: #888; font-size: 18px; margin: 0 12px; }

  .winner-banner {
    text-align: center; padding: 20px;
    background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
    border-radius: 10px; margin-bottom: 20px;
  }
  .winner-banner .label { font-size: 13px; color: #a5d6a7; letter-spacing: 2px; }
  .winner-banner .team { font-size: 32px; font-weight: 800; color: #fff; margin: 6px 0; }
  .winner-banner .conf { font-size: 18px; color: #c8e6c9; }

  .prob-bar-container { margin-bottom: 20px; }
  .prob-bar-label { display: flex; justify-content: space-between; font-size: 14px; margin-bottom: 4px; }
  .prob-bar-outer {
    height: 36px; background: #1a1f4e; border-radius: 18px;
    overflow: hidden; position: relative; border: 1px solid #2a2f5e;
  }
  .prob-bar-inner {
    height: 100%; border-radius: 18px; transition: width 0.6s ease;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 14px; color: #fff; min-width: 50px;
  }
  .prob-bar-home { background: linear-gradient(90deg, #1565c0, #1976d2); }
  .prob-bar-away { background: linear-gradient(90deg, #c62828, #d32f2f); float: right; }

  .models-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 20px; }
  .model-card {
    background: #1a1f4e; border-radius: 8px; padding: 14px;
    text-align: center; border: 1px solid #2a2f5e;
  }
  .model-card .name { font-size: 12px; color: #888; margin-bottom: 4px; }
  .model-card .prob { font-size: 20px; font-weight: 700; }
  .model-card .prob.home-fav { color: #42a5f5; }
  .model-card .prob.away-fav { color: #ef5350; }

  .features-table { width: 100%; border-collapse: collapse; font-size: 14px; }
  .features-table th { text-align: left; padding: 8px 12px; color: #888; font-weight: 500; border-bottom: 1px solid #2a2f5e; }
  .features-table td { padding: 8px 12px; border-bottom: 1px solid #181d4a; }
  .features-table .val { text-align: right; font-family: monospace; font-weight: 600; }
  .val.positive { color: #4caf50; }
  .val.negative { color: #ef5350; }

  .elo-info { display: flex; justify-content: center; gap: 40px; margin-bottom: 16px; }
  .elo-info .elo-team { text-align: center; }
  .elo-info .elo-team .code { font-size: 18px; font-weight: 700; color: #fff; }
  .elo-info .elo-team .rating { font-size: 14px; color: #aaa; }

  .live-badge {
    display: inline-block; padding: 2px 8px; background: #4caf50;
    color: #fff; font-size: 11px; font-weight: 700; border-radius: 4px;
    margin-left: 8px; animation: pulse 2s infinite;
  }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }

  .box-score-info {
    background: #1a1f4e; border-radius: 8px; padding: 14px;
    margin-bottom: 16px; border: 1px solid #2a2f5e;
    display: none;
  }
  .box-score-info.visible { display: block; }
  .box-score-info h3 { font-size: 14px; color: #f57c00; margin-bottom: 8px; }
  .box-score-info .score-display {
    font-size: 28px; font-weight: 800; text-align: center;
    color: #fff; margin: 10px 0;
  }
</style>
</head>
<body>

<div class="header">
  <h1>NBA Win Predictor</h1>
  <p>Powered by 107-feature ensemble model | Trained on 28,500+ games (2003-2026)</p>
</div>

<div class="container">

  <!-- Live Games -->
  <div class="live-section">
    <h2>Today's Games <span id="live-badge" class="live-badge" style="display:none">LIVE</span></h2>
    <div id="live-loading">Fetching today's games from NBA...</div>
    <div id="live-error">Could not load live games (NBA CDN may be blocked in this environment). Use the manual picker below - predictions work fine without live data!</div>
    <div id="live-games" class="live-games"></div>
  </div>

  <!-- Manual Picker -->
  <div class="picker-section">
    <h2>Pick a Matchup</h2>
    <div class="picker-row">
      <div class="picker-group">
        <label>Away Team</label>
        <select id="away-select">
          {% for code, name in teams|dictsort %}
          <option value="{{ code }}">{{ code }} - {{ name }} (Elo: {{ elo.get(code, 1500)|round(0)|int }})</option>
          {% endfor %}
        </select>
      </div>
      <div class="picker-group">
        <label>@ Home Team</label>
        <select id="home-select">
          {% for code, name in teams|dictsort %}
          <option value="{{ code }}">{{ code }} - {{ name }} (Elo: {{ elo.get(code, 1500)|round(0)|int }})</option>
          {% endfor %}
        </select>
      </div>
      <div>
        <button class="btn-predict" id="btn-predict" onclick="predictGame()">Predict Winner</button>
      </div>
    </div>
  </div>

  <!-- Results -->
  <div class="result-section" id="result-section">
    <div class="matchup-header">
      <div class="vs"><span id="r-away"></span> <span>@</span> <span id="r-home"></span></div>
    </div>

    <div class="elo-info">
      <div class="elo-team">
        <div class="code" id="r-away-code"></div>
        <div class="rating">Elo: <span id="r-elo-away"></span></div>
      </div>
      <div class="elo-team">
        <div class="code" id="r-home-code"></div>
        <div class="rating">Elo: <span id="r-elo-home"></span></div>
      </div>
    </div>

    <div class="box-score-info" id="box-score-info">
      <h3>Live Score</h3>
      <div class="score-display" id="live-score-display"></div>
      <div id="live-game-status-text" style="text-align:center; color:#aaa; font-size:13px;"></div>
    </div>

    <div class="winner-banner">
      <div class="label">PREDICTED WINNER</div>
      <div class="team" id="r-winner"></div>
      <div class="conf" id="r-confidence"></div>
    </div>

    <div class="prob-bar-container">
      <div class="prob-bar-label">
        <span id="r-bar-home-label"></span>
        <span id="r-bar-away-label"></span>
      </div>
      <div class="prob-bar-outer">
        <div class="prob-bar-inner prob-bar-home" id="r-bar-home"></div>
      </div>
    </div>

    <div class="models-grid">
      <div class="model-card">
        <div class="name">Logistic Regression</div>
        <div class="prob" id="r-lr"></div>
      </div>
      <div class="model-card">
        <div class="name">XGBoost</div>
        <div class="prob" id="r-xgb"></div>
      </div>
      <div class="model-card">
        <div class="name">LightGBM</div>
        <div class="prob" id="r-lgb"></div>
      </div>
      <div class="model-card" style="border-color: #f57c00;">
        <div class="name" style="color: #f57c00;">Ensemble</div>
        <div class="prob" id="r-ens"></div>
      </div>
      <div class="model-card" style="border-color: #4caf50;">
        <div class="name" style="color: #4caf50;">RL-Adjusted</div>
        <div class="prob" id="r-rl"></div>
      </div>
    </div>

    <div id="rl-info" style="background:#1a2e1a; border:1px solid #2e7d32; border-radius:8px; padding:14px; margin-bottom:16px; display:none;">
      <h3 style="font-size:14px; color:#4caf50; margin-bottom:8px;">RL Corrections Applied</h3>
      <div id="rl-adj-text" style="font-size:13px; color:#aaa;"></div>
      <div id="rl-rules-list" style="font-size:12px; color:#888; margin-top:6px;"></div>
    </div>

    <h3 style="font-size:15px; color:#f57c00; margin-bottom:10px;">Key Features</h3>
    <table class="features-table">
      <thead><tr><th>Feature</th><th style="text-align:right">Value</th><th style="text-align:right">Favors</th></tr></thead>
      <tbody id="r-features"></tbody>
    </table>
  </div>

</div>

<script>
// Use our Flask proxy to avoid CORS issues with NBA CDN
const NBA_SCOREBOARD_URL = "/api/nba/scoreboard";
const NBA_BOXSCORE_URL = "/api/nba/boxscore/{GAME_ID}";

let todaysGames = [];
let selectedGameId = null;
let selectedBoxScore = null;

// Feature display names
const FEAT_LABELS = {
  "elo_rating_diff": "Elo Rating Diff",
  "season_win_pct_diff": "Season Win% Diff",
  "pyth_win_exp_diff": "Pythagorean Diff",
  "last3_net_rating_diff": "Last 3 Games Net Rtg",
  "last5_net_rating_diff": "Last 5 Games Net Rtg",
  "last10_net_rating_diff": "Last 10 Games Net Rtg",
  "weighted_net_rating_momentum": "Momentum",
  "rest_diff": "Rest Days Diff",
  "fatigue_diff": "Fatigue Diff",
  "srs_diff": "SRS Diff",
  "team_ortg_diff": "Team Off Rating Diff",
  "team_drtg_diff": "Team Def Rating Diff",
  "team_bpm_diff": "Team BPM Diff",
  "spread": "Spread",
};

// Fetch today's live games
async function fetchLiveGames() {
  try {
    const resp = await fetch(NBA_SCOREBOARD_URL);
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const data = await resp.json();
    todaysGames = data.scoreboard.games;

    document.getElementById("live-loading").style.display = "none";

    if (todaysGames.length === 0) {
      document.getElementById("live-loading").style.display = "block";
      document.getElementById("live-loading").textContent = "No games scheduled today.";
      return;
    }

    document.getElementById("live-badge").style.display = "inline-block";
    const container = document.getElementById("live-games");
    container.innerHTML = "";

    todaysGames.forEach(g => {
      const card = document.createElement("div");
      card.className = "live-game-card";
      card.dataset.gameId = g.gameId;
      card.dataset.home = g.homeTeam.teamTricode;
      card.dataset.away = g.awayTeam.teamTricode;

      const statusClass = g.gameStatus === 2 ? "live" : "";
      let statusText = g.gameStatusText;

      card.innerHTML = `
        <div class="live-game-teams">${g.awayTeam.teamTricode} @ ${g.homeTeam.teamTricode}</div>
        <div class="live-game-score">${g.awayTeam.score || 0} - ${g.homeTeam.score || 0}</div>
        <div class="live-game-status ${statusClass}">${statusText}</div>
      `;

      card.onclick = () => selectLiveGame(g);
      container.appendChild(card);
    });

  } catch (e) {
    document.getElementById("live-loading").style.display = "none";
    document.getElementById("live-error").style.display = "block";
    console.error("Failed to fetch live games:", e);
  }
}

async function selectLiveGame(game) {
  // Highlight selected card
  document.querySelectorAll(".live-game-card").forEach(c => c.classList.remove("selected"));
  document.querySelector(`[data-game-id="${game.gameId}"]`).classList.add("selected");

  // Set dropdowns
  document.getElementById("home-select").value = game.homeTeam.teamTricode;
  document.getElementById("away-select").value = game.awayTeam.teamTricode;

  selectedGameId = game.gameId;
  selectedBoxScore = null;

  // Fetch box score
  try {
    const url = NBA_BOXSCORE_URL.replace("{GAME_ID}", game.gameId);
    const resp = await fetch(url);
    if (resp.ok) {
      selectedBoxScore = await resp.json();
    }
  } catch (e) {
    console.error("Failed to fetch box score:", e);
  }

  predictGame();
}

async function predictGame() {
  const home = document.getElementById("home-select").value;
  const away = document.getElementById("away-select").value;

  if (home === away) { alert("Pick different teams!"); return; }

  const btn = document.getElementById("btn-predict");
  btn.disabled = true;
  btn.textContent = "Predicting...";

  try {
    const resp = await fetch("/api/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({home, away}),
    });
    const r = await resp.json();
    if (r.error) { alert(r.error); return; }

    // Fill in results
    document.getElementById("r-away").textContent = r.away_name + " (" + r.away_team + ")";
    document.getElementById("r-home").textContent = r.home_name + " (" + r.home_team + ")";
    document.getElementById("r-away-code").textContent = r.away_team;
    document.getElementById("r-home-code").textContent = r.home_team;
    document.getElementById("r-elo-home").textContent = r.elo_home;
    document.getElementById("r-elo-away").textContent = r.elo_away;

    document.getElementById("r-winner").textContent = r.winner + " " + r.winner_name;
    document.getElementById("r-confidence").textContent = (r.confidence * 100).toFixed(1) + "% confidence";

    const homeProb = r.ensemble_prob;
    const awayProb = 1 - r.ensemble_prob;
    const homePct = (homeProb * 100).toFixed(1);
    const awayPct = (awayProb * 100).toFixed(1);

    document.getElementById("r-bar-home-label").textContent = r.home_team + " " + homePct + "%";
    document.getElementById("r-bar-away-label").textContent = r.away_team + " " + awayPct + "%";
    document.getElementById("r-bar-home").style.width = homePct + "%";

    const fmt = (p, isHome) => {
      const pct = isHome ? (p * 100).toFixed(1) + "%" : ((1 - p) * 100).toFixed(1) + "%";
      const cls = (isHome ? p >= 0.5 : p < 0.5) ? "home-fav" : "away-fav";
      return `<span class="${p >= 0.5 ? 'home-fav' : 'away-fav'}">${r.home_team} ${(p*100).toFixed(1)}%</span>`;
    };

    document.getElementById("r-lr").innerHTML = probHTML(r.lr_prob, r.home_team, r.away_team);
    document.getElementById("r-xgb").innerHTML = probHTML(r.xgb_prob, r.home_team, r.away_team);
    document.getElementById("r-lgb").innerHTML = probHTML(r.lgb_prob, r.home_team, r.away_team);
    document.getElementById("r-ens").innerHTML = probHTML(r.raw_ensemble_prob || r.ensemble_prob, r.home_team, r.away_team);
    document.getElementById("r-rl").innerHTML = probHTML(r.ensemble_prob, r.home_team, r.away_team);

    // Show RL adjustment info
    const rlInfo = document.getElementById("rl-info");
    if (r.rl_adjustment && r.rl_adjustment !== 0) {
      rlInfo.style.display = "block";
      const adjSign = r.rl_adjustment > 0 ? "+" : "";
      document.getElementById("rl-adj-text").textContent =
        `Adjustment: ${adjSign}${(r.rl_adjustment * 100).toFixed(2)}% | ${r.rl_rules_matched} pattern(s) matched | Raw ensemble: ${(r.raw_ensemble_prob * 100).toFixed(1)}% → Adjusted: ${(r.ensemble_prob * 100).toFixed(1)}%`;
      let rulesHtml = "";
      if (r.rl_rules && r.rl_rules.length > 0) {
        rulesHtml = r.rl_rules.map(rule =>
          `<span style="display:inline-block;background:#1a3a1a;padding:2px 8px;border-radius:4px;margin:2px;">${rule.pattern}: ${rule.correction > 0 ? '+' : ''}${(rule.correction * 100).toFixed(2)}%</span>`
        ).join(" ");
      }
      document.getElementById("rl-rules-list").innerHTML = rulesHtml;
    } else {
      rlInfo.style.display = "none";
    }

    // Features table
    const tbody = document.getElementById("r-features");
    tbody.innerHTML = "";
    for (const [feat, val] of Object.entries(r.key_features)) {
      if (val === 0) continue;
      const label = FEAT_LABELS[feat] || feat;
      const cls = val > 0 ? "positive" : "negative";
      const favors = val > 0 ? r.home_team : r.away_team;
      tbody.innerHTML += `<tr><td>${label}</td><td class="val ${cls}">${val > 0 ? '+' : ''}${val.toFixed(3)}</td><td class="val" style="color:#aaa">${favors}</td></tr>`;
    }

    // Show live score if available
    const bsInfo = document.getElementById("box-score-info");
    if (selectedBoxScore) {
      const g = selectedBoxScore.game;
      const homeScore = g.homeTeam.score;
      const awayScore = g.awayTeam.score;
      document.getElementById("live-score-display").textContent =
        g.awayTeam.teamTricode + " " + awayScore + " - " + homeScore + " " + g.homeTeam.teamTricode;
      document.getElementById("live-game-status-text").textContent = g.gameStatusText || "";
      bsInfo.classList.add("visible");
    } else {
      bsInfo.classList.remove("visible");
    }

    document.getElementById("result-section").classList.add("visible");

  } catch (e) {
    alert("Prediction failed: " + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Predict Winner";
  }
}

function probHTML(prob, home, away) {
  const homePct = (prob * 100).toFixed(1);
  const awayPct = ((1-prob) * 100).toFixed(1);
  if (prob >= 0.5) {
    return `<span class="home-fav">${home} ${homePct}%</span>`;
  } else {
    return `<span class="away-fav">${away} ${awayPct}%</span>`;
  }
}

// Auto-fetch live games on load
fetchLiveGames();

// Refresh live games every 30 seconds
setInterval(fetchLiveGames, 30000);
</script>

</body>
</html>
"""

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  NBA Win Predictor - Web App")
    print("  Open http://localhost:5050 in your browser")
    print("=" * 50 + "\n")
    app.run(host="0.0.0.0", port=5050, debug=False)

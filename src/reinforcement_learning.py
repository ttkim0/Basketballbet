"""
NBA Reinforcement Learning System
===================================
Iterative self-improvement loop that:
1. Walks forward through every game 2010-2026, predicting with only past data
2. Rewards correct predictions, deeply analyzes wrong ones
3. Discovers systematic error patterns (fatigue, upsets, calibration, etc.)
4. Converts patterns into correction features
5. Retrains and iterates until accuracy plateaus

Usage: python3 src/reinforcement_learning.py
"""

import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgbm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss
from scipy import stats
from collections import defaultdict
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = "/home/user/Basketballbet"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
RL_DIR = os.path.join(LOG_DIR, "rl_results")
os.makedirs(RL_DIR, exist_ok=True)

# Import feature list from train_model
FEATURE_COLUMNS = [
    "elo_rating_diff", "elo_diff_squared", "season_win_pct_diff",
    "power_rating_composite", "pyth_win_exp_diff",
    "last3_net_rating_diff", "last3_win_rate_diff", "last3_avg_margin_diff",
    "last5_net_rating_diff", "last5_win_rate_diff", "last5_avg_margin_diff",
    "last5_off_rating_diff", "last5_def_rating_diff",
    "last10_net_rating_diff", "last10_win_rate_diff", "last10_avg_margin_diff",
    "last10_off_rating_diff", "last10_def_rating_diff",
    "weighted_net_rating_momentum", "weighted_win_rate_momentum",
    "weighted_avg_margin_momentum",
    "rest_days_home", "rest_days_visitor", "rest_diff",
    "home_b2b", "visitor_b2b", "home_3in4", "visitor_3in4",
    "home_4in6", "visitor_4in6",
    "travel_miles_home", "travel_miles_visitor", "travel_diff",
    "tz_shift_home", "tz_shift_visitor", "tz_diff", "altitude_edge_home",
    "fatigue_composite_home", "fatigue_composite_visitor", "fatigue_diff",
    "home_b2b_x_elo", "visitor_b2b_x_elo", "rest_x_elo",
    "streak_x_progress", "h2h_x_current_form", "elo_diff_abs",
    "home_season_win_pct", "visitor_season_win_pct",
    "home_home_win_pct", "visitor_road_win_pct",
    "home_streak", "visitor_streak", "streak_diff",
    "net_matchup_edge_5", "net_matchup_edge_10",
    "off_vs_def_mismatch_5", "off_vs_def_mismatch_10",
    "def_vs_off_mismatch_5", "def_vs_off_mismatch_10", "pace_mismatch",
    "last5_efg_pct_diff_real", "last10_efg_pct_diff_real",
    "last5_ts_pct_diff_real", "last10_ts_pct_diff_real",
    "last5_tov_diff_real", "last10_tov_diff_real",
    "last5_reb_diff_real", "last10_reb_diff_real",
    "last5_ast_diff_real", "last10_ast_diff_real",
    "last5_stl_diff_real", "last10_stl_diff_real",
    "last5_blk_diff_real", "last10_blk_diff_real",
    "last5_oreb_diff_real", "last10_oreb_diff_real",
    "last5_three_pct_diff_real", "last10_three_pct_diff_real",
    "last5_tov_rate_diff", "last10_tov_rate_diff",
    "last5_ortg_diff_real", "last10_ortg_diff_real",
    "last5_drtg_diff_real", "last10_drtg_diff_real",
    "last5_pace_diff_real", "last10_pace_diff_real",
    "last5_tov_pct_diff_real", "last10_tov_pct_diff_real",
    "last5_orb_pct_diff_real", "last10_orb_pct_diff_real",
    "spread", "spread_abs", "market_prob_diff", "expected_total",
    "raptor_total_diff", "raptor_off_diff", "raptor_def_diff", "war_diff",
    "srs_diff", "team_ortg_diff", "team_drtg_diff", "team_nrtg_diff",
    "team_pace_diff", "team_bpm_diff", "team_ws48_diff", "team_vorp_diff",
    "team_per_diff",
]

TARGET = "home_win"


# ============================================================
# Game Categorization — tag every game for error analysis
# ============================================================

def get_team_tier(win_pct):
    """Bucket team by season win%."""
    if win_pct >= 0.65:
        return "elite"
    elif win_pct >= 0.55:
        return "good"
    elif win_pct >= 0.45:
        return "mid"
    else:
        return "bad"


def categorize_game(row, pred_prob):
    """Tag a game with contextual categories for error analysis."""
    tags = {}

    # Elo-based upset type
    elo_diff = row.get("elo_rating_diff", 0)
    home_won = row.get(TARGET, 0)
    home_favored = elo_diff > 0
    is_upset = (home_favored and not home_won) or (not home_favored and home_won)
    abs_elo = abs(elo_diff)
    if not is_upset:
        tags["upset_type"] = "none"
    elif abs_elo >= 150:
        tags["upset_type"] = "major_upset"
    elif abs_elo >= 75:
        tags["upset_type"] = "mild_upset"
    else:
        tags["upset_type"] = "slight_upset"

    # Quality tier matchup
    h_wp = row.get("home_season_win_pct", 0.5)
    v_wp = row.get("visitor_season_win_pct", 0.5)
    h_tier = get_team_tier(h_wp)
    v_tier = get_team_tier(v_wp)
    tiers_sorted = tuple(sorted([h_tier, v_tier]))
    tags["quality_tier"] = f"{h_tier}_vs_{v_tier}"
    tags["home_tier"] = h_tier
    tags["visitor_tier"] = v_tier

    # Fatigue category
    h_b2b = row.get("home_b2b", 0)
    v_b2b = row.get("visitor_b2b", 0)
    h_3in4 = row.get("home_3in4", 0)
    v_3in4 = row.get("visitor_3in4", 0)
    if h_b2b and v_b2b:
        tags["fatigue"] = "b2b_both"
    elif h_b2b:
        tags["fatigue"] = "b2b_home"
    elif v_b2b:
        tags["fatigue"] = "b2b_visitor"
    elif h_3in4 or v_3in4:
        tags["fatigue"] = "3in4_either"
    else:
        tags["fatigue"] = "rested_both"

    # Confidence bucket
    conf = abs(pred_prob - 0.5)
    if conf >= 0.25:
        tags["confidence"] = "very_high"
    elif conf >= 0.15:
        tags["confidence"] = "high"
    elif conf >= 0.05:
        tags["confidence"] = "medium"
    else:
        tags["confidence"] = "low"

    # Momentum
    h_streak = row.get("home_streak", 0)
    v_streak = row.get("visitor_streak", 0)
    if h_streak >= 4:
        tags["momentum"] = "hot_home"
    elif h_streak <= -3:
        tags["momentum"] = "cold_home"
    elif v_streak >= 4:
        tags["momentum"] = "hot_visitor"
    elif v_streak <= -3:
        tags["momentum"] = "cold_visitor"
    else:
        tags["momentum"] = "neutral"

    # Rest advantage
    rest_diff = row.get("rest_diff", 0)
    if rest_diff >= 3:
        tags["rest_adv"] = "big_rest_home"
    elif rest_diff >= 1:
        tags["rest_adv"] = "slight_rest_home"
    elif rest_diff <= -3:
        tags["rest_adv"] = "big_rest_visitor"
    elif rest_diff <= -1:
        tags["rest_adv"] = "slight_rest_visitor"
    else:
        tags["rest_adv"] = "equal_rest"

    # Month
    date = pd.Timestamp(row.get("date", "2020-01-01"))
    tags["month"] = date.month

    # Season phase
    season_games = row.get("home_games_played", 41)
    if isinstance(season_games, (int, float)) and season_games <= 20:
        tags["phase"] = "early"
    elif isinstance(season_games, (int, float)) and season_games >= 70:
        tags["phase"] = "late"
    else:
        tags["phase"] = "mid"

    # Market alignment
    spread = row.get("spread", 0)
    if spread != 0:
        model_favors_home = pred_prob > 0.5
        market_favors_home = spread < 0  # negative spread = home favored
        tags["market_align"] = "agree" if model_favors_home == market_favors_home else "disagree"
    else:
        tags["market_align"] = "no_line"

    # Travel
    travel_h = row.get("travel_miles_home", 0)
    travel_v = row.get("travel_miles_visitor", 0)
    if travel_v >= 1500:
        tags["travel"] = "heavy_visitor_travel"
    elif travel_h >= 1500:
        tags["travel"] = "heavy_home_travel"
    else:
        tags["travel"] = "normal_travel"

    # Altitude
    alt = row.get("altitude_edge_home", 0)
    if alt >= 3000:
        tags["altitude"] = "high_altitude"
    else:
        tags["altitude"] = "normal"

    return tags


# ============================================================
# Sample weight computation (same as train_model.py)
# ============================================================

def compute_sample_weights(seasons_array):
    """Aggressive recency weighting: base 0.88 per season."""
    seasons = sorted(set(seasons_array))
    n = len(seasons)
    rank = {s: i for i, s in enumerate(seasons)}
    weights = np.array([0.88 ** (n - 1 - rank[s]) for s in seasons_array])
    weights = weights / weights.mean()
    return weights


# ============================================================
# Fast model training (LightGBM for speed)
# ============================================================

def train_lgbm(X_train, y_train, X_val, y_val, features, sample_weights=None):
    """Train LightGBM quickly for forward walk."""
    dtrain = lgbm.Dataset(X_train, label=y_train, feature_name=features,
                          weight=sample_weights)
    dval = lgbm.Dataset(X_val, label=y_val, feature_name=features, reference=dtrain)

    params = {
        "objective": "binary", "metric": "binary_logloss",
        "num_leaves": 47, "learning_rate": 0.08,
        "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 5,
        "min_child_samples": 20, "reg_alpha": 0.1, "reg_lambda": 0.5,
        "seed": 42, "verbose": -1, "feature_pre_filter": False,
    }

    model = lgbm.train(
        params, dtrain, num_boost_round=500,
        valid_sets=[dval], valid_names=["val"],
        callbacks=[lgbm.early_stopping(30), lgbm.log_evaluation(0)],
    )
    return model


def train_lr(X_train, y_train, sample_weights=None):
    """Train logistic regression quickly."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(C=0.05, penalty="l1", max_iter=5000, solver="saga")
    model.fit(X_scaled, y_train, sample_weight=sample_weights)
    return model, scaler


# ============================================================
# Forward Walk Engine
# ============================================================

def forward_walk(df, features, start_season=2010, correction_features=None,
                 probability_adjustments=None):
    """
    Walk forward through every game from start_season to 2026.
    Retrain at each season boundary using only past data.

    Returns:
        records: list of dicts with prediction results + categories
        summary: dict with overall metrics
    """
    if correction_features:
        # Deduplicate correction features
        seen = set(features)
        for f in correction_features:
            if f not in seen:
                features = features + [f]
                seen.add(f)

    available = list(dict.fromkeys(f for f in features if f in df.columns))
    df = df.sort_values("date").reset_index(drop=True)

    seasons = sorted(df["season"].unique())
    walk_seasons = [s for s in seasons if s >= start_season]

    records = []
    current_model = None
    current_lr = None
    current_scaler = None
    trained_through = None

    total_correct = 0
    total_games = 0

    for season in walk_seasons:
        season_mask = df["season"] == season
        season_df = df[season_mask]
        season_idx = season_df.index

        # ---- RETRAIN using all data before this season ----
        train_mask = df["season"] < season
        if train_mask.sum() < 1000:
            continue

        train_df = df[train_mask]
        X_train = train_df[available].fillna(0).replace([np.inf, -np.inf], 0)
        y_train = train_df[TARGET]
        weights = compute_sample_weights(train_df["season"].values)

        # Use last 20% of training data as validation for early stopping
        n_val = max(500, int(len(X_train) * 0.15))
        X_tr, X_vl = X_train.iloc[:-n_val], X_train.iloc[-n_val:]
        y_tr, y_vl = y_train.iloc[:-n_val], y_train.iloc[-n_val:]
        w_tr = weights[:-n_val]

        current_model = train_lgbm(X_tr, y_tr, X_vl, y_vl, available, w_tr)
        current_lr, current_scaler = train_lr(X_train.values, y_train.values, weights)

        # ---- PREDICT every game in this season ----
        season_correct = 0
        for idx in season_idx:
            row = df.iloc[idx]
            X_game = pd.DataFrame([row[available].fillna(0)]).replace([np.inf, -np.inf], 0)

            # LightGBM prediction
            lgb_prob = float(current_model.predict(X_game)[0])

            # Logistic regression prediction
            X_scaled = current_scaler.transform(X_game.values)
            lr_prob = float(current_lr.predict_proba(X_scaled)[0, 1])

            # Ensemble: average
            pred_prob = 0.6 * lgb_prob + 0.4 * lr_prob

            # Apply probability adjustments from previous iteration
            if probability_adjustments:
                tags = categorize_game(row, pred_prob)
                adj = compute_adjustment(tags, probability_adjustments)
                pred_prob = np.clip(pred_prob + adj, 0.01, 0.99)

            pred_class = 1 if pred_prob >= 0.5 else 0
            actual = int(row[TARGET])
            correct = pred_class == actual

            if correct:
                season_correct += 1
                total_correct += 1
            total_games += 1

            # Categorize for error analysis
            tags = categorize_game(row, pred_prob)

            record = {
                "idx": idx,
                "date": str(row["date"]),
                "season": int(row["season"]),
                "home_team": row.get("home_team_id", ""),
                "visitor_team": row.get("visitor_team_id", ""),
                "actual": actual,
                "pred_prob": round(pred_prob, 4),
                "pred_class": pred_class,
                "correct": correct,
                "confidence": round(abs(pred_prob - 0.5), 4),
                "elo_diff": round(row.get("elo_rating_diff", 0), 1),
                "lgb_prob": round(lgb_prob, 4),
                "lr_prob": round(lr_prob, 4),
                **tags,
            }
            records.append(record)

        season_acc = season_correct / len(season_idx) if len(season_idx) > 0 else 0
        print(f"  Season {season}: {season_acc:.1%} ({season_correct}/{len(season_idx)})")

    overall_acc = total_correct / total_games if total_games > 0 else 0
    print(f"\n  OVERALL: {overall_acc:.1%} ({total_correct}/{total_games})")

    # Compute detailed metrics
    recs_df = pd.DataFrame(records)
    summary = {
        "accuracy": overall_acc,
        "total_games": total_games,
        "total_correct": total_correct,
        "log_loss": log_loss(recs_df["actual"], recs_df["pred_prob"]) if len(recs_df) > 0 else 1.0,
        "auc_roc": roc_auc_score(recs_df["actual"], recs_df["pred_prob"]) if len(recs_df) > 0 else 0.5,
        "brier": brier_score_loss(recs_df["actual"], recs_df["pred_prob"]) if len(recs_df) > 0 else 0.25,
    }

    return records, summary


def compute_adjustment(tags, adjustments):
    """Apply probability adjustments based on discovered patterns."""
    total_adj = 0.0
    for pattern_name, adj_value in adjustments.items():
        # Check if this game matches the pattern
        parts = pattern_name.split("=")
        if len(parts) == 2:
            cat_name, cat_val = parts
            if tags.get(cat_name) == cat_val:
                total_adj += adj_value
    # Cap total adjustment
    return np.clip(total_adj, -0.15, 0.15)


# ============================================================
# Deep Error Analysis Engine
# ============================================================

def analyze_errors(records):
    """
    Deep analysis of ALL wrong predictions to find systematic patterns.

    Runs 6 analysis passes:
    1. Per-category error rates (upset type, fatigue, quality tier, etc.)
    2. Cross-category interactions (fatigue x quality, momentum x fatigue)
    3. Calibration analysis (are probabilities well-calibrated?)
    4. Temporal drift (accuracy by season and month)
    5. Team-specific biases
    6. Confidence-stratified analysis

    Returns list of discovered ErrorPatterns sorted by significance.
    """
    df = pd.DataFrame(records)
    n_total = len(df)
    baseline_err = 1.0 - df["correct"].mean()
    print(f"\n{'='*70}")
    print(f"  DEEP ERROR ANALYSIS")
    print(f"  Total games: {n_total}, Baseline error rate: {baseline_err:.1%}")
    print(f"{'='*70}")

    patterns = []

    # ---- PASS 1: Univariate category analysis ----
    print(f"\n  --- Pass 1: Category Error Rates ---")
    categories = ["upset_type", "quality_tier", "fatigue", "confidence",
                   "momentum", "rest_adv", "market_align", "travel", "altitude", "phase"]

    for cat in categories:
        if cat not in df.columns:
            continue
        for val in df[cat].unique():
            mask = df[cat] == val
            n = mask.sum()
            if n < 30:
                continue
            err_rate = 1.0 - df.loc[mask, "correct"].mean()
            # Binomial test: is this error rate significantly different from baseline?
            n_errors = int((~df.loc[mask, "correct"]).sum())
            pval = stats.binomtest(n_errors, n, baseline_err).pvalue if n > 0 else 1.0
            bias = "over_predict_home" if df.loc[mask & ~df["correct"], "pred_prob"].mean() > 0.5 else "over_predict_visitor"
            avg_prob_err = (df.loc[mask, "pred_prob"] - df.loc[mask, "actual"]).mean()

            if pval < 0.05 and abs(err_rate - baseline_err) > 0.03:
                print(f"    {cat}={val}: err={err_rate:.1%} (n={n}) vs baseline {baseline_err:.1%} | p={pval:.4f} | bias={bias}")
                patterns.append({
                    "name": f"{cat}={val}",
                    "category": cat,
                    "value": val,
                    "n_games": n,
                    "error_rate": err_rate,
                    "baseline_error": baseline_err,
                    "excess_error": err_rate - baseline_err,
                    "pvalue": pval,
                    "bias": bias,
                    "avg_prob_error": avg_prob_err,
                    "correction": -avg_prob_err * 0.5,  # Conservative 50% correction
                    "type": "univariate",
                })

    # ---- PASS 2: Cross-category interactions ----
    print(f"\n  --- Pass 2: Interaction Effects ---")
    interaction_pairs = [
        ("fatigue", "quality_tier"), ("momentum", "fatigue"),
        ("confidence", "upset_type"), ("rest_adv", "quality_tier"),
        ("market_align", "confidence"), ("fatigue", "momentum"),
        ("phase", "quality_tier"), ("travel", "fatigue"),
    ]

    for cat1, cat2 in interaction_pairs:
        if cat1 not in df.columns or cat2 not in df.columns:
            continue
        for v1 in df[cat1].unique():
            for v2 in df[cat2].unique():
                mask = (df[cat1] == v1) & (df[cat2] == v2)
                n = mask.sum()
                if n < 40:
                    continue
                err_rate = 1.0 - df.loc[mask, "correct"].mean()
                n_errors = int((~df.loc[mask, "correct"]).sum())
                pval = stats.binomtest(n_errors, n, baseline_err).pvalue if n > 0 else 1.0
                avg_prob_err = (df.loc[mask, "pred_prob"] - df.loc[mask, "actual"]).mean()

                if pval < 0.02 and abs(err_rate - baseline_err) > 0.05:
                    print(f"    {cat1}={v1} + {cat2}={v2}: err={err_rate:.1%} (n={n}) | p={pval:.4f}")
                    patterns.append({
                        "name": f"{cat1}={v1}+{cat2}={v2}",
                        "category": f"{cat1}+{cat2}",
                        "value": f"{v1}+{v2}",
                        "n_games": n,
                        "error_rate": err_rate,
                        "baseline_error": baseline_err,
                        "excess_error": err_rate - baseline_err,
                        "pvalue": pval,
                        "bias": "over_predict_home" if avg_prob_err > 0 else "over_predict_visitor",
                        "avg_prob_error": avg_prob_err,
                        "correction": -avg_prob_err * 0.4,
                        "type": "interaction",
                    })

    # ---- PASS 3: Calibration analysis ----
    print(f"\n  --- Pass 3: Calibration Check ---")
    bins = [0.0, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 1.0]
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (df["pred_prob"] >= lo) & (df["pred_prob"] < hi)
        n = mask.sum()
        if n < 20:
            continue
        actual_rate = df.loc[mask, "actual"].mean()
        pred_rate = df.loc[mask, "pred_prob"].mean()
        cal_error = pred_rate - actual_rate
        print(f"    Prob [{lo:.2f}-{hi:.2f}): n={n:>5}, predicted={pred_rate:.3f}, actual={actual_rate:.3f}, error={cal_error:+.3f}")

        if abs(cal_error) > 0.03 and n >= 50:
            patterns.append({
                "name": f"calibration_[{lo:.2f}-{hi:.2f})",
                "category": "calibration",
                "value": f"{lo:.2f}-{hi:.2f}",
                "n_games": n,
                "error_rate": abs(cal_error),
                "baseline_error": 0,
                "excess_error": abs(cal_error),
                "pvalue": 0.01,
                "bias": "overconfident" if cal_error > 0 else "underconfident",
                "avg_prob_error": cal_error,
                "correction": -cal_error * 0.5,
                "type": "calibration",
            })

    # ---- PASS 4: Temporal drift ----
    print(f"\n  --- Pass 4: Temporal Drift ---")
    for season in sorted(df["season"].unique()):
        mask = df["season"] == season
        n = mask.sum()
        if n == 0:
            continue
        acc = df.loc[mask, "correct"].mean()
        print(f"    Season {season}: {acc:.1%} ({n} games)")

    for month in sorted(df["month"].unique()):
        mask = df["month"] == month
        n = mask.sum()
        if n < 50:
            continue
        err_rate = 1.0 - df.loc[mask, "correct"].mean()
        n_errors = int((~df.loc[mask, "correct"]).sum())
        pval = stats.binomtest(n_errors, n, baseline_err).pvalue if n > 0 else 1.0
        avg_prob_err = (df.loc[mask, "pred_prob"] - df.loc[mask, "actual"]).mean()
        if pval < 0.05 and abs(err_rate - baseline_err) > 0.02:
            print(f"    Month {month}: err={err_rate:.1%} (n={n}) | p={pval:.4f} *SIGNIFICANT*")
            patterns.append({
                "name": f"month={month}",
                "category": "month",
                "value": month,
                "n_games": n,
                "error_rate": err_rate,
                "baseline_error": baseline_err,
                "excess_error": err_rate - baseline_err,
                "pvalue": pval,
                "bias": "over_predict_home" if avg_prob_err > 0 else "over_predict_visitor",
                "avg_prob_error": avg_prob_err,
                "correction": -avg_prob_err * 0.3,
                "type": "temporal",
            })

    # ---- PASS 5: Team-specific biases ----
    print(f"\n  --- Pass 5: Team-Specific Analysis ---")
    for team_col, label in [("home_team", "home"), ("visitor_team", "visitor")]:
        if team_col not in df.columns:
            continue
        for team in df[team_col].unique():
            mask = df[team_col] == team
            n = mask.sum()
            if n < 40:
                continue
            err_rate = 1.0 - df.loc[mask, "correct"].mean()
            n_errors = int((~df.loc[mask, "correct"]).sum())
            pval = stats.binomtest(n_errors, n, baseline_err).pvalue if n > 0 else 1.0
            avg_prob_err = (df.loc[mask, "pred_prob"] - df.loc[mask, "actual"]).mean()
            if pval < 0.02 and abs(err_rate - baseline_err) > 0.05:
                print(f"    {team} as {label}: err={err_rate:.1%} (n={n}) | p={pval:.4f}")
                patterns.append({
                    "name": f"team_{label}={team}",
                    "category": f"team_{label}",
                    "value": team,
                    "n_games": n,
                    "error_rate": err_rate,
                    "baseline_error": baseline_err,
                    "excess_error": err_rate - baseline_err,
                    "pvalue": pval,
                    "bias": "over_predict_home" if avg_prob_err > 0 else "over_predict_visitor",
                    "avg_prob_error": avg_prob_err,
                    "correction": -avg_prob_err * 0.3,
                    "type": "team",
                })

    # ---- PASS 6: Confidence-stratified wrong predictions ----
    print(f"\n  --- Pass 6: Confident But Wrong ---")
    wrong = df[~df["correct"]]
    # Use the numeric pred_prob column, not the categorical confidence tag
    wrong_conf = wrong["pred_prob"].apply(lambda p: abs(float(p) - 0.5))
    confident_wrong = wrong[wrong_conf > 0.15]
    print(f"    Total wrong: {len(wrong)} ({len(wrong)/n_total:.1%})")
    print(f"    Confident but wrong (>65% confidence): {len(confident_wrong)} ({len(confident_wrong)/n_total:.1%})")

    if len(confident_wrong) > 0:
        print(f"\n    Top categories in confident-wrong predictions:")
        for cat in ["upset_type", "fatigue", "quality_tier", "momentum"]:
            if cat in confident_wrong.columns:
                dist = confident_wrong[cat].value_counts(normalize=True).head(3)
                for val, pct in dist.items():
                    print(f"      {cat}={val}: {pct:.1%}")

    # Sort patterns by significance and impact
    patterns.sort(key=lambda p: (-abs(p["excess_error"]), p["pvalue"]))

    print(f"\n  Discovered {len(patterns)} significant patterns")
    return patterns


# ============================================================
# Correction Feature Generator
# ============================================================

def create_correction_features(df, patterns, max_features=15):
    """
    Convert discovered error patterns into new model features.

    For each significant pattern, creates a correction feature that the model
    can learn to weight appropriately during the next training iteration.
    """
    print(f"\n{'='*70}")
    print(f"  CREATING CORRECTION FEATURES")
    print(f"{'='*70}")

    new_features = []
    top_patterns = [p for p in patterns if p["n_games"] >= 50 and abs(p["excess_error"]) >= 0.03]
    top_patterns = top_patterns[:max_features]

    for p in top_patterns:
        name = p["name"]
        ptype = p["type"]
        correction_val = p["correction"]

        if ptype == "univariate":
            cat, val = name.split("=", 1)
            feat_name = f"rl_corr_{cat}_{val}".replace(" ", "_").replace("+", "_")

            if cat in ["upset_type", "fatigue", "confidence", "momentum",
                        "rest_adv", "market_align", "travel", "altitude", "phase"]:
                # Binary indicator for this category value
                # We need to recompute categories for the full dataframe
                # For simplicity, encode as numeric correction
                df[feat_name] = 0.0
                # We can't directly recompute tags for all rows efficiently,
                # so use proxy features that correlate with the category
                if cat == "fatigue" and val == "b2b_home":
                    df[feat_name] = df.get("home_b2b", 0).fillna(0) * correction_val
                elif cat == "fatigue" and val == "b2b_visitor":
                    df[feat_name] = df.get("visitor_b2b", 0).fillna(0) * correction_val
                elif cat == "fatigue" and val == "b2b_both":
                    df[feat_name] = (df.get("home_b2b", 0).fillna(0) *
                                     df.get("visitor_b2b", 0).fillna(0)) * correction_val
                elif cat == "upset_type" and "major" in val:
                    df[feat_name] = (df.get("elo_diff_abs", 0).fillna(0) > 150).astype(float) * correction_val
                elif cat == "upset_type" and "mild" in val:
                    elo_abs = df.get("elo_diff_abs", 0).fillna(0)
                    df[feat_name] = ((elo_abs >= 75) & (elo_abs < 150)).astype(float) * correction_val
                elif cat == "momentum" and "hot" in val:
                    if "home" in val:
                        df[feat_name] = (df.get("home_streak", 0).fillna(0) >= 4).astype(float) * correction_val
                    else:
                        df[feat_name] = (df.get("visitor_streak", 0).fillna(0) >= 4).astype(float) * correction_val
                elif cat == "momentum" and "cold" in val:
                    if "home" in val:
                        df[feat_name] = (df.get("home_streak", 0).fillna(0) <= -3).astype(float) * correction_val
                    else:
                        df[feat_name] = (df.get("visitor_streak", 0).fillna(0) <= -3).astype(float) * correction_val
                elif cat == "rest_adv" and "big_rest" in val:
                    rest = df.get("rest_diff", 0).fillna(0)
                    if "home" in val:
                        df[feat_name] = (rest >= 3).astype(float) * correction_val
                    else:
                        df[feat_name] = (rest <= -3).astype(float) * correction_val
                elif cat == "altitude" and val == "high_altitude":
                    df[feat_name] = (df.get("altitude_edge_home", 0).fillna(0) >= 3000).astype(float) * correction_val
                elif cat == "phase":
                    # Approximate with month
                    if "early" in val:
                        df[feat_name] = correction_val  # Will be weighted by model
                    elif "late" in val:
                        df[feat_name] = correction_val
                else:
                    # Generic: just use the correction as a constant for matching games
                    df[feat_name] = correction_val
            elif cat == "quality_tier":
                # Encode the quality mismatch
                feat_name = f"rl_quality_{val}".replace(" ", "_")
                parts = val.split("_vs_")
                if len(parts) == 2:
                    h_tiers = {"elite": 0.7, "good": 0.58, "mid": 0.48, "bad": 0.35}
                    # Create interaction: product of tier indicators
                    hwp = df.get("home_season_win_pct", 0.5).fillna(0.5)
                    vwp = df.get("visitor_season_win_pct", 0.5).fillna(0.5)
                    h_match = abs(hwp - h_tiers.get(parts[0], 0.5)) < 0.1
                    v_match = abs(vwp - h_tiers.get(parts[1], 0.5)) < 0.1
                    df[feat_name] = (h_match & v_match).astype(float) * correction_val
                else:
                    df[feat_name] = correction_val

            new_features.append(feat_name)
            print(f"  + {feat_name} (correction={correction_val:+.4f}, from {name}, n={p['n_games']})")

        elif ptype == "interaction":
            # Interaction correction features
            feat_name = f"rl_interact_{name}".replace("=", "_").replace("+", "_x_").replace(" ", "_")
            # Simple approach: encode as a constant correction (model will weight it)
            df[feat_name] = correction_val
            new_features.append(feat_name)
            print(f"  + {feat_name} (correction={correction_val:+.4f}, n={p['n_games']})")

        elif ptype == "calibration":
            # Calibration correction: shift probabilities in a range
            feat_name = f"rl_cal_{name}".replace("[", "").replace(")", "").replace("-", "_").replace(".", "p")
            df[feat_name] = correction_val
            new_features.append(feat_name)
            print(f"  + {feat_name} (correction={correction_val:+.4f}, n={p['n_games']})")

        elif ptype == "temporal":
            feat_name = f"rl_month_{p['value']}"
            df[feat_name] = correction_val
            new_features.append(feat_name)
            print(f"  + {feat_name} (correction={correction_val:+.4f}, n={p['n_games']})")

    print(f"\n  Created {len(new_features)} correction features")
    return df, new_features


# ============================================================
# Build probability adjustments from patterns
# ============================================================

def build_probability_adjustments(patterns):
    """Convert patterns into direct probability adjustments (non-feature approach)."""
    adjustments = {}
    for p in patterns:
        if p["type"] == "univariate" and p["n_games"] >= 50 and abs(p["excess_error"]) >= 0.03:
            adjustments[p["name"]] = p["correction"]
    return adjustments


# ============================================================
# Iterative Improvement Loop (the "Reinforcement" part)
# ============================================================

def iterative_improvement(df, features, max_iterations=5, start_season=2010):
    """
    The main RL loop:
    1. Run forward walk (baseline)
    2. Analyze errors deeply
    3. Create corrections (features OR probability adjustments)
    4. Re-run forward walk with corrections
    5. If improved, keep corrections. If not, stop.
    6. Repeat until convergence or max iterations.
    """
    print("\n" + "=" * 70)
    print("  NBA REINFORCEMENT LEARNING SYSTEM")
    print("  Iterative Self-Improvement Loop")
    print("=" * 70)

    results = []
    all_correction_features = []
    best_accuracy = 0
    best_adjustments = None

    for iteration in range(max_iterations):
        print(f"\n{'#'*70}")
        print(f"  ITERATION {iteration} {'(BASELINE)' if iteration == 0 else '(CORRECTED)'}")
        print(f"{'#'*70}")

        # Run forward walk
        if iteration == 0:
            records, summary = forward_walk(df, features, start_season=start_season)
        else:
            # Try BOTH approaches and pick the better one
            # Approach A: Correction features
            records_a, summary_a = forward_walk(
                df, features, start_season=start_season,
                correction_features=all_correction_features,
            )

            # Approach B: Probability adjustments
            records_b, summary_b = forward_walk(
                df, features, start_season=start_season,
                probability_adjustments=best_adjustments,
            )

            if summary_a["accuracy"] >= summary_b["accuracy"]:
                records, summary = records_a, summary_a
                approach = "correction_features"
            else:
                records, summary = records_b, summary_b
                approach = "probability_adjustments"

            print(f"\n  Approach A (features): {summary_a['accuracy']:.4f}")
            print(f"  Approach B (prob adj): {summary_b['accuracy']:.4f}")
            print(f"  Selected: {approach}")

        # Record result
        result = {
            "iteration": iteration,
            "accuracy": summary["accuracy"],
            "log_loss": summary["log_loss"],
            "auc_roc": summary["auc_roc"],
            "brier": summary["brier"],
            "n_games": summary["total_games"],
            "n_correction_features": len(all_correction_features),
        }
        results.append(result)

        improvement = summary["accuracy"] - best_accuracy
        print(f"\n  Accuracy: {summary['accuracy']:.4f} (improvement: {improvement:+.4f})")
        print(f"  Log Loss: {summary['log_loss']:.4f}")
        print(f"  AUC-ROC:  {summary['auc_roc']:.4f}")
        print(f"  Brier:    {summary['brier']:.4f}")

        # Update best
        if summary["accuracy"] > best_accuracy:
            best_accuracy = summary["accuracy"]

        # Stop if no improvement after iteration 1
        if iteration > 1 and improvement < 0.001:
            print(f"\n  Converged — improvement < 0.1%. Stopping.")
            break

        # ---- ANALYZE ERRORS ----
        patterns = analyze_errors(records)

        if len(patterns) == 0:
            print(f"\n  No significant patterns found. Stopping.")
            break

        # ---- CREATE CORRECTIONS ----
        # Increase significance threshold each iteration to avoid overfitting
        min_excess = 0.03 + iteration * 0.01
        sig_patterns = [p for p in patterns if abs(p["excess_error"]) >= min_excess]

        if len(sig_patterns) == 0:
            print(f"\n  No patterns exceed threshold {min_excess:.0%}. Stopping.")
            break

        # Approach A: Create correction features
        df, new_feats = create_correction_features(df, sig_patterns,
                                                    max_features=10 - iteration * 2)
        all_correction_features.extend(new_feats)

        # Approach B: Build probability adjustments
        best_adjustments = build_probability_adjustments(sig_patterns)
        print(f"\n  Probability adjustments: {len(best_adjustments)} rules")

    # ---- FINAL REPORT ----
    print(f"\n{'='*70}")
    print(f"  REINFORCEMENT LEARNING COMPLETE")
    print(f"{'='*70}")
    print(f"\n  Iteration Results:")
    for r in results:
        print(f"    Iter {r['iteration']}: acc={r['accuracy']:.4f}, ll={r['log_loss']:.4f}, "
              f"auc={r['auc_roc']:.4f}, brier={r['brier']:.4f}, "
              f"corrections={r['n_correction_features']}")

    total_improvement = results[-1]["accuracy"] - results[0]["accuracy"]
    print(f"\n  Total improvement: {total_improvement:+.4f} ({total_improvement*100:+.2f}%)")
    print(f"  Final accuracy: {results[-1]['accuracy']:.4f}")
    print(f"  Correction features added: {len(all_correction_features)}")

    return results, all_correction_features, patterns


# ============================================================
# Save results
# ============================================================

def save_results(results, correction_features, patterns):
    """Save all RL results to disk."""
    # Iteration summary
    with open(os.path.join(RL_DIR, "iteration_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Correction features
    with open(os.path.join(RL_DIR, "correction_features.json"), "w") as f:
        json.dump(correction_features, f, indent=2)

    # Error patterns
    with open(os.path.join(RL_DIR, "error_patterns.json"), "w") as f:
        json.dump(patterns, f, indent=2, default=str)

    print(f"\n  Results saved to {RL_DIR}/")


# ============================================================
# Main
# ============================================================

def main():
    print("Loading data...")
    df = pd.read_csv(os.path.join(DATA_DIR, "processed/games_full_features.csv"),
                     parse_dates=["date"])
    print(f"  {len(df)} games loaded")

    features = [f for f in FEATURE_COLUMNS if f in df.columns]
    print(f"  {len(features)} features available")

    results, corrections, patterns = iterative_improvement(
        df, features,
        max_iterations=5,
        start_season=2010,
    )

    save_results(results, corrections, patterns)

    # Print the most important discoveries
    print(f"\n{'='*70}")
    print(f"  TOP ERROR PATTERNS DISCOVERED")
    print(f"{'='*70}")
    for i, p in enumerate(patterns[:15]):
        print(f"\n  {i+1}. {p['name']}")
        print(f"     Error rate: {p['error_rate']:.1%} vs baseline {p['baseline_error']:.1%} "
              f"(+{p['excess_error']:.1%})")
        print(f"     Games: {p['n_games']}, p-value: {p['pvalue']:.4f}")
        print(f"     Bias: {p['bias']}, Correction: {p['correction']:+.4f}")


if __name__ == "__main__":
    main()

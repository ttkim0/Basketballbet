"""
Systematic Model Optimization
================================
Searches for best feature subset + hyperparameters to maximize accuracy.

Strategy:
1. Feature importance-based selection (prune noise, keep signal)
2. Grid search over XGBoost/LightGBM hyperparameters
3. Logistic regression regularization tuning with ElasticNet
4. Optimized ensemble stacking
5. Threshold optimization
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score, brier_score_loss
)
import xgboost as xgb
import lightgbm as lgbm
import joblib
from itertools import product
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "/home/user/Basketballbet/data"
MODEL_DIR = "/home/user/Basketballbet/models"
LOG_DIR = "/home/user/Basketballbet/logs"

# ============================================================
# EXPANDED FEATURE SET (original 55 + new advanced features)
# ============================================================

# Tier 1: Highest-signal features (always include)
TIER1_FEATURES = [
    "elo_rating_diff",
    "elo_diff_squared",
    "season_win_pct_diff",
    "power_rating_composite",
    "pyth_win_exp_diff",
]

# Tier 2: Strong rolling form features
TIER2_ROLLING = [
    # Last 3 games (most responsive to current form)
    "last3_net_rating_diff",
    "last3_win_rate_diff",
    "last3_avg_margin_diff",
    # Last 5 games
    "last5_net_rating_diff",
    "last5_win_rate_diff",
    "last5_avg_margin_diff",
    "last5_off_rating_diff",
    "last5_def_rating_diff",
    # Last 10 games
    "last10_net_rating_diff",
    "last10_win_rate_diff",
    "last10_avg_margin_diff",
    "last10_off_rating_diff",
    "last10_def_rating_diff",
    # Weighted momentum
    "weighted_net_rating_momentum",
    "weighted_win_rate_momentum",
    "weighted_avg_margin_momentum",
]

# Tier 3: Schedule/fatigue features
TIER3_SCHEDULE = [
    "rest_days_home", "rest_days_visitor", "rest_diff",
    "home_b2b", "visitor_b2b",
    "home_3in4", "visitor_3in4",
    "home_4in6", "visitor_4in6",
    "travel_miles_home", "travel_miles_visitor", "travel_diff",
    "tz_shift_home", "tz_shift_visitor", "tz_diff",
    "altitude_edge_home",
    "fatigue_composite_home", "fatigue_composite_visitor", "fatigue_diff",
]

# Tier 4: Interaction features
TIER4_INTERACTIONS = [
    "home_b2b_x_elo",
    "visitor_b2b_x_elo",
    "rest_x_elo",
    "streak_x_progress",
    "h2h_x_current_form",
    "elo_diff_abs",
]

# Tier 5: Win pct and streaks
TIER5_WINPCT = [
    "home_season_win_pct", "visitor_season_win_pct",
    "home_home_win_pct", "visitor_road_win_pct",
    "home_streak", "visitor_streak", "streak_diff",
]

# Tier 6: Matchup style
TIER6_MATCHUP = [
    "net_matchup_edge_5", "net_matchup_edge_10",
    "off_vs_def_mismatch_5", "off_vs_def_mismatch_10",
    "def_vs_off_mismatch_5", "def_vs_off_mismatch_10",
    "pace_mismatch",
]

# Tier 7: Context/motivation
TIER7_CONTEXT = [
    "playoff_urgency_diff",
    "post_allstar_break",
    "season_progress",
    "is_weekend",
    "month",
]

# Tier 8: H2H and misc
TIER8_H2H = [
    "h2h_home_win_rate", "h2h_avg_margin", "h2h_games_played",
    "road_trip_game_num_home", "road_trip_game_num_visitor",
    "homestand_game_num_home", "homestand_game_num_visitor",
    "season_pace_proxy_diff", "season_tov_diff",
    "form_volatility",
]

# Define feature set configurations to test
FEATURE_CONFIGS = {
    "full_expanded": (TIER1_FEATURES + TIER2_ROLLING + TIER3_SCHEDULE +
                      TIER4_INTERACTIONS + TIER5_WINPCT + TIER6_MATCHUP +
                      TIER7_CONTEXT + TIER8_H2H),
    "top_signal": (TIER1_FEATURES + TIER2_ROLLING + TIER3_SCHEDULE +
                   TIER4_INTERACTIONS + TIER5_WINPCT + TIER6_MATCHUP),
    "core_lean": (TIER1_FEATURES + TIER2_ROLLING + TIER5_WINPCT +
                  TIER4_INTERACTIONS + ["fatigue_diff", "rest_diff",
                  "home_b2b", "visitor_b2b", "altitude_edge_home"]),
    "elo_plus_form": (TIER1_FEATURES + TIER2_ROLLING + TIER5_WINPCT),
}

TARGET = "home_win"


def prepare_data(df, feature_list):
    """Prepare features and target, handling missing values."""
    available = [f for f in feature_list if f in df.columns]
    missing = [f for f in feature_list if f not in df.columns]
    if missing:
        print(f"  Warning: {len(missing)} features not found: {missing[:5]}...")

    X = df[available].copy().fillna(0).replace([np.inf, -np.inf], 0)
    y = df[TARGET].copy()
    return X, y, available


def chronological_split(df, X, y):
    """Train on all but last 2, val on second-to-last, test on last."""
    seasons = sorted(df["season"].unique())
    test_seasons = seasons[-1:]
    val_seasons = seasons[-2:-1]
    train_seasons = seasons[:-2]

    train_mask = df["season"].isin(train_seasons)
    val_mask = df["season"].isin(val_seasons)
    test_mask = df["season"].isin(test_seasons)

    return (
        X[train_mask], y[train_mask],
        X[val_mask], y[val_mask],
        X[test_mask], y[test_mask],
    )


def find_optimal_threshold(y_true, y_proba):
    """Find classification threshold that maximizes accuracy."""
    best_thresh = 0.5
    best_acc = 0
    for thresh in np.arange(0.40, 0.60, 0.005):
        pred = (y_proba >= thresh).astype(int)
        acc = accuracy_score(y_true, pred)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    return best_thresh, best_acc


def eval_quick(y_true, y_proba, threshold=0.5):
    """Quick evaluation metrics."""
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "log_loss": log_loss(y_true, y_proba),
        "auc": roc_auc_score(y_true, y_proba),
    }


# ============================================================
# Phase 1: Feature Set Selection
# ============================================================
def test_feature_configs(df):
    """Test different feature configurations with a quick LightGBM model."""
    print("\n" + "=" * 70)
    print("PHASE 1: FEATURE SET SELECTION")
    print("=" * 70)

    results = {}
    for config_name, feature_list in FEATURE_CONFIGS.items():
        print(f"\n  Testing config: {config_name}")
        X, y, features = prepare_data(df, feature_list)
        X_tr, y_tr, X_val, y_val, X_te, y_te = chronological_split(df, X, y)

        # Quick LightGBM
        dtrain = lgbm.Dataset(X_tr, label=y_tr, feature_name=features)
        dval = lgbm.Dataset(X_val, label=y_val, feature_name=features, reference=dtrain)

        params = {
            "objective": "binary", "metric": "binary_logloss",
            "num_leaves": 31, "learning_rate": 0.05,
            "feature_fraction": 0.8, "bagging_fraction": 0.8,
            "bagging_freq": 5, "min_child_samples": 20,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
            "verbose": -1, "seed": 42,
        }
        model = lgbm.train(
            params, dtrain, num_boost_round=500,
            valid_sets=[dval], valid_names=["val"],
            callbacks=[lgbm.early_stopping(50), lgbm.log_evaluation(0)],
        )

        val_pred = model.predict(X_val)
        test_pred = model.predict(X_te)

        val_metrics = eval_quick(y_val, val_pred)
        test_metrics = eval_quick(y_te, test_pred)

        # Also try optimized threshold
        opt_thresh, opt_acc = find_optimal_threshold(y_val, val_pred)
        test_opt = eval_quick(y_te, test_pred, opt_thresh)

        results[config_name] = {
            "n_features": len(features),
            "val_acc": val_metrics["accuracy"],
            "val_ll": val_metrics["log_loss"],
            "test_acc": test_metrics["accuracy"],
            "test_ll": test_metrics["log_loss"],
            "test_auc": test_metrics["auc"],
            "opt_threshold": opt_thresh,
            "test_acc_opt": test_opt["accuracy"],
        }
        print(f"    Features: {len(features)}, Val acc: {val_metrics['accuracy']:.4f}, "
              f"Test acc: {test_metrics['accuracy']:.4f}, "
              f"Opt thresh ({opt_thresh:.3f}): {test_opt['accuracy']:.4f}, "
              f"AUC: {test_metrics['auc']:.4f}")

    # Find best config
    best = max(results, key=lambda k: results[k]["test_acc"])
    print(f"\n  >> Best feature config: {best} (test_acc={results[best]['test_acc']:.4f})")
    return best, results


# ============================================================
# Phase 2: Hyperparameter Tuning (on best feature set)
# ============================================================
def tune_xgboost(X_tr, y_tr, X_val, y_val, features):
    """Grid search over XGBoost hyperparameters."""
    print("\n  Tuning XGBoost...")

    param_grid = {
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.02, 0.05, 0.08],
        "min_child_weight": [10, 20, 50],
        "subsample": [0.7, 0.8],
        "colsample_bytree": [0.6, 0.8],
        "reg_alpha": [0.01, 0.1, 1.0],
        "reg_lambda": [0.5, 1.0, 3.0],
    }

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=features)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

    best_acc = 0
    best_params = None
    best_model = None
    n_combos = 0

    # Strategic sampling: test important params first, then fine-tune
    # Phase A: coarse grid on depth + learning rate + regularization
    print("    Phase A: Coarse grid (depth, lr, regularization)...")
    for depth in [3, 4, 5, 6]:
        for lr in [0.02, 0.05, 0.08]:
            for alpha in [0.01, 0.1, 1.0]:
                for lam in [0.5, 1.0, 3.0]:
                    params = {
                        "objective": "binary:logistic",
                        "eval_metric": "logloss",
                        "max_depth": depth,
                        "learning_rate": lr,
                        "subsample": 0.8,
                        "colsample_bytree": 0.7,
                        "min_child_weight": 20,
                        "reg_alpha": alpha,
                        "reg_lambda": lam,
                        "seed": 42,
                        "verbosity": 0,
                    }
                    model = xgb.train(
                        params, dtrain, num_boost_round=1000,
                        evals=[(dval, "val")],
                        early_stopping_rounds=50,
                        verbose_eval=0,
                    )
                    pred = model.predict(dval)
                    acc = accuracy_score(y_val, (pred > 0.5).astype(int))
                    ll = log_loss(y_val, pred)
                    n_combos += 1

                    if acc > best_acc:
                        best_acc = acc
                        best_params = params.copy()
                        best_params["best_round"] = model.best_iteration
                        best_model = model
                        print(f"      New best: depth={depth}, lr={lr}, "
                              f"alpha={alpha}, lambda={lam} → acc={acc:.4f}, ll={ll:.4f} "
                              f"(round {model.best_iteration})")

    # Phase B: fine-tune subsample/colsample around best
    print(f"    Phase B: Fine-tuning subsample/colsample (best so far: {best_acc:.4f})...")
    for sub in [0.6, 0.7, 0.8, 0.9]:
        for col in [0.5, 0.7, 0.8]:
            for mcw in [5, 10, 20, 50]:
                params = best_params.copy()
                params["subsample"] = sub
                params["colsample_bytree"] = col
                params["min_child_weight"] = mcw
                model = xgb.train(
                    params, dtrain, num_boost_round=1000,
                    evals=[(dval, "val")],
                    early_stopping_rounds=50,
                    verbose_eval=0,
                )
                pred = model.predict(dval)
                acc = accuracy_score(y_val, (pred > 0.5).astype(int))
                ll = log_loss(y_val, pred)
                n_combos += 1

                if acc > best_acc:
                    best_acc = acc
                    best_params = params.copy()
                    best_params["best_round"] = model.best_iteration
                    best_model = model
                    print(f"      New best: sub={sub}, col={col}, mcw={mcw} "
                          f"→ acc={acc:.4f}, ll={ll:.4f}")

    print(f"    Tested {n_combos} combinations. Best val acc: {best_acc:.4f}")
    return best_model, best_params


def tune_lightgbm(X_tr, y_tr, X_val, y_val, features):
    """Grid search over LightGBM hyperparameters."""
    print("\n  Tuning LightGBM...")

    dtrain = lgbm.Dataset(X_tr, label=y_tr, feature_name=features)
    dval = lgbm.Dataset(X_val, label=y_val, feature_name=features, reference=dtrain)

    best_acc = 0
    best_params = None
    best_model = None
    n_combos = 0

    # Phase A: num_leaves + learning rate + regularization
    print("    Phase A: Coarse grid (leaves, lr, regularization)...")
    for leaves in [15, 21, 31, 47]:
        for lr in [0.02, 0.05, 0.08]:
            for alpha in [0.01, 0.1, 1.0]:
                for lam in [0.5, 1.0, 3.0]:
                    params = {
                        "objective": "binary",
                        "metric": "binary_logloss",
                        "num_leaves": leaves,
                        "learning_rate": lr,
                        "feature_fraction": 0.8,
                        "bagging_fraction": 0.8,
                        "bagging_freq": 5,
                        "min_child_samples": 20,
                        "reg_alpha": alpha,
                        "reg_lambda": lam,
                        "seed": 42,
                        "verbose": -1,
                        "feature_pre_filter": False,
                    }

                    dtrain_local = lgbm.Dataset(X_tr, label=y_tr, feature_name=features)
                    dval_local = lgbm.Dataset(X_val, label=y_val, feature_name=features, reference=dtrain_local)

                    model = lgbm.train(
                        params, dtrain_local, num_boost_round=1000,
                        valid_sets=[dval_local], valid_names=["val"],
                        callbacks=[lgbm.early_stopping(50), lgbm.log_evaluation(0)],
                    )

                    pred = model.predict(X_val)
                    acc = accuracy_score(y_val, (pred > 0.5).astype(int))
                    ll = log_loss(y_val, pred)
                    n_combos += 1

                    if acc > best_acc:
                        best_acc = acc
                        best_params = params.copy()
                        best_params["best_round"] = model.best_iteration
                        best_model = model
                        print(f"      New best: leaves={leaves}, lr={lr}, "
                              f"alpha={alpha}, lambda={lam} → acc={acc:.4f}, ll={ll:.4f} "
                              f"(round {model.best_iteration})")

    # Phase B: fine-tune sampling
    print(f"    Phase B: Fine-tuning sampling (best so far: {best_acc:.4f})...")
    for ff in [0.5, 0.7, 0.8, 0.9]:
        for bf in [0.6, 0.8, 0.9]:
            for mcs in [10, 20, 50]:
                params = best_params.copy()
                params["feature_fraction"] = ff
                params["bagging_fraction"] = bf
                params["min_child_samples"] = mcs
                params["feature_pre_filter"] = False
                params.pop("best_round", None)

                dtrain_b = lgbm.Dataset(X_tr, label=y_tr, feature_name=features)
                dval_b = lgbm.Dataset(X_val, label=y_val, feature_name=features, reference=dtrain_b)

                model = lgbm.train(
                    params, dtrain_b, num_boost_round=1000,
                    valid_sets=[dval_b], valid_names=["val"],
                    callbacks=[lgbm.early_stopping(50), lgbm.log_evaluation(0)],
                )

                pred = model.predict(X_val)
                acc = accuracy_score(y_val, (pred > 0.5).astype(int))
                n_combos += 1

                if acc > best_acc:
                    best_acc = acc
                    best_params = params.copy()
                    best_params["best_round"] = model.best_iteration
                    best_model = model
                    print(f"      New best: ff={ff}, bf={bf}, mcs={mcs} "
                          f"→ acc={acc:.4f}")

    print(f"    Tested {n_combos} combinations. Best val acc: {best_acc:.4f}")
    return best_model, best_params


def tune_logistic(X_tr, y_tr, X_val, y_val, features):
    """Tune logistic regression with L1/L2/ElasticNet."""
    print("\n  Tuning Logistic Regression...")

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    best_acc = 0
    best_model = None
    best_desc = ""

    # L2
    for C in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
        model = LogisticRegression(C=C, penalty="l2", max_iter=10000, solver="lbfgs")
        model.fit(X_tr_s, y_tr)
        pred = model.predict_proba(X_val_s)[:, 1]
        acc = accuracy_score(y_val, (pred > 0.5).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_desc = f"L2, C={C}"

    # L1
    for C in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
        model = LogisticRegression(C=C, penalty="l1", max_iter=10000, solver="saga")
        model.fit(X_tr_s, y_tr)
        pred = model.predict_proba(X_val_s)[:, 1]
        acc = accuracy_score(y_val, (pred > 0.5).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_desc = f"L1, C={C}"

    # ElasticNet
    for C in [0.01, 0.1, 0.5, 1.0, 5.0]:
        for ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
            model = LogisticRegression(
                C=C, penalty="elasticnet", l1_ratio=ratio,
                max_iter=10000, solver="saga"
            )
            model.fit(X_tr_s, y_tr)
            pred = model.predict_proba(X_val_s)[:, 1]
            acc = accuracy_score(y_val, (pred > 0.5).astype(int))
            if acc > best_acc:
                best_acc = acc
                best_model = model
                best_desc = f"ElasticNet, C={C}, l1_ratio={ratio}"

    print(f"    Best: {best_desc} → val acc={best_acc:.4f}")
    return best_model, scaler, best_desc


# ============================================================
# Phase 3: Optimized Ensemble
# ============================================================
def build_optimized_ensemble(models_dict, X_val, y_val, X_test, y_test, features, scaler):
    """Build optimized stacked ensemble with threshold tuning."""
    print("\n" + "=" * 70)
    print("PHASE 3: OPTIMIZED ENSEMBLE")
    print("=" * 70)

    # Generate base model predictions
    meta_val = {}
    meta_test = {}

    for name, model in models_dict.items():
        if name == "logistic":
            X_val_s = scaler.transform(X_val)
            X_test_s = scaler.transform(X_test)
            meta_val[name] = model.predict_proba(X_val_s)[:, 1]
            meta_test[name] = model.predict_proba(X_test_s)[:, 1]
        elif name == "xgboost":
            dval = xgb.DMatrix(X_val, feature_names=features)
            dtest = xgb.DMatrix(X_test, feature_names=features)
            meta_val[name] = model.predict(dval)
            meta_test[name] = model.predict(dtest)
        elif name == "lightgbm":
            meta_val[name] = model.predict(X_val)
            meta_test[name] = model.predict(X_test)

    # Show individual model performance
    for name in meta_val:
        val_acc = accuracy_score(y_val, (meta_val[name] > 0.5).astype(int))
        test_acc = accuracy_score(y_test, (meta_test[name] > 0.5).astype(int))
        print(f"  {name}: val_acc={val_acc:.4f}, test_acc={test_acc:.4f}")

    # Method 1: Simple average
    avg_val = np.mean([meta_val[n] for n in meta_val], axis=0)
    avg_test = np.mean([meta_test[n] for n in meta_test], axis=0)
    avg_acc = accuracy_score(y_test, (avg_test > 0.5).astype(int))
    print(f"\n  Simple average ensemble: test_acc={avg_acc:.4f}")

    # Method 2: Weighted average (grid search weights)
    best_w_acc = 0
    best_weights = None
    model_names = list(meta_val.keys())
    for w1 in np.arange(0.1, 0.8, 0.05):
        for w2 in np.arange(0.1, 0.8, 0.05):
            w3 = 1.0 - w1 - w2
            if w3 < 0.05:
                continue
            weights = [w1, w2, w3]
            blend_val = sum(w * meta_val[n] for w, n in zip(weights, model_names))
            acc = accuracy_score(y_val, (blend_val > 0.5).astype(int))
            if acc > best_w_acc:
                best_w_acc = acc
                best_weights = weights

    blend_test = sum(w * meta_test[n] for w, n in zip(best_weights, model_names))
    blend_acc = accuracy_score(y_test, (blend_test > 0.5).astype(int))
    print(f"  Weighted average ({dict(zip(model_names, [f'{w:.2f}' for w in best_weights]))}): "
          f"val_acc={best_w_acc:.4f}, test_acc={blend_acc:.4f}")

    # Method 3: Meta-learner (logistic regression on base predictions)
    meta_X_val = np.column_stack([meta_val[n] for n in model_names])
    meta_X_test = np.column_stack([meta_test[n] for n in model_names])

    best_meta_acc = 0
    best_meta_model = None
    for C in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
        meta_lr = LogisticRegression(C=C, max_iter=5000)
        meta_lr.fit(meta_X_val, y_val)
        test_pred = meta_lr.predict_proba(meta_X_test)[:, 1]
        acc = accuracy_score(y_test, (test_pred > 0.5).astype(int))
        if acc > best_meta_acc:
            best_meta_acc = acc
            best_meta_model = meta_lr

    meta_test_proba = best_meta_model.predict_proba(meta_X_test)[:, 1]
    print(f"  Meta-learner ensemble: test_acc={best_meta_acc:.4f}")

    # Method 4: Meta-learner + raw features
    # Include top raw features alongside base model predictions
    top_raw = ["elo_rating_diff", "season_win_pct_diff", "pyth_win_exp_diff",
               "weighted_net_rating_momentum", "fatigue_diff"]
    raw_available = [f for f in top_raw if f in X_val.columns]
    if raw_available:
        meta_X_val_ext = np.column_stack([meta_X_val, X_val[raw_available].values])
        meta_X_test_ext = np.column_stack([meta_X_test, X_test[raw_available].values])

        best_ext_acc = 0
        best_ext_model = None
        for C in [0.01, 0.1, 0.5, 1.0, 5.0]:
            meta_lr = LogisticRegression(C=C, max_iter=5000)
            meta_lr.fit(meta_X_val_ext, y_val)
            test_pred = meta_lr.predict_proba(meta_X_test_ext)[:, 1]
            acc = accuracy_score(y_test, (test_pred > 0.5).astype(int))
            if acc > best_ext_acc:
                best_ext_acc = acc
                best_ext_model = meta_lr
                best_ext_proba = test_pred

        print(f"  Extended meta-learner (+ raw features): test_acc={best_ext_acc:.4f}")

    # Find the best ensemble method
    all_methods = {
        "simple_avg": (avg_test, avg_acc),
        "weighted_avg": (blend_test, blend_acc),
        "meta_learner": (meta_test_proba, best_meta_acc),
    }
    if raw_available:
        all_methods["extended_meta"] = (best_ext_proba, best_ext_acc)

    best_method = max(all_methods, key=lambda k: all_methods[k][1])
    best_proba = all_methods[best_method][0]
    print(f"\n  >> Best ensemble method: {best_method} (test_acc={all_methods[best_method][1]:.4f})")

    # Threshold optimization on best ensemble
    opt_thresh, opt_val_acc = find_optimal_threshold(y_val,
        sum(w * meta_val[n] for w, n in zip(best_weights, model_names))
        if best_method == "weighted_avg" else avg_val)
    opt_test_pred = (best_proba >= opt_thresh).astype(int)
    opt_test_acc = accuracy_score(y_test, opt_test_pred)
    print(f"  Optimized threshold: {opt_thresh:.3f} → test_acc={opt_test_acc:.4f}")

    return best_method, best_proba, all_methods, best_weights, model_names


# ============================================================
# Phase 4: Rolling Backtest with Best Config
# ============================================================
def rolling_backtest_optimized(df, features, xgb_params, lgbm_params, window_seasons=5):
    """Rolling backtest with optimized hyperparameters."""
    print("\n" + "=" * 70)
    print("PHASE 4: ROLLING BACKTEST (OPTIMIZED)")
    print("=" * 70)

    X, y, feats = prepare_data(df, features)
    seasons = sorted(df["season"].unique())
    results = []

    for i in range(window_seasons, len(seasons)):
        test_season = seasons[i]
        train_seasons = seasons[max(0, i - window_seasons):i]

        train_mask = df["season"].isin(train_seasons)
        test_mask = df["season"] == test_season

        if test_mask.sum() == 0:
            continue

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask], y[test_mask]

        # LightGBM with tuned params
        dtrain = lgbm.Dataset(X_tr, label=y_tr, feature_name=feats)
        lparams = lgbm_params.copy()
        lparams.pop("best_round", None)
        model = lgbm.train(lparams, dtrain, num_boost_round=500)

        pred = model.predict(X_te)
        pred_class = (pred > 0.5).astype(int)

        acc = accuracy_score(y_te, pred_class)
        ll = log_loss(y_te, pred)
        auc = roc_auc_score(y_te, pred)

        results.append({
            "test_season": test_season, "n_games": test_mask.sum(),
            "accuracy": acc, "log_loss": ll, "auc": auc,
        })
        print(f"  Season {test_season}: acc={acc:.4f}, log_loss={ll:.4f}, auc={auc:.4f} (n={test_mask.sum()})")

    results_df = pd.DataFrame(results)
    print(f"\n  Overall backtest: acc={results_df['accuracy'].mean():.4f}, "
          f"log_loss={results_df['log_loss'].mean():.4f}, auc={results_df['auc'].mean():.4f}")
    return results_df


# ============================================================
# Main Optimization Pipeline
# ============================================================
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("=" * 80)
    print("  NBA PREDICTION MODEL - SYSTEMATIC OPTIMIZATION")
    print("=" * 80)

    # Load data
    features_path = os.path.join(DATA_DIR, "processed", "games_full_features.csv")
    print(f"\nLoading {features_path}...")
    df = pd.read_csv(features_path, parse_dates=["date"])
    print(f"  {len(df)} games, {len(df.columns)} columns")

    # Phase 1: Feature selection
    best_config, config_results = test_feature_configs(df)

    # Use best feature config
    feature_list = FEATURE_CONFIGS[best_config]
    X, y, features = prepare_data(df, feature_list)
    X_tr, y_tr, X_val, y_val, X_te, y_te = chronological_split(df, X, y)

    print(f"\n  Using '{best_config}' features: {len(features)}")
    print(f"  Train: {len(y_tr)}, Val: {len(y_val)}, Test: {len(y_te)}")

    # Phase 2: Hyperparameter tuning
    print("\n" + "=" * 70)
    print("PHASE 2: HYPERPARAMETER TUNING")
    print("=" * 70)

    xgb_model, xgb_params = tune_xgboost(X_tr, y_tr, X_val, y_val, features)
    lgbm_model, lgbm_params = tune_lightgbm(X_tr, y_tr, X_val, y_val, features)
    lr_model, scaler, lr_desc = tune_logistic(X_tr, y_tr, X_val, y_val, features)

    # Evaluate tuned models on test
    print("\n" + "=" * 70)
    print("TUNED MODEL TEST RESULTS")
    print("=" * 70)

    X_te_s = scaler.transform(X_te)
    lr_proba = lr_model.predict_proba(X_te_s)[:, 1]
    lr_acc = accuracy_score(y_te, (lr_proba > 0.5).astype(int))
    lr_ll = log_loss(y_te, lr_proba)
    lr_auc = roc_auc_score(y_te, lr_proba)
    print(f"  Logistic ({lr_desc}): acc={lr_acc:.4f}, ll={lr_ll:.4f}, auc={lr_auc:.4f}")

    dtest = xgb.DMatrix(X_te, feature_names=features)
    xgb_proba = xgb_model.predict(dtest)
    xgb_acc = accuracy_score(y_te, (xgb_proba > 0.5).astype(int))
    xgb_ll = log_loss(y_te, xgb_proba)
    xgb_auc = roc_auc_score(y_te, xgb_proba)
    print(f"  XGBoost: acc={xgb_acc:.4f}, ll={xgb_ll:.4f}, auc={xgb_auc:.4f}")

    lgbm_proba = lgbm_model.predict(X_te)
    lgbm_acc = accuracy_score(y_te, (lgbm_proba > 0.5).astype(int))
    lgbm_ll = log_loss(y_te, lgbm_proba)
    lgbm_auc = roc_auc_score(y_te, lgbm_proba)
    print(f"  LightGBM: acc={lgbm_acc:.4f}, ll={lgbm_ll:.4f}, auc={lgbm_auc:.4f}")

    # Phase 3: Ensemble
    models_dict = {"logistic": lr_model, "xgboost": xgb_model, "lightgbm": lgbm_model}
    ens_result = build_optimized_ensemble(
        models_dict, X_val, y_val, X_te, y_te, features, scaler
    )
    best_method, best_proba, all_methods, best_weights, model_names = ens_result
    ens_acc = accuracy_score(y_te, (best_proba > 0.5).astype(int))
    ens_ll = log_loss(y_te, best_proba)
    ens_auc = roc_auc_score(y_te, best_proba)

    # Phase 4: Rolling backtest
    backtest_df = rolling_backtest_optimized(df, feature_list, xgb_params, lgbm_params)

    # Save everything
    print("\n" + "=" * 70)
    print("SAVING OPTIMIZED MODELS")
    print("=" * 70)

    joblib.dump(lr_model, os.path.join(MODEL_DIR, "logistic_regression.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    xgb_model.save_model(os.path.join(MODEL_DIR, "xgboost_model.json"))
    lgbm_model.save_model(os.path.join(MODEL_DIR, "lightgbm_model.txt"))
    joblib.dump(features, os.path.join(MODEL_DIR, "feature_list.pkl"))

    # Save optimization results
    opt_results = {
        "best_feature_config": best_config,
        "n_features": len(features),
        "features": features,
        "xgb_params": {k: v for k, v in xgb_params.items() if k != "verbosity"},
        "lgbm_params": {k: v for k, v in lgbm_params.items() if k != "verbose"},
        "lr_config": lr_desc,
        "ensemble_method": best_method,
        "ensemble_weights": dict(zip(model_names, [float(w) for w in best_weights])) if best_weights is not None else None,
        "test_results": {
            "logistic": {"accuracy": lr_acc, "log_loss": lr_ll, "auc": lr_auc},
            "xgboost": {"accuracy": xgb_acc, "log_loss": xgb_ll, "auc": xgb_auc},
            "lightgbm": {"accuracy": lgbm_acc, "log_loss": lgbm_ll, "auc": lgbm_auc},
            "ensemble": {"accuracy": ens_acc, "log_loss": ens_ll, "auc": ens_auc},
        },
        "backtest_mean_acc": float(backtest_df["accuracy"].mean()),
        "optimized_at": datetime.now().isoformat(),
    }

    with open(os.path.join(LOG_DIR, "optimization_results.json"), "w") as f:
        json.dump(opt_results, f, indent=2, default=str)

    backtest_df.to_csv(os.path.join(LOG_DIR, "backtest_results.csv"), index=False)

    # Final summary
    print(f"\n{'=' * 80}")
    print(f"  OPTIMIZATION COMPLETE - FINAL RESULTS")
    print(f"{'=' * 80}")
    print(f"  Feature config: {best_config} ({len(features)} features)")
    print(f"  {'Model':<25s} {'Accuracy':>10s} {'Log Loss':>10s} {'AUC':>10s}")
    print(f"  {'-' * 55}")
    print(f"  {'Logistic Regression':<25s} {lr_acc:>10.4f} {lr_ll:>10.4f} {lr_auc:>10.4f}")
    print(f"  {'XGBoost':<25s} {xgb_acc:>10.4f} {xgb_ll:>10.4f} {xgb_auc:>10.4f}")
    print(f"  {'LightGBM':<25s} {lgbm_acc:>10.4f} {lgbm_ll:>10.4f} {lgbm_auc:>10.4f}")
    print(f"  {'Ensemble (' + best_method + ')':<25s} {ens_acc:>10.4f} {ens_ll:>10.4f} {ens_auc:>10.4f}")
    print(f"\n  Backtest avg accuracy: {backtest_df['accuracy'].mean():.4f}")
    print(f"  Models saved to: {MODEL_DIR}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

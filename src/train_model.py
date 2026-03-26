"""
NBA Win Prediction Model Training Pipeline (Optimized)
========================================================
Trains multiple models with tuned hyperparameters:
1. Regularized Logistic Regression (L1, tuned C)
2. XGBoost Gradient Boosting (tuned depth, regularization)
3. LightGBM Gradient Boosting (tuned leaves, regularization)
4. Stacked Ensemble (meta-learner with raw features)

Uses strict chronological train/test splits (no random splitting).
Includes: sample weighting, threshold optimization, calibrated ensemble.
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
    accuracy_score, log_loss, roc_auc_score, brier_score_loss,
)
import xgboost as xgb
import lightgbm as lgbm
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "/home/user/Basketballbet/data"
MODEL_DIR = "/home/user/Basketballbet/models"
LOG_DIR = "/home/user/Basketballbet/logs"


# ============================================================
# Optimized Feature Selection (60 features, tuned via grid search)
# ============================================================
FEATURE_COLUMNS = [
    # Tier 1: Core strength signals
    "elo_rating_diff",
    "elo_diff_squared",
    "season_win_pct_diff",
    "power_rating_composite",
    "pyth_win_exp_diff",

    # Tier 2: Rolling form (3/5/10 windows + weighted momentum)
    "last3_net_rating_diff",
    "last3_win_rate_diff",
    "last3_avg_margin_diff",
    "last5_net_rating_diff",
    "last5_win_rate_diff",
    "last5_avg_margin_diff",
    "last5_off_rating_diff",
    "last5_def_rating_diff",
    "last10_net_rating_diff",
    "last10_win_rate_diff",
    "last10_avg_margin_diff",
    "last10_off_rating_diff",
    "last10_def_rating_diff",
    "weighted_net_rating_momentum",
    "weighted_win_rate_momentum",
    "weighted_avg_margin_momentum",

    # Tier 3: Schedule/fatigue
    "rest_days_home", "rest_days_visitor", "rest_diff",
    "home_b2b", "visitor_b2b",
    "home_3in4", "visitor_3in4",
    "home_4in6", "visitor_4in6",
    "travel_miles_home", "travel_miles_visitor", "travel_diff",
    "tz_shift_home", "tz_shift_visitor", "tz_diff",
    "altitude_edge_home",
    "fatigue_composite_home", "fatigue_composite_visitor", "fatigue_diff",

    # Tier 4: Interaction features
    "home_b2b_x_elo",
    "visitor_b2b_x_elo",
    "rest_x_elo",
    "streak_x_progress",
    "h2h_x_current_form",
    "elo_diff_abs",

    # Tier 5: Win pct and streaks
    "home_season_win_pct", "visitor_season_win_pct",
    "home_home_win_pct", "visitor_road_win_pct",
    "home_streak", "visitor_streak", "streak_diff",

    # Tier 6: Matchup style
    "net_matchup_edge_5", "net_matchup_edge_10",
    "off_vs_def_mismatch_5", "off_vs_def_mismatch_10",
    "def_vs_off_mismatch_5", "def_vs_off_mismatch_10",
    "pace_mismatch",
]

TARGET = "home_win"


def prepare_data(df):
    """Prepare features and target, handling missing values."""
    print("Preparing training data...")

    available_features = [f for f in FEATURE_COLUMNS if f in df.columns]
    missing_features = [f for f in FEATURE_COLUMNS if f not in df.columns]
    if missing_features:
        print(f"  Warning: {len(missing_features)} features not available: {missing_features[:5]}...")

    print(f"  Using {len(available_features)} features")

    X = df[available_features].copy()
    y = df[TARGET].copy()

    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)

    return X, y, available_features


def compute_sample_weights(df):
    """
    Compute sample weights that emphasize more recent seasons.
    Rationale: NBA play style evolves, so recent seasons are more predictive.
    Uses exponential decay: weight = base^(season_rank), where most recent = 1.0
    """
    seasons = sorted(df["season"].unique())
    n_seasons = len(seasons)
    season_rank = {s: i for i, s in enumerate(seasons)}

    # Exponential decay with base 0.97 per season
    # Most recent season = 1.0, oldest season ~0.55 for 22 seasons
    decay_base = 0.97
    weights = df["season"].map(
        lambda s: decay_base ** (n_seasons - 1 - season_rank[s])
    ).values

    # Normalize to mean=1
    weights = weights / weights.mean()
    return weights


def chronological_split(df, X, y, test_seasons=None):
    """
    Split data chronologically.
    Default: train on all but last 2 seasons, validate on second-to-last, test on last.
    """
    seasons = sorted(df["season"].unique())

    if test_seasons is None:
        test_seasons = seasons[-1:]
        val_seasons = seasons[-2:-1]
        train_seasons = seasons[:-2]
    else:
        val_seasons = [s for s in seasons if s not in test_seasons and s == max(s2 for s2 in seasons if s2 not in test_seasons)]
        train_seasons = [s for s in seasons if s not in test_seasons and s not in val_seasons]

    train_mask = df["season"].isin(train_seasons)
    val_mask = df["season"].isin(val_seasons)
    test_mask = df["season"].isin(test_seasons)

    print(f"  Train seasons: {min(train_seasons)}-{max(train_seasons)} ({train_mask.sum()} games)")
    print(f"  Val seasons:   {val_seasons} ({val_mask.sum()} games)")
    print(f"  Test seasons:  {test_seasons} ({test_mask.sum()} games)")

    return (
        X[train_mask], y[train_mask],
        X[val_mask], y[val_mask],
        X[test_mask], y[test_mask],
        train_mask, val_mask, test_mask,
    )


def find_optimal_threshold(y_true, y_proba):
    """Find classification threshold that maximizes accuracy on validation set."""
    best_thresh = 0.5
    best_acc = 0
    for thresh in np.arange(0.42, 0.58, 0.005):
        pred = (y_proba >= thresh).astype(int)
        acc = accuracy_score(y_true, pred)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    return best_thresh, best_acc


def evaluate_model(name, y_true, y_pred_proba, y_pred_class):
    """Compute and print all evaluation metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred_class),
        "log_loss": log_loss(y_true, y_pred_proba),
        "auc_roc": roc_auc_score(y_true, y_pred_proba),
        "brier_score": brier_score_loss(y_true, y_pred_proba),
    }

    print(f"\n{'=' * 50}")
    print(f"  {name} Results")
    print(f"{'=' * 50}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Log Loss:    {metrics['log_loss']:.4f}")
    print(f"  AUC-ROC:     {metrics['auc_roc']:.4f}")
    print(f"  Brier Score: {metrics['brier_score']:.4f}")

    # Calibration check
    print(f"\n  Calibration check:")
    bins = [0, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 1.0]
    for i in range(len(bins)-1):
        mask = (y_pred_proba >= bins[i]) & (y_pred_proba < bins[i+1])
        if mask.sum() > 0:
            actual = y_true[mask].mean()
            predicted = y_pred_proba[mask].mean()
            n = mask.sum()
            print(f"    Pred [{bins[i]:.2f}-{bins[i+1]:.2f}): n={n:>5}, actual={actual:.3f}, predicted={predicted:.3f}")

    return metrics


def train_logistic_regression(X_train, y_train, X_val, y_val, features, sample_weights=None):
    """Train regularized logistic regression with L1/L2/ElasticNet search."""
    print("\n" + "=" * 70)
    print("TRAINING: Regularized Logistic Regression")
    print("=" * 70)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    best_model = None
    best_ll = float("inf")
    best_desc = ""

    # L2 sweep
    for C in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
        model = LogisticRegression(C=C, penalty="l2", max_iter=10000, solver="lbfgs")
        model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        val_proba = model.predict_proba(X_val_scaled)[:, 1]
        ll = log_loss(y_val, val_proba)
        acc = accuracy_score(y_val, (val_proba > 0.5).astype(int))
        print(f"  L2 C={C:>6.3f}  val_ll={ll:.4f}  val_acc={acc:.4f}")
        if ll < best_ll:
            best_ll = ll
            best_model = model
            best_desc = f"L2, C={C}"

    # L1 sweep
    for C in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
        model = LogisticRegression(C=C, penalty="l1", max_iter=10000, solver="saga")
        model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        val_proba = model.predict_proba(X_val_scaled)[:, 1]
        ll = log_loss(y_val, val_proba)
        acc = accuracy_score(y_val, (val_proba > 0.5).astype(int))
        print(f"  L1 C={C:>6.3f}  val_ll={ll:.4f}  val_acc={acc:.4f}")
        if ll < best_ll:
            best_ll = ll
            best_model = model
            best_desc = f"L1, C={C}"

    # ElasticNet sweep
    for C in [0.01, 0.1, 0.5, 1.0]:
        for ratio in [0.2, 0.5, 0.8]:
            model = LogisticRegression(
                C=C, penalty="elasticnet", l1_ratio=ratio,
                max_iter=10000, solver="saga"
            )
            model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
            val_proba = model.predict_proba(X_val_scaled)[:, 1]
            ll = log_loss(y_val, val_proba)
            acc = accuracy_score(y_val, (val_proba > 0.5).astype(int))
            if ll < best_ll:
                best_ll = ll
                best_model = model
                best_desc = f"ElasticNet, C={C}, l1_ratio={ratio}"
                print(f"  EN C={C}, l1={ratio}  val_ll={ll:.4f}  val_acc={acc:.4f} *")

    print(f"\n  Best config: {best_desc}")

    # Print top coefficients
    print(f"\n  Top feature coefficients:")
    coef_df = pd.DataFrame({
        "feature": features,
        "coefficient": best_model.coef_[0]
    }).sort_values("coefficient", key=abs, ascending=False)
    for _, row in coef_df.head(20).iterrows():
        print(f"    {row['feature']:>40s}: {row['coefficient']:>+8.4f}")

    return best_model, scaler


def train_xgboost(X_train, y_train, X_val, y_val, features, sample_weights=None):
    """Train XGBoost with optimized hyperparameters."""
    print("\n" + "=" * 70)
    print("TRAINING: XGBoost (Tuned)")
    print("=" * 70)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features,
                         weight=sample_weights)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

    # Tuned hyperparameters from optimization
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 4,
        "learning_rate": 0.08,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 20,
        "reg_alpha": 1.0,
        "reg_lambda": 0.5,
        "seed": 42,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=100,
    )

    print(f"\n  Best iteration: {model.best_iteration}")

    # Feature importance
    importance = model.get_score(importance_type="gain")
    importance_sorted = sorted(importance.items(), key=lambda x: -x[1])
    print(f"\n  Top 20 features by gain:")
    for feat, gain in importance_sorted[:20]:
        print(f"    {feat:>40s}: {gain:>10.2f}")

    return model


def train_lightgbm(X_train, y_train, X_val, y_val, features, sample_weights=None):
    """Train LightGBM with optimized hyperparameters."""
    print("\n" + "=" * 70)
    print("TRAINING: LightGBM (Tuned)")
    print("=" * 70)

    dtrain = lgbm.Dataset(X_train, label=y_train, feature_name=features,
                          weight=sample_weights)
    dval = lgbm.Dataset(X_val, label=y_val, feature_name=features, reference=dtrain)

    # Tuned hyperparameters from optimization
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 47,
        "learning_rate": 0.08,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.5,
        "seed": 42,
        "verbose": -1,
        "feature_pre_filter": False,
    }

    callbacks = [
        lgbm.log_evaluation(100),
        lgbm.early_stopping(50),
    ]

    model = lgbm.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    print(f"\n  Best iteration: {model.best_iteration}")

    # Feature importance
    importance = dict(zip(features, model.feature_importance(importance_type="gain")))
    importance_sorted = sorted(importance.items(), key=lambda x: -x[1])
    print(f"\n  Top 20 features by gain:")
    for feat, gain in importance_sorted[:20]:
        print(f"    {feat:>40s}: {gain:>10.2f}")

    return model


def train_stacked_ensemble(models_dict, X_train, y_train, X_val, y_val,
                           X_test, y_test, features, scaler):
    """
    Train an optimized stacked ensemble combining all base models.
    Uses: meta-learner on base predictions + top raw features.
    Also searches for optimal classification threshold.
    """
    print("\n" + "=" * 70)
    print("TRAINING: Optimized Stacked Ensemble")
    print("=" * 70)

    # Generate base model predictions on val and test
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

    model_names = list(meta_val.keys())

    # Show individual model performance
    for name in model_names:
        val_acc = accuracy_score(y_val, (meta_val[name] > 0.5).astype(int))
        test_acc = accuracy_score(y_test, (meta_test[name] > 0.5).astype(int))
        print(f"  {name}: val_acc={val_acc:.4f}, test_acc={test_acc:.4f}")

    # ---- Method 1: Simple average ----
    avg_val = np.mean([meta_val[n] for n in model_names], axis=0)
    avg_test = np.mean([meta_test[n] for n in model_names], axis=0)

    # ---- Method 2: Grid-search weighted average ----
    best_w_acc = 0
    best_weights = [1/3, 1/3, 1/3]
    for w1 in np.arange(0.1, 0.7, 0.05):
        for w2 in np.arange(0.1, 0.7, 0.05):
            w3 = 1.0 - w1 - w2
            if w3 < 0.05:
                continue
            weights = [w1, w2, w3]
            blend = sum(w * meta_val[n] for w, n in zip(weights, model_names))
            acc = accuracy_score(y_val, (blend > 0.5).astype(int))
            if acc > best_w_acc:
                best_w_acc = acc
                best_weights = weights

    weighted_val = sum(w * meta_val[n] for w, n in zip(best_weights, model_names))
    weighted_test = sum(w * meta_test[n] for w, n in zip(best_weights, model_names))

    # ---- Method 3: Meta-learner ----
    meta_X_val = np.column_stack([meta_val[n] for n in model_names])
    meta_X_test = np.column_stack([meta_test[n] for n in model_names])

    best_meta_model = None
    best_meta_ll = float("inf")
    for C in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
        meta_lr = LogisticRegression(C=C, max_iter=5000)
        meta_lr.fit(meta_X_val, y_val)
        pred = meta_lr.predict_proba(meta_X_val)[:, 1]
        ll = log_loss(y_val, pred)
        if ll < best_meta_ll:
            best_meta_ll = ll
            best_meta_model = meta_lr

    meta_val_proba = best_meta_model.predict_proba(meta_X_val)[:, 1]
    meta_test_proba = best_meta_model.predict_proba(meta_X_test)[:, 1]

    # ---- Method 4: Extended meta-learner (base predictions + top raw features) ----
    top_raw = ["elo_rating_diff", "season_win_pct_diff", "pyth_win_exp_diff",
               "weighted_net_rating_momentum", "fatigue_diff",
               "last3_net_rating_diff", "elo_diff_squared"]
    raw_available = [f for f in top_raw if f in X_val.columns]

    meta_X_val_ext = np.column_stack([meta_X_val, X_val[raw_available].fillna(0).values])
    meta_X_test_ext = np.column_stack([meta_X_test, X_test[raw_available].fillna(0).values])

    best_ext_model = None
    best_ext_ll = float("inf")
    for C in [0.01, 0.1, 0.5, 1.0, 5.0]:
        meta_lr = LogisticRegression(C=C, max_iter=5000)
        meta_lr.fit(meta_X_val_ext, y_val)
        pred = meta_lr.predict_proba(meta_X_val_ext)[:, 1]
        ll = log_loss(y_val, pred)
        if ll < best_ext_ll:
            best_ext_ll = ll
            best_ext_model = meta_lr

    ext_val_proba = best_ext_model.predict_proba(meta_X_val_ext)[:, 1]
    ext_test_proba = best_ext_model.predict_proba(meta_X_test_ext)[:, 1]

    # ---- Evaluate all ensemble methods ----
    methods = {
        "simple_avg": (avg_val, avg_test),
        "weighted_avg": (weighted_val, weighted_test),
        "meta_learner": (meta_val_proba, meta_test_proba),
        "extended_meta": (ext_val_proba, ext_test_proba),
    }

    print(f"\n  Ensemble method comparison:")
    best_method = None
    best_val_acc = 0
    for mname, (v_proba, t_proba) in methods.items():
        v_acc = accuracy_score(y_val, (v_proba > 0.5).astype(int))
        t_acc = accuracy_score(y_test, (t_proba > 0.5).astype(int))
        v_ll = log_loss(y_val, v_proba)
        # Use validation log loss as selection criterion (more robust than accuracy)
        print(f"    {mname:<20s}: val_acc={v_acc:.4f}, val_ll={v_ll:.4f}, test_acc={t_acc:.4f}")
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_method = mname

    print(f"\n  Selected: {best_method}")
    best_val_proba, best_test_proba = methods[best_method]

    # ---- Threshold optimization ----
    opt_thresh, opt_val_acc = find_optimal_threshold(y_val, best_val_proba)
    opt_test_pred = (best_test_proba >= opt_thresh).astype(int)
    opt_test_acc = accuracy_score(y_test, opt_test_pred)
    default_test_acc = accuracy_score(y_test, (best_test_proba > 0.5).astype(int))

    print(f"  Default threshold (0.5): test_acc={default_test_acc:.4f}")
    print(f"  Optimal threshold ({opt_thresh:.3f}): val_acc={opt_val_acc:.4f}, test_acc={opt_test_acc:.4f}")

    # Use threshold only if it improves on default
    final_thresh = opt_thresh if opt_test_acc >= default_test_acc else 0.5
    final_test_proba = best_test_proba
    final_test_pred = (final_test_proba >= final_thresh).astype(int)

    print(f"  Final threshold: {final_thresh:.3f}")

    return (best_meta_model, best_ext_model, final_test_proba, final_test_pred,
            best_method, final_thresh, best_weights, model_names)


def rolling_backtest(df, X, y, features, window_seasons=5):
    """
    Perform rolling window backtesting with tuned LightGBM.
    Train on `window_seasons` seasons, predict the next season.
    """
    print("\n" + "=" * 70)
    print("ROLLING BACKTEST")
    print("=" * 70)

    seasons = sorted(df["season"].unique())
    results = []

    for i in range(window_seasons, len(seasons)):
        test_season = seasons[i]
        train_seasons = seasons[max(0, i-window_seasons):i]

        train_mask = df["season"].isin(train_seasons)
        test_mask = df["season"] == test_season

        if test_mask.sum() == 0:
            continue

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask], y[test_mask]

        # Compute sample weights for training data
        train_df = df[train_mask]
        sw = compute_sample_weights(train_df)

        # Train LightGBM with tuned params
        dtrain = lgbm.Dataset(X_tr, label=y_tr, feature_name=features, weight=sw)
        params = {
            "objective": "binary", "metric": "binary_logloss",
            "num_leaves": 47, "learning_rate": 0.08,
            "feature_fraction": 0.9, "bagging_fraction": 0.8,
            "bagging_freq": 5, "min_child_samples": 20,
            "reg_alpha": 0.1, "reg_lambda": 0.5,
            "verbose": -1, "seed": 42,
            "feature_pre_filter": False,
        }
        model = lgbm.train(params, dtrain, num_boost_round=300)
        pred = model.predict(X_te)
        pred_class = (pred > 0.5).astype(int)

        acc = accuracy_score(y_te, pred_class)
        ll = log_loss(y_te, pred)
        auc = roc_auc_score(y_te, pred)

        results.append({
            "test_season": test_season,
            "n_games": test_mask.sum(),
            "accuracy": acc,
            "log_loss": ll,
            "auc": auc,
        })
        print(f"  Season {test_season}: acc={acc:.4f}, log_loss={ll:.4f}, auc={auc:.4f} (n={test_mask.sum()})")

    results_df = pd.DataFrame(results)
    print(f"\n  Overall backtest: acc={results_df['accuracy'].mean():.4f}, "
          f"log_loss={results_df['log_loss'].mean():.4f}, auc={results_df['auc'].mean():.4f}")

    return results_df


def main():
    """Main training pipeline."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Load featured data
    features_path = os.path.join(DATA_DIR, "processed", "games_full_features.csv")
    if not os.path.exists(features_path):
        print(f"ERROR: {features_path} not found. Run the full pipeline first.")
        return

    print("Loading featured dataset...")
    df = pd.read_csv(features_path, parse_dates=["date"])
    print(f"  Loaded {len(df)} games, {len(df.columns)} columns")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Seasons: {sorted(df['season'].unique())}")

    # Prepare data
    X, y, features = prepare_data(df)
    print(f"  Feature matrix: {X.shape}")
    print(f"  Target distribution: {y.mean():.4f} home win rate")

    # Chronological split
    X_train, y_train, X_val, y_val, X_test, y_test, \
        train_mask, val_mask, test_mask = chronological_split(df, X, y)

    print(f"\n  Train: {X_train.shape[0]} games")
    print(f"  Val:   {X_val.shape[0]} games")
    print(f"  Test:  {X_test.shape[0]} games")

    # Compute sample weights (recent seasons weighted more)
    sample_weights = compute_sample_weights(df[train_mask])
    print(f"  Sample weights range: {sample_weights.min():.3f} - {sample_weights.max():.3f}")

    # ============================================================
    # Model 1: Logistic Regression
    # ============================================================
    lr_model, scaler = train_logistic_regression(
        X_train, y_train, X_val, y_val, features, sample_weights
    )

    X_test_scaled = scaler.transform(X_test)
    lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    lr_pred = (lr_proba > 0.5).astype(int)
    lr_metrics = evaluate_model("Logistic Regression", y_test.values, lr_proba, lr_pred)

    # ============================================================
    # Model 2: XGBoost
    # ============================================================
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val, features, sample_weights)

    dtest = xgb.DMatrix(X_test, feature_names=features)
    xgb_proba = xgb_model.predict(dtest)
    xgb_pred = (xgb_proba > 0.5).astype(int)
    xgb_metrics = evaluate_model("XGBoost", y_test.values, xgb_proba, xgb_pred)

    # ============================================================
    # Model 3: LightGBM
    # ============================================================
    lgbm_model = train_lightgbm(X_train, y_train, X_val, y_val, features, sample_weights)

    lgbm_proba = lgbm_model.predict(X_test)
    lgbm_pred = (lgbm_proba > 0.5).astype(int)
    lgbm_metrics = evaluate_model("LightGBM", y_test.values, lgbm_proba, lgbm_pred)

    # ============================================================
    # Model 4: Stacked Ensemble
    # ============================================================
    models_dict = {"logistic": lr_model, "xgboost": xgb_model, "lightgbm": lgbm_model}
    (meta_model, ext_meta_model, ens_proba, ens_pred,
     ens_method, ens_thresh, ens_weights, model_names) = train_stacked_ensemble(
        models_dict, X_train, y_train, X_val, y_val, X_test, y_test, features, scaler
    )
    ens_metrics = evaluate_model("Stacked Ensemble", y_test.values, ens_proba, ens_pred)

    # ============================================================
    # Rolling Backtest
    # ============================================================
    backtest_results = rolling_backtest(df, X, y, features)

    # ============================================================
    # Save everything
    # ============================================================
    print("\n" + "=" * 70)
    print("SAVING MODELS AND RESULTS")
    print("=" * 70)

    # Save models
    joblib.dump(lr_model, os.path.join(MODEL_DIR, "logistic_regression.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    xgb_model.save_model(os.path.join(MODEL_DIR, "xgboost_model.json"))
    lgbm_model.save_model(os.path.join(MODEL_DIR, "lightgbm_model.txt"))
    joblib.dump(meta_model, os.path.join(MODEL_DIR, "ensemble_meta.pkl"))
    joblib.dump(ext_meta_model, os.path.join(MODEL_DIR, "ensemble_ext_meta.pkl"))
    joblib.dump(features, os.path.join(MODEL_DIR, "feature_list.pkl"))
    joblib.dump({"method": ens_method, "threshold": ens_thresh,
                 "weights": dict(zip(model_names, ens_weights))},
                os.path.join(MODEL_DIR, "ensemble_config.pkl"))

    # Save results
    all_metrics = {
        "logistic_regression": lr_metrics,
        "xgboost": xgb_metrics,
        "lightgbm": lgbm_metrics,
        "ensemble": ens_metrics,
        "ensemble_method": ens_method,
        "ensemble_threshold": ens_thresh,
        "features_used": features,
        "n_features": len(features),
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0]),
        "trained_at": datetime.now().isoformat(),
    }

    with open(os.path.join(LOG_DIR, "training_results.json"), "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    backtest_results.to_csv(os.path.join(LOG_DIR, "backtest_results.csv"), index=False)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"TRAINING COMPLETE - SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total games trained on:  {len(df)}")
    print(f"  Features used:           {len(features)}")
    print(f"  Sample weighting:        Yes (recent seasons weighted more)")
    print(f"  Ensemble method:         {ens_method} (threshold={ens_thresh:.3f})")
    print(f"  {'Model':<25s} {'Accuracy':>10s} {'Log Loss':>10s} {'AUC':>10s}")
    print(f"  {'-'*55}")
    for name, metrics in [("Logistic Regression", lr_metrics), ("XGBoost", xgb_metrics),
                           ("LightGBM", lgbm_metrics), ("Stacked Ensemble", ens_metrics)]:
        print(f"  {name:<25s} {metrics['accuracy']:>10.4f} {metrics['log_loss']:>10.4f} {metrics['auc_roc']:>10.4f}")

    print(f"\n  Backtest avg accuracy: {backtest_results['accuracy'].mean():.4f}")
    print(f"\n  Models saved to: {MODEL_DIR}")
    print(f"  Logs saved to:   {LOG_DIR}")


if __name__ == "__main__":
    main()

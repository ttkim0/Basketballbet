"""
NBA Win Prediction Model Training Pipeline
=============================================
Trains multiple models:
1. Regularized Logistic Regression (Bradley-Terry style)
2. XGBoost Gradient Boosting
3. LightGBM Gradient Boosting
4. Stacked Ensemble

Uses strict chronological train/test splits (no random splitting).
Evaluates: accuracy, log loss, AUC, calibration.
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
    classification_report, confusion_matrix
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import calibration_curve
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
# Feature selection
# ============================================================

# Core features for the model (from the covariate spec)
FEATURE_COLUMNS = [
    # Elo
    "elo_rating_diff",

    # Home court
    "home_flag",

    # Schedule features
    "rest_days_home", "rest_days_visitor", "rest_diff",
    "home_b2b", "visitor_b2b",
    "home_3in4", "visitor_3in4",
    "home_4in6", "visitor_4in6",
    "road_trip_game_num_home", "road_trip_game_num_visitor",
    "homestand_game_num_home", "homestand_game_num_visitor",
    "travel_miles_home", "travel_miles_visitor", "travel_diff",
    "tz_shift_home", "tz_shift_visitor", "tz_diff",
    "altitude_edge_home",

    # Rolling form (last 5)
    "last5_net_rating_diff",
    "last5_off_rating_diff",
    "last5_def_rating_diff",
    "last5_win_rate_diff",
    "last5_avg_margin_diff",

    # Rolling form (last 10)
    "last10_net_rating_diff",
    "last10_off_rating_diff",
    "last10_def_rating_diff",
    "last10_win_rate_diff",
    "last10_avg_margin_diff",

    # Season cumulative stats
    "season_pace_proxy_diff",
    "season_tov_diff",

    # Matchup style
    "pace_mismatch",
    "off_vs_def_mismatch_5",
    "off_vs_def_mismatch_10",
    "def_vs_off_mismatch_5",
    "def_vs_off_mismatch_10",

    # Head-to-head
    "h2h_home_win_rate",
    "h2h_avg_margin",
    "h2h_games_played",

    # Streaks
    "home_streak", "visitor_streak", "streak_diff",

    # Season win %
    "home_season_win_pct", "visitor_season_win_pct",
    "season_win_pct_diff",
    "home_home_win_pct", "visitor_road_win_pct",

    # Motivation
    "playoff_urgency_diff",
    "post_allstar_break",
    "season_progress",
    "is_weekend",
    "month",
]

TARGET = "home_win"


def prepare_data(df):
    """Prepare features and target, handling missing values."""
    print("Preparing training data...")

    # Filter to available features
    available_features = [f for f in FEATURE_COLUMNS if f in df.columns]
    missing_features = [f for f in FEATURE_COLUMNS if f not in df.columns]
    if missing_features:
        print(f"  Warning: {len(missing_features)} features not available: {missing_features[:5]}...")

    print(f"  Using {len(available_features)} features")

    X = df[available_features].copy()
    y = df[TARGET].copy()

    # Fill NaN with 0 for features that may not have enough history
    X = X.fillna(0)

    # Replace infinities
    X = X.replace([np.inf, -np.inf], 0)

    return X, y, available_features


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
    )


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

    # Calibration check: bin predictions and compare
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


def train_logistic_regression(X_train, y_train, X_val, y_val, features):
    """Train regularized logistic regression (Bradley-Terry backbone)."""
    print("\n" + "=" * 70)
    print("TRAINING: Regularized Logistic Regression")
    print("=" * 70)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Grid search over regularization
    best_model = None
    best_ll = float("inf")
    best_C = None

    for C in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
        model = LogisticRegression(C=C, penalty="l2", max_iter=5000, solver="lbfgs")
        model.fit(X_train_scaled, y_train)
        val_proba = model.predict_proba(X_val_scaled)[:, 1]
        ll = log_loss(y_val, val_proba)
        acc = accuracy_score(y_val, (val_proba > 0.5).astype(int))
        print(f"  C={C:>6.3f}  val_log_loss={ll:.4f}  val_acc={acc:.4f}")
        if ll < best_ll:
            best_ll = ll
            best_model = model
            best_C = C

    print(f"\n  Best C: {best_C}")

    # Print coefficients
    print(f"\n  Top feature coefficients:")
    coef_df = pd.DataFrame({
        "feature": features,
        "coefficient": best_model.coef_[0]
    }).sort_values("coefficient", key=abs, ascending=False)
    for _, row in coef_df.head(20).iterrows():
        print(f"    {row['feature']:>40s}: {row['coefficient']:>+8.4f}")

    return best_model, scaler


def train_xgboost(X_train, y_train, X_val, y_val, features):
    """Train XGBoost gradient boosting model."""
    print("\n" + "=" * 70)
    print("TRAINING: XGBoost")
    print("=" * 70)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
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


def train_lightgbm(X_train, y_train, X_val, y_val, features):
    """Train LightGBM gradient boosting model."""
    print("\n" + "=" * 70)
    print("TRAINING: LightGBM")
    print("=" * 70)

    dtrain = lgbm.Dataset(X_train, label=y_train, feature_name=features)
    dval = lgbm.Dataset(X_val, label=y_val, feature_name=features, reference=dtrain)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "seed": 42,
        "verbose": -1,
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


def train_stacked_ensemble(models_dict, X_val, y_val, X_test, y_test, features, scaler):
    """
    Train a stacked ensemble that combines predictions from all base models.
    Uses logistic regression as the meta-learner.
    """
    print("\n" + "=" * 70)
    print("TRAINING: Stacked Ensemble")
    print("=" * 70)

    # Generate base model predictions on validation set
    meta_features_val = []
    meta_features_test = []

    for name, model in models_dict.items():
        if name == "logistic":
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            val_pred = model.predict_proba(X_val_scaled)[:, 1]
            test_pred = model.predict_proba(X_test_scaled)[:, 1]
        elif name == "xgboost":
            dval = xgb.DMatrix(X_val, feature_names=features)
            dtest = xgb.DMatrix(X_test, feature_names=features)
            val_pred = model.predict(dval)
            test_pred = model.predict(dtest)
        elif name == "lightgbm":
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)

        meta_features_val.append(val_pred)
        meta_features_test.append(test_pred)
        print(f"  {name}: val_acc={accuracy_score(y_val, (val_pred > 0.5).astype(int)):.4f}")

    meta_X_val = np.column_stack(meta_features_val)
    meta_X_test = np.column_stack(meta_features_test)

    # Train meta-learner
    meta_model = LogisticRegression(C=1.0, max_iter=5000)
    meta_model.fit(meta_X_val, y_val)

    # Evaluate on test
    test_proba = meta_model.predict_proba(meta_X_test)[:, 1]
    test_pred = (test_proba > 0.5).astype(int)

    print(f"\n  Meta-learner weights: {dict(zip(models_dict.keys(), meta_model.coef_[0]))}")

    return meta_model, test_proba, test_pred


def rolling_backtest(df, X, y, features, window_seasons=5):
    """
    Perform rolling window backtesting.
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

        # Train LightGBM (fastest)
        dtrain = lgbm.Dataset(X_tr, label=y_tr, feature_name=features)
        params = {
            "objective": "binary", "metric": "binary_logloss",
            "num_leaves": 31, "learning_rate": 0.05,
            "feature_fraction": 0.8, "bagging_fraction": 0.8,
            "bagging_freq": 5, "min_child_samples": 20,
            "verbose": -1, "seed": 42,
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
    X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(df, X, y)

    print(f"\n  Train: {X_train.shape[0]} games")
    print(f"  Val:   {X_val.shape[0]} games")
    print(f"  Test:  {X_test.shape[0]} games")

    # ============================================================
    # Model 1: Logistic Regression
    # ============================================================
    lr_model, scaler = train_logistic_regression(X_train, y_train, X_val, y_val, features)

    X_test_scaled = scaler.transform(X_test)
    lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    lr_pred = (lr_proba > 0.5).astype(int)
    lr_metrics = evaluate_model("Logistic Regression", y_test.values, lr_proba, lr_pred)

    # ============================================================
    # Model 2: XGBoost
    # ============================================================
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val, features)

    dtest = xgb.DMatrix(X_test, feature_names=features)
    xgb_proba = xgb_model.predict(dtest)
    xgb_pred = (xgb_proba > 0.5).astype(int)
    xgb_metrics = evaluate_model("XGBoost", y_test.values, xgb_proba, xgb_pred)

    # ============================================================
    # Model 3: LightGBM
    # ============================================================
    lgbm_model = train_lightgbm(X_train, y_train, X_val, y_val, features)

    lgbm_proba = lgbm_model.predict(X_test)
    lgbm_pred = (lgbm_proba > 0.5).astype(int)
    lgbm_metrics = evaluate_model("LightGBM", y_test.values, lgbm_proba, lgbm_pred)

    # ============================================================
    # Model 4: Stacked Ensemble
    # ============================================================
    models_dict = {"logistic": lr_model, "xgboost": xgb_model, "lightgbm": lgbm_model}
    meta_model, ens_proba, ens_pred = train_stacked_ensemble(
        models_dict, X_val, y_val, X_test, y_test, features, scaler
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
    joblib.dump(features, os.path.join(MODEL_DIR, "feature_list.pkl"))

    # Save results
    all_metrics = {
        "logistic_regression": lr_metrics,
        "xgboost": xgb_metrics,
        "lightgbm": lgbm_metrics,
        "ensemble": ens_metrics,
        "features_used": features,
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
    print(f"  {'Model':<25s} {'Accuracy':>10s} {'Log Loss':>10s} {'AUC':>10s}")
    print(f"  {'-'*55}")
    for name, metrics in [("Logistic Regression", lr_metrics), ("XGBoost", xgb_metrics),
                           ("LightGBM", lgbm_metrics), ("Stacked Ensemble", ens_metrics)]:
        print(f"  {name:<25s} {metrics['accuracy']:>10.4f} {metrics['log_loss']:>10.4f} {metrics['auc_roc']:>10.4f}")

    print(f"\n  Models saved to: {MODEL_DIR}")
    print(f"  Logs saved to:   {LOG_DIR}")


if __name__ == "__main__":
    main()

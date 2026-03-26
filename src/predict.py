"""
NBA Game Prediction Interface
================================
Load trained models and predict win probabilities for new matchups.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import lightgbm as lgbm

MODEL_DIR = "/home/user/Basketballbet/models"
DATA_DIR = "/home/user/Basketballbet/data/processed"


class NBAPredictor:
    """Load trained models and make predictions."""

    def __init__(self):
        print("Loading NBA Win Prediction Models...")

        self.lr_model = joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl"))
        self.scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        self.features = joblib.load(os.path.join(MODEL_DIR, "feature_list.pkl"))
        self.meta_model = joblib.load(os.path.join(MODEL_DIR, "ensemble_meta.pkl"))

        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(os.path.join(MODEL_DIR, "xgboost_model.json"))

        self.lgbm_model = lgbm.Booster(model_file=os.path.join(MODEL_DIR, "lightgbm_model.txt"))

        # Load final Elo ratings
        elo_path = os.path.join(DATA_DIR, "final_elo_ratings.csv")
        if os.path.exists(elo_path):
            self.elo_ratings = dict(zip(
                pd.read_csv(elo_path)["team"],
                pd.read_csv(elo_path)["elo"]
            ))
        else:
            self.elo_ratings = {}

        print(f"  Loaded {len(self.features)} features")
        print(f"  Elo ratings for {len(self.elo_ratings)} teams")
        print("  Models: Logistic Regression, XGBoost, LightGBM, Stacked Ensemble")

    def predict_game(self, home_team, visitor_team, features_override=None):
        """
        Predict win probability for a single game.

        Args:
            home_team: 3-letter abbreviation (e.g., 'BOS')
            visitor_team: 3-letter abbreviation (e.g., 'LAL')
            features_override: dict of feature overrides (optional)

        Returns:
            dict with predictions from all models
        """
        # Build feature vector with defaults
        feature_values = {f: 0.0 for f in self.features}

        # Set Elo difference
        home_elo = self.elo_ratings.get(home_team, 1500)
        visitor_elo = self.elo_ratings.get(visitor_team, 1500)
        if "elo_rating_diff" in feature_values:
            feature_values["elo_rating_diff"] = home_elo - visitor_elo

        # Home flag
        if "home_flag" in feature_values:
            feature_values["home_flag"] = 1

        # Default rest days
        if "rest_days_home" in feature_values:
            feature_values["rest_days_home"] = 1
        if "rest_days_visitor" in feature_values:
            feature_values["rest_days_visitor"] = 1

        # Apply overrides
        if features_override:
            for k, v in features_override.items():
                if k in feature_values:
                    feature_values[k] = v

        # Create feature array
        X = np.array([[feature_values[f] for f in self.features]])

        # Get predictions from each model
        X_scaled = self.scaler.transform(X)
        lr_prob = self.lr_model.predict_proba(X_scaled)[0, 1]

        dmatrix = xgb.DMatrix(X, feature_names=self.features)
        xgb_prob = float(self.xgb_model.predict(dmatrix)[0])

        lgbm_prob = float(self.lgbm_model.predict(X)[0])

        # Ensemble
        meta_X = np.array([[lr_prob, xgb_prob, lgbm_prob]])
        ens_prob = float(self.meta_model.predict_proba(meta_X)[0, 1])

        result = {
            "home_team": home_team,
            "visitor_team": visitor_team,
            "home_elo": round(home_elo, 1),
            "visitor_elo": round(visitor_elo, 1),
            "elo_diff": round(home_elo - visitor_elo, 1),
            "predictions": {
                "logistic_regression": round(lr_prob, 4),
                "xgboost": round(xgb_prob, 4),
                "lightgbm": round(lgbm_prob, 4),
                "ensemble": round(ens_prob, 4),
            },
            "predicted_winner": home_team if ens_prob > 0.5 else visitor_team,
            "win_probability": round(max(ens_prob, 1 - ens_prob), 4),
        }

        return result


def print_prediction(result):
    """Pretty-print a prediction result."""
    print(f"\n{'=' * 50}")
    print(f"  {result['visitor_team']} @ {result['home_team']}")
    print(f"{'=' * 50}")
    print(f"  Elo: {result['home_team']}={result['home_elo']}, "
          f"{result['visitor_team']}={result['visitor_elo']} "
          f"(diff={result['elo_diff']:+.1f})")
    print(f"\n  Model Predictions (home win %):")
    for model, prob in result["predictions"].items():
        bar = "#" * int(prob * 40)
        print(f"    {model:<25s}: {prob:.1%} {bar}")
    print(f"\n  >>> Predicted Winner: {result['predicted_winner']} "
          f"({result['win_probability']:.1%} confidence)")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    predictor = NBAPredictor()

    # Example predictions
    matchups = [
        ("BOS", "LAL"),
        ("GSW", "MIL"),
        ("DEN", "PHX"),
        ("CLE", "NYK"),
        ("OKC", "MIN"),
    ]

    for home, visitor in matchups:
        result = predictor.predict_game(home, visitor)
        print_prediction(result)

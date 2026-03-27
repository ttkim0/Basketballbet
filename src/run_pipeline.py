"""
Master Pipeline: NBA Win Prediction Model v2
===============================================
Runs the complete pipeline end-to-end:
1. Build complete game dataset from deepshot + BBRef + kyleskom
2. Compute Elo ratings
3. Engineer all features (real box score + odds + RAPTOR + schedule + advanced)
4. Train all models
5. Evaluate and save

Usage: python3 src/run_pipeline.py
"""

import sys
import os
import time
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from integrate_deepshot_data import run_full_integration
from elo_ratings import compute_elo_ratings
from feature_engineering import compute_all_features
from train_model import main as train_models


def run_full_pipeline():
    """Execute the complete NBA prediction pipeline."""
    total_start = time.time()

    print("\n" + "=" * 80)
    print("  NBA WIN PREDICTION MODEL - FULL TRAINING PIPELINE v2")
    print("  Building from 2003-04 through 2025-26 | 27K+ real games")
    print("  Real box scores + advanced stats + odds + RAPTOR")
    print("=" * 80)

    processed_dir = "/home/user/Basketballbet/data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    # ============================================================
    # Step 1: Build Enriched Game Dataset
    # ============================================================
    print("\n\n" + "#" * 80)
    print("# STEP 1/4: BUILD ENRICHED GAME DATASET")
    print("#" * 80)
    step_start = time.time()

    df = run_full_integration()
    print(f"\n  Step 1 completed in {time.time() - step_start:.1f}s")
    print(f"  Dataset: {len(df)} games, {len(df.columns)} columns")

    # ============================================================
    # Step 2: Compute Elo Ratings
    # ============================================================
    print("\n\n" + "#" * 80)
    print("# STEP 2/4: COMPUTE MARGIN-AWARE ELO RATINGS")
    print("#" * 80)
    step_start = time.time()

    df["date"] = pd.to_datetime(df["date"])
    df_elo, final_ratings = compute_elo_ratings(df)

    # Save Elo data
    df_elo.to_csv(os.path.join(processed_dir, "games_with_elo.csv"), index=False)

    ratings_df = pd.DataFrame([
        {"team": k, "elo": v} for k, v in sorted(final_ratings.items(), key=lambda x: -x[1])
    ])
    ratings_df.to_csv(os.path.join(processed_dir, "final_elo_ratings.csv"), index=False)

    print(f"\n  Step 2 completed in {time.time() - step_start:.1f}s")
    print(f"\n  Final Elo Rankings (Top 10):")
    for _, row in ratings_df.head(10).iterrows():
        print(f"    {row['team']}: {row['elo']:.1f}")

    # ============================================================
    # Step 3: Engineer All Features
    # ============================================================
    print("\n\n" + "#" * 80)
    print("# STEP 3/4: ENGINEER ALL FEATURES")
    print("#" * 80)
    step_start = time.time()

    df_featured = compute_all_features(df_elo)
    df_featured.to_csv(os.path.join(processed_dir, "games_full_features.csv"), index=False)

    print(f"\n  Step 3 completed in {time.time() - step_start:.1f}s")
    print(f"  Feature matrix: {df_featured.shape}")

    # ============================================================
    # Step 4: Train Models and Evaluate
    # ============================================================
    print("\n\n" + "#" * 80)
    print("# STEP 4/4: TRAIN MODELS AND EVALUATE")
    print("#" * 80)
    step_start = time.time()

    train_models()

    print(f"\n  Step 4 completed in {time.time() - step_start:.1f}s")

    # ============================================================
    # Final Summary
    # ============================================================
    total_time = time.time() - total_start
    print("\n\n" + "=" * 80)
    print("  PIPELINE COMPLETE")
    print("=" * 80)
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"  Games processed: {len(df_featured)}")
    print(f"  Features engineered: {len(df_featured.columns)}")
    print(f"  Models trained: 4 (Logistic, XGBoost, LightGBM, Ensemble)")
    print(f"  Output directory: /home/user/Basketballbet/models/")
    print("=" * 80)


if __name__ == "__main__":
    run_full_pipeline()

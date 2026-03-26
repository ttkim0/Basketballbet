"""
Master Pipeline: NBA Win Prediction Model
=============================================
Runs the complete pipeline end-to-end:
1. Build complete game dataset
2. Compute Elo ratings
3. Engineer all features
4. Train all models
5. Evaluate and save

Usage: python3 src/run_pipeline.py
"""

import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from build_complete_dataset import build_complete_dataset
from integrate_real_data import merge_all_into_games
from elo_ratings import compute_elo_ratings
from feature_engineering import compute_all_features
from train_model import main as train_models


def run_full_pipeline():
    """Execute the complete NBA prediction pipeline."""
    total_start = time.time()

    print("\n" + "=" * 80)
    print("  NBA WIN PREDICTION MODEL - FULL TRAINING PIPELINE (OPTIMIZED)")
    print("  Building from 2003-04 through 2024-25 | 28K+ real games")
    print("=" * 80)

    # ============================================================
    # Step 1: Build Complete Game Dataset
    # ============================================================
    print("\n\n" + "#" * 80)
    print("# STEP 1/6: BUILD COMPLETE GAME DATASET")
    print("#" * 80)
    step_start = time.time()

    df = build_complete_dataset()
    print(f"\n  Step 1 completed in {time.time() - step_start:.1f}s")
    print(f"  Dataset: {len(df)} games, {len(df.columns)} columns")

    # ============================================================
    # Step 1b: Enrich with real box scores, odds, RAPTOR
    # ============================================================
    print("\n\n" + "#" * 80)
    print("# STEP 1b/6: MERGE REAL DATA (BOX SCORES, ODDS, RAPTOR)")
    print("#" * 80)
    step_start = time.time()

    df = merge_all_into_games(df)
    print(f"\n  Step 1b completed in {time.time() - step_start:.1f}s")
    print(f"  Enriched dataset: {len(df)} games, {len(df.columns)} columns")

    # ============================================================
    # Step 2: Compute Elo Ratings
    # ============================================================
    print("\n\n" + "#" * 80)
    print("# STEP 2/6: COMPUTE MARGIN-AWARE ELO RATINGS")
    print("#" * 80)
    step_start = time.time()

    import pandas as pd
    df["date"] = pd.to_datetime(df["date"])
    df_elo, final_ratings = compute_elo_ratings(df)

    # Save Elo data
    processed_dir = "/home/user/Basketballbet/data/processed"
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
    print("# STEP 3/5: ENGINEER ALL FEATURES")
    print("#" * 80)
    step_start = time.time()

    df_featured = compute_all_features(df_elo)
    df_featured.to_csv(os.path.join(processed_dir, "games_full_features.csv"), index=False)

    print(f"\n  Step 3 completed in {time.time() - step_start:.1f}s")
    print(f"  Feature matrix: {df_featured.shape}")

    # ============================================================
    # Step 4 & 5: Train Models and Evaluate
    # ============================================================
    print("\n\n" + "#" * 80)
    print("# STEP 4-5/5: TRAIN MODELS AND EVALUATE")
    print("#" * 80)
    step_start = time.time()

    train_models()

    print(f"\n  Steps 4-5 completed in {time.time() - step_start:.1f}s")

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

"""
Layer A: Margin-Aware Elo Rating System
========================================
Implements FiveThirtyEight-style Elo with:
- Margin of victory multiplier (diminishing returns)
- Home court advantage
- Season-to-season regression to mean
- K-factor tuning
"""

import pandas as pd
import numpy as np
from tqdm import tqdm


# ============================================================
# Elo Parameters (tuned for NBA)
# ============================================================
INITIAL_ELO = 1500
K_FACTOR = 20
HOME_ADVANTAGE = 100  # Elo points for home court
SEASON_REGRESSION = 0.75  # Regress 25% toward mean each season
MOV_MULTIPLIER_POWER = 0.8  # Diminishing returns on margin


def expected_score(rating_a, rating_b, home_advantage=0):
    """Calculate expected win probability for team A."""
    d = rating_a - rating_b + home_advantage
    return 1.0 / (1.0 + 10.0 ** (-d / 400.0))


def margin_multiplier(mov, elo_diff):
    """
    Margin of victory multiplier with diminishing returns.
    Based on FiveThirtyEight's NBA Elo methodology.
    Prevents blowouts from over-influencing ratings.
    """
    abs_mov = abs(mov)
    # Diminishing returns: log-based scaling
    # Also adjust for expectation - upsets with big margins count more
    if abs_mov == 0:
        return 1.0
    mult = ((abs_mov + 3.0) ** MOV_MULTIPLIER_POWER) / (7.5 + 0.006 * abs(elo_diff))
    return mult


def update_elo(rating_a, rating_b, score_a, margin, is_home_a=True):
    """
    Update Elo ratings after a game.

    Args:
        rating_a: Current rating of team A
        rating_b: Current rating of team B
        score_a: 1 if A wins, 0 if A loses
        margin: Point differential (A's score - B's score)
        is_home_a: Whether team A is the home team

    Returns:
        (new_rating_a, new_rating_b)
    """
    ha = HOME_ADVANTAGE if is_home_a else -HOME_ADVANTAGE
    expected = expected_score(rating_a, rating_b, ha)

    elo_diff = rating_a - rating_b + ha
    mov_mult = margin_multiplier(margin, elo_diff)

    update = K_FACTOR * mov_mult * (score_a - expected)

    new_a = rating_a + update
    new_b = rating_b - update

    return new_a, new_b


def regress_to_mean(ratings, factor=SEASON_REGRESSION):
    """Regress all ratings toward the mean at season boundary."""
    mean_rating = np.mean(list(ratings.values()))
    return {
        team: mean_rating + factor * (rating - mean_rating)
        for team, rating in ratings.items()
    }


def compute_elo_ratings(games_df):
    """
    Compute Elo ratings for all teams across all games chronologically.

    Args:
        games_df: DataFrame with columns: date, home_team_id, visitor_team_id,
                  home_pts, visitor_pts, home_win, margin, season

    Returns:
        games_df with added columns: home_elo_pre, visitor_elo_pre, elo_diff,
                                      home_elo_post, visitor_elo_post
    """
    print("Computing margin-aware Elo ratings...")

    # Sort by date
    df = games_df.sort_values("date").reset_index(drop=True)

    # Initialize ratings
    ratings = {}

    # Storage for pregame ratings
    home_elo_pre = []
    visitor_elo_pre = []
    home_elo_post = []
    visitor_elo_post = []
    elo_diff = []
    home_expected = []

    current_season = None

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Elo Ratings"):
        home = row["home_team_id"]
        visitor = row["visitor_team_id"]
        season = row["season"]

        # Season transition: regress ratings
        if current_season is not None and season != current_season:
            ratings = regress_to_mean(ratings)
            print(f"  Season {current_season} -> {season}: regressed {len(ratings)} team ratings")
        current_season = season

        # Initialize new teams
        if home not in ratings:
            ratings[home] = INITIAL_ELO
        if visitor not in ratings:
            ratings[visitor] = INITIAL_ELO

        # Pre-game ratings
        r_home = ratings[home]
        r_visitor = ratings[visitor]

        home_elo_pre.append(r_home)
        visitor_elo_pre.append(r_visitor)
        elo_diff.append(r_home - r_visitor)

        # Expected score
        exp = expected_score(r_home, r_visitor, HOME_ADVANTAGE)
        home_expected.append(exp)

        # Update
        home_win = 1 if row["home_pts"] > row["visitor_pts"] else 0
        mov = row["home_pts"] - row["visitor_pts"]

        new_home, new_visitor = update_elo(r_home, r_visitor, home_win, mov, is_home_a=True)

        home_elo_post.append(new_home)
        visitor_elo_post.append(new_visitor)

        ratings[home] = new_home
        ratings[visitor] = new_visitor

    df["home_elo_pre"] = home_elo_pre
    df["visitor_elo_pre"] = visitor_elo_pre
    df["home_elo_post"] = home_elo_post
    df["visitor_elo_post"] = visitor_elo_post
    df["elo_diff"] = elo_diff
    df["home_elo_expected"] = home_expected

    # Rating difference (positive = home team stronger)
    df["elo_rating_diff"] = df["home_elo_pre"] - df["visitor_elo_pre"]

    print(f"  Computed Elo for {len(df)} games across {len(ratings)} teams")
    print(f"  Final rating range: {min(ratings.values()):.0f} - {max(ratings.values()):.0f}")

    # Accuracy check
    correct = ((df["home_elo_expected"] > 0.5) == (df["home_win"] == 1)).mean()
    print(f"  Elo-only accuracy: {correct:.4f}")

    return df, ratings


if __name__ == "__main__":
    import os
    processed_dir = "/home/user/Basketballbet/data/processed"
    games_path = os.path.join(processed_dir, "games_clean.csv")

    if os.path.exists(games_path):
        df = pd.read_csv(games_path, parse_dates=["date"])
        df_elo, final_ratings = compute_elo_ratings(df)
        df_elo.to_csv(os.path.join(processed_dir, "games_with_elo.csv"), index=False)

        # Save final ratings
        ratings_df = pd.DataFrame([
            {"team": k, "elo": v} for k, v in sorted(final_ratings.items(), key=lambda x: -x[1])
        ])
        ratings_df.to_csv(os.path.join(processed_dir, "final_elo_ratings.csv"), index=False)
        print("\nFinal Elo Ratings:")
        print(ratings_df.to_string(index=False))
    else:
        print("No clean games file found. Run build_dataset_from_kaggle.py first.")

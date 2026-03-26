"""
NBA Game Data Scraper - Basketball Reference
Scrapes every NBA game from 2003-04 season through 2025-26 season.
Collects: date, teams, scores, box score stats, home/away, overtime info.
"""

import requests
import time
import re
import json
import os
import csv
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

BASE_URL = "https://www.basketball-reference.com"
DATA_DIR = "/home/user/Basketballbet/data/raw"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Seasons to scrape: 2003-04 through 2025-26
SEASONS = list(range(2004, 2027))  # Basketball-Reference uses end year


def get_season_schedule(season_year):
    """Scrape the full season schedule for a given season end year."""
    games = []
    months_to_try = [
        "october", "november", "december", "january", "february",
        "march", "april", "may", "june"
    ]

    for month in months_to_try:
        url = f"{BASE_URL}/leagues/NBA_{season_year}_games-{month}.html"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 404:
                continue
            if resp.status_code == 429:
                print(f"    Rate limited, waiting 60s...")
                time.sleep(60)
                resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code != 200:
                continue

            soup = BeautifulSoup(resp.text, "lxml")
            table = soup.find("table", {"id": "schedule"})
            if not table:
                continue

            tbody = table.find("tbody")
            if not tbody:
                continue

            rows = tbody.find_all("tr")
            for row in rows:
                # Skip playoff header rows
                if row.get("class") and "thead" in row.get("class", []):
                    continue

                cells = row.find_all(["th", "td"])
                if len(cells) < 6:
                    continue

                try:
                    date_cell = cells[0]
                    date_text = date_cell.get_text(strip=True)

                    visitor_cell = cells[1]
                    visitor_pts_cell = cells[2]
                    home_cell = cells[3]
                    home_pts_cell = cells[4]

                    visitor_team = visitor_cell.get_text(strip=True)
                    home_team = home_cell.get_text(strip=True)
                    visitor_pts = visitor_pts_cell.get_text(strip=True)
                    home_pts = home_pts_cell.get_text(strip=True)

                    # Skip games not yet played
                    if not visitor_pts or not home_pts:
                        continue

                    # Check for OT
                    ot_cell = cells[5] if len(cells) > 5 else None
                    ot_text = ot_cell.get_text(strip=True) if ot_cell else ""

                    # Get box score link if available
                    box_link = ""
                    link_tag = date_cell.find("a")
                    if link_tag and link_tag.get("href"):
                        box_link = link_tag["href"]

                    # Parse date
                    try:
                        game_date = pd.to_datetime(date_text).strftime("%Y-%m-%d")
                    except Exception:
                        game_date = date_text

                    game = {
                        "date": game_date,
                        "season": season_year,
                        "visitor_team": visitor_team,
                        "visitor_pts": int(visitor_pts),
                        "home_team": home_team,
                        "home_pts": int(home_pts),
                        "overtime": ot_text,
                        "box_score_url": box_link,
                    }
                    games.append(game)

                except (ValueError, IndexError, AttributeError) as e:
                    continue

            time.sleep(3.5)  # Respect rate limits

        except requests.exceptions.RequestException as e:
            print(f"    Error fetching {month} {season_year}: {e}")
            time.sleep(10)
            continue

    return games


def get_team_season_stats(season_year):
    """Scrape team per-game and advanced stats for a season."""
    stats = {}

    # Per-game stats
    url = f"{BASE_URL}/leagues/NBA_{season_year}.html"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code == 429:
            time.sleep(60)
            resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            return stats

        soup = BeautifulSoup(resp.text, "lxml")

        # Team per-game stats table
        per_game_table = soup.find("table", {"id": "per_game-team"})
        if per_game_table:
            rows = per_game_table.find("tbody").find_all("tr")
            for row in rows:
                if row.get("class") and "thead" in row.get("class", []):
                    continue
                cells = row.find_all(["th", "td"])
                if len(cells) < 2:
                    continue
                team_name = cells[0].get_text(strip=True).replace("*", "")
                team_stats = {}
                headers_row = per_game_table.find("thead").find_all("th")
                for i, cell in enumerate(cells):
                    if i < len(headers_row):
                        col_name = headers_row[i].get_text(strip=True)
                        val = cell.get_text(strip=True)
                        try:
                            team_stats[col_name] = float(val)
                        except ValueError:
                            team_stats[col_name] = val
                stats[team_name] = team_stats

        time.sleep(3.5)

        # Opponent per-game stats
        opp_table = soup.find("table", {"id": "per_game-opponent"})
        if opp_table:
            rows = opp_table.find("tbody").find_all("tr")
            for row in rows:
                if row.get("class") and "thead" in row.get("class", []):
                    continue
                cells = row.find_all(["th", "td"])
                if len(cells) < 2:
                    continue
                team_name = cells[0].get_text(strip=True).replace("*", "")
                if team_name not in stats:
                    stats[team_name] = {}
                headers_row = opp_table.find("thead").find_all("th")
                for i, cell in enumerate(cells):
                    if i < len(headers_row):
                        col_name = "opp_" + headers_row[i].get_text(strip=True)
                        val = cell.get_text(strip=True)
                        try:
                            stats[team_name][col_name] = float(val)
                        except ValueError:
                            stats[team_name][col_name] = val

    except Exception as e:
        print(f"  Error fetching team stats for {season_year}: {e}")

    return stats


def get_advanced_team_stats(season_year):
    """Scrape advanced team stats (ORtg, DRtg, Pace, etc.) from the misc stats page."""
    stats = {}
    url = f"{BASE_URL}/leagues/NBA_{season_year}.html"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code == 429:
            time.sleep(60)
            resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            return stats

        soup = BeautifulSoup(resp.text, "lxml")

        # Advanced stats are often in comments - parse them out
        comments = soup.find_all(string=lambda text: isinstance(text, type(soup.new_string(""))) == False and text and "misc_stats" in str(text))

        # Try direct table first
        misc_table = soup.find("table", {"id": "advanced-team"})

        if not misc_table:
            # Look in HTML comments
            for comment in soup.find_all(string=lambda text: isinstance(text, str) and "advanced-team" in text):
                comment_soup = BeautifulSoup(comment, "lxml")
                misc_table = comment_soup.find("table", {"id": "advanced-team"})
                if misc_table:
                    break

        if not misc_table:
            # Try misc_stats
            for comment in soup.find_all(string=lambda text: isinstance(text, str) and "misc_stats" in text):
                comment_soup = BeautifulSoup(comment, "lxml")
                misc_table = comment_soup.find("table", {"id": "misc_stats"})
                if misc_table:
                    break

        if misc_table:
            thead = misc_table.find("thead")
            header_rows = thead.find_all("tr") if thead else []
            # Get the last header row (actual column names)
            if header_rows:
                headers = [th.get_text(strip=True) for th in header_rows[-1].find_all("th")]
            else:
                headers = []

            tbody = misc_table.find("tbody")
            if tbody:
                for row in tbody.find_all("tr"):
                    if row.get("class") and "thead" in row.get("class", []):
                        continue
                    cells = row.find_all(["th", "td"])
                    if len(cells) < 2:
                        continue
                    team_name = cells[0].get_text(strip=True).replace("*", "")
                    team_stats = {}
                    for i, cell in enumerate(cells):
                        if i < len(headers):
                            val = cell.get_text(strip=True)
                            try:
                                team_stats[f"adv_{headers[i]}"] = float(val)
                            except ValueError:
                                team_stats[f"adv_{headers[i]}"] = val
                    stats[team_name] = team_stats

    except Exception as e:
        print(f"  Error fetching advanced stats for {season_year}: {e}")

    return stats


def scrape_all_games():
    """Main function to scrape all games from 2003-04 to 2025-26."""
    all_games = []
    os.makedirs(DATA_DIR, exist_ok=True)

    output_file = os.path.join(DATA_DIR, "all_nba_games_2003_2026.csv")

    # Check for existing progress
    existing_seasons = set()
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        existing_seasons = set(existing_df["season"].unique())
        all_games = existing_df.to_dict("records")
        print(f"Found existing data with {len(all_games)} games from seasons: {sorted(existing_seasons)}")

    print("=" * 70)
    print("NBA GAME DATA SCRAPER - Basketball Reference")
    print(f"Scraping seasons: 2003-04 through 2025-26")
    print("=" * 70)

    for season in tqdm(SEASONS, desc="Seasons"):
        season_label = f"{season-1}-{str(season)[2:]}"

        if season in existing_seasons:
            count = len([g for g in all_games if g["season"] == season])
            print(f"\n[SKIP] Season {season_label}: already have {count} games")
            continue

        print(f"\n[SCRAPING] Season {season_label}...")
        games = get_season_schedule(season)
        print(f"  Found {len(games)} games")

        all_games.extend(games)

        # Save after each season
        df = pd.DataFrame(all_games)
        df.to_csv(output_file, index=False)
        print(f"  Saved cumulative total: {len(all_games)} games")

        time.sleep(5)  # Extra pause between seasons

    # Final save
    df = pd.DataFrame(all_games)
    df.to_csv(output_file, index=False)
    print(f"\n{'=' * 70}")
    print(f"COMPLETE: {len(all_games)} total games saved to {output_file}")
    print(f"{'=' * 70}")

    return df


def scrape_all_team_stats():
    """Scrape per-game and advanced team stats for all seasons."""
    os.makedirs(DATA_DIR, exist_ok=True)
    all_stats = []
    output_file = os.path.join(DATA_DIR, "all_team_stats_2003_2026.csv")

    print("=" * 70)
    print("NBA TEAM STATS SCRAPER - Basketball Reference")
    print("=" * 70)

    for season in tqdm(SEASONS, desc="Team Stats"):
        season_label = f"{season-1}-{str(season)[2:]}"
        print(f"\n[SCRAPING] Team stats for {season_label}...")

        per_game = get_team_season_stats(season)
        time.sleep(4)
        advanced = get_advanced_team_stats(season)

        for team_name in per_game:
            row = {"season": season, "team": team_name}
            row.update(per_game.get(team_name, {}))
            row.update(advanced.get(team_name, {}))
            all_stats.append(row)

        print(f"  Got stats for {len(per_game)} teams")
        time.sleep(5)

    df = pd.DataFrame(all_stats)
    df.to_csv(output_file, index=False)
    print(f"\nSaved {len(all_stats)} team-season records to {output_file}")
    return df


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "team_stats":
        scrape_all_team_stats()
    else:
        scrape_all_games()

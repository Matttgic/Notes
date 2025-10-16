"""
Script to fetch upcoming matches for top 5 European leagues, compute team ratings
using SofaScore's public API, and apply back‑tested predictive rules to
generate football match predictions.  The script reads a CSV of rules
(complete_predictive_rules_summary.csv) to determine which rules are
profitable and uses their ROI and success rate when generating
predictions.  It outputs a CSV file containing only the matches for
which a profitable rule applies.

Usage:
    python predict.py --output predictions.csv

The script will automatically determine the next weekend based on
Europe/Paris timezone, fetch the fixtures from worldfootball.net for
each of the top five leagues (Premier League, Ligue 1, Bundesliga,
Serie A, La Liga), retrieve the average team ratings from SofaScore,
apply the rules and write out predictions.

Requirements:
    - pandas
    - requests
    - beautifulsoup4
    - python-dateutil

Note:
    This script uses public endpoints from SofaScore and
    worldfootball.net.  These endpoints are subject to change and may
    occasionally fail.  Error handling is included to skip matches
    when data cannot be retrieved.
"""

import argparse
import csv
import datetime as dt
from dateutil import tz
from dateutil.relativedelta import relativedelta, FR, MO
import os
import re
import sys
import time
from typing import Dict, List, Tuple, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup


# Constants for the five major European leagues.  Each entry maps the
# league name to its SofaScore unique tournament ID, season ID for
# 2025/26, and the corresponding worldfootball.net slug used to
# construct the fixture URL.
LEAGUES = {
    "Premier League": {
        "ut_id": 17,
        "season_id": 76986,
        "wf_slug": "eng-premier-league-2025-2026",
    },
    "Ligue 1": {
        "ut_id": 34,
        "season_id": 77356,
        "wf_slug": "fra-ligue-1-2025-2026",
    },
    "Bundesliga": {
        "ut_id": 35,
        "season_id": 77333,
        "wf_slug": "ger-bundesliga-2025-2026",
    },
    "Serie A": {
        "ut_id": 23,
        "season_id": 76457,
        "wf_slug": "ita-serie-a-2025-2026",
    },
    "La Liga": {
        "ut_id": 8,
        "season_id": 77559,
        "wf_slug": "esp-primera-division-2025-2026",
    },
}


def load_rules(csv_path: str) -> pd.DataFrame:
    """Load rules from the CSV file and filter for profitable ones.

    Returns a DataFrame sorted by ROI descending.
    """
    df = pd.read_csv(csv_path)
    # Keep only rules marked as profitable
    df_profitable = df[df["Profitability"].str.contains("Profitable", na=False)].copy()
    # Ensure ROI_Percent numeric
    df_profitable["ROI_Percent"] = pd.to_numeric(df_profitable["ROI_Percent"], errors="coerce")
    df_profitable.sort_values(by="ROI_Percent", ascending=False, inplace=True)
    return df_profitable


def next_weekend(date: dt.date, tzinfo: tz.tzfile) -> Tuple[dt.date, dt.date]:
    """Return the start and end date (inclusive) of the next weekend.

    The weekend is considered from Friday to Monday.  The returned dates
    are naive dates (no timezone) but correspond to the local dates in
    the provided timezone.
    """
    # Convert date to timezone aware
    dt_date = dt.datetime.combine(date, dt.time(0), tzinfo)
    # Next Friday
    next_friday = dt_date + relativedelta(weekday=FR(+1))
    friday = next_friday.date()
    # Monday after that weekend
    next_monday = dt_date + relativedelta(weekday=MO(+2))  # Monday after weekend
    monday = next_monday.date()
    return friday, monday


def fetch_fixture_list(league_slug: str, start_date: dt.date, end_date: dt.date) -> List[Tuple[str, str, dt.date]]:
    """Fetch fixtures between start_date and end_date inclusive for a league.

    Returns a list of tuples (home_team, away_team, match_date).
    Uses worldfootball.net schedule page for the league.
    """
    url = f"https://www.worldfootball.net/schedule/{league_slug}/"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
    except Exception as exc:
        print(f"Error fetching schedule for {league_slug}: {exc}", file=sys.stderr)
        return []
    soup = BeautifulSoup(resp.content, "html.parser")
    # The schedule table contains rows with date in first column
    fixtures = []
    table = soup.find("table", class_="standard_tabelle")
    if not table:
        return fixtures
    for row in table.find_all("tr"):
        cols = [c.get_text().strip() for c in row.find_all(["th", "td"])]
        if not cols or len(cols) < 3:
            continue
        # Date is the first column in format 'dd/mm/yyyy'
        # Some rows may have multiple lines; ensure day-month-year pattern
        date_match = re.match(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", cols[0])
        if not date_match:
            continue
        day, month, year = map(int, date_match.groups())
        match_date = dt.date(year, month, day)
        if not (start_date <= match_date <= end_date):
            continue
        # Next columns: home vs away
        # Format could be "Home Team - Away Team" or with hyphen/trailing spaces
        matchup = cols[1]
        if '-' not in matchup:
            continue
        home, away = [t.strip() for t in matchup.split('-', 1)]
        fixtures.append((home, away, match_date))
    return fixtures


def search_team(team_name: str) -> Optional[int]:
    """Search SofaScore for a team and return its ID.

    The search API returns multiple results; we pick the first entry where
    the type is 'team', sport is 'Football' and gender is male.
    """
    url = f"https://www.sofascore.com/api/v1/search/all?q={requests.utils.quote(team_name)}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
    except Exception as exc:
        print(f"Error searching for team {team_name}: {exc}", file=sys.stderr)
        return None
    try:
        data = resp.json()
    except Exception:
        return None
    for category in data.get("categories", []):
        for item in category.get("items", []):
            if item.get("type") == "team":
                sport = item.get("sport", {}).get("name")
                gender = item.get("team", {}).get("gender")
                if sport == "Football" and gender == "M":
                    return item.get("id")
    return None


def get_team_rating(team_id: int, ut_id: int, season_id: int) -> Optional[float]:
    """Get the average rating for a team in a specific league and season.
    Returns None if not found.
    """
    url = f"https://www.sofascore.com/api/v1/team/{team_id}/unique-tournament/{ut_id}/season/{season_id}/statistics/overall"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
    except Exception as exc:
        print(f"Error fetching statistics for team {team_id}: {exc}", file=sys.stderr)
        return None
    try:
        data = resp.json()
    except Exception:
        return None
    try:
        return float(data.get("statistics", {}).get("avgRating"))
    except Exception:
        return None


def apply_rules(home_rating: float, away_rating: float, rules_df: pd.DataFrame) -> Optional[Tuple[str, str, float, float]]:
    """Apply predictive rules to the given ratings.

    Returns a tuple (prediction, rule_name, ROI_Percent, Success_Rate) if a rule
    applies, otherwise None.
    The rules are evaluated in descending order of ROI to favour the most
    profitable.
    """
    rating_diff = home_rating - away_rating
    for _, row in rules_df.iterrows():
        rule = row["Rule_Name"]
        roi = row["ROI_Percent"]
        win_rate = row["Success_Rate"]
        # Elite home team
        if rule == "elite_home_team" and home_rating > 6.95:
            return ("Home Win", rule, roi, win_rate)
        # Strong home combo
        if rule == "strong_home_combo" and home_rating > 6.85 and rating_diff > 0.15:
            return ("Home Win", rule, roi, win_rate)
        # Home strong advantage
        if rule == "home_strong_advantage" and rating_diff > 0.25:
            return ("Home Win", rule, roi, win_rate)
        # Weak vs strong
        if rule == "weak_vs_strong" and home_rating < 6.7 and away_rating > 6.9:
            return ("Away Win", rule, roi, win_rate)
        # Away strong advantage
        if rule == "away_strong_advantage" and rating_diff < -0.25:
            return ("Away Win", rule, roi, win_rate)
        # Balanced match
        if rule == "balanced_match" and abs(rating_diff) < 0.05:
            return ("Draw", rule, roi, win_rate)
    return None


def main(output_path: str):
    # Load rules
    rules_df = load_rules(os.path.join(os.path.dirname(__file__), "complete_predictive_rules_summary.csv"))
    # Determine next weekend dates in Europe/Paris timezone
    paris_tz = tz.gettz("Europe/Paris")
    today = dt.datetime.now(paris_tz).date()
    start_date, end_date = next_weekend(today, paris_tz)
    print(f"Fetching fixtures from {start_date} to {end_date}")
    predictions = []
    for league, info in LEAGUES.items():
        fixtures = fetch_fixture_list(info["wf_slug"], start_date, end_date)
        print(f"{league}: Found {len(fixtures)} fixtures")
        for home_team, away_team, match_date in fixtures:
            # Search for team IDs
            home_id = search_team(home_team)
            time.sleep(0.2)  # Rate limiting
            away_id = search_team(away_team)
            time.sleep(0.2)
            if home_id is None or away_id is None:
                print(f"Skipping {home_team} vs {away_team}: team ID not found")
                continue
            # Get ratings
            home_rating = get_team_rating(home_id, info["ut_id"], info["season_id"])
            time.sleep(0.2)
            away_rating = get_team_rating(away_id, info["ut_id"], info["season_id"])
            time.sleep(0.2)
            if home_rating is None or away_rating is None:
                print(f"Skipping {home_team} vs {away_team}: rating not found")
                continue
            # Apply rules
            result = apply_rules(home_rating, away_rating, rules_df)
            if result:
                prediction, rule_name, roi, win_rate = result
                predictions.append({
                    "League": league,
                    "Date": match_date.isoformat(),
                    "HomeTeam": home_team,
                    "AwayTeam": away_team,
                    "HomeRating": round(home_rating, 4),
                    "AwayRating": round(away_rating, 4),
                    "Prediction": prediction,
                    "RuleName": rule_name,
                    "ROI": roi,
                    "WinRate": win_rate,
                })
    # Write output CSV
    if predictions:
        fieldnames = [
            "League", "Date", "HomeTeam", "AwayTeam", "HomeRating",
            "AwayRating", "Prediction", "RuleName", "ROI", "WinRate"
        ]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in predictions:
                writer.writerow(row)
        print(f"Wrote {len(predictions)} predictions to {output_path}")
    else:
        print("No predictions met profitable rule criteria during this period.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate football match predictions using SofaScore data and backtested rules.")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path")
    args = parser.parse_args()
    main(args.output)
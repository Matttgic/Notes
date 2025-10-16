#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
predict.py
- Récupère les fixtures du prochain week-end (ven→lun) depuis football-data.org (stable)
- Mappe les noms vers IDs SofaScore via /unique-tournament/{uid}/season/{sid}/teams
- Récupère avgRating chez SofaScore /team/{id}/unique-tournament/{uid}/season/{sid}/statistics/overall
- Applique les règles ROI (CSV) et écrit un CSV: date, league, home, away, prediction, rule, ROI, success_rate

Usage:
  python predict.py --output predictions.csv
"""

import argparse
import datetime as dt
import difflib
import os
import sys
import time
import unicodedata
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

# ----------- Config ----------
HEADERS = {
    "User-Agent": "Mozilla/5.0 (GitHub Actions; +https://github.com)",
    "Accept": "application/json",
}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)
TIMEOUT = 25

# Leagues mapping: football-data.org code -> (SofaScore uid, SofaScore season_id, display name)
LEAGUES = {
    "PL":  {"uid": 17, "season_id": 76986, "name": "Premier League"},
    "FL1": {"uid": 34, "season_id": 77356, "name": "Ligue 1"},
    "BL1": {"uid": 35, "season_id": 77333, "name": "Bundesliga"},
    "SA":  {"uid": 23, "season_id": 76457, "name": "Serie A"},
    "PD":  {"uid": 8,  "season_id": 77559, "name": "La Liga"},
}
FD_COMP_LIST = ",".join(LEAGUES.keys())

RULE_PRIORITY = [
    "weak_vs_strong",
    "away_strong_advantage",
    "balanced_match",
    "elite_home_team",
    "strong_home_combo",
    "home_strong_advantage",
]

# ----------- Dates ----------
def next_weekend_dates(today_utc: dt.date) -> List[dt.date]:
    dow = today_utc.weekday()  # Mon=0 .. Sun=6
    days_until_friday = (4 - dow) % 7
    fri = today_utc + dt.timedelta(days=days_until_friday)
    return [fri + dt.timedelta(days=i) for i in range(4)]  # Fri..Mon

# ----------- Helpers ----------
def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII")
    return s.lower().replace("&", "and").replace(".", "").replace("  ", " ").strip()

def best_match(name: str, candidates: List[str]) -> Optional[str]:
    if not candidates:
        return None
    nm = norm(name)
    cand_norm = {c: norm(c) for c in candidates}
    # exact
    for raw, cn in cand_norm.items():
        if cn == nm:
            return raw
    # close
    close = difflib.get_close_matches(nm, list(cand_norm.values()), n=1, cutoff=0.82)
    if close:
        target = close[0]
        for raw, cn in cand_norm.items():
            if cn == target:
                return raw
    return None

# ----------- SofaScore ----------
def ss_get(url: str) -> dict:
    r = SESSION.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def sofascore_teams(uid: int, season_id: int) -> Dict[str, int]:
    url = f"https://www.sofascore.com/api/v1/unique-tournament/{uid}/season/{season_id}/teams"
    js = ss_get(url)
    mapping = {}
    for t in js.get("teams", []):
        name = t.get("name")
        tid = t.get("id")
        if name and tid:
            mapping[name] = int(tid)
    return mapping

def sofascore_avg_rating(team_id: int, uid: int, season_id: int) -> Optional[float]:
    url = f"https://www.sofascore.com/api/v1/team/{team_id}/unique-tournament/{uid}/season/{season_id}/statistics/overall"
    js = ss_get(url)
    return js.get("statistics", {}).get("avgRating")

# ----------- Football-data.org (fixtures) ----------
def fd_get_matches(api_token: str, date_from: dt.date, date_to: dt.date) -> List[dict]:
    url = (
        "https://api.football-data.org/v4/matches"
        f"?competitions={FD_COMP_LIST}&dateFrom={date_from.isoformat()}&dateTo={date_to.isoformat()}"
    )
    headers = {"X-Auth-Token": api_token, **HEADERS}
    r = requests.get(url, headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data.get("matches", [])

def collect_fixtures(api_token: str, dates: List[dt.date]) -> List[dict]:
    start, end = min(dates), max(dates)
    matches = fd_get_matches(api_token, start, end)
    fixtures = []
    for m in matches:
        try:
            comp = m["competition"]["code"]  # PL/FL1/BL1/SA/PD
            if comp not in LEAGUES:
                continue
            status = m.get("status")
            if status not in ("SCHEDULED", "TIMED", "POSTPONED"):
                continue
            utc_date = m["utcDate"][:10]  # 'YYYY-MM-DD'
            home = m["homeTeam"]["name"]
            away = m["awayTeam"]["name"]
            fixtures.append(
                {
                    "date": utc_date,
                    "fd_code": comp,
                    "league": LEAGUES[comp]["name"],
                    "uid": LEAGUES[comp]["uid"],
                    "season_id": LEAGUES[comp]["season_id"],
                    "home_name": home,
                    "away_name": away,
                }
            )
        except Exception:
            continue
    return fixtures

# ----------- Règles ----------
def load_rules(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["Rule_Name"].isin(RULE_PRIORITY)].copy()
    for col in ("ROI_Percent", "Success_Rate"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.set_index("Rule_Name")

def apply_rules(home_rating: Optional[float], away_rating: Optional[float], rules_df: pd.DataFrame):
    if home_rating is None or away_rating is None:
        return None
    diff = home_rating - away_rating
    for rule in RULE_PRIORITY:
        if rule not in rules_df.index:
            continue
        if rule == "weak_vs_strong":
            if home_rating < 6.7 and away_rating > 6.9:
                return rule, "Away Win"
        elif rule == "away_strong_advantage":
            if diff < -0.25:
                return rule, "Away Win"
        elif rule == "balanced_match":
            if abs(diff) < 0.05:
                return rule, "Draw"
        elif rule == "elite_home_team":
            if home_rating > 6.95:
                return rule, "Home Win"
        elif rule == "strong_home_combo":
            if home_rating > 6.85 and diff > 0.15:
                return rule, "Home Win"
        elif rule == "home_strong_advantage":
            if diff > 0.25:
                return rule, "Home Win"
    return None

# ----------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="predictions.csv")
    args = ap.parse_args()

    rules_path = os.path.join(os.getcwd(), "complete_predictive_rules_summary.csv")
    if not os.path.exists(rules_path):
        print(f"[ERR] Rules CSV introuvable: {rules_path}", file=sys.stderr)
        sys.exit(1)
    rules_df = load_rules(rules_path)

    fd_token = os.getenv("FOOTBALL_DATA_TOKEN", "").strip()
    if not fd_token:
        print(
            "[ERR] FOOTBALL_DATA_TOKEN manquant. "
            "Ajoutez un secret de dépôt nommé FOOTBALL_DATA_TOKEN et relancez.",
            file=sys.stderr,
        )
        # On sort quand même un CSV vide pour l’artefact
        pd.DataFrame([]).to_csv(args.output, index=False)
        return

    today_utc = dt.datetime.utcnow().date()
    dates = next_weekend_dates(today_utc)
    print("Fenêtre (UTC):", ", ".join(d.isoformat() for d in dates))

    # 1) fixtures depuis football-data
    fixtures = collect_fixtures(fd_token, dates)
    print(f"Fixtures trouvés: {len(fixtures)}")
    if not fixtures:
        pd.DataFrame([]).to_csv(args.output, index=False)
        print("Aucun match à traiter.")
        return

    # 2) pour chaque league, construire mapping name -> sofascore_id
    league_team_maps: Dict[Tuple[int,int], Dict[str,int]] = {}  # (uid, season_id) -> {name:id}
    def get_team_map(uid: int, season_id: int) -> Dict[str,int]:
        key = (uid, season_id)
        if key not in league_team_maps:
            for attempt in range(3):
                try:
                    league_team_maps[key] = sofascore_teams(uid, season_id)
                    break
                except Exception as e:
                    if attempt == 2:
                        print(f"[WARN] teams {uid}/{season_id} a échoué: {e}", file=sys.stderr)
                        league_team_maps[key] = {}
                    time.sleep(0.5)
        return league_team_maps[key]

    # 3) ratings cache
    rating_cache: Dict[Tuple[int,int,int], Optional[float]] = {}

    rows = []
    for fx in fixtures:
        uid = fx["uid"]; season_id = fx["season_id"]
        team_map = get_team_map(uid, season_id)
        candidate_names = list(team_map.keys())

        # match des noms (fd -> sofascore)
        h_match = best_match(fx["home_name"], candidate_names)
        a_match = best_match(fx["away_name"], candidate_names)
        if not h_match or not a_match:
            # log et skip
            print(f"[SKIP] Mapping échoué: {fx['league']} | {fx['home_name']} vs {fx['away_name']}", file=sys.stderr)
            continue

        home_id = team_map[h_match]
        away_id = team_map[a_match]

        # ratings
        for key in [(home_id, uid, season_id), (away_id, uid, season_id)]:
            if key not in rating_cache:
                try:
                    rating_cache[key] = sofascore_avg_rating(*key)
                    time.sleep(0.2)
                except Exception as e:
                    print(f"[WARN] rating {key} échoue: {e}", file=sys.stderr)
                    rating_cache[key] = None

        hr = rating_cache[(home_id, uid, season_id)]
        ar = rating_cache[(away_id, uid, season_id)]
        res = apply_rules(hr, ar, rules_df)
        if not res:
            continue

        rule_name, prediction = res
        roi = rules_df.loc[rule_name]["ROI_Percent"] if "ROI_Percent" in rules_df.columns else None
        sr = rules_df.loc[rule_name]["Success_Rate"] if "Success_Rate" in rules_df.columns else None

        rows.append({
            "date": fx["date"],
            "league": fx["league"],
            "home_team": h_match,
            "away_team": a_match,
            "home_rating": hr,
            "away_rating": ar,
            "rating_diff": None if hr is None or ar is None else (hr - ar),
            "prediction": prediction,
            "method_rule": rule_name,
            "rule_roi_percent": roi,
            "rule_success_rate": sr,
        })

    out = pd.DataFrame(rows).sort_values(["date", "league", "home_team"])
    out.to_csv(args.output, index=False)
    print(f"Ecrit: {args.output} ({len(out)} lignes)")

if __name__ == "__main__":
    main()

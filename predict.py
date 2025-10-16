#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
predict.py (mapping statique)
- Fixtures: football-data.org (secret FOOTBALL_DATA_TOKEN)
- Ratings: SofaScore /team/{id}/unique-tournament/{uid}/season/{sid}/statistics/overall
- Mapping nom->team_id SofaScore: fichier local sofa_team_map.csv (pas d'appel /teams)
- Règles ROI depuis complete_predictive_rules_summary.csv
- Sortie: CSV avec date, league, home, away, prediction, rule, ROI, success_rate
"""

import argparse
import datetime as dt
import os
import sys
import time
import unicodedata
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (GitHub Actions; +https://github.com)",
    "Accept": "application/json",
}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)
TIMEOUT = 25

# football-data -> SofaScore
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

def next_weekend_dates(today_utc: dt.date) -> List[dt.date]:
    dow = today_utc.weekday()  # Mon=0..Sun=6
    days_until_friday = (4 - dow) % 7
    fri = today_utc + dt.timedelta(days=days_until_friday)
    return [fri + dt.timedelta(days=i) for i in range(4)]  # Fri..Mon

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
            utc_date = m["utcDate"][:10]
            home = m["homeTeam"]["name"]
            away = m["awayTeam"]["name"]
            fixtures.append(
                {
                    "date": utc_date,
                    "fd_code": comp,
                    "league": LEAGUES[comp]["name"],
                    "uid": LEAGUES[comp]["uid"],
                    "season_id": LEAGUES[comp]["season_id"],
                    "home_name_fd": home,
                    "away_name_fd": away,
                }
            )
        except Exception:
            continue
    return fixtures

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

# ---------- SofaScore ratings ----------
def sofascore_avg_rating(team_id: int, uid: int, season_id: int) -> Optional[float]:
    url = f"https://www.sofascore.com/api/v1/team/{team_id}/unique-tournament/{uid}/season/{season_id}/statistics/overall"
    r = SESSION.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    js = r.json()
    return js.get("statistics", {}).get("avgRating")

# ---------- Mapping loader ----------
def load_sofa_team_map(path: str) -> pd.DataFrame:
    """
    Charge sofa_team_map.csv avec colonnes:
    league,season_id,team_name_fd,team_name_ss,team_id_ss
    """
    if not os.path.exists(path):
        print(f"[WARN] Mapping {path} introuvable. Aucun match ne sera mappé.", file=sys.stderr)
        return pd.DataFrame(columns=["league","season_id","team_name_fd","team_name_ss","team_id_ss"])
    df = pd.read_csv(path, dtype={"team_id_ss": str})
    # nettoie/trims
    for c in ("league","team_name_fd","team_name_ss"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    if "season_id" in df.columns:
        df["season_id"] = pd.to_numeric(df["season_id"], errors="coerce").astype("Int64")
    if "team_id_ss" in df.columns:
        df["team_id_ss"] = df["team_id_ss"].str.extract(r"(\d+)")[0]
    return df

def get_team_id_from_map(map_df: pd.DataFrame, league: str, season_id: int, team_name_fd: str) -> Optional[int]:
    if map_df.empty:
        return None
    try:
        candidates = map_df[(map_df["league"]==league) & (map_df["season_id"]==season_id)]
        # essai match direct sur team_name_fd
        row = candidates[candidates["team_name_fd"].str.casefold()==team_name_fd.casefold()].head(1)
        if row.empty:
            # essai sur team_name_ss (au cas où on a mis le nom SofaScore à la place)
            row = candidates[candidates["team_name_ss"].str.casefold()==team_name_fd.casefold()].head(1)
        if row.empty:
            return None
        tid = row.iloc[0]["team_id_ss"]
        return int(tid) if pd.notna(tid) else None
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="predictions.csv")
    args = ap.parse_args()

    # Règles
    rules_path = os.path.join(os.getcwd(), "complete_predictive_rules_summary.csv")
    if not os.path.exists(rules_path):
        print(f"[ERR] Rules CSV introuvable: {rules_path}", file=sys.stderr)
        sys.exit(1)
    rules_df = load_rules(rules_path)

    # Token football-data
    fd_token = os.getenv("FOOTBALL_DATA_TOKEN", "").strip()
    if not fd_token:
        print("[ERR] FOOTBALL_DATA_TOKEN manquant (secret repo).", file=sys.stderr)
        pd.DataFrame([], columns=[
            "date","league","home_team","away_team","prediction","method_rule",
            "rule_roi_percent","rule_success_rate","home_rating","away_rating","rating_diff"
        ]).to_csv(args.output, index=False)
        return

    # Dates
    today_utc = dt.datetime.utcnow().date()
    dates = next_weekend_dates(today_utc)
    print("Fenêtre (UTC):", ", ".join(d.isoformat() for d in dates))

    # Fixtures
    fixtures = collect_fixtures(fd_token, dates)
    print(f"Fixtures trouvés: {len(fixtures)}")
    if not fixtures:
        pd.DataFrame([], columns=[
            "date","league","home_team","away_team","prediction","method_rule",
            "rule_roi_percent","rule_success_rate","home_rating","away_rating","rating_diff"
        ]).to_csv(args.output, index=False)
        print("Aucun match à traiter.")
        return

    # Mapping statique
    map_path = os.path.join(os.getcwd(), "sofa_team_map.csv")
    map_df = load_sofa_team_map(map_path)

    rating_cache: Dict[Tuple[int,int,int], Optional[float]] = {}
    rows = []

    skipped = 0
    for fx in fixtures:
        league = fx["league"]; uid = fx["uid"]; season_id = fx["season_id"]
        h_fd = fx["home_name_fd"]; a_fd = fx["away_name_fd"]

        home_id = get_team_id_from_map(map_df, league, season_id, h_fd)
        away_id = get_team_id_from_map(map_df, league, season_id, a_fd)

        if not home_id or not away_id:
            print(f"[SKIP] Mapping manquant: {league} | {h_fd} vs {a_fd}", file=sys.stderr)
            skipped += 1
            continue

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
            "league": league,
            "home_team": h_fd,
            "away_team": a_fd,
            "prediction": prediction,
            "method_rule": rule_name,
            "rule_roi_percent": roi,
            "rule_success_rate": sr,
            "home_rating": hr,
            "away_rating": ar,
            "rating_diff": None if (hr is None or ar is None) else (hr - ar),
        })

    # Sortie (safe même vide)
    cols = ["date","league","home_team","away_team","prediction","method_rule",
            "rule_roi_percent","rule_success_rate","home_rating","away_rating","rating_diff"]
    out = pd.DataFrame(rows, columns=cols)
    if not out.empty:
        out = out.sort_values(["date", "league", "home_team"])
    out.to_csv(args.output, index=False)
    print(f"Ecrit: {args.output} ({len(out)} lignes, {skipped} matches ignorés faute de mapping)")

if __name__ == "__main__":
    main()

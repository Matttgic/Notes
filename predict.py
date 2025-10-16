#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
predict.py (saison 2025-2026, mapping statique robuste)
- Fixtures: football-data.org (secret FOOTBALL_DATA_TOKEN)
- Ratings: SofaScore /team/{id}/unique-tournament/{uid}/season/{sid}/statistics/overall
- Mapping nom -> team_id SofaScore: CSV local (par défaut data/team_map_2025_26.csv)
  Colonnes acceptées (au moins celles marquées *):
    *team_name_fd, *team_id_ss
     team_name_ss (optionnel)
     league (nom complet, ex "Bundesliga") OU fd_code (PL/FL1/BL1/SA/PD)
     season_id (optionnel)
- Règles ROI: complete_predictive_rules_summary.csv
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

# ------------ HTTP session ------------
HEADERS = {
    "User-Agent": "Mozilla/5.0 (GitHub Actions; +https://github.com)",
    "Accept": "application/json",
}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)
TIMEOUT = 25

# ------------ Ligues (football-data -> SofaScore) ------------
LEAGUES = {
    "PL":  {"uid": 17, "season_id": 76986, "name": "Premier League"},
    "FL1": {"uid": 34, "season_id": 77356, "name": "Ligue 1"},
    "BL1": {"uid": 35, "season_id": 77333, "name": "Bundesliga"},
    "SA":  {"uid": 23, "season_id": 76457, "name": "Serie A"},
    "PD":  {"uid": 8,  "season_id": 77559, "name": "La Liga"},
}
FD_COMP_LIST = ",".join(LEAGUES.keys())

# ------------ Priorité des règles (doit correspondre au CSV) ------------
RULE_PRIORITY = [
    "weak_vs_strong",
    "away_strong_advantage",
    "balanced_match",
    "elite_home_team",
    "strong_home_combo",
    "home_strong_advantage",
]

# ------------ Helpers de normalisation ------------
_STOP_TOKENS = {
    "fc","cf","ac","as","rc","rcd","sd","ud","cd","ssc","tsg","sv","sc",
    "club","de","the","and","u.","v.","a.","b.","c.","d.",
    # années fréquentes / numéros
    "04","05","06","07","08","09","1899","1901","1903","1907","1909","1910",
}

def _strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

def _norm(s: str) -> str:
    s = str(s or "")
    s = _strip_accents(s.lower())
    for ch in "-'’.,()&/+":
        s = s.replace(ch, " ")
    s = s.replace("saint-germain", "saint germain")  # PSG variantes
    tokens = [t for t in s.split() if t and t not in _STOP_TOKENS and not t.isdigit()]
    return " ".join(tokens)

def _eq(a: str, b: str) -> bool:
    return _norm(a) == _norm(b)

# ------------ Dates ------------
def next_weekend_dates(today_utc: dt.date) -> List[dt.date]:
    """Vendredi -> Lundi inclus"""
    dow = today_utc.weekday()  # Mon=0..Sun=6
    fri = today_utc + dt.timedelta(days=(4 - dow) % 7)
    return [fri + dt.timedelta(days=i) for i in range(4)]

# ------------ football-data.org (fixtures) ------------
def fd_get_matches(api_token: str, date_from: dt.date, date_to: dt.date) -> List[dict]:
    url = (
        "https://api.football-data.org/v4/matches"
        f"?competitions={FD_COMP_LIST}&dateFrom={date_from.isoformat()}&dateTo={date_to.isoformat()}"
    )
    headers = {"X-Auth-Token": api_token, **HEADERS}
    r = requests.get(url, headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json().get("matches", [])

def collect_fixtures(api_token: str, dates: List[dt.date]) -> List[dict]:
    start, end = min(dates), max(dates)
    matches = fd_get_matches(api_token, start, end)
    fixtures = []
    for m in matches:
        try:
            comp = m["competition"]["code"]
            if comp not in LEAGUES:
                continue
            status = m.get("status")
            if status not in ("SCHEDULED", "TIMED", "POSTPONED"):
                continue
            fixtures.append(
                {
                    "date": m["utcDate"][:10],
                    "fd_code": comp,
                    "league": LEAGUES[comp]["name"],
                    "uid": LEAGUES[comp]["uid"],
                    "season_id": LEAGUES[comp]["season_id"],
                    "home_name_fd": m["homeTeam"]["name"],
                    "away_name_fd": m["awayTeam"]["name"],
                }
            )
        except Exception:
            continue
    return fixtures

# ------------ Règles ------------
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

# ------------ SofaScore (ratings équipe) ------------
def sofascore_avg_rating(team_id: int, uid: int, season_id: int) -> Optional[float]:
    url = f"https://www.sofascore.com/api/v1/team/{team_id}/unique-tournament/{uid}/season/{season_id}/statistics/overall"
    r = SESSION.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    js = r.json()
    return js.get("statistics", {}).get("avgRating")

# ------------ Mapping nom -> team_id (CSV local) ------------
def load_team_map(path: str) -> pd.DataFrame:
    """
    Colonnes possibles (min requis: team_name_fd, team_id_ss):
      league (nom complet) OU fd_code (PL/FL1/BL1/SA/PD)
      season_id (optionnel)
      team_name_fd (nom FD)
      team_name_ss (optionnel, purement informatif)
      team_id_ss (numérique)
    """
    if not os.path.exists(path):
        print(f"[WARN] Mapping introuvable: {path}. Aucun match ne sera mappé.", file=sys.stderr)
        return pd.DataFrame(columns=["fd_code","league","season_id","team_name_fd","team_name_ss","team_id_ss"])

    df = pd.read_csv(path, dtype={"team_id_ss": str})
    # Normalisation douce
    for c in ("league","fd_code","team_name_fd","team_name_ss"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    if "season_id" in df.columns:
        df["season_id"] = pd.to_numeric(df["season_id"], errors="coerce").astype("Int64")
    if "team_id_ss" in df.columns:
        df["team_id_ss"] = df["team_id_ss"].str.extract(r"(\d+)")[0]

    # Ajout colonnes de travail
    if "fd_code" not in df.columns:
        df["fd_code"] = None
    if "league" not in df.columns:
        df["league"] = None

    # Clé normalisée pour accélérer les recherches par nom
    df["name_key"] = df["team_name_fd"].apply(_norm)
    return df

def _match_pool(map_df: pd.DataFrame, fd_code: str, league_name: str, season_id: int) -> pd.DataFrame:
    pool = map_df.copy()
    # Filtrage ligue: accepte code OU nom, ou rien (mapping global)
    pool = pool[
        (pool["fd_code"].isna() | (pool["fd_code"] == fd_code)) &
        (pool["league"].isna() | (pool["league"] == league_name))
    ]
    # Filtrage saison si présente dans le CSV
    if "season_id" in pool.columns and pool["season_id"].notna().any():
        pool = pool[(pool["season_id"].isna()) | (pool["season_id"] == season_id)]
    return pool

def get_team_id(map_df: pd.DataFrame, fd_code: str, league_name: str, season_id: int, team_name_fd: str) -> Optional[int]:
    if map_df.empty:
        return None
    pool = _match_pool(map_df, fd_code, league_name, season_id)
    if pool.empty:
        return None

    key = _norm(team_name_fd)
    # 1) match exact normalisé sur team_name_fd
    row = pool[pool["name_key"] == key].head(1)
    if row.empty and "team_name_ss" in pool.columns:
        # 2) match exact normalisé sur team_name_ss
        pool2 = pool.copy()
        pool2["name_key_ss"] = pool2["team_name_ss"].apply(_norm)
        row = pool2[pool2["name_key_ss"] == key].head(1)

    if row.empty:
        # 3) fallback "commence par" (utile pour longs noms)
        row = pool[pool["name_key"].str.startswith(key)].head(1)
    if row.empty:
        row = pool[pool["name_key"].str.contains(key)].head(1)

    if row.empty:
        return None
    tid = row.iloc[0]["team_id_ss"]
    try:
        return int(tid) if pd.notna(tid) else None
    except Exception:
        return None

# ------------ Main ------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="predictions.csv", help="Nom du fichier CSV de sortie")
    parser.add_argument("--map", default=os.path.join("data", "team_map_2025_26.csv"),
                        help="Chemin du fichier mapping nom->team_id SofaScore")
    args = parser.parse_args()

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
        empty_cols = ["date","league","home_team","away_team","prediction","method_rule",
                      "rule_roi_percent","rule_success_rate","home_rating","away_rating","rating_diff"]
        pd.DataFrame([], columns=empty_cols).to_csv(args.output, index=False)
        return

    # Fenêtre de dates: prochain week-end en UTC
    today_utc = dt.datetime.utcnow().date()
    dates = next_weekend_dates(today_utc)
    print("Fenêtre (UTC):", ", ".join(d.isoformat() for d in dates))

    # Fixtures
    fixtures = collect_fixtures(fd_token, dates)
    print(f"Fixtures trouvés: {len(fixtures)}")
    if not fixtures:
        empty_cols = ["date","league","home_team","away_team","prediction","method_rule",
                      "rule_roi_percent","rule_success_rate","home_rating","away_rating","rating_diff"]
        pd.DataFrame([], columns=empty_cols).to_csv(args.output, index=False)
        print("Aucun match à traiter.")
        return

    # Mapping local
    map_path = os.path.abspath(args.map)
    print(f"Mapping: {map_path}")
    map_df = load_team_map(map_path)

    rating_cache: Dict[Tuple[int,int,int], Optional[float]] = {}
    rows = []
    skipped = 0

    for fx in fixtures:
        league_name = fx["league"]
        fd_code = fx["fd_code"]
        uid = fx["uid"]
        season_id = fx["season_id"]
        h_fd = fx["home_name_fd"]
        a_fd = fx["away_name_fd"]

        home_id = get_team_id(map_df, fd_code, league_name, season_id, h_fd)
        away_id = get_team_id(map_df, fd_code, league_name, season_id, a_fd)

        if not home_id or not away_id:
            print(f"[SKIP] Mapping manquant: {league_name} | {h_fd} vs {a_fd}", file=sys.stderr)
            skipped += 1
            continue

        # Ratings (avec cache)
        for key in [(home_id, uid, season_id), (away_id, uid, season_id)]:
            if key not in rating_cache:
                try:
                    rating_cache[key] = sofascore_avg_rating(*key)
                    time.sleep(0.2)  # douceur
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
            "league": league_name,
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

    # Sortie CSV (safe même si vide)
    cols = ["date","league","home_team","away_team","prediction","method_rule",
            "rule_roi_percent","rule_success_rate","home_rating","away_rating","rating_diff"]
    out = pd.DataFrame(rows, columns=cols)
    if not out.empty:
        out = out.sort_values(["date", "league", "home_team"])
    out.to_csv(args.output, index=False)
    print(f"Ecrit: {args.output} ({len(out)} lignes, {skipped} matches ignorés faute de mapping)")

if __name__ == "__main__":
    main()

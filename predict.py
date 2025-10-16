#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
predict.py
- Récupère les fixtures du prochain week-end via l'API SofaScore (scheduled-events par date)
- Calcule les avgRating des équipes via l'API SofaScore (statistics/overall)
- Applique les règles les plus rentables de complete_predictive_rules_summary.csv
- Écrit un CSV avec: date, league, home, away, prediction, rule, ROI, success_rate

Usage:
  python predict.py --output predictions.csv
"""

import argparse
import datetime as dt
from datetime import timezone, timedelta
import os
import sys
import time
import pandas as pd
import requests

# ---------- Config de base ----------
HEADERS = {
    "User-Agent": "Mozilla/5.0 (GitHub Actions; +https://github.com)",
    "Accept": "application/json",
}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)
TIMEOUT = 20

# UniqueTournament IDs et Season IDs 2025/26 (confirmés par appels précédents)
LEAGUES = {
    "Premier League": {"uid": 17, "season_id": 76986, "slug": "ENG"},
    "Ligue 1": {"uid": 34, "season_id": 77356, "slug": "FRA"},
    "Bundesliga": {"uid": 35, "season_id": 77333, "slug": "GER"},
    "Serie A": {"uid": 23, "season_id": 76457, "slug": "ITA"},
    "La Liga": {"uid": 8, "season_id": 77559, "slug": "ESP"},
}

# Règles à ROI élevé: ordre de priorité (décroissant ROI) — les noms doivent matcher Rule_Name
RULE_PRIORITY = [
    "weak_vs_strong",
    "away_strong_advantage",
    "balanced_match",
    "elite_home_team",
    "strong_home_combo",
    "home_strong_advantage",
]

# ---------- Utilitaires dates ----------
def next_weekend_window(today_utc: dt.date) -> list[dt.date]:
    """
    Retourne la liste des dates (UTC) du prochain week-end: vendredi -> lundi inclus.
    Les runners GitHub sont en UTC, ça convient.
    """
    # Trouver prochain samedi (weekday 5 = samedi? Non: Monday=0). Saturday = 5.
    # En Python: Monday=0 ... Sunday=6, donc Saturday=5, Friday=4.
    dow = today_utc.weekday()
    days_until_friday = (4 - dow) % 7
    friday = today_utc + dt.timedelta(days=days_until_friday)
    saturday = friday + dt.timedelta(days=1)
    sunday = friday + dt.timedelta(days=2)
    monday = friday + dt.timedelta(days=3)
    return [friday, saturday, sunday, monday]

# ---------- Récup fixtures ----------
def fetch_scheduled_events_for_date(date_obj: dt.date) -> dict:
    """
    GET https://www.sofascore.com/api/v1/sport/football/scheduled-events/{YYYY-MM-DD}
    Retourne le JSON dict ou {}.
    """
    url = f"https://www.sofascore.com/api/v1/sport/football/scheduled-events/{date_obj.isoformat()}"
    r = SESSION.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def collect_fixtures_window(dates: list[dt.date]) -> list[dict]:
    """
    Agrège les événements sur l’intervalle de dates.
    Filtre uniquement nos 5 ligues (par uniqueTournament.id) et les matches non démarrés.
    Retourne une liste d'objets: {date, league, tournament_uid, season_id, home_id, away_id, home_name, away_name}
    """
    fixtures = []
    valid_uids = {cfg["uid"] for cfg in LEAGUES.values()}

    for d in dates:
        try:
            data = fetch_scheduled_events_for_date(d)
        except Exception as e:
            print(f"Warning: scheduled-events {d} échoue: {e}", file=sys.stderr)
            continue

        events = data.get("events", []) or data.get("scheduledEvents", []) or []
        for ev in events:
            try:
                tu = ev.get("tournament", {}).get("uniqueTournament", {}) or ev.get("uniqueTournament", {})
                uid = tu.get("id")
                if uid not in valid_uids:
                    continue

                status = ev.get("status", {}).get("type")
                if status not in ("notstarted", "postponed", "canceled"):
                    # on ne prédit que les matches à venir / pas commencés
                    if status != "notstarted":
                        continue

                home = ev.get("homeTeam") or ev.get("home")
                away = ev.get("awayTeam") or ev.get("away")
                if not home or not away:
                    continue

                # Résolution du nom de league + season_id
                league_name = None
                season_id = None
                for name, cfg in LEAGUES.items():
                    if cfg["uid"] == uid:
                        league_name = name
                        season_id = cfg["season_id"]
                        break
                if not league_name:
                    continue

                fixtures.append({
                    "date": dt.datetime.fromtimestamp(ev.get("startTimestamp", 0), tz=timezone.utc).date().isoformat(),
                    "league": league_name,
                    "tournament_uid": uid,
                    "season_id": season_id,
                    "home_id": home.get("id"),
                    "away_id": away.get("id"),
                    "home_name": home.get("name"),
                    "away_name": away.get("name"),
                })
            except Exception:
                # ignorer l'event si structure inattendue
                continue
    return fixtures

# ---------- Notes d’équipes ----------
def fetch_team_avg_rating(team_id: int, tournament_uid: int, season_id: int) -> float | None:
    """
    GET /api/v1/team/{team}/unique-tournament/{uid}/season/{sid}/statistics/overall
    Retourne avgRating ou None.
    """
    url = f"https://www.sofascore.com/api/v1/team/{team_id}/unique-tournament/{tournament_uid}/season/{season_id}/statistics/overall"
    r = SESSION.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    js = r.json()
    return js.get("statistics", {}).get("avgRating")

# ---------- Règles ----------
def load_rules(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # On garde seulement les règles du top ROI (liste RULE_PRIORITY) si présentes
    df = df[df["Rule_Name"].isin(RULE_PRIORITY)].copy()
    # on s’assure que ROI_Percent et Success_Rate sont numériques
    for col in ("ROI_Percent", "Success_Rate"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # on indexe par Rule_Name pour lookup rapide
    return df.set_index("Rule_Name")

def apply_rules(home_rating: float, away_rating: float, rules_df: pd.DataFrame) -> tuple[str, str] | None:
    """
    Applique les règles par ordre RULE_PRIORITY.
    Retourne (rule_name, prediction_text) ou None si aucune règle ne matche.
    """
    if home_rating is None or away_rating is None:
        return None
    diff = home_rating - away_rating

    # mapping des conditions
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

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="predictions.csv", help="Nom du fichier CSV de sortie")
    args = parser.parse_args()

    rules_path = os.path.join(os.getcwd(), "complete_predictive_rules_summary.csv")
    if not os.path.exists(rules_path):
        print(f"Rules CSV introuvable à {rules_path}", file=sys.stderr)
        sys.exit(1)

    rules_df = load_rules(rules_path)

    today_utc = dt.datetime.now(tz=timezone.utc).date()
    window = next_weekend_window(today_utc)
    print(f"Fetching fixtures (SofaScore scheduled-events) for: {', '.join(d.isoformat() for d in window)}")

    fixtures = collect_fixtures_window(window)
    print(f"Fixtures trouvés: {len(fixtures)}")

    rows = []
    # Cache simple pour éviter de recharger 2x la même note
    rating_cache: dict[tuple[int,int,int], float | None] = {}

    for fx in fixtures:
        key_home = (fx["home_id"], fx["tournament_uid"], fx["season_id"])
        key_away = (fx["away_id"], fx["tournament_uid"], fx["season_id"])

        if key_home not in rating_cache:
            try:
                rating_cache[key_home] = fetch_team_avg_rating(*key_home)
                time.sleep(0.2)
            except Exception as e:
                print(f"Warn: rating home {key_home} failed: {e}", file=sys.stderr)
                rating_cache[key_home] = None
        if key_away not in rating_cache:
            try:
                rating_cache[key_away] = fetch_team_avg_rating(*key_away)
                time.sleep(0.2)
            except Exception as e:
                print(f"Warn: rating away {key_away} failed: {e}", file=sys.stderr)
                rating_cache[key_away] = None

        hr = rating_cache[key_home]
        ar = rating_cache[key_away]
        result = apply_rules(hr, ar, rules_df)
        if not result:
            continue

        rule_name, prediction = result
        roi = rules_df.loc[rule_name]["ROI_Percent"] if "ROI_Percent" in rules_df.columns else None
        sr = rules_df.loc[rule_name]["Success_Rate"] if "Success_Rate" in rules_df.columns else None

        rows.append({
            "date": fx["date"],
            "league": fx["league"],
            "home_team": fx["home_name"],
            "away_team": fx["away_name"],
            "home_rating": hr,
            "away_rating": ar,
            "rating_diff": None if hr is None or ar is None else (hr - ar),
            "prediction": prediction,
            "method_rule": rule_name,
            "rule_roi_percent": roi,
            "rule_success_rate": sr,
        })

    if not rows:
        print("Aucune rencontre ne satisfait les règles profitables sur cette fenêtre.")
        # On sort quand même un CSV minimal pour que l’artefact existe
        pd.DataFrame(rows).to_csv(args.output, index=False)
        return

    out_df = pd.DataFrame(rows).sort_values(["date", "league", "home_team"])
    out_df.to_csv(args.output, index=False)
    print(f"Écrit: {args.output} ({len(out_df)} lignes)")

if __name__ == "__main__":
    main()

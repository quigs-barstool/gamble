"""
Excel output:
- SeasonAverages: all players season per-game averages (PTS/REB/AST/MIN)
- TodaysPlayers: players on rosters of teams that play today
  + Projected_MIN from SportsLine (diacritics-safe matching)
  + PRA, ProjMIN_minus_SeasonMIN, PRA_per_min_times_ProjMIN
  + Opp_PTS_Allowed_Rank, Opp_REB_Allowed_Rank
  + Opp_PTS_Allowed_Scaler, Weighted_PTS
  + Opp_REB_Allowed_Scaler, Weighted_REB
  + DraftKings props via The Odds API (bookmaker=draftkings):
      DK_PTS_Line, DK_PTS_Over_Odds, DK_PTS_Under_Odds
      DK_PRA_Line, DK_PRA_Over_Odds, DK_PRA_Under_Odds
  + NEW:
      Weighted_PTS_Proj = (Weighted_PTS / MIN) * Projected_MIN
      Weighted_PRA      = ((Weighted_PTS + Weighted_REB + AST) / MIN) * Projected_MIN
      Weighted_Differential = Weighted_PRA - DK_PRA_Line
"""

import os
import re
import time
import requests
import pandas as pd
import unicodedata
from io import StringIO
from datetime import datetime, date, time as dtime
from zoneinfo import ZoneInfo

from nba_api.stats.endpoints import (
    LeagueDashPlayerStats,
    LeagueDashTeamStats,
    ScoreboardV2,
    CommonTeamRoster,
)
from nba_api.stats.library.parameters import SeasonTypeAllStar
from nba_api.stats.library.http import NBAStatsHTTP


# --------------------------------------------------
# CONFIG
# --------------------------------------------------
SEASON = "2025-26"
SEASON_TYPE = SeasonTypeAllStar.regular
SLEEP_BETWEEN_CALLS = 0.6

OUTPUT_XLSX = f"player_avgs_plus_today_{SEASON.replace('-', '')}.xlsx"
MIN_GP_FOR_SEASON_SHEET = 0

SPORTSLINE_URL = "https://www.sportsline.com/nba/expert-projections/simulation/"

# Scaler mapping: Rank 1 -> 0.8, Rank 30 -> 1.2
RANK_MIN = 1
RANK_MAX = 30
SCALER_MIN = 0.8
SCALER_MAX = 1.2

# The Odds API
ODDS_API_KEY_ENV = "ODDS_API_KEY"
ODDS_API_HOST = "https://api.the-odds-api.com"
ODDS_SPORT_KEY = "basketball_nba"
ODDS_REGIONS = "us"
ODDS_BOOKMAKERS = "draftkings"
ODDS_ODDS_FORMAT = "american"

# Only what you want:
ODDS_MARKETS = [
    "player_points",
    "player_points_rebounds_assists",
]

# Browser-like headers (important)
NBA_HEADERS = {
    "Host": "stats.nba.com",
    "Connection": "keep-alive",
    "Accept": "application/json, text/plain, */*",
    "x-nba-stats-token": "true",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept-Language": "en-US,en;q=0.9",
}
NBAStatsHTTP.headers.update(NBA_HEADERS)

SPORTSLINE_HEADERS = {
    "User-Agent": NBA_HEADERS["User-Agent"],
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
    "Connection": "keep-alive",
}

ODDS_HEADERS = {
    "User-Agent": NBA_HEADERS["User-Agent"],
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}


# --------------------------------------------------
# NAME NORMALIZATION
# --------------------------------------------------
def strip_accents(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def normalize_player_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    n = strip_accents(name).lower().strip()
    n = n.replace("`", "'").replace("’", "'").replace("‘", "'")
    n = re.sub(r"[^a-z0-9\s']", " ", n)
    n = n.replace("'", "")
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


# --------------------------------------------------
# RANK -> SCALER
# --------------------------------------------------
def rank_to_scaler(rank_value) -> float | None:
    if pd.isna(rank_value):
        return None
    try:
        r = int(rank_value)
    except Exception:
        return None

    r = max(RANK_MIN, min(RANK_MAX, r))
    step = (SCALER_MAX - SCALER_MIN) / (RANK_MAX - RANK_MIN)
    return SCALER_MIN + (r - RANK_MIN) * step


# --------------------------------------------------
# DATE / SCOREBOARD HELPERS
# --------------------------------------------------
def get_today_date_str_chicago() -> str:
    now = datetime.now(ZoneInfo("America/Chicago"))
    return now.strftime("%m/%d/%Y")


def fetch_today_games_df(game_date: str) -> pd.DataFrame:
    sb = ScoreboardV2(game_date=game_date)
    return sb.get_data_frames()[0]  # GameHeader


def build_team_to_opponent_map(games_df: pd.DataFrame) -> dict[int, int]:
    if games_df is None or games_df.empty:
        return {}

    home = pd.to_numeric(games_df["HOME_TEAM_ID"], errors="coerce")
    away = pd.to_numeric(games_df["VISITOR_TEAM_ID"], errors="coerce")

    team_to_opp: dict[int, int] = {}
    for h, a in zip(home, away):
        if pd.isna(h) or pd.isna(a):
            continue
        h = int(h)
        a = int(a)
        team_to_opp[h] = a
        team_to_opp[a] = h

    return team_to_opp


def chicago_day_bounds_utc(dt_chicago: date) -> tuple[str, str]:
    tz = ZoneInfo("America/Chicago")
    start_local = datetime.combine(dt_chicago, dtime(0, 0, 0), tzinfo=tz)
    end_local = datetime.combine(dt_chicago, dtime(23, 59, 59), tzinfo=tz)
    start_utc = start_local.astimezone(ZoneInfo("UTC"))
    end_utc = end_local.astimezone(ZoneInfo("UTC"))
    return (start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"), end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"))


# --------------------------------------------------
# NBA DATA FETCHERS
# --------------------------------------------------
def fetch_season_per_game_averages(season: str, season_type: str) -> pd.DataFrame:
    resp = LeagueDashPlayerStats(
        season=season,
        season_type_all_star=season_type,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Base",
    )
    df = resp.get_data_frames()[0]

    keep = [
        "PLAYER_ID", "PLAYER_NAME",
        "TEAM_ID", "TEAM_ABBREVIATION",
        "GP", "MIN", "PTS", "REB", "AST",
    ]
    df = df[keep].copy()

    for c in ["PLAYER_ID", "TEAM_ID", "GP", "MIN", "PTS", "REB", "AST"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def fetch_today_team_ids(games_df: pd.DataFrame) -> list[int]:
    if games_df is None or games_df.empty:
        return []

    team_ids = pd.concat(
        [games_df["HOME_TEAM_ID"], games_df["VISITOR_TEAM_ID"]],
        ignore_index=True,
    )

    return (
        pd.to_numeric(team_ids, errors="coerce")
        .dropna()
        .astype(int)
        .drop_duplicates()
        .tolist()
    )


def fetch_roster_players_for_team(team_id: int, season: str) -> pd.DataFrame:
    roster = CommonTeamRoster(team_id=team_id, season=season)
    df = roster.get_data_frames()[0]

    if df is None or df.empty or "PLAYER_ID" not in df.columns:
        return pd.DataFrame(columns=["PLAYER_ID", "PLAYER", "TEAM_ID"])

    out = df.copy()
    out["TEAM_ID"] = team_id
    out["PLAYER_ID"] = pd.to_numeric(out["PLAYER_ID"], errors="coerce")
    out = out.dropna(subset=["PLAYER_ID"]).copy()
    out["PLAYER_ID"] = out["PLAYER_ID"].astype(int)

    return out[["PLAYER_ID", "PLAYER", "TEAM_ID"]].copy()


# --------------------------------------------------
# SportsLine projected minutes
# --------------------------------------------------
def fetch_sportsline_projected_minutes() -> pd.DataFrame:
    r = requests.get(SPORTSLINE_URL, headers=SPORTSLINE_HEADERS, timeout=30)
    r.raise_for_status()

    tables = pd.read_html(StringIO(r.text))

    target = None
    for t in tables:
        cols = [str(c).upper() for c in t.columns]
        if "PLAYER" in cols and "MIN" in cols:
            target = t
            break

    if target is None:
        return pd.DataFrame(columns=["NAME_KEY", "Projected_MIN"])

    target.columns = [str(c).strip() for c in target.columns]
    out = target[["PLAYER", "MIN"]].copy().rename(columns={"MIN": "Projected_MIN"})
    out["Projected_MIN"] = pd.to_numeric(out["Projected_MIN"], errors="coerce")
    out["NAME_KEY"] = out["PLAYER"].apply(normalize_player_name)

    return (
        out.sort_values("Projected_MIN", ascending=False)
        .drop_duplicates(subset=["NAME_KEY"])
        .reset_index(drop=True)
    )


# --------------------------------------------------
# Opponent allowed ranks
# --------------------------------------------------
def fetch_opponent_allowed_ranks(season: str, season_type: str) -> pd.DataFrame:
    resp = LeagueDashTeamStats(
        season=season,
        season_type_all_star=season_type,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Opponent",
    )
    df = resp.get_data_frames()[0].copy()

    def pick_col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    pts_col = pick_col(["OPP_PTS", "OPP_PTS_PG"])
    reb_col = pick_col(["OPP_REB", "OPP_REB_PG"])

    if "TEAM_ID" not in df.columns or pts_col is None or reb_col is None:
        return pd.DataFrame(columns=["TEAM_ID", "OPP_PTS_Allowed_Rank", "OPP_REB_Allowed_Rank"])

    df["TEAM_ID"] = pd.to_numeric(df["TEAM_ID"], errors="coerce")
    df[pts_col] = pd.to_numeric(df[pts_col], errors="coerce")
    df[reb_col] = pd.to_numeric(df[reb_col], errors="coerce")

    df["OPP_PTS_Allowed_Rank"] = df[pts_col].rank(method="min", ascending=True).astype("Int64")
    df["OPP_REB_Allowed_Rank"] = df[reb_col].rank(method="min", ascending=True).astype("Int64")

    return df[["TEAM_ID", "OPP_PTS_Allowed_Rank", "OPP_REB_Allowed_Rank"]].copy()


# --------------------------------------------------
# The Odds API helpers (DraftKings props)
# --------------------------------------------------
def odds_api_get(path: str, params: dict, timeout: int = 30):
    url = f"{ODDS_API_HOST}{path}"
    r = requests.get(url, params=params, headers=ODDS_HEADERS, timeout=timeout)
    r.raise_for_status()
    return r, r.json()


def _first_valid(series: pd.Series):
    for v in series:
        if pd.notna(v):
            return v
    return pd.NA


def fetch_dk_player_props_today() -> pd.DataFrame:
    """
    Player-prop parsing (Odds API):
      outcome.name        -> "Over"/"Under"
      outcome.description -> player name
      outcome.point       -> line
      outcome.price       -> american odds
    """
    api_key = os.getenv(ODDS_API_KEY_ENV, "").strip()
    if not api_key:
        print(f"  -> Missing {ODDS_API_KEY_ENV} env var. Skipping DK props.")
        return pd.DataFrame(columns=["NAME_KEY"])

    now_chi = datetime.now(ZoneInfo("America/Chicago"))
    today_chi = now_chi.date()
    start_utc, end_utc = chicago_day_bounds_utc(today_chi)

    print(f"Fetching Odds API events between {start_utc} and {end_utc} ...")
    _, events = odds_api_get(
        f"/v4/sports/{ODDS_SPORT_KEY}/events",
        params={
            "apiKey": api_key,
            "dateFormat": "iso",
            "commenceTimeFrom": start_utc,
            "commenceTimeTo": end_utc,
        },
    )

    if not events:
        print("  -> No Odds API events returned for today window.")
        return pd.DataFrame(columns=["NAME_KEY"])

    market_to_prefix = {
        "player_points": "DK_PTS",
        "player_points_rebounds_assists": "DK_PRA",
    }

    rows = []
    for idx, ev in enumerate(events, 1):
        ev_id = ev.get("id")
        if not ev_id:
            continue

        print(f"  [{idx}/{len(events)}] Fetching DK props for event {ev_id}...")
        try:
            _, ev_odds = odds_api_get(
                f"/v4/sports/{ODDS_SPORT_KEY}/events/{ev_id}/odds",
                params={
                    "apiKey": api_key,
                    "regions": ODDS_REGIONS,
                    "markets": ",".join(ODDS_MARKETS),
                    "oddsFormat": ODDS_ODDS_FORMAT,
                    "dateFormat": "iso",
                    "bookmakers": ODDS_BOOKMAKERS,
                },
            )
        except requests.HTTPError as e:
            print(f"    -> Odds API event fetch failed: {e}")
            continue
        except Exception as e:
            print(f"    -> Odds API event fetch error: {e}")
            continue

        bookmakers = ev_odds.get("bookmakers") or []
        if not bookmakers:
            continue

        for bm in bookmakers:
            if (bm.get("key") or "").lower() != ODDS_BOOKMAKERS.lower():
                continue

            markets = bm.get("markets") or []
            for m in markets:
                mkey = m.get("key")
                if mkey not in market_to_prefix:
                    continue

                prefix = market_to_prefix[mkey]
                outcomes = m.get("outcomes") or []
                for o in outcomes:
                    side = (o.get("name") or "").strip().lower()  # over/under
                    player = o.get("description")  # player name
                    line = o.get("point")
                    price = o.get("price")

                    if side not in ("over", "under"):
                        continue
                    if not isinstance(player, str) or not player.strip():
                        continue

                    nk = normalize_player_name(player)
                    rec = {
                        "NAME_KEY": nk,
                        f"{prefix}_Line": line,
                    }
                    if side == "over":
                        rec[f"{prefix}_Over_Odds"] = price
                    else:
                        rec[f"{prefix}_Under_Odds"] = price

                    rows.append(rec)

        time.sleep(0.20)

    if not rows:
        return pd.DataFrame(columns=["NAME_KEY"])

    df = pd.DataFrame(rows)

    for base in ["DK_PTS", "DK_PRA"]:
        for col in [f"{base}_Line", f"{base}_Over_Odds", f"{base}_Under_Odds"]:
            if col not in df.columns:
                df[col] = pd.NA

    agg_cols = [c for c in df.columns if c != "NAME_KEY"]
    out = df.groupby("NAME_KEY", as_index=False).agg({c: _first_valid for c in agg_cols})

    for c in out.columns:
        if c.endswith("_Line") or c.endswith("_Odds"):
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    today_str = get_today_date_str_chicago()
    print("Chicago 'today' =", today_str)

    games_df = fetch_today_games_df(today_str)
    team_to_opp = build_team_to_opponent_map(games_df)

    season_avgs = fetch_season_per_game_averages(SEASON, SEASON_TYPE)
    if MIN_GP_FOR_SEASON_SHEET > 0:
        season_avgs = season_avgs[season_avgs["GP"] >= MIN_GP_FOR_SEASON_SHEET].copy()
    season_avgs = season_avgs.sort_values(["PTS", "GP"], ascending=[False, False]).reset_index(drop=True)

    team_ids = fetch_today_team_ids(games_df)

    roster_frames = []
    for i, tid in enumerate(team_ids, 1):
        print(f"[{i}/{len(team_ids)}] Fetching roster for team {tid}...")
        try:
            rf = fetch_roster_players_for_team(tid, SEASON)
            if not rf.empty:
                roster_frames.append(rf)
        except Exception as e:
            print(f"  -> Skipping team {tid}: {e}")
        time.sleep(SLEEP_BETWEEN_CALLS)

    if roster_frames:
        today_roster = pd.concat(roster_frames, ignore_index=True).drop_duplicates(subset=["PLAYER_ID"])
        todays_players = season_avgs.merge(today_roster[["PLAYER_ID"]], on="PLAYER_ID", how="inner")
    else:
        todays_players = season_avgs.iloc[0:0].copy()

    # SportsLine projected minutes
    print("Fetching SportsLine projected minutes...")
    sl = fetch_sportsline_projected_minutes()
    todays_players["NAME_KEY"] = todays_players["PLAYER_NAME"].apply(normalize_player_name)
    todays_players = (
        todays_players.merge(sl[["NAME_KEY", "Projected_MIN"]], on="NAME_KEY", how="left")
        .drop(columns=["NAME_KEY"])
    )

    # PRA + minutes transforms
    todays_players["PRA"] = todays_players["PTS"] + todays_players["REB"] + todays_players["AST"]
    todays_players["ProjMIN_minus_SeasonMIN"] = todays_players["Projected_MIN"] - todays_players["MIN"]

    season_min = pd.to_numeric(todays_players["MIN"], errors="coerce")
    proj_min = pd.to_numeric(todays_players["Projected_MIN"], errors="coerce")
    pra = pd.to_numeric(todays_players["PRA"], errors="coerce")

    rate = pra / season_min.replace({0: pd.NA})
    todays_players["PRA_per_min_times_ProjMIN"] = rate * proj_min

    # Opponent ranks
    ranks_df = fetch_opponent_allowed_ranks(SEASON, SEASON_TYPE)
    todays_players["OPP_TEAM_ID"] = todays_players["TEAM_ID"].map(
        lambda tid: team_to_opp.get(int(tid)) if pd.notna(tid) else None
    )
    todays_players["OPP_TEAM_ID"] = pd.to_numeric(todays_players["OPP_TEAM_ID"], errors="coerce")

    todays_players = todays_players.merge(
        ranks_df.rename(columns={
            "TEAM_ID": "OPP_TEAM_ID",
            "OPP_PTS_Allowed_Rank": "Opp_PTS_Allowed_Rank",
            "OPP_REB_Allowed_Rank": "Opp_REB_Allowed_Rank",
        }),
        on="OPP_TEAM_ID",
        how="left",
    )

    # Weighted PTS / REB scalers
    todays_players["Opp_PTS_Allowed_Scaler"] = todays_players["Opp_PTS_Allowed_Rank"].apply(rank_to_scaler)
    todays_players["Opp_REB_Allowed_Scaler"] = todays_players["Opp_REB_Allowed_Rank"].apply(rank_to_scaler)

    todays_players["Weighted_PTS"] = (
        todays_players["PTS"] *
        pd.to_numeric(todays_players["Opp_PTS_Allowed_Scaler"], errors="coerce")
    )
    todays_players["Weighted_REB"] = (
        todays_players["REB"] *
        pd.to_numeric(todays_players["Opp_REB_Allowed_Scaler"], errors="coerce")
    )

    # DraftKings props via The Odds API (PTS + PRA only)
    print("Fetching DraftKings props (via The Odds API)...")
    dk = fetch_dk_player_props_today()

    todays_players["NAME_KEY"] = todays_players["PLAYER_NAME"].apply(normalize_player_name)
    if dk is not None and not dk.empty:
        todays_players = todays_players.merge(dk, on="NAME_KEY", how="left")
    else:
        for base in ["DK_PTS", "DK_PRA"]:
            for col in [f"{base}_Line", f"{base}_Over_Odds", f"{base}_Under_Odds"]:
                if col not in todays_players.columns:
                    todays_players[col] = pd.NA

    todays_players = todays_players.drop(columns=["NAME_KEY"])

    # --------------------------------------------------
    # NEW: Weighted Points / Weighted PRA / Differential
    # --------------------------------------------------
    for c in ["Weighted_PTS", "Weighted_REB", "AST", "MIN", "Projected_MIN", "DK_PRA_Line"]:
        if c in todays_players.columns:
            todays_players[c] = pd.to_numeric(todays_players[c], errors="coerce")

    # Weighted Points category: (Weighted_PTS / MIN) * Projected_MIN
    todays_players["Weighted_PTS_Proj"] = (
        todays_players["Weighted_PTS"] / todays_players["MIN"].replace({0: pd.NA})
    ) * todays_players["Projected_MIN"]

    # Weighted PRA: ((Weighted_PTS + Weighted_REB + AST) / MIN) * Projected_MIN
    todays_players["Weighted_PRA"] = (
        (todays_players["Weighted_PTS"] + todays_players["Weighted_REB"] + todays_players["AST"])
        / todays_players["MIN"].replace({0: pd.NA})
    ) * todays_players["Projected_MIN"]

    # Differential: Weighted_PRA - DK_PRA_Line
    if "DK_PRA_Line" in todays_players.columns:
        todays_players["Weighted_Differential"] = todays_players["Weighted_PRA"] - todays_players["DK_PRA_Line"]
    else:
        todays_players["Weighted_Differential"] = pd.NA

    # Sort
    sort_cols = ["Weighted_Differential", "Weighted_PRA", "PRA_per_min_times_ProjMIN", "Projected_MIN"]
    sort_cols = [c for c in sort_cols if c in todays_players.columns]
    todays_players = todays_players.sort_values(
        sort_cols,
        ascending=[False] * len(sort_cols),
        na_position="last",
    ).reset_index(drop=True)

    # Write Excel
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        season_avgs.to_excel(writer, sheet_name="SeasonAverages", index=False)
        todays_players.to_excel(writer, sheet_name="TodaysPlayers", index=False)

    print("Saved:", OUTPUT_XLSX)
    print("SeasonAverages rows:", len(season_avgs))
    print("TodaysPlayers rows:", len(todays_players))
    print("Projected_MIN matches:", todays_players["Projected_MIN"].notna().sum())
    print("Opp PTS rank matches:", todays_players["Opp_PTS_Allowed_Rank"].notna().sum())
    print("Opp REB rank matches:", todays_players["Opp_REB_Allowed_Rank"].notna().sum())
    print("DK PTS line matches:", todays_players["DK_PTS_Line"].notna().sum() if "DK_PTS_Line" in todays_players.columns else 0)
    print("DK PRA line matches:", todays_players["DK_PRA_Line"].notna().sum() if "DK_PRA_Line" in todays_players.columns else 0)
    print("Weighted_PTS_Proj non-null:", todays_players["Weighted_PTS_Proj"].notna().sum() if "Weighted_PTS_Proj" in todays_players.columns else 0)
    print("Weighted_PRA non-null:", todays_players["Weighted_PRA"].notna().sum() if "Weighted_PRA" in todays_players.columns else 0)


if __name__ == "__main__":
    main()

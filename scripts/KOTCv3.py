from __future__ import annotations

import os
import re
import time
import random
import unicodedata
from io import StringIO
from datetime import datetime, date, time as dtime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from requests.exceptions import ReadTimeout, ConnectionError, Timeout


# --------------------------------------------------
# CONFIG
# --------------------------------------------------
SEASON = "2025-26"
SEASON_TYPE = "Regular Season"  # stats.nba.com expects e.g. "Regular Season"

SLEEP_BETWEEN_CALLS = 0.65

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "public/data")
OUTPUT_XLSX = os.path.join(OUTPUT_DIR, f"player_avgs_plus_today_{SEASON.replace('-', '')}.xlsx")

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

ODDS_MARKETS = [
    "player_points",
    "player_points_rebounds_assists",
]


# --------------------------------------------------
# SESSIONS / HEADERS
# --------------------------------------------------
SESSION = requests.Session()
CDN_SESSION = requests.Session()

# Browser-like headers are critical for stats.nba.com.
NBA_HEADERS = {
    "Host": "stats.nba.com",
    "Connection": "keep-alive",
    "Accept": "application/json, text/plain, */*",
    "x-nba-stats-token": "true",
    "x-nba-stats-origin": "stats",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "Accept-Language": "en-US,en;q=0.9",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

SPORTSLINE_HEADERS = {
    "User-Agent": NBA_HEADERS["User-Agent"],
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}

ODDS_HEADERS = {
    "User-Agent": NBA_HEADERS["User-Agent"],
    "Accept": "application/json",
}


# --------------------------------------------------
# RETRIES
# --------------------------------------------------
def with_retries(fn, label: str, max_retries: int = 6, base_sleep: float = 2.0):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except (ReadTimeout, ConnectionError, TimeoutError, Timeout) as e:
            last_err = e
            sleep_s = base_sleep * (2 ** (attempt - 1)) + random.uniform(0.0, 0.8)
            print(f"  {label} timeout (attempt {attempt}/{max_retries}) — sleeping {sleep_s:.1f}s then retrying...")
            time.sleep(sleep_s)
    raise last_err


def _looks_like_html(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.strip().lower()
    return t.startswith("<!doctype html") or t.startswith("<html") or "<head" in t[:500] or "<body" in t[:500]


def nba_stats_get(endpoint: str, params: dict, timeout_s: int = 90) -> pd.DataFrame:
    """Call a stats.nba.com endpoint and return the first resultSet/resultSets as a DataFrame."""
    url = f"https://stats.nba.com/stats/{endpoint}"

    def do_call():
        r = SESSION.get(url, params=params, headers=NBA_HEADERS, timeout=timeout_s)
        r.raise_for_status()
        txt = r.text or ""
        if _looks_like_html(txt):
            raise ReadTimeout(f"stats.nba.com returned HTML (blocked?) for {endpoint}")
        return r.json()

    raw = with_retries(lambda: do_call(), label=f"stats:{endpoint}", max_retries=5, base_sleep=2.5)

    rs = None
    if isinstance(raw, dict):
        if "resultSets" in raw and isinstance(raw["resultSets"], list) and raw["resultSets"]:
            rs = raw["resultSets"][0]
        elif "resultSet" in raw and isinstance(raw["resultSet"], dict):
            rs = raw["resultSet"]

    if not rs or "headers" not in rs or "rowSet" not in rs:
        raise ValueError(f"Unexpected payload for {endpoint}")

    return pd.DataFrame(rs["rowSet"], columns=rs["headers"])


# --------------------------------------------------
# NAME NORMALIZATION
# --------------------------------------------------
def normalize_player_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    n = unicodedata.normalize("NFKD", name)
    n = "".join(ch for ch in n if not unicodedata.combining(ch))
    n = n.lower().strip()
    n = re.sub(r"[^a-z\s'-]", "", n)
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


def fetch_today_games_df(game_date_mmddyyyy: str) -> pd.DataFrame:
    """Return GameHeader-like df with HOME_TEAM_ID/VISITOR_TEAM_ID.

    In Actions, ScoreboardV2 often times out. Use the NBA CDN for today's scoreboard.
    """
    # Convert to YYYYMMDD for CDN
    dt = datetime.strptime(game_date_mmddyyyy, "%m/%d/%Y").date()
    yyyymmdd = dt.strftime("%Y%m%d")

    cdn_url = f"https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_{yyyymmdd}.json"

    def cdn_call():
        r = CDN_SESSION.get(cdn_url, timeout=25)
        r.raise_for_status()
        return r.json()

    try:
        raw = with_retries(cdn_call, label=f"CDN scoreboard {yyyymmdd}", max_retries=3, base_sleep=1.5)
        games = (((raw or {}).get("scoreboard") or {}).get("games")) or []
        if games:
            df = pd.DataFrame(games)
            # Map to the columns this script expects
            keep = {
                "gameId": "GAME_ID",
                "homeTeam.teamId": "HOME_TEAM_ID",
                "awayTeam.teamId": "VISITOR_TEAM_ID",
            }
            # Flatten minimal fields
            out = pd.DataFrame({
                "GAME_ID": [g.get("gameId") for g in games],
                "HOME_TEAM_ID": [((g.get("homeTeam") or {}).get("teamId")) for g in games],
                "VISITOR_TEAM_ID": [((g.get("awayTeam") or {}).get("teamId")) for g in games],
            })
            return out
        print("  ⚠️ CDN scoreboard returned no games; falling back to stats.nba.com...")
    except Exception as e:
        print(f"  ⚠️ CDN scoreboard failed: {e}. Falling back to stats.nba.com...")

    # Fallback: stats.nba.com scoreboardv2
    df = nba_stats_get(
        "scoreboardv2",
        {
            "GameDate": game_date_mmddyyyy,
            "LeagueID": "00",
            "DayOffset": "0",
        },
        timeout_s=90,
    )
    return df


def build_team_to_opponent_map(games_df: pd.DataFrame) -> dict[int, int]:
    if games_df is None or games_df.empty:
        return {}

    home = pd.to_numeric(games_df.get("HOME_TEAM_ID"), errors="coerce")
    away = pd.to_numeric(games_df.get("VISITOR_TEAM_ID"), errors="coerce")

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
    df = nba_stats_get(
        "leaguedashplayerstats",
        {
            "Season": season,
            "SeasonType": season_type,
            "PerMode": "PerGame",
            "MeasureType": "Base",
            "PlusMinus": "N",
            "PaceAdjust": "N",
            "Rank": "N",
            "LeagueID": "00",
            "Outcome": "",
            "Location": "",
            "Month": "0",
            "SeasonSegment": "",
            "DateFrom": "",
            "DateTo": "",
            "OpponentTeamID": "0",
            "VsConference": "",
            "VsDivision": "",
            "GameSegment": "",
            "Period": "0",
            "LastNGames": "0",
            "TeamID": "0",
        },
        timeout_s=120,
    ).copy()

    # Keep and rename what you use
    keep = [
        "PLAYER_ID",
        "PLAYER_NAME",
        "TEAM_ID",
        "GP",
        "MIN",
        "PTS",
        "REB",
        "AST",
    ]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()

    for c in ["GP", "MIN", "PTS", "REB", "AST"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out[out.get("GP", 0).fillna(0) >= MIN_GP_FOR_SEASON_SHEET].copy()
    return out


def fetch_opponent_allowed_ranks(season: str, season_type: str) -> pd.DataFrame:
    df = nba_stats_get(
        "leaguedashteamstats",
        {
            "Season": season,
            "SeasonType": season_type,
            "PerMode": "PerGame",
            "MeasureType": "Opponent",
            "LeagueID": "00",
            "PlusMinus": "N",
            "PaceAdjust": "N",
            "Rank": "N",
            "Outcome": "",
            "Location": "",
            "Month": "0",
            "SeasonSegment": "",
            "DateFrom": "",
            "DateTo": "",
            "OpponentTeamID": "0",
            "VsConference": "",
            "VsDivision": "",
            "GameSegment": "",
            "Period": "0",
            "LastNGames": "0",
            "TeamID": "0",
        },
        timeout_s=120,
    ).copy()

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


def fetch_team_roster(team_id: int, season: str) -> pd.DataFrame:
    df = nba_stats_get(
        "commonteamroster",
        {
            "TeamID": str(team_id),
            "Season": season,
            "LeagueID": "00",
        },
        timeout_s=90,
    )
    # commonteamroster returns multiple resultSets; first one is roster.
    # The helper currently grabs first set already.
    return df


# --------------------------------------------------
# SportsLine projected minutes
# --------------------------------------------------
def fetch_sportsline_projected_minutes() -> pd.DataFrame:
    r = requests.get(SPORTSLINE_URL, headers=SPORTSLINE_HEADERS, timeout=45)
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
# The Odds API helpers (DraftKings props)
# --------------------------------------------------
def odds_api_get(path: str, params: dict, timeout: int = 30):
    url = f"{ODDS_API_HOST}{path}"
    r = requests.get(url, params=params, headers=ODDS_HEADERS, timeout=timeout)
    r.raise_for_status()
    return r, r.json()


def fetch_dk_player_props_today() -> pd.DataFrame:
    """Return per-player DK prop lines/odds for PTS and PRA (if available)."""
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
        {
            "apiKey": api_key,
            "regions": ODDS_REGIONS,
            "dateFormat": "iso",
        },
        timeout=30,
    )

    if not isinstance(events, list) or not events:
        return pd.DataFrame(columns=["NAME_KEY"])

    # Filter today in Chicago day bounds
    event_ids = []
    for ev in events:
        commence = (ev or {}).get("commence_time")
        if not commence:
            continue
        if start_utc <= commence <= end_utc:
            if ev.get("id"):
                event_ids.append(ev["id"])

    if not event_ids:
        return pd.DataFrame(columns=["NAME_KEY"])

    rows = []
    for eid in event_ids:
        time.sleep(0.2)
        try:
            _, odds = odds_api_get(
                f"/v4/sports/{ODDS_SPORT_KEY}/events/{eid}/odds",
                {
                    "apiKey": api_key,
                    "regions": ODDS_REGIONS,
                    "markets": ",".join(ODDS_MARKETS),
                    "bookmakers": ODDS_BOOKMAKERS,
                    "oddsFormat": ODDS_ODDS_FORMAT,
                    "dateFormat": "iso",
                },
                timeout=30,
            )
        except Exception as e:
            print(f"  ⚠️ Odds API failed for event {eid}: {e}")
            continue

        bookmakers = (odds or {}).get("bookmakers") or []
        for bm in bookmakers:
            markets = (bm or {}).get("markets") or []
            for m in markets:
                key = (m or {}).get("key")
                outcomes = (m or {}).get("outcomes") or []
                for o in outcomes:
                    player = (o or {}).get("description")
                    side = (o or {}).get("name")
                    point = (o or {}).get("point")
                    price = (o or {}).get("price")
                    if not player or side not in ("Over", "Under"):
                        continue
                    rows.append(
                        {
                            "NAME_KEY": normalize_player_name(player),
                            "market": key,
                            "side": side,
                            "point": point,
                            "price": price,
                        }
                    )

    if not rows:
        return pd.DataFrame(columns=["NAME_KEY"])

    df = pd.DataFrame(rows)
    df["point"] = pd.to_numeric(df["point"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    def pack_market(market_key: str, prefix: str):
        sub = df[df["market"] == market_key].copy()
        if sub.empty:
            return pd.DataFrame(columns=["NAME_KEY"])

        # One player may have multiple offers; take first by point then odds
        over = sub[sub["side"] == "Over"].copy()
        under = sub[sub["side"] == "Under"].copy()

        over = over.sort_values(["point", "price"], ascending=[True, False])
        under = under.sort_values(["point", "price"], ascending=[True, False])

        out = pd.DataFrame({"NAME_KEY": pd.unique(sub["NAME_KEY"])})
        out = out.merge(over.groupby("NAME_KEY")["point"].first().rename(f"{prefix}_Line"), on="NAME_KEY", how="left")
        out = out.merge(over.groupby("NAME_KEY")["price"].first().rename(f"{prefix}_Over_Odds"), on="NAME_KEY", how="left")
        out = out.merge(under.groupby("NAME_KEY")["price"].first().rename(f"{prefix}_Under_Odds"), on="NAME_KEY", how="left")
        return out

    pts = pack_market("player_points", "DK_PTS")
    pra = pack_market("player_points_rebounds_assists", "DK_PRA")

    out = pts.merge(pra, on="NAME_KEY", how="outer")
    return out


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    today_str = get_today_date_str_chicago()
    print(f"Chicago 'today' = {today_str}")

    # Today games
    games_df = fetch_today_games_df(today_str)
    team_to_opp = build_team_to_opponent_map(games_df)

    # Season averages
    print("Fetching season per-game averages...")
    season_avgs = fetch_season_per_game_averages(SEASON, SEASON_TYPE)
    time.sleep(SLEEP_BETWEEN_CALLS)

    # Opponent ranks
    print("Fetching opponent allowed ranks...")
    opp_ranks = fetch_opponent_allowed_ranks(SEASON, SEASON_TYPE)
    time.sleep(SLEEP_BETWEEN_CALLS)

    # Build today's players by rosters of teams playing today
    teams_today = sorted({int(t) for t in team_to_opp.keys()})
    print(f"Teams playing today: {len(teams_today)}")

    roster_rows = []
    for tid in teams_today:
        print(f"  fetching roster TeamID={tid}...")
        try:
            r = fetch_team_roster(tid, SEASON)
            if not r.empty:
                r["TEAM_ID"] = tid
                roster_rows.append(r)
        except Exception as e:
            print(f"    ⚠️ roster failed for {tid}: {e}")
        time.sleep(SLEEP_BETWEEN_CALLS)

    if roster_rows:
        rosters = pd.concat(roster_rows, ignore_index=True)
    else:
        rosters = pd.DataFrame(columns=["PLAYER_ID", "PLAYER", "TEAM_ID"])

    # Try to normalize roster schema across seasons
    player_name_col = "PLAYER"
    if "PLAYER" not in rosters.columns and "PLAYER_NAME" in rosters.columns:
        player_name_col = "PLAYER_NAME"
    if "PLAYER" not in rosters.columns and "PLAYER_NAME" not in rosters.columns:
        # Some schema: "PLAYER" may be missing if API changed
        for cand in rosters.columns:
            if "PLAYER" in cand.upper() and "NAME" in cand.upper():
                player_name_col = cand
                break

    rosters = rosters.rename(columns={player_name_col: "PLAYER_NAME"})

    # Merge rosters with season averages to get PTS/REB/AST/MIN
    todays_players = rosters.merge(season_avgs, on=["PLAYER_ID", "TEAM_ID"], how="left", suffixes=("", "_SEASON"))

    # Projected minutes (SportsLine)
    print("Fetching SportsLine projected minutes...")
    sl = fetch_sportsline_projected_minutes()
    todays_players["NAME_KEY"] = todays_players["PLAYER_NAME"].apply(normalize_player_name)
    todays_players = todays_players.merge(sl[["NAME_KEY", "Projected_MIN"]], on="NAME_KEY", how="left")

    # PRA + helpers
    for c in ["PTS", "REB", "AST", "MIN", "Projected_MIN"]:
        if c in todays_players.columns:
            todays_players[c] = pd.to_numeric(todays_players[c], errors="coerce")

    todays_players["PRA"] = todays_players[["PTS", "REB", "AST"]].sum(axis=1, min_count=1)
    todays_players["ProjMIN_minus_SeasonMIN"] = todays_players["Projected_MIN"] - todays_players["MIN"]
    todays_players["PRA_per_min_times_ProjMIN"] = (
        todays_players["PRA"] / todays_players["MIN"].replace({0: pd.NA})
    ) * todays_players["Projected_MIN"]

    # Opponent ranks & scalers
    todays_players["OPP_TEAM_ID"] = todays_players["TEAM_ID"].map(team_to_opp)
    todays_players = todays_players.merge(
        opp_ranks.rename(columns={
            "TEAM_ID": "OPP_TEAM_ID",
            "OPP_PTS_Allowed_Rank": "Opp_PTS_Allowed_Rank",
            "OPP_REB_Allowed_Rank": "Opp_REB_Allowed_Rank",
        }),
        on="OPP_TEAM_ID",
        how="left",
    )

    todays_players["Opp_PTS_Allowed_Scaler"] = todays_players["Opp_PTS_Allowed_Rank"].apply(rank_to_scaler)
    todays_players["Opp_REB_Allowed_Scaler"] = todays_players["Opp_REB_Allowed_Rank"].apply(rank_to_scaler)

    # Weighted categories
    todays_players["Weighted_PTS"] = (
        pd.to_numeric(todays_players["Opp_PTS_Allowed_Scaler"], errors="coerce") * todays_players["PTS"]
    )
    todays_players["Weighted_REB"] = (
        pd.to_numeric(todays_players["Opp_REB_Allowed_Scaler"], errors="coerce") * todays_players["REB"]
    )

    # DK props
    print("Fetching DraftKings player props (Odds API)...")
    dk = fetch_dk_player_props_today()
    if dk is not None and not dk.empty:
        todays_players = todays_players.merge(dk, on="NAME_KEY", how="left")
    else:
        for base in ["DK_PTS", "DK_PRA"]:
            for col in [f"{base}_Line", f"{base}_Over_Odds", f"{base}_Under_Odds"]:
                if col not in todays_players.columns:
                    todays_players[col] = pd.NA

    # NEW: Weighted Points / Weighted PRA / Differential
    for c in ["Weighted_PTS", "Weighted_REB", "AST", "MIN", "Projected_MIN", "DK_PRA_Line"]:
        if c in todays_players.columns:
            todays_players[c] = pd.to_numeric(todays_players[c], errors="coerce")

    todays_players["Weighted_PTS_Proj"] = (
        todays_players["Weighted_PTS"] / todays_players["MIN"].replace({0: pd.NA})
    ) * todays_players["Projected_MIN"]

    todays_players["Weighted_PRA"] = (
        (todays_players["Weighted_PTS"] + todays_players["Weighted_REB"] + todays_players["AST"])
        / todays_players["MIN"].replace({0: pd.NA})
    ) * todays_players["Projected_MIN"]

    if "DK_PRA_Line" in todays_players.columns:
        todays_players["Weighted_Differential"] = todays_players["Weighted_PRA"] - todays_players["DK_PRA_Line"]
    else:
        todays_players["Weighted_Differential"] = pd.NA

    # Sort
    sort_cols = ["Weighted_Differential", "Weighted_PRA", "PRA_per_min_times_ProjMIN", "Projected_MIN"]
    sort_cols = [c for c in sort_cols if c in todays_players.columns]
    todays_players = todays_players.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last").reset_index(drop=True)

    # Drop helper
    todays_players = todays_players.drop(columns=["NAME_KEY"], errors="ignore")

    # Write Excel
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        season_avgs.to_excel(writer, sheet_name="SeasonAverages", index=False)
        todays_players.to_excel(writer, sheet_name="TodaysPlayers", index=False)

    print("Saved:", OUTPUT_XLSX)
    print("SeasonAverages rows:", len(season_avgs))
    print("TodaysPlayers rows:", len(todays_players))
    print("Projected_MIN matches:", todays_players.get("Projected_MIN").notna().sum() if "Projected_MIN" in todays_players.columns else 0)
    print("Opp PTS rank matches:", todays_players.get("Opp_PTS_Allowed_Rank").notna().sum() if "Opp_PTS_Allowed_Rank" in todays_players.columns else 0)
    print("Opp REB rank matches:", todays_players.get("Opp_REB_Allowed_Rank").notna().sum() if "Opp_REB_Allowed_Rank" in todays_players.columns else 0)
    print("DK PTS line matches:", todays_players.get("DK_PTS_Line").notna().sum() if "DK_PTS_Line" in todays_players.columns else 0)
    print("DK PRA line matches:", todays_players.get("DK_PRA_Line").notna().sum() if "DK_PRA_Line" in todays_players.columns else 0)


if __name__ == "__main__":
    main()

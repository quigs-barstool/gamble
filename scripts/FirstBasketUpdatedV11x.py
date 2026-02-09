from __future__ import annotations

import datetime
import os
import random
import re
import time
from typing import Any

import numpy as np
import pandas as pd
import requests
from requests.exceptions import ConnectionError, ReadTimeout, Timeout


# -----------------------------
# CONFIG
# -----------------------------
SEASON = "2025-26"
SEASON_TYPE = "Regular Season"  # stats.nba.com parameter value

# Throttles
SLEEP_BETWEEN_CALLS = 0.55  # between PBP calls when backfilling new games
SCOREBOARD_SLEEP = 0.25     # between scoreboard calls for ThisWeeksMatchups

OUTPUT_DIR = os.path.join("public", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cache file: per-team per-game results so we only fetch NEW games next time
CACHE_FILE = os.path.join(OUTPUT_DIR, f"first_shot_cache_{SEASON.replace('-', '')}.csv")

# Sessions (persistent)
SESSION = requests.Session()
CDN_SESSION = requests.Session()


# -----------------------------
# Headers for stats.nba.com
# -----------------------------
NBA_HEADERS = {
    "Host": "stats.nba.com",
    "Connection": "keep-alive",
    "Accept": "application/json, text/plain, */*",
    "x-nba-stats-token": "true",
    "x-nba-stats-origin": "stats",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "Accept-Language": "en-US,en;q=0.9",
    # Use a "desktop" UA to reduce block likelihood
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}


# ---------------------------------------------------------
#   GENERIC RETRY WRAPPER
# ---------------------------------------------------------
def with_retries(fn, label: str, max_retries: int = 5, base_sleep: float = 2.0):
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except (ReadTimeout, ConnectionError, TimeoutError, Timeout, requests.HTTPError) as e:
            last_err = e
            sleep_s = base_sleep * (2 ** (attempt - 1)) + random.uniform(0.0, 0.6)
            print(f"  {label} failed (attempt {attempt}/{max_retries}) — sleeping {sleep_s:.1f}s then retrying...\n    -> {e}")
            time.sleep(sleep_s)
    if last_err:
        raise last_err
    raise RuntimeError("Unknown retry failure")


def _looks_like_html(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.strip().lower()
    return t.startswith("<!doctype html") or t.startswith("<html") or "<head" in t[:500] or "<body" in t[:500]


def nba_stats_json(endpoint: str, params: dict[str, Any], timeout: int = 90) -> dict[str, Any]:
    """GET https://stats.nba.com/stats/{endpoint} with retries."""
    url = f"https://stats.nba.com/stats/{endpoint}"

    def _call():
        r = SESSION.get(url, params=params, headers=NBA_HEADERS, timeout=timeout)
        # Some blocks return HTML with 200
        txt = r.text or ""
        if r.status_code >= 400:
            r.raise_for_status()
        if _looks_like_html(txt):
            raise requests.HTTPError("stats.nba.com returned HTML (likely blocked)")
        return r.json()

    return with_retries(_call, label=f"stats:{endpoint}", max_retries=5, base_sleep=2.0)


def resultset_to_df(payload: dict[str, Any], idx: int = 0) -> pd.DataFrame:
    """Convert a stats.nba.com payload with resultSets/resultSet to a DataFrame."""
    rs = None
    if isinstance(payload, dict):
        if "resultSets" in payload and isinstance(payload["resultSets"], list) and payload["resultSets"]:
            rs = payload["resultSets"][idx]
        elif "resultSet" in payload and isinstance(payload["resultSet"], dict):
            rs = payload["resultSet"]

    if not rs or "headers" not in rs or "rowSet" not in rs:
        return pd.DataFrame()

    return pd.DataFrame(rs["rowSet"], columns=rs["headers"])  # type: ignore[arg-type]


# ---------------------------------------------------------
#   HELPERS
# ---------------------------------------------------------
def is_gleague_game_id(game_id: str) -> bool:
    return str(game_id).startswith("20")


def safe_upper(x: Any) -> str:
    return str(x).upper() if isinstance(x, str) else ""


def sec_remaining_from_clock(clock_val) -> int | None:
    """CDN clock looks like PT11M34.00S; sometimes you see 11:34."""
    if clock_val is None:
        return None
    s = str(clock_val).strip()

    # ISO-ish duration e.g. PT11M34.00S
    m = re.match(r"^PT(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?$", s)
    if m:
        mins = int(m.group(1) or 0)
        secs = float(m.group(2) or 0)
        return int(round(mins * 60 + secs))

    # MM:SS
    m2 = re.match(r"^(\d+):(\d+)$", s)
    if m2:
        return int(m2.group(1)) * 60 + int(m2.group(2))

    return None


def dedupe_game_ids(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "GAME_ID" not in df.columns:
        return df
    return df.drop_duplicates(subset=["GAME_ID"], keep="first").copy()


# ---------------------------------------------------------
#   SCOREBOARD (CDN FOR TODAY)
# ---------------------------------------------------------
def fetch_scoreboard_game_ids(date_str: str) -> pd.DataFrame:
    """Return GAME_ID / HOME_TEAM_ID / VISITOR_TEAM_ID for a date.

    - If date_str == today (local), use NBA CDN scoreboard to avoid stats.nba.com timeouts on runners.
    - Otherwise, use stats.nba.com scoreboardv2.
    """
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    if date_str == today_str:
        ymd = datetime.date.today().strftime("%Y%m%d")
        cdn_url = f"https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_{ymd}.json"

        def cdn_call():
            r = CDN_SESSION.get(cdn_url, timeout=25)
            r.raise_for_status()
            return r.json()

        try:
            payload = with_retries(cdn_call, label=f"CDN scoreboard {ymd}", max_retries=3, base_sleep=1.2)
            games = (((payload or {}).get("scoreboard") or {}).get("games")) or []
            rows = []
            for g in games:
                gid = str(g.get("gameId") or "").strip()
                if not gid or is_gleague_game_id(gid):
                    continue
                home_id = (g.get("homeTeam") or {}).get("teamId")
                away_id = (g.get("awayTeam") or {}).get("teamId")
                rows.append({
                    "GAME_ID": gid,
                    "HOME_TEAM_ID": home_id,
                    "VISITOR_TEAM_ID": away_id,
                })
            df = pd.DataFrame(rows)
            df["GAME_ID"] = df["GAME_ID"].astype(str)
            df = df[~df["GAME_ID"].str.startswith("20")].copy()
            return df
        except Exception as e:
            print(f"  ⚠️ CDN scoreboard failed for today: {e}. Falling back to stats.nba.com...")

    # stats.nba.com fallback
    payload = nba_stats_json(
        "scoreboardv2",
        {
            "GameDate": date_str,
            "LeagueID": "00",
            "DayOffset": 0,
        },
        timeout=90,
    )
    game_header = resultset_to_df(payload, idx=0)
    if game_header is None or game_header.empty:
        return pd.DataFrame(columns=["GAME_ID", "HOME_TEAM_ID", "VISITOR_TEAM_ID"])

    need = {"GAME_ID", "HOME_TEAM_ID", "VISITOR_TEAM_ID"}
    if not need.issubset(set(game_header.columns)):
        return pd.DataFrame(columns=["GAME_ID", "HOME_TEAM_ID", "VISITOR_TEAM_ID"])

    out = game_header[["GAME_ID", "HOME_TEAM_ID", "VISITOR_TEAM_ID"]].copy()
    out["GAME_ID"] = out["GAME_ID"].astype(str)
    out = out[~out["GAME_ID"].str.startswith("20")].copy()
    return out


def get_matchups_for_date(team_id_map, date_str: str) -> pd.DataFrame:
    try:
        games = fetch_scoreboard_game_ids(date_str)
    except Exception as e:
        print(f"  ❌ scoreboard({date_str}) failed after retries: {e}")
        return pd.DataFrame(columns=["GAME_ID", "Home", "Away"])

    if games is None or games.empty:
        return pd.DataFrame(columns=["GAME_ID", "Home", "Away"])

    df = games[["GAME_ID", "HOME_TEAM_ID", "VISITOR_TEAM_ID"]].copy()
    df["GAME_ID"] = df["GAME_ID"].astype(str)
    df = df[~df["GAME_ID"].str.startswith("20")].copy()

    df["Home"] = pd.to_numeric(df["HOME_TEAM_ID"], errors="coerce").map(team_id_map)
    df["Away"] = pd.to_numeric(df["VISITOR_TEAM_ID"], errors="coerce").map(team_id_map)
    df = df[["GAME_ID", "Home", "Away"]].dropna().copy()
    return dedupe_game_ids(df)


# ---------------------------------------------------------
#   LEAGUE GAME FINDER (ALL GAMES FOR SEASON)
# ---------------------------------------------------------
def get_games_for_season(season: str, season_type: str = SEASON_TYPE) -> pd.DataFrame:
    payload = nba_stats_json(
        "leaguegamefinder",
        {
            "LeagueID": "00",
            "Season": season,
            "SeasonType": season_type,
        },
        timeout=120,
    )
    games = resultset_to_df(payload, idx=0)

    cols_to_keep = ["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "GAME_DATE", "MATCHUP"]
    games = games[cols_to_keep].drop_duplicates(subset=["GAME_ID", "TEAM_ID"]).copy()
    games["GAME_ID"] = games["GAME_ID"].astype(str)
    games = games[~games["GAME_ID"].str.startswith("20")].copy()
    return games


# ---------------------------------------------------------
#   PLAYBYPLAY FETCH (CDN FIRST, stats.nba.com FALLBACK)
# ---------------------------------------------------------
def fetch_playbyplay_df(game_id: str) -> pd.DataFrame | None:
    game_id = str(game_id)
    if is_gleague_game_id(game_id):
        return None

    cdn_url = f"https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"

    def cdn_call():
        r = CDN_SESSION.get(cdn_url, timeout=20)
        r.raise_for_status()
        return r

    try:
        r = with_retries(cdn_call, label=f"CDN PBP {game_id}", max_retries=3, base_sleep=1.2)
        raw = r.json()
        actions = (((raw or {}).get("game") or {}).get("actions")) or []
        if not actions:
            print(f"  ❌ Skipping PlayByPlay for game {game_id}: CDN returned no actions")
            return None

        df = pd.DataFrame(actions)
        if "period" not in df.columns or "clock" not in df.columns:
            print(f"  ❌ Skipping PlayByPlay for game {game_id}: CDN format missing period/clock")
            return None

        rename_map = {
            "period": "PERIOD",
            "clock": "CLOCK",
            "teamId": "TEAM_ID",
            "description": "DESCRIPTION",
            "actionType": "ACTION_TYPE",
            "shotValue": "SHOT_VALUE",
            "shotResult": "SHOT_RESULT",
            "subType": "SUB_TYPE",
            "actionNumber": "actionNumber",
        }
        for k, v in rename_map.items():
            if k in df.columns:
                df.rename(columns={k: v}, inplace=True)

        if "TEAM_ID" in df.columns:
            df["TEAM_ID"] = pd.to_numeric(df["TEAM_ID"], errors="coerce")

        df["SEC_REMAINING"] = df["CLOCK"].apply(sec_remaining_from_clock)
        df["DESC_UPPER"] = df.get("DESCRIPTION", "").apply(safe_upper)

        if "actionNumber" in df.columns:
            df["actionNumber"] = pd.to_numeric(df["actionNumber"], errors="coerce")
            df.sort_values(["PERIOD", "actionNumber"], inplace=True)

        return df

    except Exception as e:
        print(f"  ⚠️ CDN PBP failed for {game_id}: {e}. Trying stats.nba.com fallback...")

    # stats.nba.com fallback
    payload = nba_stats_json(
        "playbyplayv2",
        {"GameID": game_id, "StartPeriod": 0, "EndPeriod": 14},
        timeout=120,
    )

    df = resultset_to_df(payload, idx=0)
    if df.empty:
        print(f"  ❌ Skipping PlayByPlay for game {game_id}: empty stats.nba.com PBP")
        return None

    if "PCTIMESTRING" in df.columns:
        df["SEC_REMAINING"] = df["PCTIMESTRING"].apply(sec_remaining_from_clock)
    else:
        df["SEC_REMAINING"] = None

    def desc_u(row):
        for c in ("HOMEDESCRIPTION", "VISITORDESCRIPTION", "NEUTRALDESCRIPTION"):
            v = row.get(c, "")
            if isinstance(v, str) and v.strip():
                return v.upper()
        return ""

    df["DESC_UPPER"] = df.apply(desc_u, axis=1)
    for c in ("EVENTMSGTYPE", "PERIOD", "EVENTNUM", "PLAYER1_TEAM_ID"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# ---------------------------------------------------------
#   FIRST SHOT / FIRST MINUTE METRICS
# ---------------------------------------------------------
def compute_first_shot_and_first_minute(df_pbp: pd.DataFrame, home_team_id: int, away_team_id: int) -> pd.DataFrame:
    """Compute:
      - first shot type and time (sec into game)
      - first minute baskets (FG makes + FT trips with a made FT)
      - first minute points
    """
    if df_pbp is None or df_pbp.empty:
        return pd.DataFrame()

    result = pd.DataFrame({"TEAM_ID": [home_team_id, away_team_id]})
    result["TEAM_ID"] = pd.to_numeric(result["TEAM_ID"], errors="coerce")

    # Normalize team id column name for CDN PBP
    if "TEAM_ID" not in df_pbp.columns and "PLAYER1_TEAM_ID" in df_pbp.columns:
        df_pbp = df_pbp.rename(columns={"PLAYER1_TEAM_ID": "TEAM_ID"})

    # Determine first shot attempt
    # CDN actionType/subType varies; we mostly rely on text matching.
    df = df_pbp.copy()
    df["TEAM_ID"] = pd.to_numeric(df.get("TEAM_ID"), errors="coerce")

    # Find first FG attempt rows by description keywords
    df["DESC_UPPER"] = df.get("DESC_UPPER", "").astype(str).str.upper()
    shot_mask = df["DESC_UPPER"].str.contains(r"\b(\d-PT|3PT|2PT|JUMPER|LAYUP|DUNK|HOOK|TIP|SHOT)\b", regex=True)
    shot_rows = df[shot_mask].copy()

    # Compute seconds elapsed from clock remaining (period 1)
    if "PERIOD" in shot_rows.columns and "SEC_REMAINING" in shot_rows.columns:
        shot_rows = shot_rows[shot_rows["PERIOD"] == 1].copy()
        shot_rows["SEC_ELAPSED"] = 12 * 60 - pd.to_numeric(shot_rows["SEC_REMAINING"], errors="coerce")
    else:
        shot_rows["SEC_ELAPSED"] = pd.NA

    shot_rows = shot_rows.dropna(subset=["SEC_ELAPSED"]).sort_values("SEC_ELAPSED").copy()

    if not shot_rows.empty:
        first = shot_rows.iloc[0]
        first_team = int(first["TEAM_ID"]) if pd.notna(first.get("TEAM_ID")) else None
        first_desc = str(first.get("DESC_UPPER", ""))
        first_time = float(first.get("SEC_ELAPSED"))

        def shot_type_from_desc(desc: str) -> str:
            if "3" in desc and "3PT" in desc:
                return "3PT"
            if "2" in desc and "2PT" in desc:
                return "2PT"
            # Heuristic
            if "3" in desc and "PT" in desc:
                return "3PT"
            return "2PT"

        result["FIRST_SHOT_TYPE"] = ""
        result["FIRST_SHOT_TIME_SEC"] = pd.NA
        if first_team in (home_team_id, away_team_id):
            result.loc[result["TEAM_ID"] == first_team, "FIRST_SHOT_TYPE"] = shot_type_from_desc(first_desc)
            result.loc[result["TEAM_ID"] == first_team, "FIRST_SHOT_TIME_SEC"] = first_time
    else:
        result["FIRST_SHOT_TYPE"] = ""
        result["FIRST_SHOT_TIME_SEC"] = pd.NA

    # FIRST MINUTE BASKETS and POINTS
    # Define rows within first 60 seconds of Q1
    if "PERIOD" in df.columns and "SEC_REMAINING" in df.columns:
        df1 = df[(pd.to_numeric(df["PERIOD"], errors="coerce") == 1)].copy()
        df1["SEC_ELAPSED"] = 12 * 60 - pd.to_numeric(df1["SEC_REMAINING"], errors="coerce")
        fm = df1[(df1["SEC_ELAPSED"] >= 0) & (df1["SEC_ELAPSED"] <= 60)].copy()
    else:
        fm = pd.DataFrame()

    # FG makes: look for "MISS" vs made, crude but works well in practice.
    if not fm.empty:
        fg_mask = fm["DESC_UPPER"].str.contains(r"\b(3PT|2PT|DUNK|LAYUP|JUMPER|HOOK|TIP)\b")
        made_mask = fg_mask & ~fm["DESC_UPPER"].str.contains("MISS")
        fg_counts = fm[made_mask].groupby("TEAM_ID").size().to_dict()

        # FT trips: count trips where any FT is made
        ft = fm[fm["DESC_UPPER"].str.contains(r"\bFREE THROW\b")].copy()
        ft_trip_counts: dict[int, int] = {}
        if not ft.empty:
            # group by team and attempt to detect trips by contiguous FTs in the feed
            for tid, g in ft.groupby("TEAM_ID"):
                if pd.isna(tid):
                    continue
                g = g.sort_values("SEC_ELAPSED")
                trips = []
                cur = None
                last_t = None
                for _, row in g.iterrows():
                    t = float(row.get("SEC_ELAPSED") or 0)
                    desc = str(row.get("DESC_UPPER") or "")
                    made = "MISS" not in desc
                    if cur is None:
                        cur = {"start": t, "made_any": made}
                        last_t = t
                        continue
                    # new trip if gap > 6s
                    if last_t is not None and t - last_t > 6:
                        trips.append(cur)
                        cur = {"start": t, "made_any": made}
                    else:
                        cur["made_any"] = cur["made_any"] or made
                    last_t = t
                if cur is not None:
                    trips.append(cur)
                ft_trip_counts[int(tid)] = sum(1 for tr in trips if tr["made_any"])

        fmb = {}
        for tid in fm["TEAM_ID"].dropna().unique():
            tid_i = int(tid)
            fmb[tid_i] = int(fg_counts.get(tid_i, 0)) + int(ft_trip_counts.get(tid_i, 0))

        result["FIRST_MINUTE_BASKETS"] = result["TEAM_ID"].map(fmb).fillna(0).astype(int)

        # points: infer 2/3 for made FGs + count made FTs
        pts = {home_team_id: 0, away_team_id: 0}
        # made fg points
        for tid, g in fm[made_mask].groupby("TEAM_ID"):
            if pd.isna(tid):
                continue
            p = 0
            for desc in g["DESC_UPPER"].astype(str):
                if "3PT" in desc:
                    p += 3
                else:
                    p += 2
            pts[int(tid)] = pts.get(int(tid), 0) + p
        # made ft points
        made_ft = ft[~ft["DESC_UPPER"].str.contains("MISS")].groupby("TEAM_ID").size().to_dict() if not fm.empty else {}
        for tid, c in (made_ft or {}).items():
            if pd.isna(tid):
                continue
            pts[int(tid)] = pts.get(int(tid), 0) + int(c)

        result["FIRST_MINUTE_POINTS"] = result["TEAM_ID"].map(pts).fillna(0).astype(int)

        out = result.copy()
        out.insert(0, "GAME_ID", pd.NA)
        return out[[
            "GAME_ID",
            "TEAM_ID",
            "FIRST_SHOT_TYPE",
            "FIRST_SHOT_TIME_SEC",
            "FIRST_MINUTE_BASKETS",
            "FIRST_MINUTE_POINTS",
        ]]

    # default if fm empty
    result["FIRST_MINUTE_BASKETS"] = 0
    result["FIRST_MINUTE_POINTS"] = 0
    out = result.copy()
    out.insert(0, "GAME_ID", pd.NA)
    return out[[
        "GAME_ID",
        "TEAM_ID",
        "FIRST_SHOT_TYPE",
        "FIRST_SHOT_TIME_SEC",
        "FIRST_MINUTE_BASKETS",
        "FIRST_MINUTE_POINTS",
    ]]


# ---------------------------------------------------------
#   TEAM MAP
# ---------------------------------------------------------
def fetch_team_id_map() -> dict[int, str]:
    """Use the CDN teams list to avoid extra stats calls."""
    url = "https://cdn.nba.com/static/json/staticData/teams.json"

    def call():
        r = CDN_SESSION.get(url, timeout=25)
        r.raise_for_status()
        return r.json()

    payload = with_retries(call, label="CDN teams", max_retries=3, base_sleep=1.0)
    teams = (((payload or {}).get("league") or {}).get("standard")) or []
    out: dict[int, str] = {}
    for t in teams:
        try:
            tid = int(t.get("teamId"))
        except Exception:
            continue
        abbr = (t.get("tricode") or "").strip().upper()
        if tid and abbr:
            out[tid] = abbr
    return out


# ---------------------------------------------------------
#   ROLLUPS + QPS
# ---------------------------------------------------------
def build_team_aggregates(games: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team counts/means from the cached per-game computations."""
    if games is None or games.empty:
        return pd.DataFrame()

    df = games.copy()
    df["TEAM_ID"] = pd.to_numeric(df["TEAM_ID"], errors="coerce")
    df["FIRST_SHOT_TIME_SEC"] = pd.to_numeric(df["FIRST_SHOT_TIME_SEC"], errors="coerce")
    df["FIRST_MINUTE_BASKETS"] = pd.to_numeric(df["FIRST_MINUTE_BASKETS"], errors="coerce")
    df["FIRST_MINUTE_POINTS"] = pd.to_numeric(df["FIRST_MINUTE_POINTS"], errors="coerce")

    # First shot 2PT/3PT counts
    df["IS_2PT"] = (df["FIRST_SHOT_TYPE"] == "2PT").astype(int)
    df["IS_3PT"] = (df["FIRST_SHOT_TYPE"] == "3PT").astype(int)

    agg = df.groupby("TEAM_ABBREVIATION").agg(
        Num_2PT=("IS_2PT", "sum"),
        Num_3PT=("IS_3PT", "sum"),
        First_Minute_Baskets=("FIRST_MINUTE_BASKETS", "sum"),
        First_Minute_Points=("FIRST_MINUTE_POINTS", "sum"),
        Games_Played=("GAME_ID", pd.Series.nunique),
    )

    # Allowed baskets (swap perspective)
    opp = df.copy()
    # infer opponent by GAME_ID within the two rows
    opp_map = opp.groupby("GAME_ID")["TEAM_ABBREVIATION"].apply(list).to_dict()

    allowed_rows = []
    for gid, teams in opp_map.items():
        if len(teams) != 2:
            continue
        t1, t2 = teams
        g = opp[opp["GAME_ID"] == gid]
        b1 = int(g.loc[g["TEAM_ABBREVIATION"] == t1, "FIRST_MINUTE_BASKETS"].fillna(0).iloc[0])
        b2 = int(g.loc[g["TEAM_ABBREVIATION"] == t2, "FIRST_MINUTE_BASKETS"].fillna(0).iloc[0])
        allowed_rows.append({"TEAM_ABBREVIATION": t1, "First_Minute_Baskets_Allowed": b2})
        allowed_rows.append({"TEAM_ABBREVIATION": t2, "First_Minute_Baskets_Allowed": b1})

    allowed = pd.DataFrame(allowed_rows)
    allowed_sum = allowed.groupby("TEAM_ABBREVIATION")["First_Minute_Baskets_Allowed"].sum()
    agg["First_Minute_Baskets_Allowed"] = allowed_sum
    agg["First_Minute_Baskets_Allowed"] = agg["First_Minute_Baskets_Allowed"].fillna(0).astype(int)

    # AVG_Time_FirstShot: average only when team had FIRST shot attempt of game
    first_shots = df.dropna(subset=["FIRST_SHOT_TIME_SEC"]).copy()
    first_shots["FIRST_SHOT_TIME_SEC"] = pd.to_numeric(first_shots["FIRST_SHOT_TIME_SEC"], errors="coerce")
    avg_time_firstshot = first_shots.groupby("TEAM_ABBREVIATION")["FIRST_SHOT_TIME_SEC"].mean().rename("AVG_Time_FirstShot")
    agg = agg.join(avg_time_firstshot)
    agg["AVG_Time_FirstShot"] = pd.to_numeric(agg["AVG_Time_FirstShot"], errors="coerce")

    denom = (agg["Num_2PT"] + agg["Num_3PT"]).replace({0: pd.NA})
    agg["Rate_2PT_raw"] = agg["Num_2PT"] / denom
    agg["FMBPG_raw"] = agg["First_Minute_Baskets"] / agg["Games_Played"].replace({0: pd.NA})

    allowed_per_game = agg["First_Minute_Baskets_Allowed"] / agg["Games_Played"].replace({0: pd.NA})

    # Your current QPS_raw formula
    time_cap = (
        pd.to_numeric(agg["AVG_Time_FirstShot"], errors="coerce")
        .fillna(20)
        .clip(lower=0, upper=20)
    )

    agg["QPS_raw"] = -1 / (
        (agg["FMBPG_raw"] / 1.6)
        - (time_cap / 12.0)
        + ((agg["FMBPG_raw"] * allowed_per_game) / 3.5)
    )

    agg["Rate_2PT"] = agg["Rate_2PT_raw"].round(2).fillna(0)
    agg["First_Minute_Baskets_Per_Game"] = agg["FMBPG_raw"].round(2).fillna(0)
    agg["AVG_Time_FirstShot"] = agg["AVG_Time_FirstShot"].round(2)
    agg["QUIGS_POWER_SCORE"] = agg["QPS_raw"].round(2).fillna(0)

    agg.reset_index(inplace=True)
    agg.rename(columns={"TEAM_ABBREVIATION": "Team"}, inplace=True)

    agg = agg[
        [
            "Team",
            "Num_2PT",
            "Num_3PT",
            "Rate_2PT",
            "First_Minute_Baskets",
            "First_Minute_Baskets_Allowed",
            "Games_Played",
            "First_Minute_Baskets_Per_Game",
            "AVG_Time_FirstShot",
            "QUIGS_POWER_SCORE",
        ]
    ]

    return agg


# ---------------------------------------------------------
#   MATCHUP ENRICHMENT
# ---------------------------------------------------------
def add_rates_and_combined(df_matchups: pd.DataFrame, agg_filtered: pd.DataFrame) -> pd.DataFrame:
    if df_matchups is None or df_matchups.empty:
        return df_matchups

    out = df_matchups.merge(
        agg_filtered[["Team", "Rate_2PT", "QUIGS_POWER_SCORE"]],
        left_on="Home",
        right_on="Team",
        how="left",
    )
    out = out.merge(
        agg_filtered[["Team", "Rate_2PT", "QUIGS_POWER_SCORE"]],
        left_on="Away",
        right_on="Team",
        how="left",
        suffixes=("_Home", "_Away"),
    )
    out.drop(columns=["Team_Home", "Team_Away"], inplace=True, errors="ignore")

    out["Combined_Rate"] = out["Rate_2PT_Home"] + out["Rate_2PT_Away"]
    out["Combined_QUIGS_POWER_SCORE"] = out["QUIGS_POWER_SCORE_Home"] + out["QUIGS_POWER_SCORE_Away"]
    return out


# ---------------------------------------------------------
#   WIN/Loss + BINNED WIN RATE
# ---------------------------------------------------------
def compute_binned_win_rates(
    df: pd.DataFrame,
    score_col: str = "Combined_QUIGS_POWER_SCORE",
    outcome_col: str = "Win/Loss",
    bins: int = 5,
):
    if df is None or df.empty:
        return pd.DataFrame(), np.array([]), []

    tmp = df.copy()
    tmp[score_col] = pd.to_numeric(tmp[score_col], errors="coerce")
    tmp = tmp.dropna(subset=[score_col]).copy()
    if tmp.empty:
        return pd.DataFrame(), np.array([]), []

    edges = np.quantile(tmp[score_col], np.linspace(0, 1, bins + 1))
    # ensure monotonic
    edges = np.unique(edges)
    if len(edges) < 2:
        return pd.DataFrame(), np.array([]), []

    labels = [f"{edges[i]:.2f} to {edges[i+1]:.2f}" for i in range(len(edges) - 1)]
    tmp["Bin"] = pd.cut(tmp[score_col], bins=edges, labels=labels, include_lowest=True)

    # map win/loss to 1/0
    tmp["Win"] = tmp[outcome_col].astype(str).str.upper().str.startswith("W").astype(int)
    binned = tmp.groupby("Bin").agg(Games=("Win", "size"), Wins=("Win", "sum"))
    binned["Win_Rate"] = (binned["Wins"] / binned["Games"]).round(3)
    binned = binned.reset_index().rename(columns={"Bin": "Bin"})
    return binned, edges, labels


def add_win_probabilities(
    df_matchups: pd.DataFrame,
    binned_table: pd.DataFrame,
    edges: np.ndarray,
    score_col: str = "Combined_QUIGS_POWER_SCORE",
    prob_col: str = "Win_Prob",
) -> pd.DataFrame:
    if df_matchups is None or df_matchups.empty:
        return df_matchups

    out = df_matchups.copy()
    out[score_col] = pd.to_numeric(out[score_col], errors="coerce")

    if binned_table is None or binned_table.empty or edges is None or len(edges) < 2:
        out[prob_col] = pd.NA
        return out

    labels = [f"{edges[i]:.2f} to {edges[i+1]:.2f}" for i in range(len(edges) - 1)]
    bin_labels = pd.cut(out[score_col], bins=edges, labels=labels, include_lowest=True)

    rate_map = dict(zip(binned_table["Bin"].astype(str), binned_table["Win_Rate"]))
    out[prob_col] = bin_labels.astype(str).map(rate_map)
    out.loc[out[score_col].isna(), prob_col] = pd.NA
    out[prob_col] = pd.to_numeric(out[prob_col], errors="coerce").round(3)
    return out


# ---------------------------------------------------------
#   FORMATTING/CHART (OPTIONAL)
# ---------------------------------------------------------
def format_workbook_and_chart(_path: str):
    """Kept minimal; if openpyxl exists (it should), you can extend formatting here."""
    # The original file had extensive formatting. To keep the deploy stable,
    # we leave the workbook mostly unformatted.
    return


# ---------------------------------------------------------
#   MAIN
# ---------------------------------------------------------
def main():
    print(f"Season: {SEASON} | Type: {SEASON_TYPE}")

    # Team map
    team_id_map = fetch_team_id_map()
    if not team_id_map:
        raise RuntimeError("Failed to load team map")

    # Season games list
    print("Fetching season games list...")
    games = get_games_for_season(SEASON, SEASON_TYPE)
    print("Games rows:", len(games))

    # Load existing cache
    if os.path.exists(CACHE_FILE):
        cache = pd.read_csv(CACHE_FILE, dtype={"GAME_ID": str})
    else:
        cache = pd.DataFrame(columns=[
            "GAME_ID",
            "TEAM_ID",
            "TEAM_ABBREVIATION",
            "GAME_DATE",
            "MATCHUP",
            "FIRST_SHOT_TYPE",
            "FIRST_SHOT_TIME_SEC",
            "FIRST_MINUTE_BASKETS",
            "FIRST_MINUTE_POINTS",
        ])

    cached_game_ids = set(cache.get("GAME_ID", pd.Series(dtype=str)).astype(str).unique())
    new_game_ids = [gid for gid in games["GAME_ID"].astype(str).unique() if gid not in cached_game_ids]
    print(f"New games to backfill: {len(new_game_ids)}")

    # Backfill new games
    new_rows = []
    if new_game_ids:
        # Need mapping from game -> home/away team ids
        # Use games df: two rows per game, parse MATCHUP to determine home.
        for gid in new_game_ids:
            g = games[games["GAME_ID"].astype(str) == str(gid)].copy()
            if g.empty or g["TEAM_ID"].nunique() != 2:
                continue

            # Determine home/away team id by '@' in matchup string
            home_tid = None
            away_tid = None
            for _, row in g.iterrows():
                matchup = str(row.get("MATCHUP") or "")
                tid = int(row.get("TEAM_ID"))
                if "vs." in matchup:
                    home_tid = tid
                elif "@" in matchup:
                    away_tid = tid
            if home_tid is None or away_tid is None:
                # fallback: pick arbitrary
                tids = list(map(int, g["TEAM_ID"].tolist()))
                home_tid, away_tid = tids[0], tids[1]

            print(f"\nBackfilling game {gid} (home={home_tid}, away={away_tid})")
            df_pbp = fetch_playbyplay_df(str(gid))
            if df_pbp is None or df_pbp.empty:
                continue

            per_team = compute_first_shot_and_first_minute(df_pbp, home_tid, away_tid)
            per_team["GAME_ID"] = str(gid)

            # Merge season game meta info
            meta = g[["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "GAME_DATE", "MATCHUP"]].copy()
            meta["GAME_ID"] = meta["GAME_ID"].astype(str)
            meta["TEAM_ID"] = pd.to_numeric(meta["TEAM_ID"], errors="coerce")

            out = meta.merge(per_team, on=["GAME_ID", "TEAM_ID"], how="left")
            new_rows.append(out)

            time.sleep(SLEEP_BETWEEN_CALLS)

    if new_rows:
        new_df = pd.concat(new_rows, ignore_index=True)
        cache = pd.concat([cache, new_df], ignore_index=True)

        # Persist cache
        cache.to_csv(CACHE_FILE, index=False)
        print("Updated cache:", CACHE_FILE)

    # Build team aggregates
    cache["TEAM_ID"] = pd.to_numeric(cache.get("TEAM_ID"), errors="coerce")
    cache["TEAM_ABBREVIATION"] = cache.get("TEAM_ABBREVIATION")
    agg = build_team_aggregates(cache)

    agg_filtered = agg[agg["Games_Played"] > 10].copy()
    agg_filtered = agg_filtered.sort_values("QUIGS_POWER_SCORE", ascending=False).reset_index(drop=True)

    # TODAY'S MATCHUPS
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    matchups_base = get_matchups_for_date(team_id_map, today_str)
    matchups = add_rates_and_combined(matchups_base, agg_filtered)
    matchups = dedupe_game_ids(matchups)
    if matchups is not None and not matchups.empty and "Combined_QUIGS_POWER_SCORE" in matchups.columns:
        matchups = matchups.sort_values("Combined_QUIGS_POWER_SCORE", ascending=False).reset_index(drop=True)

    # THIS WEEK'S MATCHUPS (today + next 5)
    week_rows = []
    for i in range(0, 6):
        d_str = (datetime.date.today() + datetime.timedelta(days=i)).strftime("%Y-%m-%d")
        print(f"\nFetching scoreboard for {d_str} (ThisWeeksMatchups) ...")
        day_df = get_matchups_for_date(team_id_map, d_str)
        if day_df is None or day_df.empty:
            continue
        day_df = dedupe_game_ids(day_df)
        day_df.insert(0, "Game_Date", d_str)
        week_rows.append(day_df)
        time.sleep(SCOREBOARD_SLEEP)

    if week_rows:
        this_week_base = pd.concat(week_rows, ignore_index=True)
        this_week_base = this_week_base.drop_duplicates(subset=["GAME_ID"], keep="first").copy()

        tmp = add_rates_and_combined(this_week_base.drop(columns=["Game_Date"]), agg_filtered)
        this_week = tmp.copy()
        this_week.insert(0, "Game_Date", this_week_base["Game_Date"].values)

        this_week = this_week[
            [
                "Game_Date",
                "Home",
                "Away",
                "Rate_2PT_Home",
                "Rate_2PT_Away",
                "Combined_Rate",
                "QUIGS_POWER_SCORE_Home",
                "QUIGS_POWER_SCORE_Away",
                "Combined_QUIGS_POWER_SCORE",
                "GAME_ID",
            ]
        ].copy()

        this_week = this_week.sort_values("Combined_QUIGS_POWER_SCORE", ascending=False).reset_index(drop=True)
    else:
        this_week = pd.DataFrame(columns=[
            "Game_Date", "Home", "Away",
            "Rate_2PT_Home", "Rate_2PT_Away", "Combined_Rate",
            "QUIGS_POWER_SCORE_Home", "QUIGS_POWER_SCORE_Away", "Combined_QUIGS_POWER_SCORE",
            "GAME_ID",
        ])

    # LAST 80 DAYS (excluding today): use cache game_date column
    cache_dates = pd.to_datetime(cache.get("GAME_DATE"), errors="coerce")
    cutoff = pd.Timestamp(datetime.date.today() - datetime.timedelta(days=80))
    mask_last80 = (cache_dates >= cutoff) & (cache_dates.dt.date != datetime.date.today())
    last80_raw = cache.loc[mask_last80].copy()
    # Convert cache to matchup-level (one row per GAME_ID)
    if not last80_raw.empty:
        # Build matchup view from season games list
        # Use scoreboard for each date? Too slow. Instead parse from games list.
        g80 = games[games["GAME_ID"].astype(str).isin(last80_raw["GAME_ID"].astype(str).unique())].copy()
        # Create game -> (home, away) using MATCHUP parsing
        rows = []
        for gid, gg in g80.groupby("GAME_ID"):
            home = None
            away = None
            for _, r in gg.iterrows():
                m = str(r.get("MATCHUP") or "")
                ab = str(r.get("TEAM_ABBREVIATION") or "")
                if "vs." in m:
                    home = ab
                elif "@" in m:
                    away = ab
            if home and away:
                rows.append({"GAME_ID": str(gid), "Home": home, "Away": away})
        last80_matchups = pd.DataFrame(rows)
        last80_matchups = add_rates_and_combined(last80_matchups, agg_filtered)
        last80_matchups = dedupe_game_ids(last80_matchups)
        last80 = last80_matchups
        # No win/loss column available without standings; keep placeholder
        last80["Win/Loss"] = pd.NA
    else:
        last80 = pd.DataFrame(columns=["Home", "Away", "Combined_QUIGS_POWER_SCORE", "GAME_ID", "Win/Loss"])

    # Binned win rates (will be empty without Win/Loss data)
    binned_table, edges, _labels = compute_binned_win_rates(
        last80,
        score_col="Combined_QUIGS_POWER_SCORE",
        outcome_col="Win/Loss",
        bins=5,
    )
    matchups = add_win_probabilities(matchups, binned_table, edges, score_col="Combined_QUIGS_POWER_SCORE", prob_col="Win_Prob")
    this_week = add_win_probabilities(this_week, binned_table, edges, score_col="Combined_QUIGS_POWER_SCORE", prob_col="Win_Prob")

    if matchups is not None and not matchups.empty and "Combined_QUIGS_POWER_SCORE" in matchups.columns:
        matchups = matchups.sort_values("Combined_QUIGS_POWER_SCORE", ascending=False).reset_index(drop=True)
    if this_week is not None and not this_week.empty and "Combined_QUIGS_POWER_SCORE" in this_week.columns:
        this_week = this_week.sort_values("Combined_QUIGS_POWER_SCORE", ascending=False).reset_index(drop=True)

    # SAVE EXCEL
    out_path = os.path.join(OUTPUT_DIR, f"first_shot_report_{SEASON.replace('-', '')}.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        agg_filtered.to_excel(writer, sheet_name="TeamBreakdown", index=False)
        matchups.to_excel(writer, sheet_name="TodaysMatchups", index=False)
        this_week.to_excel(writer, sheet_name="ThisWeeksMatchups", index=False)
        last80.to_excel(writer, sheet_name="Last80DaysMatchups", index=False)
        binned_table.to_excel(writer, sheet_name="BinnedWinRate", index=False)

    format_workbook_and_chart(out_path)
    print("\nSaved Excel:", out_path)
    print("\nToday's matchups:")
    print(matchups)


if __name__ == "__main__":
    main()

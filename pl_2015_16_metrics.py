"""
Premier League 2015/16 - team metrics from StatsBomb event data
==============================================================
Builds ~250 team metrics (for AND against) for all 20 clubs, then reports
which ones Leicester City rank top 3 / bottom 3 in.

Needs:  statsbomb_browser.py (same folder), pandas, numpy, requests

What it does
  1. Gets the 380 match IDs via get_matches(2, 27)
  2. Downloads each event file ONCE (threaded), boils it down to small
     per-match counters and caches them in ./pl1516_cache/  (re-runs are instant)
  3. Sums counters per team, derives metrics, ranks the league
  4. Writes CSVs + a text report

Heads-up: the raw event files are big (roughly 1-2 GB total download).
Raw files are NOT kept - only the tiny counter files are cached.
If you have the events folder locally (data/events/<id>.json or events/<id>.json
under REPO_ROOT), it will use those instead of downloading.

NOT AVAILABLE in StatsBomb open data: "meters run" (needs tracking data).
Closest proxies included: metres carried with the ball, metres of passing,
pressures, and defensive-action height.
"""

import json
import math
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from statsbomb_browser import get_matches, REPO_ROOT, REMOTE_BASE

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
COMPETITION_ID, SEASON_ID = 2, 27
TARGET = "Leicester City"
CACHE_DIR = Path("pl1516_cache")
WORKERS = 8
YARD = 0.9144                     # StatsBomb pitch units are ~yards -> metres

ON_TARGET = {"Goal", "Saved", "Saved to Post"}
TACKLE_WON = {"Won", "Success In Play", "Success Out"}
GK_SAVES = {"Shot Saved", "Shot Saved to Post", "Penalty Saved"}
SET_PIECE_PATTERNS = {"From Corner", "From Free Kick", "From Throw In"}
NON_PLAY = {"Starting XI", "Half Start", "Half End", "Substitution", "Tactical Shift",
            "Injury Stoppage", "Referee Ball-Drop", "Player Off", "Player On",
            "Camera On", "Camera off", "Camera Off", "Camera on"}
TOUCH_TYPES = {"Pass", "Shot", "Dribble", "Miscontrol", "Dispossessed"}


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def nm(d):
    return d.get("name") if isinstance(d, dict) else None


def ts_sec(ts):
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def in_box(x, y):
    return x >= 102 and 18 <= y <= 62


def sdiv(a, b):
    return a / b if b else np.nan


def load_events(match_id):
    for sub in ("data/events", "events"):
        p = Path(REPO_ROOT) / sub / f"{match_id}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    url = f"{REMOTE_BASE}/events/{match_id}.json"
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=90)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


# ----------------------------------------------------------------------------
# 1. ONE MATCH -> COUNTERS (everything additive so we can sum over a season)
# ----------------------------------------------------------------------------
def process_match(events, home, away, hs, as_):
    teams = (home, away)
    C = {t: defaultdict(float) for t in teams}
    goals = []                                   # (team, period, minute)
    poss = {}
    subbed_on = {t: set() for t in teams}
    sub_minutes = {t: [] for t in teams}
    players = {t: set() for t in teams}
    starters, formations = {}, {}

    for ev in events:
        et = ev["type"]["name"]
        team = nm(ev.get("team"))
        if team not in C:
            continue
        c = C[team]
        period = ev.get("period", 1)
        loc = ev.get("location")
        pl = ev.get("player")
        if pl:
            players[team].add(pl["id"])

        if et == "Starting XI":
            tac = ev.get("tactics", {})
            starters[team] = [x["player"]["id"] for x in tac.get("lineup", [])]
            formations[team] = tac.get("formation")
            continue
        if period >= 5:                          # penalty shoot-out: ignore
            continue

        # ---- possession bookkeeping (time, passes, shots, start position)
        pkey = None
        if et not in NON_PLAY and "possession" in ev:
            pkey = (period, ev["possession"])
            t0 = ts_sec(ev["timestamp"])
            t1 = t0 + (ev.get("duration") or 0)
            p = poss.get(pkey)
            if p is None:
                p = poss[pkey] = {"team": nm(ev["possession_team"]), "start": t0, "end": t1,
                                  "passes": 0, "shots": 0, "start_x": None}
            else:
                p["start"] = min(p["start"], t0)
                p["end"] = max(p["end"], t1)
            if p["start_x"] is None and loc and team == p["team"]:
                p["start_x"] = loc[0]

        # ---- touches (approximation: passes, shots, dribbles, receipts, ...)
        if et in TOUCH_TYPES and loc:
            c["touches"] += 1
            if loc[0] >= 80:
                c["touches_final_third"] += 1
            if in_box(*loc):
                c["touches_box"] += 1
        elif et == "Ball Receipt*" and loc:
            if nm(ev.get("ball_receipt", {}).get("outcome")) != "Incomplete":
                c["touches"] += 1
                if loc[0] >= 80:
                    c["touches_final_third"] += 1
                if in_box(*loc):
                    c["touches_box"] += 1

        # ---- event specific
        if et == "Pass":
            p_ = ev["pass"]
            sx, sy = loc
            ex, ey = p_["end_location"]
            done = p_.get("outcome") is None
            c["passes"] += 1
            c["pass_len_sum_m"] += p_.get("length", 0.0) * YARD
            if done:
                c["passes_completed"] += 1
            if sx >= 80:
                c["passes_in_final_third"] += 1
            if sx <= 72:
                c["passes_own60"] += 1           # for opponent's PPDA
            if abs(p_.get("angle", 9)) < math.pi / 4:
                c["passes_forward"] += 1
            if p_.get("length", 0) >= 32:
                c["long_balls"] += 1
            if ev.get("under_pressure"):
                c["passes_under_pressure"] += 1
            if p_.get("switch"):
                c["switches"] += 1
            if p_.get("shot_assist"):
                c["key_passes"] += 1
            if p_.get("goal_assist"):
                c["assists"] += 1
            if p_.get("cross"):
                c["crosses"] += 1
                c["crosses_completed"] += int(done)
            if nm(p_.get("technique")) == "Through Ball":
                c["through_balls"] += 1
            ptype = nm(p_.get("type"))
            if ptype == "Corner":
                c["corners"] += 1
            elif ptype == "Free Kick":
                c["free_kick_passes"] += 1
            elif ptype == "Throw-in":
                c["throw_ins"] += 1
            elif ptype == "Goal Kick":
                c["goal_kicks"] += 1
            if done:
                if ex - sx >= 10 and ex >= 40:
                    c["passes_progressive"] += 1
                if sx < 80 <= ex:
                    c["passes_into_final_third"] += 1
                if in_box(ex, ey) and not in_box(sx, sy):
                    c["passes_into_box"] += 1
            if pkey and team == poss[pkey]["team"]:
                poss[pkey]["passes"] += 1

        elif et == "Shot":
            s = ev["shot"]
            xg = s.get("statsbomb_xg") or 0.0
            out, stype = nm(s.get("outcome")), nm(s.get("type"))
            pattern = nm(ev.get("play_pattern"))
            x, y = loc
            is_pen = stype == "Penalty"
            c["shots"] += 1
            c["xg"] += xg
            if is_pen:
                c["penalties"] += 1
            else:
                c["np_shots"] += 1
                c["npxg"] += xg
                c["shot_dist_sum_m"] += math.hypot(120 - x, 40 - y) * YARD
                if xg >= 0.3:
                    c["big_chances"] += 1
            if out in ON_TARGET:
                c["shots_on_target"] += 1
            if out == "Blocked":
                c["shots_blocked"] += 1
            if in_box(x, y):
                c["shots_in_box"] += 1
            if nm(s.get("body_part")) == "Head":
                c["headers"] += 1
            if stype == "Free Kick":
                c["direct_fk_shots"] += 1
            if pattern == "From Counter":
                c["counter_shots"] += 1
                c["counter_xg"] += xg
            if pattern in SET_PIECE_PATTERNS:
                c["setpiece_shots"] += 1
                c["setpiece_xg"] += xg
            if pl and pl["id"] in subbed_on[team]:
                c["sub_shots"] += 1
                c["sub_xg"] += xg
                c["sub_goals"] += int(out == "Goal")
            if out == "Goal":
                c["shot_goals"] += 1
                goals.append((team, period, ev["minute"]))
            if pkey and team == poss[pkey]["team"]:
                poss[pkey]["shots"] += 1

        elif et == "Own Goal For":
            c["own_goals_for"] += 1
            goals.append((team, period, ev["minute"]))

        elif et == "Own Goal Against":
            c["own_goals_against"] += 1

        elif et == "Carry":
            sx, sy = loc
            ex, ey = ev["carry"]["end_location"]
            c["carries"] += 1
            c["carry_dist_m"] += math.hypot(ex - sx, ey - sy) * YARD
            if ex - sx >= 5 and ex >= 40:
                c["carries_progressive"] += 1
            if sx < 80 <= ex:
                c["carries_into_final_third"] += 1
            if in_box(ex, ey) and not in_box(sx, sy):
                c["carries_into_box"] += 1

        elif et == "Dribble":
            c["dribbles"] += 1
            c["dribbles_completed"] += int(nm(ev["dribble"].get("outcome")) == "Complete")

        elif et == "Dispossessed":
            c["dispossessed"] += 1
        elif et == "Miscontrol":
            c["miscontrols"] += 1
        elif et == "Dribbled Past":
            c["dribbled_past"] += 1
        elif et == "Error":
            c["errors"] += 1
        elif et == "Offside":
            c["offsides"] += 1

        elif et == "Duel":
            d = ev["duel"]
            dt = nm(d.get("type"))
            if dt == "Tackle":
                c["tackles"] += 1
                c["tackles_won"] += int(nm(d.get("outcome")) in TACKLE_WON)
                if loc:
                    c["def_x_sum"] += loc[0]
                    c["def_x_n"] += 1
                    if loc[0] >= 48:
                        c["ppda_actions"] += 1
            elif dt == "Aerial Lost":
                c["aerial_lost"] += 1

        elif et == "Interception":
            c["interceptions"] += 1
            if loc:
                c["def_x_sum"] += loc[0]
                c["def_x_n"] += 1
                if loc[0] >= 48:
                    c["ppda_actions"] += 1

        elif et == "Pressure":
            c["pressures"] += 1
            if ev.get("counterpress"):
                c["counterpresses"] += 1
            if loc:
                c["def_x_sum"] += loc[0]
                c["def_x_n"] += 1
                if loc[0] >= 80:
                    c["pressures_att_third"] += 1

        elif et == "Clearance":
            c["clearances"] += 1
        elif et == "Block":
            c["blocks"] += 1

        elif et == "Ball Recovery":
            if not ev.get("ball_recovery", {}).get("recovery_failure"):
                c["recoveries"] += 1
                if loc and loc[0] >= 80:
                    c["recoveries_att_third"] += 1
                if loc and loc[0] >= 60:
                    c["recoveries_opp_half"] += 1

        elif et == "Foul Committed":
            fc = ev.get("foul_committed", {})
            c["fouls"] += 1
            card = nm(fc.get("card"))
            c["yellows"] += int(card == "Yellow Card")
            c["reds"] += int(card in ("Red Card", "Second Yellow"))
            c["penalties_conceded"] += int(bool(fc.get("penalty")))
            if loc and loc[0] >= 48:
                c["ppda_actions"] += 1
        elif et == "Bad Behaviour":
            card = nm(ev.get("bad_behaviour", {}).get("card"))
            c["yellows"] += int(card == "Yellow Card")
            c["reds"] += int(card in ("Red Card", "Second Yellow"))
        elif et == "Foul Won":
            c["fouls_won"] += 1

        elif et == "Goal Keeper":
            gt = nm(ev["goalkeeper"].get("type"))
            if gt in GK_SAVES:
                c["gk_saves"] += 1
            elif gt == "Collected":
                c["gk_claims"] += 1
            elif gt == "Punch":
                c["gk_punches"] += 1

        elif et == "Tactical Shift":
            c["tactical_shifts"] += 1

        elif et == "Substitution":
            s = ev["substitution"]
            c["subs"] += 1
            c["sub_minute_sum"] += ev["minute"]
            c["subs_before60"] += int(ev["minute"] < 60)
            c["subs_injury"] += int(nm(s.get("outcome")) == "Injury")
            subbed_on[team].add(s["replacement"]["id"])
            sub_minutes[team].append(ev["minute"])

    # ---- possessions -> time in possession etc.
    for p in poss.values():
        if p["team"] not in C:
            continue
        c = C[p["team"]]
        c["possessions"] += 1
        c["possession_time_s"] += max(p["end"] - p["start"], 0)
        c["possessions_10plus"] += int(p["passes"] >= 10)
        c["possessions_with_shot"] += int(p["shots"] > 0)
        c["poss_passes"] += p["passes"]
        if p["start_x"] is not None:
            c["poss_start_x_sum"] += p["start_x"]
            c["poss_start_n"] += 1

    # ---- results, game state and timing
    first = goals[0][0] if goals else None
    scores = {home: (hs, as_), away: (as_, hs)}
    for t in teams:
        c = C[t]
        gf, ga = scores[t]
        pts = 3 if gf > ga else (1 if gf == ga else 0)
        c["matches"] = 1
        c["goals"] = gf
        c["points"] = pts
        c["wins"] = int(gf > ga)
        c["clean_sheets"] = int(ga == 0)
        c["failed_to_score"] = int(gf == 0)
        mine = [g for g in goals if g[0] == t]
        c["goals_1h"] = sum(g[1] == 1 for g in mine)
        c["goals_2h"] = sum(g[1] == 2 for g in mine)
        c["goals_75plus"] = sum(g[1] == 2 and g[2] >= 75 for g in mine)
        if first == t:
            c["scored_first"] = 1
            c["pts_when_scoring_first"] = pts
        elif first is not None:
            c["conceded_first"] = 1
            c["pts_when_conceding_first"] = pts
        if sub_minutes[t]:
            c["games_with_sub"] = 1
            c["first_sub_minute_sum"] = min(sub_minutes[t])

    return {
        "counters": {t: dict(C[t]) for t in teams},
        "meta": {"starters": starters, "formations": formations,
                 "players": {t: sorted(players[t]) for t in teams}},
    }


def get_match_result(row):
    cache = CACHE_DIR / f"{row.match_id}.json"
    if cache.exists():
        return row.match_id, json.loads(cache.read_text(encoding="utf-8"))
    events = load_events(row.match_id)
    res = process_match(events, row.home_team, row.away_team,
                        int(row.home_score), int(row.away_score))
    cache.write_text(json.dumps(res), encoding="utf-8")
    return row.match_id, res


# ----------------------------------------------------------------------------
# 2. COUNTERS -> METRICS
# ----------------------------------------------------------------------------
def derive(me, other):
    """
    me / other: dicts of season-summed counters.
    Returns list of (name, value, direction, category, also_make_against_version)
    direction: +1 higher is better, -1 lower is better, 0 neutral/style
    """
    n = me["matches"]
    out = []

    def add(name, val, d, cat, opp=True):
        out.append((name, val, d, cat, opp))

    def pg(k):
        return me[k] / n

    # --- Results
    add("points_pg", pg("points"), 1, "Results", False)
    add("win_pct", 100 * pg("wins"), 1, "Results", False)
    add("goals_pg", pg("goals"), 1, "Results")
    add("clean_sheet_pct", 100 * pg("clean_sheets"), 1, "Results", False)
    add("failed_to_score_pct", 100 * pg("failed_to_score"), -1, "Results", False)
    add("scored_first_pct", 100 * pg("scored_first"), 1, "Results", False)
    add("pts_pg_when_scoring_first", sdiv(me["pts_when_scoring_first"], me["scored_first"]), 1, "Results", False)
    add("pts_pg_when_conceding_first", sdiv(me["pts_when_conceding_first"], me["conceded_first"]), 1, "Results", False)

    # --- Timing of goals
    add("goals_1st_half_pg", pg("goals_1h"), 1, "Timing")
    add("goals_2nd_half_pg", pg("goals_2h"), 1, "Timing")
    add("goals_after_75min_pg", pg("goals_75plus"), 1, "Timing")

    # --- Shooting
    add("shots_pg", pg("shots"), 1, "Shooting")
    add("shots_on_target_pg", pg("shots_on_target"), 1, "Shooting")
    add("shot_accuracy_pct", 100 * sdiv(me["shots_on_target"], me["shots"]), 1, "Shooting")
    add("goal_conversion_pct", 100 * sdiv(me["shot_goals"], me["shots"]), 1, "Shooting")
    add("xg_pg", pg("xg"), 1, "Shooting")
    add("npxg_pg", pg("npxg"), 1, "Shooting")
    add("npxg_per_shot", sdiv(me["npxg"], me["np_shots"]), 1, "Shooting")
    add("avg_shot_distance_m", sdiv(me["shot_dist_sum_m"], me["np_shots"]), 0, "Shooting")
    add("shots_in_box_pg", pg("shots_in_box"), 1, "Shooting")
    add("shots_in_box_pct", 100 * sdiv(me["shots_in_box"], me["shots"]), 1, "Shooting")
    add("big_chances_pg", pg("big_chances"), 1, "Shooting")
    add("headed_shots_pg", pg("headers"), 0, "Shooting")
    add("blocked_shots_pg", pg("shots_blocked"), 0, "Shooting")
    add("penalties_pg", pg("penalties"), 1, "Shooting")
    add("goals_minus_xg_pg", (me["shot_goals"] - me["xg"]) / n, 1, "Shooting")

    # --- Chance creation
    add("key_passes_pg", pg("key_passes"), 1, "Chance creation")
    add("assists_pg", pg("assists"), 1, "Chance creation")
    add("through_balls_pg", pg("through_balls"), 0, "Chance creation")
    add("crosses_pg", pg("crosses"), 0, "Chance creation")
    add("cross_completion_pct", 100 * sdiv(me["crosses_completed"], me["crosses"]), 1, "Chance creation")
    add("passes_into_box_pg", pg("passes_into_box"), 1, "Chance creation")
    add("carries_into_box_pg", pg("carries_into_box"), 1, "Chance creation")
    add("touches_in_box_pg", pg("touches_box"), 1, "Chance creation")

    # --- Transitions & set pieces
    add("counter_attack_shots_pg", pg("counter_shots"), 1, "Transitions & set pieces")
    add("counter_attack_xg_pg", pg("counter_xg"), 1, "Transitions & set pieces")
    add("set_piece_shots_pg", pg("setpiece_shots"), 1, "Transitions & set pieces")
    add("set_piece_xg_pg", pg("setpiece_xg"), 1, "Transitions & set pieces")
    add("corners_pg", pg("corners"), 1, "Transitions & set pieces")
    add("free_kick_passes_pg", pg("free_kick_passes"), 0, "Transitions & set pieces")
    add("throw_ins_pg", pg("throw_ins"), 0, "Transitions & set pieces")
    add("goal_kicks_pg", pg("goal_kicks"), 0, "Transitions & set pieces")
    add("direct_free_kick_shots_pg", pg("direct_fk_shots"), 0, "Transitions & set pieces")

    # --- Passing
    add("passes_pg", pg("passes"), 0, "Passing")
    add("pass_completion_pct", 100 * sdiv(me["passes_completed"], me["passes"]), 1, "Passing")
    add("avg_pass_length_m", sdiv(me["pass_len_sum_m"], me["passes"]), 0, "Passing")
    add("total_pass_distance_pg_m", pg("pass_len_sum_m"), 0, "Passing")
    add("long_ball_pct", 100 * sdiv(me["long_balls"], me["passes"]), 0, "Passing")
    add("forward_pass_pct", 100 * sdiv(me["passes_forward"], me["passes"]), 0, "Passing")
    add("progressive_passes_pg", pg("passes_progressive"), 1, "Passing")
    add("passes_into_final_third_pg", pg("passes_into_final_third"), 1, "Passing")
    add("passes_in_final_third_pg", pg("passes_in_final_third"), 1, "Passing")
    add("under_pressure_pass_pct", 100 * sdiv(me["passes_under_pressure"], me["passes"]), 0, "Passing")
    add("switches_of_play_pg", pg("switches"), 0, "Passing")

    # --- Possession & territory
    add("possession_pct_time", 100 * sdiv(me["possession_time_s"], me["possession_time_s"] + other["possession_time_s"]), 0, "Possession", False)
    add("possession_pct_passes", 100 * sdiv(me["passes"], me["passes"] + other["passes"]), 0, "Possession", False)
    add("time_in_possession_min_pg", me["possession_time_s"] / 60 / n, 0, "Possession")
    add("possessions_pg", pg("possessions"), 0, "Possession")
    add("avg_possession_duration_s", sdiv(me["possession_time_s"], me["possessions"]), 0, "Possession")
    add("passes_per_possession", sdiv(me["poss_passes"], me["possessions"]), 0, "Possession")
    add("long_possessions_10plus_passes_pg", pg("possessions_10plus"), 0, "Possession")
    add("possessions_ending_in_shot_pct", 100 * sdiv(me["possessions_with_shot"], me["possessions"]), 1, "Possession")
    add("avg_possession_start_dist_from_own_goal_m", sdiv(me["poss_start_x_sum"], me["poss_start_n"]) * YARD, 0, "Possession")
    add("field_tilt_pct", 100 * sdiv(me["passes_in_final_third"], me["passes_in_final_third"] + other["passes_in_final_third"]), 0, "Possession", False)
    add("touches_pg", pg("touches"), 0, "Possession")
    add("touches_final_third_pg", pg("touches_final_third"), 1, "Possession")

    # --- Carrying / movement (proxy for 'metres run')
    add("carries_pg", pg("carries"), 0, "Carrying")
    add("metres_carried_pg", pg("carry_dist_m"), 0, "Carrying")
    add("avg_carry_length_m", sdiv(me["carry_dist_m"], me["carries"]), 0, "Carrying")
    add("progressive_carries_pg", pg("carries_progressive"), 1, "Carrying")
    add("carries_into_final_third_pg", pg("carries_into_final_third"), 1, "Carrying")
    add("dribbles_attempted_pg", pg("dribbles"), 0, "Carrying")
    add("dribble_success_pct", 100 * sdiv(me["dribbles_completed"], me["dribbles"]), 1, "Carrying")
    add("dispossessed_pg", pg("dispossessed"), -1, "Carrying")
    add("miscontrols_pg", pg("miscontrols"), -1, "Carrying")

    # --- Defending & pressing
    add("tackles_pg", pg("tackles"), 0, "Defending")
    add("tackle_win_pct", 100 * sdiv(me["tackles_won"], me["tackles"]), 1, "Defending")
    add("interceptions_pg", pg("interceptions"), 0, "Defending")
    add("clearances_pg", pg("clearances"), 0, "Defending")
    add("blocks_pg", pg("blocks"), 0, "Defending")
    add("ball_recoveries_pg", pg("recoveries"), 1, "Defending")
    add("recoveries_att_third_pg", pg("recoveries_att_third"), 1, "Defending")
    add("recoveries_opp_half_pg", pg("recoveries_opp_half"), 1, "Defending")
    add("pressures_pg", pg("pressures"), 0, "Defending")
    add("pressures_att_third_pg", pg("pressures_att_third"), 1, "Defending")
    add("counterpressures_pg", pg("counterpresses"), 1, "Defending")
    add("ppda_lower_is_more_intense", sdiv(other["passes_own60"], me["ppda_actions"]), -1, "Defending", False)
    add("avg_defensive_action_height_m", sdiv(me["def_x_sum"], me["def_x_n"]) * YARD, 0, "Defending")
    add("aerial_win_pct", 100 * sdiv(other["aerial_lost"], other["aerial_lost"] + me["aerial_lost"]), 1, "Defending", False)
    add("dribbled_past_pg", pg("dribbled_past"), -1, "Defending")
    add("errors_pg", pg("errors"), -1, "Defending")

    # --- Discipline
    add("fouls_committed_pg", pg("fouls"), 0, "Discipline")
    add("fouls_won_pg", pg("fouls_won"), 1, "Discipline")
    add("yellow_cards_pg", pg("yellows"), -1, "Discipline")
    add("red_cards_pg", pg("reds"), -1, "Discipline")
    add("offsides_pg", pg("offsides"), -1, "Discipline")
    add("penalties_conceded_pg", pg("penalties_conceded"), -1, "Discipline", False)

    # --- Goalkeeping
    add("gk_saves_pg", pg("gk_saves"), 0, "Goalkeeping", False)
    add("save_pct", 100 * sdiv(me["gk_saves"], other["shots_on_target"]), 1, "Goalkeeping", False)
    add("gk_claims_pg", pg("gk_claims"), 0, "Goalkeeping", False)
    add("gk_punches_pg", pg("gk_punches"), 0, "Goalkeeping", False)
    add("goals_prevented_pg_xga_minus_ga", (other["xg"] - other["shot_goals"]) / n, 1, "Goalkeeping", False)

    # --- Substitutions & squad management
    add("subs_pg", pg("subs"), 0, "Substitutions", False)
    add("avg_sub_minute", sdiv(me["sub_minute_sum"], me["subs"]), 0, "Substitutions", False)
    add("avg_first_sub_minute", sdiv(me["first_sub_minute_sum"], me["games_with_sub"]), 0, "Substitutions", False)
    add("subs_before_60min_pct", 100 * sdiv(me["subs_before60"], me["subs"]), 0, "Substitutions", False)
    add("injury_subs_total", me["subs_injury"], -1, "Substitutions", False)
    add("sub_shots_pg", pg("sub_shots"), 1, "Substitutions", False)
    add("sub_xg_pg", pg("sub_xg"), 1, "Substitutions", False)
    add("goals_by_subs_total", me["sub_goals"], 1, "Substitutions", False)
    add("tactical_shifts_pg", pg("tactical_shifts"), 0, "Substitutions", False)
    return out


# ----------------------------------------------------------------------------
# 3. RUN
# ----------------------------------------------------------------------------
def main():
    CACHE_DIR.mkdir(exist_ok=True)
    matches = get_matches(COMPETITION_ID, SEASON_ID)
    print(f"{len(matches)} matches found. Processing (cached results are reused)...")

    results = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(get_match_result, r) for r in matches.itertuples()]
        for i, f in enumerate(as_completed(futs), 1):
            mid, res = f.result()
            results[mid] = res
            if i % 20 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} done")

    # ---- sum per team (for) and per opponent (against)
    sums = defaultdict(lambda: defaultdict(float))
    against = defaultdict(lambda: defaultdict(float))
    lineups = defaultdict(list)
    players = defaultdict(set)
    for r in matches.sort_values("date").itertuples():
        res = results[r.match_id]
        for t, o in ((r.home_team, r.away_team), (r.away_team, r.home_team)):
            for k, v in res["counters"][t].items():
                sums[t][k] += v
                against[o][k] += v
            lineups[t].append((res["meta"]["starters"].get(t, []),
                               res["meta"]["formations"].get(t)))
            players[t].update(res["meta"]["players"][t])

    # ---- derive metrics for every team
    rows, info = {}, {}
    for t in sums:
        me, opp = sums[t], against[t]
        row = {}
        for name, val, d, cat, mk_opp in derive(me, opp):
            row[name] = val
            info[name] = (cat, d)
        for name, val, d, cat, mk_opp in derive(opp, me):
            if mk_opp:
                row["against_" + name] = val
                info["against_" + name] = (cat + " (against)", -d)   # lower opp value = better for us
        # squad / lineup metrics
        starts = lineups[t]
        changes = [11 - len(set(a[0]) & set(b[0])) for a, b in zip(starts, starts[1:]) if a[0] and b[0]]
        forms = [f for _, f in starts if f]
        row["players_used"] = len(players[t])
        row["different_starters_used"] = len({p for s, _ in starts for p in s})
        row["avg_xi_changes_between_games"] = np.mean(changes) if changes else np.nan
        row["formations_used"] = len(set(forms))
        row["main_formation_pct"] = 100 * pd.Series(forms).value_counts(normalize=True).iloc[0] if forms else np.nan
        rows[t] = row
    for k in ("players_used", "different_starters_used", "avg_xi_changes_between_games",
              "formations_used", "main_formation_pct"):
        info[k] = ("Squad management", 0)

    df = pd.DataFrame(rows).T
    df.index.name = "team"
    df.to_csv("pl_2015_16_all_metrics.csv")

    # ---- ranks
    n_teams = len(df)
    value_rank = df.rank(ascending=False, method="min")            # 1 = highest value
    direction = pd.Series({k: v[1] for k, v in info.items()})
    category = pd.Series({k: v[0] for k, v in info.items()})
    quality = df.mul(direction, axis=1).where(direction != 0)
    quality_rank = quality.rank(ascending=False, method="min")     # 1 = best

    if TARGET not in df.index:
        raise SystemExit(f"{TARGET} not found. Teams: {sorted(df.index)}")

    L = pd.DataFrame({
        "category": category,
        "value": df.loc[TARGET],
        "league_avg": df.mean(),
        "rank_by_value": value_rank.loc[TARGET],
        "quality_rank": quality_rank.loc[TARGET],
        "direction": direction.map({1: "higher=better", -1: "lower=better", 0: "style/neutral"}),
    }).dropna(subset=["value"])
    L.index.name = "metric"
    L.to_csv("leicester_all_ranks.csv")
    value_rank.T.to_csv("pl_2015_16_ranks_by_value.csv")

    top = L[L["rank_by_value"] <= 3].sort_values(["category", "rank_by_value"])
    bottom = L[L["rank_by_value"] >= n_teams - 2].sort_values(["category", "rank_by_value"], ascending=[True, False])
    best = L[L["quality_rank"] <= 3].sort_values(["category", "quality_rank"])
    worst = L[L["quality_rank"] >= n_teams - 2].sort_values(["category", "quality_rank"], ascending=[True, False])
    pd.concat([top.assign(list="TOP 3 (highest value)"), bottom.assign(list="BOTTOM 3 (lowest value)")]).to_csv("leicester_top_bottom_3.csv")

    def fmt(frame, rank_col):
        lines, cur = [], None
        for m, r in frame.iterrows():
            if r["category"] != cur:
                cur = r["category"]
                lines.append(f"\n  [{cur}]")
            lines.append(f"    {m:<46} {r['value']:>9.2f}   (league avg {r['league_avg']:>8.2f})   "
                         f"#{int(r[rank_col])} of {n_teams}   {r['direction']}")
        return "\n".join(lines)

    report = [
        f"{TARGET.upper()} - PREMIER LEAGUE 2015/16  ({len(L)} metrics, {n_teams} teams)",
        "=" * 100,
        f"\nA) TOP 3 IN THE LEAGUE BY RAW VALUE ({len(top)} metrics)  [#1 = highest number]",
        fmt(top, "rank_by_value"),
        f"\n\nB) BOTTOM 3 IN THE LEAGUE BY RAW VALUE ({len(bottom)} metrics)  [#{n_teams} = lowest number]",
        fmt(bottom, "rank_by_value"),
        f"\n\nC) BEST 3 IN QUALITY TERMS ({len(best)} metrics) - direction-adjusted, #1 = best",
        fmt(best, "quality_rank"),
        f"\n\nD) WORST 3 IN QUALITY TERMS ({len(worst)} metrics) - direction-adjusted, #{n_teams} = worst",
        fmt(worst, "quality_rank"),
        "\n\nNotes: 'against_' = what opponents did against Leicester. 'style/neutral' metrics "
        "have no good/bad direction. 'Metres run' is not in open data; see metres_carried_pg.",
    ]
    text = "\n".join(report)
    Path("leicester_top_bottom_report.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    print("\nSaved: pl_2015_16_all_metrics.csv, leicester_all_ranks.csv, "
          "leicester_top_bottom_3.csv, pl_2015_16_ranks_by_value.csv, leicester_top_bottom_report.txt")


if __name__ == "__main__":
    main()

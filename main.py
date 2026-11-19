"""GTA VI trophy rarity forecaster (Python 3.13, standard library only).

Forecasts, for every Grand Theft Auto VI PS5 trophy, its earn rate on Exophase
at the case's target, as five quantiles (5/25/50/75/95%), scored by pinball
loss on the logit scale.

Model — earners over players, not rates over time
--------------------------------------------------
A trophy's earn rate is a ratio: earners / players-tracked. Both move, and
they move for different reasons, so the program models them separately and
divides at the target:

* players tracked N(t): the tracked player base grows after launch as more
  players sync profiles. log N is fitted against log(1 + hours since
  release) by least squares over the case's history.
* earners E_j(t) for trophy j: E = rate * N / 100, the implied headcount.
  log(E + 0.5) (a half-count continuity correction for the small launch-night
  counts) is fitted the same way, per trophy.

The median forecast is the middle value (in logit space) of three
candidates, so no single extrapolation can run away with it:

* persistence: the trophy's last rate (rates move slowly once the launch
  rush passes);
* a trend: logit(rate) fitted against log(1 + hours) over the last 24 hours
  of reads and read at the target, shrunk halfway back toward persistence
  (a launch-week rate curve saturates, so a raw extrapolation of early
  growth overshoots);
* the earners-over-players forecast: 100 * E*(target) / N*(target) from
  the two power-law fits above. This captures the dilution the raw rates
  hide: when the player base grows faster than a trophy's earners, the
  rate falls (the refresh the house saw on launch night: 638 -> 3,868
  players tracked, every rate but 100% down), and when earners catch up
  as players progress, the rate rises.

A flat history (practice cases) makes all three candidates the last rate,
so the median falls back to persistence.

Spread — honest about a refresh and about extrapolation
--------------------------------------------------------
The spread, in logit units, combines five terms per trophy:

* the residual scatter of that trophy's log-earners fit (how noisy its
  history is),
* the observed refresh jumps: the standard deviation of logit rate changes
  across reads where the players-tracked count moved (a refresh can move
  every rate at once, and the answer is the page as of the last refresh
  before the target),
* the disagreement between the three median candidates, halved and
  capped at 0.25 (when the models disagree, the honest band is wider,
  but the median-of-three already hedges, so it is not full risk),
* a random-walk term growing with the horizon (0.25 * sqrt(days)),
* a floor (0.30) so a perfectly flat history still gets a real band.

The fit-residual and refresh-jump terms are capped (0.4 and 0.6): a
misspecified power law inflates residuals beyond pure noise, and a
single huge refresh should not blow the band out for every trophy.

Quantiles are median +/- z_tau * sigma on the logit scale, clamped to
[0.01, 99.99] and rounded to 4 decimals.

Guards: fitted slopes are clamped (players 0..2.5, earners -0.5..4 in
log-log space), the forecast's logit move from the last read is clamped to
+/-3, and with fewer than 3 distinct reads the program falls back to
persistence (last rate) with the baseline-style spread. Deterministic:
pure least squares, no randomness, no network, standard library only.

Run locally: mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves this file's directory OFF sys.path; put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Spread terms, in logit units.
SIGMA_FLOOR = 0.30        # minimum spread, even on a perfectly flat history
SIGMA_RW_PER_SQRT_DAY = 0.25  # random-walk growth with the horizon
SIGMA_RESID_CAP = 0.4     # cap on the fit-residual term (a misspecified
                          # power law inflates residuals beyond pure noise)
SIGMA_JUMP_CAP = 0.6      # cap on the refresh-jump term
SIGMA_DISAGREE_CAP = 0.5  # cap on the candidate-disagreement term
SIGMA_DISAGREE_W = 0.5    # weight on disagreement (the median-of-three
                          # already hedges, so disagreement is not full risk)
MAX_LOGIT_MOVE = 3.0      # clamp on the forecast's move from the last read
B_N_MIN, B_N_MAX = 0.0, 2.5   # clamped log-log slope of players tracked
B_E_MIN, B_E_MAX = -0.5, 4.0  # clamped log-log slope of earners


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """Round to 4 decimals, half up, like JavaScript's Math.round(x*1e4)/1e4."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def least_squares(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Ordinary least squares y ~ a + b*x. Returns (intercept, slope)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return my - b * mx, b


def fit_log_log(hours: list[float], values: list[float]) -> tuple[float, float, float]:
    """Fit log(value) ~ a + b*log(1+h). Returns (a, b, residual std)."""
    xs = [math.log1p(h) for h in hours]
    ys = [math.log(v) for v in values]
    a, b = least_squares(xs, ys)
    if len(xs) >= 3:
        resid = [y - (a + b * x) for x, y in zip(xs, ys)]
        var = sum(r * r for r in resid) / (len(resid) - 2)
        std = math.sqrt(max(var, 0.0))
    else:
        std = 0.0
    return a, b, std


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")

    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    x_target = math.log1p((target - release).total_seconds() / 3600.0)

    trophies = case["trophies"]
    n_tr = len(trophies)

    # Per-read hours since release and players tracked.
    hours: list[float] = []
    players: list[float] = []
    rates: list[list[float | None]] = []
    for read in history:
        hours.append((parse_time(read["at"]) - release).total_seconds() / 3600.0)
        players.append(float(read["players_tracked"]))
        rates.append(list(read["rates"]))

    # Distinct reads (a read counts as distinct when N or any rate changed);
    # the launch-night history repeats the same page for hours, and repeats
    # would otherwise overweight one point in the fits.
    distinct_idx: list[int] = []
    prev_key = None
    for i, read in enumerate(history):
        key = (read["players_tracked"], tuple(read["rates"]))
        if key != prev_key:
            distinct_idx.append(i)
        prev_key = key
    n_distinct = len(distinct_idx)
    fit_ok = n_distinct >= 3

    # --- players-tracked fit ---
    a_n, b_n, _ = fit_log_log(
        [hours[i] for i in distinct_idx], [players[i] for i in distinct_idx]
    )
    b_n = min(B_N_MAX, max(B_N_MIN, b_n))
    n_star = math.exp(a_n + b_n * x_target) if fit_ok else players[-1]
    # Never forecast fewer players than the last read saw.
    n_star = max(n_star, players[-1])

    # --- refresh jumps: logit moves on reads where N changed ---
    jump_sq: list[float] = []
    for k in range(1, len(distinct_idx)):
        i, j = distinct_idx[k - 1], distinct_idx[k]
        if players[j] == players[i]:
            continue
        for t in range(n_tr):
            r0, r1 = rates[i][t], rates[j][t]
            if r0 is None or r1 is None:
                continue
            d = logit(r1) - logit(r0)
            jump_sq.append(d * d)
    jump_std = math.sqrt(sum(jump_sq) / len(jump_sq)) if jump_sq else 0.0
    jump_std = min(SIGMA_JUMP_CAP, jump_std)

    rw = SIGMA_RW_PER_SQRT_DAY * math.sqrt(horizon_days)
    # Distinct reads from the last 24 hours before the checkpoint, for the
    # local trend fit (recent slope, not the launch-night rush).
    at_hours = (at - release).total_seconds() / 3600.0
    recent_idx = [i for i in distinct_idx if hours[i] >= at_hours - 24.0]
    if len(recent_idx) < 3:
        recent_idx = distinct_idx
    rows = []
    for t, trophy in enumerate(trophies):
        # Last non-null rate for this trophy.
        last_rate = None
        for read_rates in reversed(rates):
            if read_rates[t] is not None:
                last_rate = read_rates[t]
                break
        if last_rate is None:
            last_rate = 0.0
        last_lg = logit(last_rate)

        # Candidate 1: persistence.
        c_persist = last_lg
        # Candidate 2: local logit trend over the last 24h, shrunk halfway
        # back toward persistence.
        c_trend = last_lg
        if fit_ok:
            pts_h = [hours[i] for i in recent_idx if rates[i][t] is not None]
            pts_lg = [logit(rates[i][t]) for i in recent_idx if rates[i][t] is not None]
            if len(pts_h) >= 3:
                a_lg, b_lg = least_squares([math.log1p(h) for h in pts_h], pts_lg)
                raw = a_lg + b_lg * x_target
                raw = min(last_lg + 2.5, max(last_lg - 2.5, raw))
                c_trend = last_lg + 0.5 * (raw - last_lg)
        # Candidate 3: earners over players.
        c_ratio = last_lg
        resid_std = 0.0
        if fit_ok:
            pts_h = []
            pts_e = []
            for i in distinct_idx:
                r = rates[i][t]
                if r is None or r <= 0:
                    continue
                pts_h.append(hours[i])
                pts_e.append(r * players[i] / 100.0 + 0.5)
            if len(pts_h) >= 3:
                a_e, b_e, resid_std = fit_log_log(pts_h, pts_e)
                b_e = min(B_E_MAX, max(B_E_MIN, b_e))
                e_star = math.exp(a_e + b_e * x_target)
                c_ratio = logit(100.0 * e_star / n_star)

        # Median: the middle candidate, so no one extrapolation runs away.
        candidates = sorted((c_persist, c_trend, c_ratio))
        med_lg = candidates[1]
        # Disagreement between the candidates, in logit units (halved and
        # capped: the median-of-three already hedges, so disagreement is
        # not full predictive risk).
        mean_c = sum(candidates) / 3.0
        disagree = SIGMA_DISAGREE_W * min(
            SIGMA_DISAGREE_CAP,
            math.sqrt(sum((c - mean_c) ** 2 for c in candidates) / 3.0),
        )
        sigma = math.sqrt(
            min(resid_std, SIGMA_RESID_CAP) ** 2
            + jump_std**2
            + disagree**2
            + rw**2
            + SIGMA_FLOOR**2
        )

        # Guard: never move the median more than MAX_LOGIT_MOVE from the
        # last read, however steep the fitted trend is.
        med_lg = min(last_lg + MAX_LOGIT_MOVE, max(last_lg - MAX_LOGIT_MOVE, med_lg))

        qs = []
        prev = -1.0
        for z in Z:
            q = round4(clamp_pct(inv_logit(med_lg + z * sigma)))
            q = max(q, prev)  # keep the row non-decreasing after rounding
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})

    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()

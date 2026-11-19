# GTA VI trophy rarity forecast — make.fun bounty entry

Entry for [Forecast how many GTA VI players earn each trophy in week one](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) (`gta6-trophy-rarity-forecast-01`, $250, deadline 21 Nov 2026 11:30 UTC).

`main.py` (Python 3.13, standard library only) reads one case (`/input/<id>.in`)
and writes `/output/<id>.out`: for each of the game's 55 PS5 trophies, five
quantiles (5/25/50/75/95%) of its earn rate on Exophase at the case's target.
It runs offline, deterministically, in well under the 8-second cap (~0.07 s
on the practice cases).

## Model

A trophy's earn rate is a ratio — earners / players tracked — and the two
sides move for different reasons, so they are modelled separately:

- **Players tracked N(t):** log N fitted against log(1 + hours since
  release) by least squares over the case's history.
- **Earners E(t)** per trophy (E = rate × N / 100, with a half-count
  correction for the small launch-night counts): log(E + 0.5) fitted the
  same way.

The median is the middle value (in logit space) of three candidates, so no
single extrapolation can run away with it:

1. **Persistence** — the trophy's last rate.
2. **Local trend** — logit(rate) fitted against log(1 + hours) over the
   last 24 hours of reads and read at the target, shrunk halfway back
   toward persistence (a launch-week rate curve saturates, so raw
   extrapolation of early growth overshoots).
3. **Earners over players** — 100 × E\*(target) / N\*(target). This
   captures the dilution the raw rates hide: when the player base grows
   faster than a trophy's earners, the rate falls (launch night's refresh:
   638 → 3,868 players tracked, every rate but 100% down); when earners
   catch up as players progress, the rate rises.

A flat history (the practice cases) makes all three candidates the last
rate, so the median falls back to persistence.

## Spread

In logit units, combining the residual scatter of the earners fit (capped),
the observed refresh jumps (logit moves on reads where the player count
changed, capped), the disagreement between the three median candidates
(halved, capped), a random-walk term (0.25 × √days), and a floor (0.30).

## Validation

- Practice cases `p00`/`p01`: valid outputs (55 rows, 5 non-decreasing
  quantiles each, within [0.01, 99.99], under the 65,536-byte cap),
  byte-identical across repeated runs.
- Synthetic end-to-end test: four simulated launch-week worlds (different
  player-growth and trophy-progression regimes) with a final-like case cut
  at the deadline and scored by the contest's pinball loss. Mean loss
  **0.214** vs **0.335** for the house baseline (starter), beating it in
  every world; typical miss 2.3 pp vs 6.2 pp, 90% coverage 0.98–1.00.
  (Synthetic worlds are smooth by construction, so these are optimistic;
  the margins above the baseline are the point, not the absolute numbers.)

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

with `input/` holding the kit's practice cases.

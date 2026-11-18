# Battery or reinforcement on a medium-voltage feeder (v0.2: N-1 security and reliability)

Should a wind-heavy rural feeder get a battery, a parallel cable or nothing, when weather, prices, renewable
growth and load growth are all uncertain, and which uncertainty is worth resolving first? A two-stage planning
study on an open benchmark feeder with illustrative costs. Not a network plan and not an investment case.

## Setup

- **Feeder:** SimBench `1-MV-rural--0-no_sw`, 12 buses, 12.7 km, 3.7 MW wind, 1.6 MW PV, 0.8 MW peak load.
- **Operation:** linearized branch flow (LinDistFlow) in a daily LP on hourly ERA5 capacity factors and SMARD
  day-ahead prices, solved with HiGHS. Voltage band 0.965 to 1.055 pu, substation setpoint 1.02 pu.
- **Plans (36):** nothing, batteries of 2, 4 and 8 MWh (two-hour) at five sites, parallel cable on the shared
  trunk (5.0 km) or on all wind lines (5.4 km), and combinations.
- **Scenarios (81):** weather years 2010, 2024 and 1995 (tercile midpoints of 35 ERA5 years), price years 2023 to
  2025, today's wind and PV times 1, 2 or 3 and today's load times 1, 1.3 or 1.6. Growth levels have weights
  25%, 50% and 25%. These weights are assumptions, and information values depend on them.
- **Costs:** battery 300 EUR/kWh, cable 120,000 EUR/km, curtailed energy 80 EUR/MWh, wear 5 EUR/MWh. All are
  illustrative planning values in `src/feederrisk/config.py`. Annual cost means annualized capital cost plus
  curtailment and wear, minus arbitrage income.

## Results

The base case runs on all 365 days of every scenario (`scripts/run_full_year.py`). Sensitivities on setpoint and
curtailment value use representative days (`scripts/run_study.py`) and are marked below.

### Which plan

| Plan | Annual capital, kEUR | Expected annual cost, kEUR | CVaR 90%, kEUR | Worst case, kEUR | Arbitrage, kEUR | Curtailed, MWh |
|---|---|---|---|---|---|---|
| Cable on the trunk | 34.6 | **34.9** | 36.5 | 37.1 | 0 | 4 |
| Cable on all wind lines | 37.4 | 37.4 | 37.5 | 37.6 | 0 | 0 |
| 2 MWh at near wind bus + trunk cable | 109.5 | 43.7 | 55.2 | 56.4 | 71.7 | 2 |
| Nothing | 0 | 56.1 | 232.7 | 248.9 | 0 | 702 |
| 2 MWh at far wind bus | 74.9 | 62.5 | 235.6 | 261.0 | 69.5 | 639 |
| 8 MWh upstream of wind + trunk cable | 334.1 | 70.0 | 114.2 | 115.6 | 287.4 | 4 |
| 8 MWh upstream of wind | 299.5 | 91.0 | 276.2 | 326.4 | 287.2 | 696 |
| 8 MWh at far wind bus | 299.5 | 116.4 | 300.8 | 346.2 | 253.4 | 614 |

- **The cable wins at the base costs,** on expected cost, on CVaR and on maximum regret alike, so risk aversion
  changes nothing here.
- **The battery wins only when it pays for itself as a market asset.** The best plan contains a battery up to
  265 EUR/kWh on a 5 EUR grid (270 when the cable costs 240,000 EUR/km). A battery removes little curtailment: at
  the far wind bus 2 MWh removes 9% and 8 MWh removes 12%.
- **The site depends on size.** At 2 MWh the wind bus is better than upstream by 2.3 kEUR per year, because some
  curtailment is relieved. At 8 MWh it is worse by 25.5 kEUR, because voltage limits at that bus cut 34 kEUR of
  arbitrage. A model without network limits cannot see this.
- **The value of curtailed energy decides between cable and nothing.** From the table, the trunk cable pays above
  about 50 EUR/MWh. Representative-day runs agree in direction: nothing is best at 40, the cable at 120.

### Which uncertainty is worth resolving

| Source | EVPI, kEUR per year |
|---|---|
| Renewable growth | 14.2 |
| Prices | 1.2 |
| Weather year | 0.6 |
| Load growth | 0 |
| All together | 15.6 |

Planning on the central scenario chooses nothing, because the cable pays only at the high growth level. The cost
of ignoring uncertainty (VSS) is 21.2 kEUR per year. A study of renewable growth is worth nothing up to 60%
accuracy, then 1.6, 5.7 and 10.0 kEUR at 70%, 80% and 90%.

The ranking is not fixed. Close to the battery's break-even, at 250 EUR/kWh, the EVPI of prices rises to 9.5 kEUR,
and at 350 EUR/kWh it is zero. Prices matter only where they can change the decision.

### A setpoint may beat both (representative days)

Voltage, not current, is the limit that binds, so the substation setpoint moves the whole problem.

| Setpoint, pu | Best plan | Expected annual cost, kEUR | Curtailed with no action, MWh | EVPI of growth, kEUR |
|---|---|---|---|---|
| 1.00 | Nothing | 7.8 | 97 | 0 |
| 1.01 | Nothing | 20.9 | 261 | 11.9 |
| 1.02 (base) | Trunk cable | 34.9 | 671 | 14.3 |
| 1.03 | Trunk cable | 49.0 | 1,524 | 9.8 |

At 1.00 pu the feeder needs no investment and the growth question stops mattering. This is the cheapest option in
the model, but the model charges nothing for it and holds no other feeder on the same transformer, whose customers
may need the higher voltage. It shows what to check first, not that a lower setpoint is available.

## v0.2: N-1 security, restoration and reliability in the plan choice

A feeder plan is also judged by what happens after a fault. v0.2 adds the N-1 criterion and an expected reliability
cost to the plan choice (`src/feederrisk/postfault.py`, `python scripts/run_n1.py`, results in `results/n1_*.csv` and
`results/summary_n1.json`).

**Model.** After the outage of one line the feeder is a tree again: a line with a parallel circuit keeps one cable in
service (no interruption), a normally open tie between the far ends of the two branches (3 km, 21.0 kEUR per year,
the open point of a ring) is closed after manual switching (1 hour), and everything else waits for the repair (8
hours). For every line, hour of the year and scenario, voltage band and thermal limits of the post-fault tree are
checked with linearized flow, allowing the renewable generation to be curtailed in four steps; hours with no feasible
step are not restored. The cost of a fault is energy not supplied at 10 EUR/kWh plus lost and curtailed renewable
energy at 80 EUR/MWh, weighted by 0.05 faults per km and year and averaged over the start hour. Fault rate, repair and
switching times, tie length and value of lost load are illustrative assumptions in `postfault.Reliability`.
Batteries are grid-following and do not island, so they carry no reliability value.

| Plan | Annual capital, kEUR | Reliability cost, kEUR | Expected annual cost, kEUR | Energy not supplied, MWh/y |
|---|---|---|---|---|
| Cable on the trunk | 34.6 | 6.9 | **41.9** | 0.69 |
| Cable on all wind lines | 37.4 | 6.7 | 44.1 | 0.66 |
| Cable on the trunk + tie | 55.6 | 1.0 | 56.9 | 0.09 |
| Nothing | 0 | 11.1 | 67.2 | 1.08 |
| Tie only | 21.0 | 2.9 | 80.0 | 0.27 |
| Cable on every line (12.7 km) | 88.9 | 0.0 | 88.9 | 0 |

- **The trunk cable is also the N-1 plan.** It protects the four trunk lines (0.75 km plus the 4.2 km wind line) in all hours at any load, which
  removes 4.2 of the 11.1 kEUR reliability cost of doing nothing. The ranking of plans does not change; the lead over
  nothing grows from 21.2 to 25.4 kEUR per year, and the value of learning the renewable growth falls from 14.2 to
  11.1 kEUR per year, because the cable now pays under every growth level.
- **Full N-1 is expensive here.** The only plan in which every line fault is restored without interruption is a parallel
  cable on every line: 47 kEUR per year more than the best plan (88.9 against 41.9). The tie restores every branch
  fault that has a loop through it (the long lines 3 and 4 carry 0.46 of the 0.64 faults per year) but not a fault on
  the 0.75 km head, where no other source exists, and not on a leaf line.
- **The tie pays only at a high value of lost load.** With the trunk cable in place, adding the tie wins above 36
  EUR/kWh at 0.05 faults per km and year, above 30 EUR/kWh at twice that rate and above 10 EUR/kWh at twice the rate
  and half the tie cost (`results/n1_sensitivity.csv`). At 100 EUR/kWh the tie is in the best plan in every case.
- **AC check of the post-fault states** (pandapower, 70 combinations of plan structure, line and scenario, 36,120
  hours, `results/n1_ac_check.csv`): the linear verdict (restored or not, at the curtailment step it chose) agrees
  with AC power flow in every hour, with no voltage violation and no line above 87% loading. The linearized voltage is
  off by up to 0.001 pu on the feeder as built and by up to 0.023 pu on the reverse-fed branch through the tie, where
  the path is 7 to 8 km long and losses matter; the verdict is not sensitive to that error at these loads.

## Checks

- **Representative days against all 365 days.** Same best plan, but they understate arbitrage by about 15% and the
  cost of no action by 4% on average and up to 18% in a single scenario. The break-even battery cost moved from
  between 200 and 250 EUR/kWh to 265, which is why the base case was rerun. The setpoint and curtailment
  sensitivities keep this bias. Their best plans are cable or nothing, and a correction of this size does not change
  them, but their costs are approximate.
- **AC power flow (pandapower) on the LP dispatch.** At 1.02 pu, on the representative days of three scenarios and
  four plans: no voltage or thermal violations, and voltage error at most 0.0011 pu. At 1.00 pu, every third day:
  none for nothing or the cable. An 8 MWh battery at the far wind bus falls below the band in 4% to 7% of hours, by
  at most 0.0009 pu, because the LP neglects losses.
- **An earlier version** took only representative days without the windiest ones and understated curtailment by up
  to 64%. The 60 windiest days of each year are now solved one by one in the representative-day runs.

## Limits of the study

One synthetic feeder and illustrative costs, so the numbers do not transfer and the logic is what carries over.
Three weather and three price years. Each day starts and ends the battery half full and knows that day's prices,
wind and load, so arbitrage is an upper bound with no forecast error. Price-taker, day-ahead only, no balancing
markets. Losses are neglected and reactive flows fixed in the LP. Inverter reactive power, controllable
transformers and flexible connections are not among the plans. Reliability uses assumed fault rates, repair times and value of lost load, no outage data, and no islanding by batteries. Plans are built once, with no staging over time, so
the option to wait, which growth uncertainty makes valuable, is not priced.

## Run

`pip install -e ".[grid,test]"`, `pytest`, then `python scripts/run_n1.py` for v0.2 (needs the base-case table; about 25 minutes for the one new plan on six cores), and `python scripts/run_study.py` (sensitivities and checks, about 3
hours on six cores) and `python scripts/run_full_year.py` (base case, about 1 h 45 min, compares against the
table of the first script). `data/inputs.npz` holds the prepared inputs; `scripts/prepare_data.py` rebuilds them
from SimBench and local clones of [de-power-adequacy](https://github.com/serikov-energy-systems/de-power-adequacy) and
[de-power-market-ppa](https://github.com/serikov-power-markets/de-power-market-ppa).

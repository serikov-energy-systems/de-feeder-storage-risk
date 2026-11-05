# Battery or reinforcement on a medium-voltage feeder

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
transformers and flexible connections are not among the plans. Plans are built once, with no staging over time, so
the option to wait, which growth uncertainty makes valuable, is not priced.

## Run

`pip install -e ".[grid,test]"`, `pytest`, then `python scripts/run_study.py` (sensitivities and checks, about 3
hours on six cores) and `python scripts/run_full_year.py` (base case, about 1 h 45 min, compares against the
table of the first script). `data/inputs.npz` holds the prepared inputs; `scripts/prepare_data.py` rebuilds them
from SimBench and local clones of [de-power-adequacy](https://github.com/michaelserikov/de-power-adequacy) and
[de-power-market-ppa](https://github.com/michaelserikov/de-power-market-ppa).

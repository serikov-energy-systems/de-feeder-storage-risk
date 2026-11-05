"""Build data/inputs.npz from SimBench, ERA5 capacity factors and SMARD prices.

ERA5 capacity factors come from https://github.com/michaelserikov/de-power-adequacy and SMARD
day-ahead prices from https://github.com/michaelserikov/de-power-market-ppa, as local sibling clones.
The study itself reads only data/inputs.npz.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from feederrisk.config import FEEDER_CODE, FEEDER_N_BUSES, PRICE_YEARS
from feederrisk.network import load_simbench_feeder

ROOT = Path(__file__).resolve().parents[1]
SIB = ROOT.parent


def no_leap(values: np.ndarray, index: pd.DatetimeIndex) -> np.ndarray:
    mask = ~((index.month == 2) & (index.day == 29))
    return values[mask]


def main() -> None:
    feeder, load_h, _ = load_simbench_feeder(FEEDER_CODE, FEEDER_N_BUSES)

    base = pd.read_parquet(SIB / "de-power-adequacy" / "data" / "base.parquet")
    annual = base.groupby("year")["cf_onshore"].mean().sort_values()
    # Tercile midpoints of 35 ERA5 years ranked by mean onshore capacity factor.
    picks = [int(annual.index[i]) for i in (5, 17, 29)]
    wind, solar = [], []
    for y in picks:
        part = base[base["year"] == y]
        wind.append(no_leap(part["cf_onshore"].to_numpy(), part.index)[:8760])
        solar.append(no_leap(part["cf_solar"].to_numpy(), part.index)[:8760])

    px = pd.read_parquet(SIB / "de-power-market-ppa" / "data" / "hourly_raw.parquet")
    px["ts"] = pd.to_datetime(px["ts_utc"], utc=True)
    prices = []
    for y in PRICE_YEARS:
        part = px[px["ts"].dt.year == y].set_index("ts")["price"].interpolate()
        prices.append(no_leap(part.to_numpy(), part.index)[:8760])

    np.savez_compressed(
        ROOT / "data" / "inputs.npz",
        parent=feeder.parent, r_ohm=feeder.r_ohm, x_ohm=feeder.x_ohm, s_max_mva=feeder.s_max_mva,
        length_km=feeder.length_km, load_p_mw=feeder.load_p_mw, load_q_share=feeder.load_q_share,
        wind_mw=feeder.wind_mw, pv_mw=feeder.pv_mw, flat_mw=feeder.flat_mw, bus_id=feeder.bus_id,
        load_h=load_h, wind_cf=np.array(wind), solar_cf=np.array(solar), price=np.array(prices),
        weather_years=np.array(picks), weather_annual_cf=annual.loc[picks].to_numpy(),
        all_years_cf=annual.to_numpy(),
    )
    print("weather years", picks, annual.loc[picks].round(4).tolist())
    print("price means", [round(float(p.mean()), 1) for p in prices])


if __name__ == "__main__":
    main()

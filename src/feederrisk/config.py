"""Every assumption of the study in one place. Costs are illustrative planning values."""

from dataclasses import dataclass


def annuity(rate: float, years: int) -> float:
    return rate / (1 - (1 + rate) ** -years)


@dataclass(frozen=True)
class Assumptions:
    v_nominal_kv: float = 20.0
    v_slack_pu: float = 1.02
    v_min_pu: float = 0.965  # SimBench medium-voltage band
    v_max_pu: float = 1.055
    eta: float = 0.92  # one way, about 85% round trip
    duration_h: float = 2.0
    wear_eur_mwh: float = 5.0  # per MWh discharged
    curtail_eur_mwh: float = 80.0  # compensation paid for curtailed renewable energy
    voll_eur_mwh: float = 10_000.0
    battery_capex_eur_kwh: float = 300.0
    battery_life_y: int = 15
    battery_rate: float = 0.07
    battery_om_share: float = 0.015
    cable_capex_eur_km: float = 120_000.0
    cable_life_y: int = 40
    cable_rate: float = 0.05

    def battery_annual_eur_per_mwh(self) -> float:
        capex = self.battery_capex_eur_kwh * 1000
        return capex * (annuity(self.battery_rate, self.battery_life_y) + self.battery_om_share)

    def cable_annual_eur_per_km(self) -> float:
        return self.cable_capex_eur_km * annuity(self.cable_rate, self.cable_life_y)


DER_LEVELS = {"low": 1.0, "mid": 2.0, "high": 3.0}  # multiple of today's wind and PV on the feeder
LOAD_LEVELS = {"low": 1.0, "mid": 1.3, "high": 1.6}  # multiple of today's load, same shape
LEVEL_PROB = {"low": 0.25, "mid": 0.5, "high": 0.25}
PRICE_YEARS = (2023, 2024, 2025)
BATTERY_SIZES_MWH = (2.0, 4.0, 8.0)
FEEDER_CODE = "1-MV-rural--0-no_sw"
FEEDER_N_BUSES = 12

"""
simulator.py
------------
Digital Twin of a Caterpillar diesel engine running in a test cell.
Models the physics of airflow, turbocharging, intercooling, and exhaust.
Outputs realistic sensor readings every 100ms, plus leak injection.
"""

import numpy as np
import pandas as pd
import time
from dataclasses import dataclass, field
from typing import Optional, Dict


# ─── Engine Constants ──────────────────────────────────────────────────────────
DISPLACEMENT_L = 7.2          # Engine displacement in litres (e.g. CAT C7 equivalent)
NUM_CYLINDERS  = 6
COMPRESSION_RATIO = 17.5
R_AIR = 287.05                # J/(kg·K) — specific gas constant for air
GAMMA = 1.4                   # Ratio of specific heats

# ─── Volumetric Efficiency Table  [RPM → VE fraction] ─────────────────────────
VE_TABLE_RPM = np.array([600, 800, 1000, 1200, 1400, 1600, 1800, 2000,
                          2200, 2400, 2600, 2800, 3000])
VE_TABLE_VE  = np.array([0.70, 0.74, 0.78, 0.82, 0.85, 0.87, 0.89, 0.91,
                          0.90, 0.88, 0.85, 0.81, 0.76])

# ─── Turbocharger Compressor Map  [RPM → pressure ratio] ──────────────────────
TURBO_RPM_BREAKPOINTS  = np.array([600, 900, 1200, 1600, 2000, 2400, 2800, 3000])
TURBO_PRESSURE_RATIO   = np.array([1.05, 1.10, 1.25, 1.55, 1.85, 2.10, 2.30, 2.35])
TURBO_COMP_TEMP_RISE_K = np.array([5,   10,   20,   35,   52,   68,   80,   85])

# ─── Intercooler Model ─────────────────────────────────────────────────────────
INTERCOOLER_EFFICIENCY = 0.72   # fraction of temperature rise removed

# ─── Exhaust Back-Pressure Regression  EBP = a*fuel + b*RPM + c ───────────────
EBP_COEFF_FUEL = 0.18   # kPa per g/s fuel
EBP_COEFF_RPM  = 0.004  # kPa per RPM
EBP_INTERCEPT  = 2.5    # kPa

# ─── EGT pair baseline temps ──────────────────────────────────────────────────
EGT_BASE_K = 700.0      # Kelvin baseline at neutral load
EGT_PER_RPM = 0.05      # K per RPM increase

# ─── Noise levels (1-sigma, in sensor units) ──────────────────────────────────
NOISE_MAF  = 0.3    # g/s
NOISE_MAP  = 0.4    # kPa (absolute)
NOISE_IAT  = 0.2    # °C
NOISE_EBP  = 0.15   # kPa
NOISE_EGT  = 1.5    # °C
NOISE_BOOST_TEMP = 0.4  # °C


@dataclass
class LeakConfig:
    zone_a_severity: float = 0.0   # fraction 0..1, MAF reduction
    zone_b_severity: float = 0.0   # fraction 0..1, MAP reduction
    zone_c_severity: float = 0.0   # fraction 0..1, EBP increase (leak raises backpressure)
    zone_b_location: str   = "after_intercooler"   # "before_intercooler" | "after_intercooler"
    zone_c_bank: str       = "upstream"             # "upstream" | "downstream"


@dataclass
class EngineState:
    rpm: float = 2000.0
    load_pct: float = 60.0           # 0-100 %
    coolant_temp_c: float = 88.0
    ambient_temp_c: float = 25.0
    ambient_pressure_kpa: float = 101.325
    dpf_regen_active: bool = False
    egr_position_pct: float = 15.0   # 0-100 %
    transient: bool = False

    # computed sensor readings (filled by simulator)
    maf_actual: float = 0.0
    map_actual: float = 0.0
    iat_c: float = 0.0
    boost_temp_c: float = 0.0
    intercooler_outlet_temp_c: float = 0.0
    ebp_actual: float = 0.0
    egt_1: float = 0.0
    egt_2: float = 0.0
    fuel_rate_gs: float = 0.0

    # ECU flags for edge-case suppression
    timestamp: float = 0.0


def _air_density(temp_c: float, pressure_kpa: float) -> float:
    """ρ = P / (R·T)  [kg/m³]"""
    T = temp_c + 273.15
    P = pressure_kpa * 1000.0
    return P / (R_AIR * T)


def _ve_lookup(rpm: float) -> float:
    return float(np.interp(rpm, VE_TABLE_RPM, VE_TABLE_VE))


def _turbo_pressure_ratio(rpm: float) -> float:
    return float(np.interp(rpm, TURBO_RPM_BREAKPOINTS, TURBO_PRESSURE_RATIO))


def _turbo_comp_temp_rise(rpm: float) -> float:
    return float(np.interp(rpm, TURBO_RPM_BREAKPOINTS, TURBO_COMP_TEMP_RISE_K))


def _fuel_rate(rpm: float, load_pct: float) -> float:
    """Simple brake-specific fuel model. g/s."""
    power_fraction = load_pct / 100.0
    max_fuel_gs = 12.5                  # max at 3000 RPM full load
    return max_fuel_gs * (rpm / 3000.0) * power_fraction


class EngineSimulator:
    """
    Stateful diesel engine digital twin.
    Call step() every 100 ms to get one row of sensor readings.
    Use inject_leak() / clear_leak() to simulate air path faults.
    """

    def __init__(self, initial_rpm: float = 2000.0, initial_load: float = 60.0):
        self.state = EngineState(rpm=initial_rpm, load_pct=initial_load)
        self.leak = LeakConfig()
        self._t = 0.0
        self._rng = np.random.default_rng(seed=42)

    # ── Leak Controls ──────────────────────────────────────────────────────────

    def inject_leak(self, zone: str, severity: float = 0.15,
                    b_location: str = "after_intercooler",
                    c_bank: str = "upstream"):
        """
        zone: 'A', 'B', or 'C'
        severity: 0.0 (none) → 1.0 (total blockage)
        """
        severity = float(np.clip(severity, 0.0, 1.0))
        if zone.upper() == "A":
            self.leak.zone_a_severity = severity
        elif zone.upper() == "B":
            self.leak.zone_b_severity = severity
            self.leak.zone_b_location = b_location
        elif zone.upper() == "C":
            self.leak.zone_c_severity = severity
            self.leak.zone_c_bank = c_bank

    def clear_leak(self, zone: Optional[str] = None):
        """Clear one zone or all zones."""
        if zone is None:
            self.leak = LeakConfig()
        elif zone.upper() == "A":
            self.leak.zone_a_severity = 0.0
        elif zone.upper() == "B":
            self.leak.zone_b_severity = 0.0
        elif zone.upper() == "C":
            self.leak.zone_c_severity = 0.0

    def set_operating_point(self, rpm: float, load_pct: float):
        self.state.rpm = float(np.clip(rpm, 600, 3000))
        self.state.load_pct = float(np.clip(load_pct, 0, 100))

    def set_engine_flags(self, dpf_regen: bool = False,
                         egr_pct: float = 15.0,
                         transient: bool = False,
                         coolant_temp_c: float = 88.0):
        self.state.dpf_regen_active = dpf_regen
        self.state.egr_position_pct = egr_pct
        self.state.transient = transient
        self.state.coolant_temp_c = coolant_temp_c

    # ── Physics Calculations ──────────────────────────────────────────────────

    def _compute_physics(self) -> EngineState:
        s = self.state
        rng = self._rng
        rpm = s.rpm
        load = s.load_pct

        # 1. Fuel rate
        fuel_gs = _fuel_rate(rpm, load) + rng.normal(0, 0.05)

        # 2. Turbocharger
        pr = _turbo_pressure_ratio(rpm)
        comp_temp_rise = _turbo_comp_temp_rise(rpm)

        # MAP — turbocharger raises inlet pressure
        map_abs_kpa = s.ambient_pressure_kpa * pr
        boost_temp_c = s.ambient_temp_c + comp_temp_rise

        # 3. Intercooler
        temp_drop = comp_temp_rise * INTERCOOLER_EFFICIENCY
        intercooler_outlet_c = boost_temp_c - temp_drop

        # 4. Air density at intercooler outlet
        rho = _air_density(intercooler_outlet_c, map_abs_kpa)

        # 5. MAF — volumetric flow × air density
        ve = _ve_lookup(rpm)
        vol_flow_m3s = (DISPLACEMENT_L / 1000.0) * (rpm / 60.0) / 2.0  # 4-stroke
        vol_flow_m3s *= ve
        maf_gs = vol_flow_m3s * rho * 1000.0   # kg/s → g/s (×1000)

        # 6. EBP — exhaust back-pressure
        ebp_kpa = (EBP_COEFF_FUEL * fuel_gs
                   + EBP_COEFF_RPM * rpm
                   + EBP_INTERCEPT)
        if s.dpf_regen_active:
            ebp_kpa *= 1.35          # DPF regen raises backpressure naturally

        # 7. EGT pair
        egt_base = EGT_BASE_K + EGT_PER_RPM * (rpm - 1000) + (load / 100.0) * 150
        egt_1 = egt_base - 273.15 + rng.normal(0, NOISE_EGT)
        egt_2 = egt_base - 273.15 + rng.normal(0, NOISE_EGT)

        # 8. Intake Air Temperature
        iat_c = intercooler_outlet_c + rng.normal(0, NOISE_IAT)

        # ── Apply Leak Effects ─────────────────────────────────────────────────

        # Zone A — air path before turbo or intake restriction → MAF drops
        if self.leak.zone_a_severity > 0:
            maf_gs *= (1.0 - self.leak.zone_a_severity)

        # Zone B — charge air cooler or hose failure → MAP drops
        if self.leak.zone_b_severity > 0:
            map_abs_kpa *= (1.0 - self.leak.zone_b_severity)
            if self.leak.zone_b_location == "before_intercooler":
                # Leak is between turbo compressor outlet and intercooler inlet.
                # Boost temperature at the turbo outlet is UNCHANGED (leak is downstream of turbo).
                # However, the intercooler sees reduced flow, so it removes relatively
                # more heat per unit mass → intercooler outlet temp drops noticeably.
                # The boost_temp sensor (turbo outlet side) reads normally,
                # but the intercooler outlet temperature is lower → delta SHRINKS.
                temp_drop_factor = self.leak.zone_b_severity * 0.6
                intercooler_outlet_c -= comp_temp_rise * temp_drop_factor
            else:  # after_intercooler
                # Leak is between intercooler outlet and intake manifold.
                # Intercooler operates normally → boost_temp and ic_outlet behave normally.
                # Only a very slight pressure-driven temperature change at outlet.
                intercooler_outlet_c -= 2.0 * self.leak.zone_b_severity  # slight temp delta

        # Zone C — exhaust valve or manifold leak → EBP drops
        if self.leak.zone_c_severity > 0:
            ebp_kpa *= (1.0 - 0.6 * self.leak.zone_c_severity)
            if self.leak.zone_c_bank == "upstream":
                egt_1 -= 40 * self.leak.zone_c_severity  # upstream cylinder bank cools
            else:
                egt_2 -= 40 * self.leak.zone_c_severity  # downstream bank cools

        # ── Add Sensor Noise ──────────────────────────────────────────────────
        maf_gs        += rng.normal(0, NOISE_MAF)
        map_abs_kpa   += rng.normal(0, NOISE_MAP)
        boost_temp_c  += rng.normal(0, NOISE_BOOST_TEMP)
        ebp_kpa       += rng.normal(0, NOISE_EBP)

        # ── Build updated state ───────────────────────────────────────────────
        s.maf_actual               = round(max(0.0, maf_gs), 3)
        s.map_actual               = round(max(50.0, map_abs_kpa), 3)
        s.boost_temp_c             = round(boost_temp_c, 2)
        s.intercooler_outlet_temp_c= round(intercooler_outlet_c, 2)
        s.iat_c                    = round(iat_c, 2)
        s.ebp_actual               = round(max(0.0, ebp_kpa), 3)
        s.egt_1                    = round(egt_1, 2)
        s.egt_2                    = round(egt_2, 2)
        s.fuel_rate_gs             = round(max(0.0, fuel_gs), 3)
        s.timestamp                = round(self._t, 3)

        return s

    # ── Public Step ───────────────────────────────────────────────────────────

    def step(self) -> dict:
        """Advance simulation by one timestep (100 ms). Returns a sensor dict."""
        state = self._compute_physics()
        self._t += 0.1
        return {
            "timestamp":               state.timestamp,
            "rpm":                     state.rpm,
            "load_pct":                state.load_pct,
            "maf_gs":                  state.maf_actual,
            "map_kpa":                 state.map_actual,
            "iat_c":                   state.iat_c,
            "boost_temp_c":            state.boost_temp_c,
            "intercooler_outlet_c":    state.intercooler_outlet_temp_c,
            "ebp_kpa":                 state.ebp_actual,
            "egt_1_c":                 state.egt_1,
            "egt_2_c":                 state.egt_2,
            "fuel_rate_gs":            state.fuel_rate_gs,
            "coolant_temp_c":          state.coolant_temp_c,
            "dpf_regen":               int(state.dpf_regen_active),
            "egr_pct":                 state.egr_position_pct,
            "transient":               int(state.transient),
            # leak ground truth (for training / evaluation, hidden in real deploy)
            "leak_zone_a":             round(self.leak.zone_a_severity, 3),
            "leak_zone_b":             round(self.leak.zone_b_severity, 3),
            "leak_zone_c":             round(self.leak.zone_c_severity, 3),
        }

    def run_batch(self, duration_s: float = 60.0, dt: float = 0.1,
                  realtime: bool = False) -> pd.DataFrame:
        """
        Run the simulator for 'duration_s' seconds.
        Returns a DataFrame of sensor readings.
        """
        rows = []
        steps = int(duration_s / dt)
        for _ in range(steps):
            rows.append(self.step())
            if realtime:
                time.sleep(dt)
        return pd.DataFrame(rows)


# ─── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    sim = EngineSimulator(initial_rpm=2000, initial_load=60)

    print("=== Healthy Mode (10 s) ===")
    df_healthy = sim.run_batch(duration_s=10.0)
    print(df_healthy[["timestamp", "maf_gs", "map_kpa", "ebp_kpa"]].tail(5))

    print("\n=== Injecting Zone B leak (severity=0.20) ===")
    sim.inject_leak("B", severity=0.20, b_location="after_intercooler")
    df_leak = sim.run_batch(duration_s=5.0)
    print(df_leak[["timestamp", "maf_gs", "map_kpa", "ebp_kpa"]].tail(5))

    print("\n=== Clearing leaks ===")
    sim.clear_leak()
    df_clear = sim.run_batch(duration_s=5.0)
    print(df_clear[["timestamp", "maf_gs", "map_kpa", "ebp_kpa"]].tail(3))

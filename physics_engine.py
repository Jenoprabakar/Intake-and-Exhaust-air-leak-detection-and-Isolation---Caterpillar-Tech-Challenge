"""
physics_engine.py
-----------------
Physics-based leak detection for three zones.

Zone A — Air intake (before turbocharger)
    Expected MAF from VE table + air density.
    If actual MAF << expected → Zone A flag.

Zone B — Charge-air path (between turbo and intake manifold)
    Expected MAP from compressor map.
    If actual MAP << expected → Zone B flag.
    Sub-location from boost_temp vs intercooler_outlet_temp delta.

Zone C — Exhaust path
    Expected EBP from regression (fuel rate + RPM).
    If actual EBP << expected → Zone C flag.
    Sub-location from EGT pair comparison.

Each module also runs:
  - Drift detection  : slow residual growth → warning, not alert
  - Edge-case filter : DPF regen, EGR, coolant, transient suppression
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

from simulator import (
    VE_TABLE_RPM, VE_TABLE_VE,
    TURBO_RPM_BREAKPOINTS, TURBO_PRESSURE_RATIO,
    EBP_COEFF_FUEL, EBP_COEFF_RPM, EBP_INTERCEPT,
    DISPLACEMENT_L, INTERCOOLER_EFFICIENCY,
    _air_density, _ve_lookup, _turbo_pressure_ratio, _turbo_comp_temp_rise,
)

# ─── Detection Thresholds ─────────────────────────────────────────────────────
ZONE_A_MAF_THRESHOLD_PCT = 8.0        # % residual to flag
ZONE_B_MAP_THRESHOLD_PCT = 6.0        # % residual to flag
ZONE_C_EBP_THRESHOLD_PCT = 12.0       # % residual to flag (EBP drops on leak)
EGT_DELTA_THRESHOLD_C    = 8.0        # °C difference between EGT pair for sub-location
                                       # (reduced from 20 °C — median filter dampens raw signal)
# Zone B sub-location thresholds.
# At healthy operating conditions the boost→intercooler_outlet temp delta is
# ~37–39 °C.  A leak BEFORE the intercooler reduces hot-air mass-flow into it,
# causing the outlet temperature to fall further → delta RISES above baseline.
# A leak AFTER the intercooler leaves the intercooler unaffected → delta ≈ baseline.
BOOST_IC_DELTA_THRESHOLD     = 8.0    # °C — retained for legacy compatibility
BOOST_IC_BEFORE_IC_THRESHOLD = 43.0  # °C — delta above this → leak is BEFORE intercooler

# ─── Drift Detection ──────────────────────────────────────────────────────────
DRIFT_WINDOW  = 50          # samples (~5 s at 100 ms)
DRIFT_SLOPE_THRESHOLD = 0.008  # residual % per sample — slow creep (true drift); sudden leaks have higher slope but still alert

# ─── Confidence mapping (residual % → confidence %) ──────────────────────────
def _pct_to_confidence(residual_pct: float, threshold_pct: float) -> float:
    """Sigmoid-like mapping: at threshold → 50%, at 3× threshold → 95%."""
    ratio = residual_pct / max(threshold_pct, 0.01)
    conf  = 100.0 / (1.0 + np.exp(-3.0 * (ratio - 1.0)))
    return round(float(np.clip(conf, 0, 100)), 1)


@dataclass
class ZoneResult:
    zone:           str
    flag:           bool          = False
    residual_pct:   float         = 0.0
    expected:       float         = 0.0
    actual:         float         = 0.0
    confidence:     float         = 0.0
    sub_location:   str           = "unknown"
    drift:          bool          = False
    suppressed:     bool          = False
    suppression_reason: str       = ""
    sensor_name:    str           = ""


class ResidualTracker:
    """Tracks a sliding window of residuals for drift detection."""

    def __init__(self, window: int = DRIFT_WINDOW):
        self._buf: deque = deque(maxlen=window)

    def push(self, residual: float):
        self._buf.append(residual)

    def is_drifting(self) -> bool:
        if len(self._buf) < self._buf.maxlen:
            return False
        xs = np.arange(len(self._buf))
        ys = np.array(self._buf)
        coeffs = np.polyfit(xs, ys, 1)
        slope = abs(coeffs[0])
        return slope >= DRIFT_SLOPE_THRESHOLD


# ─── ZONE A MODULE ────────────────────────────────────────────────────────────

class ZoneADetector:
    """
    Detects air-intake leaks (pre-turbo, air filter, intake ducting).
    Uses VE table + air density to compute expected MAF.
    """

    def __init__(self):
        self._tracker = ResidualTracker()

    def _expected_maf(self, rpm: float, intercooler_outlet_c: float, map_kpa: float) -> float:
        """Expected MAF in g/s given current operating point.

        Uses intercooler_outlet_c (not iat_c) for air density — this matches the
        simulator physics exactly and eliminates RPM-dependent bias that arose when
        iat_c (post-sensor noise) was used instead.
        """
        ve            = _ve_lookup(rpm)
        rho           = _air_density(intercooler_outlet_c, map_kpa)
        vol_flow_m3s  = (DISPLACEMENT_L / 1000.0) * (rpm / 60.0) / 2.0
        vol_flow_m3s *= ve
        return vol_flow_m3s * rho * 1000.0    # g/s

    def run(self, filt: dict, ecu: dict) -> ZoneResult:
        result = ZoneResult(zone="A", sensor_name="maf_gs")

        # Edge-case suppression
        if ecu.get("transient"):
            result.suppressed = True
            result.suppression_reason = "Engine transient — MAF unstable"
            return result
        if ecu.get("egr_pct", 0) > 40:
            result.suppressed = True
            result.suppression_reason = "High EGR — bypasses MAF path"
            return result

        rpm                  = filt["rpm"]
        intercooler_outlet_c = filt["intercooler_outlet_c"]   # FIX: use intercooler temp, not iat_c
        map_kpa              = filt["map_kpa"]
        actual               = filt["maf_gs"]

        expected = self._expected_maf(rpm, intercooler_outlet_c, map_kpa)
        if expected <= 0:
            return result

        # Residual: positive means actual is LOWER than expected (leak symptom)
        residual_pct = ((expected - actual) / expected) * 100.0
        self._tracker.push(residual_pct)

        result.expected      = round(expected, 3)
        result.actual        = round(actual, 3)
        result.residual_pct  = round(residual_pct, 2)
        result.drift         = self._tracker.is_drifting()
        result.confidence    = _pct_to_confidence(residual_pct, ZONE_A_MAF_THRESHOLD_PCT)

        if residual_pct >= ZONE_A_MAF_THRESHOLD_PCT:
            result.flag = True
            result.sub_location = "pre_turbo_leak"

        return result


# ─── ZONE B MODULE ────────────────────────────────────────────────────────────

class ZoneBDetector:
    """
    Detects charge-air leaks (turbo outlet → intercooler → intake manifold).
    Uses compressor map to compute expected MAP.
    Sub-location from boost_temp vs intercooler_outlet_temp delta.
    """

    def __init__(self):
        self._tracker = ResidualTracker()

    def _expected_map(self, rpm: float, ambient_kpa: float = 101.325) -> float:
        pr = _turbo_pressure_ratio(rpm)
        return pr * ambient_kpa

    def run(self, filt: dict, ecu: dict) -> ZoneResult:
        result = ZoneResult(zone="B", sensor_name="map_kpa")

        # Edge-case suppression
        if ecu.get("transient"):
            result.suppressed = True
            result.suppression_reason = "Engine transient — boost unstable"
            return result
        if ecu.get("coolant_temp_c", 88) < 60:
            result.suppressed = True
            result.suppression_reason = "Cold engine — turbo not at operating point"
            return result

        rpm     = filt["rpm"]
        actual  = filt["map_kpa"]
        expected = self._expected_map(rpm)

        residual_pct = ((expected - actual) / expected) * 100.0
        self._tracker.push(residual_pct)

        result.expected     = round(expected, 3)
        result.actual       = round(actual, 3)
        result.residual_pct = round(residual_pct, 2)
        result.drift        = self._tracker.is_drifting()
        result.confidence   = _pct_to_confidence(residual_pct, ZONE_B_MAP_THRESHOLD_PCT)

        # Drift is informational — never suppresses a genuine residual flag
        if residual_pct >= ZONE_B_MAP_THRESHOLD_PCT:
            result.flag = True
            # Sub-location via boost→intercooler_outlet temperature delta:
            #
            # BEFORE intercooler (turbo outlet hose / compressor outlet):
            #   Compressed air escapes before reaching the intercooler.
            #   The intercooler sees reduced hot mass-flow → cools what little air
            #   it receives more aggressively → outlet temperature DROPS further
            #   → delta (boost_temp − ic_outlet) RISES above the healthy baseline.
            #
            # AFTER intercooler (intercooler outlet hose / clamp / intake fitting):
            #   Air escapes downstream of the intercooler.
            #   The intercooler operates normally on the full hot charge →
            #   delta stays near the healthy ~37–39 °C baseline.
            #
            boost_temp = filt.get("boost_temp_c", 0.0)
            ic_temp    = filt.get("intercooler_outlet_c", 0.0)
            temp_delta = boost_temp - ic_temp

            # Dynamic Baseline Calculation
            # Calculates what the temp drop SHOULD be at this exact RPM
            expected_comp_temp_rise = _turbo_comp_temp_rise(rpm)
            expected_baseline_delta = expected_comp_temp_rise * INTERCOOLER_EFFICIENCY
            
            # If the actual temp drop exceeds expected by more than 5 degrees, the intercooler is starved
            dynamic_threshold = expected_baseline_delta + 5.0

            if temp_delta > dynamic_threshold:
                # Delta is significantly higher than expected baseline → intercooler starved
                # → leak is upstream of the intercooler
                result.sub_location = "before_intercooler_hose_or_turbo_outlet"
            else:
                # Delta is near baseline → intercooler working normally → leak is downstream
                result.sub_location = "after_intercooler_hose_or_clamp"

        return result


# ─── ZONE C MODULE ────────────────────────────────────────────────────────────

class ZoneCDetector:
    """
    Detects exhaust path leaks (cracked manifold, blown gasket, loose connection).
    Uses fuel-rate + RPM regression to compute expected EBP.
    Sub-location from EGT pair comparison.
    """

    def __init__(self):
        self._tracker = ResidualTracker()

    def _expected_ebp(self, fuel_gs: float, rpm: float, dpf_regen: bool = False) -> float:
        """Expected EBP in kPa.

        When DPF regen is active the filter is being actively regenerated and
        exhaust backpressure rises naturally by ~35 %.  The model must account
        for this so the residual stays near zero during regen events and does
        not produce a false Zone C flag.
        """
        base = (EBP_COEFF_FUEL * fuel_gs
                + EBP_COEFF_RPM  * rpm
                + EBP_INTERCEPT)
        if dpf_regen:
            base *= 1.35   # mirrors the simulator's regen multiplier
        return base

    def run(self, filt: dict, ecu: dict) -> ZoneResult:
        result = ZoneResult(zone="C", sensor_name="ebp_kpa")

        # ── Edge-case suppression ─────────────────────────────────────────────
        # DPF regen raises EBP naturally — must suppress OR compensate via model.
        # We do BOTH: suppress first (fast path), then compensate in model (safety net).
        dpf_regen = bool(ecu.get("dpf_regen", False))
        if dpf_regen:
            result.suppressed = True
            result.suppression_reason = "DPF regen active — EBP naturally elevated; Zone C suppressed"
            return result
        if ecu.get("coolant_temp_c", 88) < 60:
            result.suppressed = True
            result.suppression_reason = "Cold engine — EBP model not calibrated"
            return result

        fuel_gs  = filt["fuel_rate_gs"]
        rpm      = filt["rpm"]
        actual   = filt["ebp_kpa"]
        # Pass dpf_regen=False here (regen case already returned above)
        expected = self._expected_ebp(fuel_gs, rpm, dpf_regen=False)

        # For Zone C, leak DROPS EBP — actual is lower than expected
        residual_pct = ((expected - actual) / max(expected, 0.1)) * 100.0
        self._tracker.push(residual_pct)

        result.expected     = round(expected, 3)
        result.actual       = round(actual, 3)
        result.residual_pct = round(residual_pct, 2)
        result.drift        = self._tracker.is_drifting()
        result.confidence   = _pct_to_confidence(residual_pct, ZONE_C_EBP_THRESHOLD_PCT)

        # Drift is informational — never suppresses a genuine residual flag
        if residual_pct >= ZONE_C_EBP_THRESHOLD_PCT:
            result.flag = True
            # Sub-location: which EGT sensor is reading abnormally low.
            # The simulator cools egt_1 for an upstream bank leak and egt_2 for downstream.
            # A cooled sensor reads LOWER than its pair → negative delta for upstream.
            # delta = egt1 − egt2:
            #   delta << 0  → egt1 is low  → upstream  bank exhaust port / manifold crack
            #   delta >> 0  → egt2 is low  → downstream bank / DPF or catalyst section
            #   |delta| small → no clear lateralisation → general restriction / EBP sensor
            egt1  = filt.get("egt_1_c", 0.0)
            egt2  = filt.get("egt_2_c", 0.0)
            delta = egt1 - egt2
            if delta < -EGT_DELTA_THRESHOLD_C:
                # egt1 is significantly lower → upstream bank is losing exhaust heat
                result.sub_location = "upstream_bank_cylinder_exhaust_ports"
            elif delta > EGT_DELTA_THRESHOLD_C:
                # egt2 is significantly lower → downstream / DPF or catalyst section
                result.sub_location = "downstream_bank_DPF_or_catalyst"
            else:
                result.sub_location = "general_exhaust_restriction"

        return result


# ─── Orchestrator ─────────────────────────────────────────────────────────────

class PhysicsEngine:
    """Runs all three zone detectors and returns their results."""

    def __init__(self):
        self.zone_a = ZoneADetector()
        self.zone_b = ZoneBDetector()
        self.zone_c = ZoneCDetector()

    def run(self, filt: dict, raw: dict) -> Tuple[ZoneResult, ZoneResult, ZoneResult]:
        """
        filt: filtered sensor values from pipeline
        raw:  original row (contains ECU flags)
        Returns: (zone_a_result, zone_b_result, zone_c_result)
        """
        ecu = {
            "dpf_regen":     bool(raw.get("dpf_regen", 0)),
            "egr_pct":       raw.get("egr_pct", 15.0),
            "coolant_temp_c": raw.get("coolant_temp_c", 88.0),
            "transient":     bool(raw.get("transient", 0)),
        }
        ra = self.zone_a.run(filt, ecu)
        rb = self.zone_b.run(filt, ecu)
        rc = self.zone_c.run(filt, ecu)
        return ra, rb, rc


# ─── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from simulator import EngineSimulator
    from pipeline  import DataPipeline

    sim  = EngineSimulator(2000, 60)
    pipe = DataPipeline()
    pe   = PhysicsEngine()

    # Warm up steady state
    for _ in range(110):
        pipe.process(sim.step())

    print("=== Healthy mode ===")
    for _ in range(5):
        raw = sim.step()
        pr  = pipe.process(raw)
        ra, rb, rc = pe.run(pr.filt, pr.raw)
    print(f"  Zone A flag: {ra.flag}  residual={ra.residual_pct:.1f}%")
    print(f"  Zone B flag: {rb.flag}  residual={rb.residual_pct:.1f}%")
    print(f"  Zone C flag: {rc.flag}  residual={rc.residual_pct:.1f}%")

    print("\n=== Zone B leak 25% ===")
    sim.inject_leak("B", 0.25)
    for _ in range(30):
        raw = sim.step()
        pr  = pipe.process(raw)
        ra, rb, rc = pe.run(pr.filt, pr.raw)
    print(f"  Zone B flag: {rb.flag}  residual={rb.residual_pct:.1f}%  sub={rb.sub_location}")
    print(f"  Zone B confidence: {rb.confidence}%")

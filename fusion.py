"""
fusion.py
---------
Combines physics zone results and ML anomaly result into a single
structured decision output every second.

Weighting:   Physics 60% | ML 40%
Confidence levels:
  HIGH   — both physics AND ML flag the same zone  → ALERT
  MEDIUM — only physics flags a zone               → MONITOR
  LOW    — only ML flags                           → LOG
  CLEAR  — neither flags                           → PASS
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
from physics_engine import ZoneResult
from ml_engine       import MLResult

# ─── Fusion Weights ───────────────────────────────────────────────────────────
PHYSICS_WEIGHT = 0.60
ML_WEIGHT      = 0.40

# Minimum physics residual to contribute to score (avoid noise)
PHYSICS_MIN_RESIDUAL = 3.0   # %

# Action lookup per zone × sub-location
ACTION_MAP = {
    "A": {
        "pre_turbo_leak":
            "Inspect air filter element and intake ducting from filter to turbocharger inlet.",
        "unknown":
            "Check entire Zone A air path from ambient intake to turbo inlet.",
    },
    "B": {
        "before_intercooler_hose_or_turbo_outlet":
            "Inspect turbocharger compressor outlet hose and clamp; check for cracks.",
        "after_intercooler_hose_or_clamp":
            "Inspect intercooler outlet hose, clamp and charge air cooler end-tanks.",
        "unknown":
            "Check hoses, clamps, and connections in the charge-air cooler circuit.",
    },
    "C": {
        "upstream_bank_cylinder_exhaust_ports":
            "Inspect upstream exhaust manifold gaskets, ports, and flex pipes.",
        "downstream_bank_DPF_or_catalyst":
            "Inspect DPF canister, catalyst, and downstream exhaust connections.",
        "general_exhaust_restriction":
            "Perform exhaust back-pressure test; inspect DPF condition and EBP sensor.",
        "unknown":
            "Check exhaust system from exhaust manifold to tail pipe for restrictions.",
    },
}


@dataclass
class FusionDecision:
    # Status
    status:          str    = "PASS"        # "PASS" | "LOG" | "MONITOR" | "ALERT"
    leak_detected:   bool   = False
    zone:            str    = "none"
    sub_location:    str    = "none"

    # Confidence / scores
    confidence_pct:  float  = 0.0
    physics_score:   float  = 0.0
    ml_score:        float  = 0.0
    fused_score:     float  = 0.0

    # Evidence
    triggered_by:    str    = ""            # "physics" | "ml" | "both"
    sensor_name:     str    = ""
    residual_pct:    float  = 0.0
    expected_value:  float  = 0.0
    actual_value:    float  = 0.0

    # ML info
    ml_flag:           bool  = False
    ml_recon_error:    float = 0.0
    ml_worst_feature:  str   = ""

    # Action
    action:    str  = "No action required — system nominal."
    drift:     bool = False

    # Suppression
    suppressed:           bool = False
    suppression_reason:   str  = ""

    # Timestamp
    timestamp: float = 0.0


def fuse(
    zone_a: ZoneResult,
    zone_b: ZoneResult,
    zone_c: ZoneResult,
    ml:     MLResult,
    timestamp: float = 0.0,
) -> FusionDecision:
    """
    Combine physics zone flags and ML result into a FusionDecision.
    """
    decision = FusionDecision(timestamp=timestamp)

    # ── 1. Collect physics flags ──────────────────────────────────────────────
    phys_flags = []
    for zr in [zone_a, zone_b, zone_c]:
        if zr.flag and not zr.suppressed:
            phys_flags.append(zr)

    # Pick highest-confidence physics flag
    best_physics: Optional[ZoneResult] = None
    if phys_flags:
        best_physics = max(phys_flags, key=lambda z: z.confidence)

    # ── 2. Compute physics score ──────────────────────────────────────────────
    if best_physics:
        # Normalise residual to a 0-100 score
        decision.physics_score = min(100.0, best_physics.confidence)
    else:
        # Even without a flag, residuals carry partial signal
        max_res = max(
            max(zr.residual_pct, 0) for zr in [zone_a, zone_b, zone_c]
            if not zr.suppressed
        ) if any(not zr.suppressed for zr in [zone_a, zone_b, zone_c]) else 0.0
        decision.physics_score = min(30.0, max_res * 1.5)

    # ── 2b. ML Suppression Override ───────────────────────────────────────────
    # If the ML model flags an anomaly in a zone that the physics engine has
    # explicitly suppressed (e.g. DPF regen), we must suppress the ML flag too.
    if ml.flag:
        if (ml.anomaly_zone == "A" and zone_a.suppressed) or \
           (ml.anomaly_zone == "B" and zone_b.suppressed) or \
           (ml.anomaly_zone == "C" and zone_c.suppressed):
            ml.flag = False

    # ── 3. ML score ───────────────────────────────────────────────────────────
    decision.ml_flag          = ml.flag
    decision.ml_recon_error   = ml.reconstruction_error
    decision.ml_worst_feature = ml.worst_feature
    decision.ml_score         = ml.confidence if ml.flag else 0.0

    # ── 4. Fused score (weighted average) ─────────────────────────────────────
    fused = (PHYSICS_WEIGHT * decision.physics_score
             + ML_WEIGHT    * decision.ml_score)
    decision.fused_score = round(fused, 1)

    # ── 5. Zone agreement check ───────────────────────────────────────────────
    # Relaxed: any flagged physics zone + ML flagging = ALERT
    # (multiple zones can flag simultaneously on compound leaks)
    same_zone = (
        best_physics is not None
        and ml.flag
        and (ml.anomaly_zone == best_physics.zone or ml.anomaly_zone == "unknown")
    )

    # ── 6. Status determination ───────────────────────────────────────────────
    if best_physics and ml.flag:
        # Both engines agree there is a problem — ALERT regardless of exact zone match
        decision.status        = "ALERT"
        decision.leak_detected = True
        decision.triggered_by  = "both"
        decision.confidence_pct= round((best_physics.confidence * 0.6
                                        + ml.confidence * 0.4), 1)

    elif best_physics and not ml.flag:
        decision.status        = "MONITOR"
        decision.leak_detected = True
        decision.triggered_by  = "physics"
        decision.confidence_pct= round(best_physics.confidence * 0.7, 1)

    elif ml.flag and not best_physics:
        # ML flags but physics hasn't confirmed — still show as MONITOR (not silent LOG)
        decision.status        = "MONITOR"
        decision.leak_detected = True
        decision.triggered_by  = "ml"
        decision.confidence_pct= round(ml.confidence * 0.5, 1)

    else:
        decision.status        = "PASS"
        decision.leak_detected = False
        decision.triggered_by  = ""
        decision.confidence_pct= 0.0

    # ── 7. Populate zone info and action ─────────────────────────────────────
    if best_physics and decision.leak_detected:
        decision.zone          = best_physics.zone
        decision.sub_location  = best_physics.sub_location
        decision.sensor_name   = best_physics.sensor_name
        decision.residual_pct  = best_physics.residual_pct
        decision.expected_value= best_physics.expected
        decision.actual_value  = best_physics.actual
        decision.drift         = best_physics.drift

        zone_actions = ACTION_MAP.get(best_physics.zone, {})
        decision.action = zone_actions.get(
            best_physics.sub_location,
            zone_actions.get("unknown", "Inspect the flagged zone.")
        )

    elif ml.flag and not best_physics:
        decision.zone         = ml.anomaly_zone
        decision.sub_location = "ML-identified — confirm with physics check"
        zone_actions = ACTION_MAP.get(ml.anomaly_zone, {})
        decision.action = zone_actions.get(
            "unknown", f"Check Zone {ml.anomaly_zone} — ML anomaly detected."
        )

    # ── 8. Suppression propagation ────────────────────────────────────────────
    all_suppressed = all(
        zr.suppressed for zr in [zone_a, zone_b, zone_c]
    )
    if all_suppressed and not ml.flag:
        decision.suppressed = True
        decision.suppression_reason = (
            zone_a.suppression_reason or
            zone_b.suppression_reason or
            zone_c.suppression_reason
        )
        decision.status = "SUPPRESSED"

    return decision


# ─── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from simulator      import EngineSimulator
    from pipeline       import DataPipeline
    from physics_engine import PhysicsEngine
    from ml_engine      import MLEngine

    sim  = EngineSimulator(2000, 60)
    pipe = DataPipeline()
    pe   = PhysicsEngine()
    ml   = MLEngine()

    try:
        ml.load()
        ml_available = True
    except FileNotFoundError:
        print("[Fusion] ML model not found — running physics-only mode.")
        ml_available = False

    # Warm up
    for _ in range(110):
        raw = sim.step()
        pr  = pipe.process(raw)

    print("=== Healthy Mode ===")
    for _ in range(5):
        raw = sim.step()
        pr  = pipe.process(raw)
        ra, rb, rc = pe.run(pr.filt, pr.raw)
        ml_result  = ml.run(pr.filt) if ml_available else MLResult()
        d = fuse(ra, rb, rc, ml_result, raw["timestamp"])
    print(f"  Status: {d.status}  Zone: {d.zone}  Confidence: {d.confidence_pct}%")

    print("\n=== Zone B Leak 30% ===")
    sim.inject_leak("B", 0.30)
    for _ in range(40):
        raw = sim.step()
        pr  = pipe.process(raw)
        ra, rb, rc = pe.run(pr.filt, pr.raw)
        ml_result  = ml.run(pr.filt) if ml_available else MLResult()
        d = fuse(ra, rb, rc, ml_result, raw["timestamp"])
    print(f"  Status: {d.status}")
    print(f"  Zone:   {d.zone}")
    print(f"  Sub:    {d.sub_location}")
    print(f"  Confidence: {d.confidence_pct}%")
    print(f"  Action: {d.action}")

"""
main.py
-------
Orchestrator for the Caterpillar Digital Twin Leak Detection System.

Runs the full pipeline in a background thread:
  simulator → pipeline → physics_engine → ml_engine → fusion → output

Also writes a shared state dict that dashboard.py reads in real time.
Controls:
  - Set RPM / load
  - Inject / clear leaks per zone
  - Toggle DPF regen, EGR position, engine transient flags
"""

import time
import threading
import json
import argparse
from pathlib import Path
from typing import Optional

from simulator      import EngineSimulator
from pipeline       import DataPipeline
from physics_engine import PhysicsEngine
from ml_engine      import MLEngine, MLResult
from fusion         import fuse, FusionDecision
from output         import OutputManager, print_banner, print_go_nogo

# ─── Shared state (written by engine thread, read by dashboard) ───────────────
STATE_FILE = Path(__file__).parent / "logs" / "live_state.json"

_shared_state: dict = {}
_state_lock         = threading.Lock()


def _write_state(decision: FusionDecision, raw: dict, filt: dict):
    """Serialize current pipeline state to shared dict and JSON file."""
    state = {
        # Sensor readings
        "timestamp":            raw.get("timestamp", 0.0),
        "rpm":                  filt.get("rpm", 0.0),
        "load_pct":             raw.get("load_pct", 0.0),
        "maf_gs":               filt.get("maf_gs", 0.0),
        "map_kpa":              filt.get("map_kpa", 0.0),
        "iat_c":                filt.get("iat_c", 0.0),
        "boost_temp_c":         filt.get("boost_temp_c", 0.0),
        "intercooler_outlet_c": filt.get("intercooler_outlet_c", 0.0),
        "ebp_kpa":              filt.get("ebp_kpa", 0.0),
        "egt_1_c":              filt.get("egt_1_c", 0.0),
        "egt_2_c":              filt.get("egt_2_c", 0.0),
        "fuel_rate_gs":         filt.get("fuel_rate_gs", 0.0),
        # ECU flags
        "dpf_regen":            raw.get("dpf_regen", 0),
        "egr_pct":              raw.get("egr_pct", 15.0),
        # Decision
        "status":               decision.status,
        "leak_detected":        decision.leak_detected,
        "zone":                 decision.zone,
        "sub_location":         decision.sub_location,
        "confidence_pct":       decision.confidence_pct,
        "physics_score":        decision.physics_score,
        "ml_score":             decision.ml_score,
        "fused_score":          decision.fused_score,
        "triggered_by":         decision.triggered_by,
        "sensor_name":          decision.sensor_name,
        "residual_pct":         decision.residual_pct,
        "expected_value":       decision.expected_value,
        "actual_value":         decision.actual_value,
        "action":               decision.action,
        "drift":                decision.drift,
        "suppressed":           decision.suppressed,
        "suppression_reason":   decision.suppression_reason,
        "ml_flag":              decision.ml_flag,
        "ml_recon_error":       decision.ml_recon_error,
        "ml_worst_feature":     decision.ml_worst_feature,
    }

    with _state_lock:
        _shared_state.clear()
        _shared_state.update(state)

    # Write to JSON for dashboard process
    STATE_FILE.parent.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def get_shared_state() -> dict:
    with _state_lock:
        return dict(_shared_state)


# ─── Engine Thread ────────────────────────────────────────────────────────────

class EngineThread(threading.Thread):
    """
    Background thread running the full pipeline loop at 10 Hz (100 ms steps).
    Aggregates 10 steps then emits one decision per second.
    """

    def __init__(self, simulator: EngineSimulator, ml_engine: MLEngine,
                 out: OutputManager, verbose: bool = True):
        super().__init__(daemon=True, name="EngineThread")
        self.sim     = simulator
        self.pipe    = DataPipeline()
        self.pe      = PhysicsEngine()
        self.ml      = ml_engine
        self.out     = out
        self.verbose = verbose
        self._running = threading.Event()
        self._running.set()
        self._last_decision: FusionDecision | None = None

    def stop(self):
        self._running.clear()

    @property
    def last_decision(self) -> FusionDecision | None:
        return self._last_decision

    def run(self):
        step_counter = 0
        last_pr  = None
        last_raw = None

        while self._running.is_set():
            loop_start = time.perf_counter()

            # Run one 100 ms simulator step
            raw = self.sim.step()
            pr  = self.pipe.process(raw)

            last_pr  = pr
            last_raw = raw
            step_counter += 1

            # Every 10 steps = 1 second → emit decision
            if step_counter % 10 == 0 and pr.steady_state:
                ra, rb, rc = self.pe.run(pr.filt, pr.raw)
                ml_res     = self.ml.run(pr.filt) if self.ml.is_loaded else MLResult()
                decision   = fuse(ra, rb, rc, ml_res, raw["timestamp"])
                self._last_decision = decision

                _write_state(decision, raw, pr.filt)

                if self.verbose:
                    self.out.emit(decision)

            elif step_counter % 10 == 0:
                # Not yet steady — write raw state for dashboard, no alert
                import dataclasses
                blank = FusionDecision(
                    status="WARMING_UP",
                    timestamp=raw["timestamp"]
                )
                _write_state(blank, raw, pr.filt)

            # Pace to real-time
            elapsed = time.perf_counter() - loop_start
            sleep   = max(0.0, 0.1 - elapsed)
            time.sleep(sleep)


# ─── Demo Scenario ────────────────────────────────────────────────────────────

def run_demo(sim: EngineSimulator, engine_thread: EngineThread,
             out: OutputManager):
    """
    Scripted demo sequence:
      0s  — healthy, all green
      30s — inject Zone B leak
      50s — clear, inject Zone A leak
      70s — clear, inject Zone C leak
      90s — clear, healthy close
    """
    print("\n[DEMO] Starting scripted scenario — engine at 2000 RPM, 60% load\n")
    time.sleep(15)   # warm-up

    print("\n[DEMO] ── Phase 1: Healthy running ──")
    time.sleep(15)

    print("\n[DEMO] ── Phase 2: Injecting Zone B leak (25%) ──")
    sim.inject_leak("B", severity=0.25, b_location="after_intercooler")
    time.sleep(20)

    print("\n[DEMO] ── Phase 3: Clearing leak → Injecting Zone A leak (20%) ──")
    sim.clear_leak()
    time.sleep(3)
    sim.inject_leak("A", severity=0.20)
    time.sleep(20)

    print("\n[DEMO] ── Phase 4: Clearing → Injecting Zone C leak (30%) ──")
    sim.clear_leak()
    time.sleep(3)
    sim.inject_leak("C", severity=0.30, c_bank="upstream")
    time.sleep(20)

    print("\n[DEMO] ── Phase 5: Clearing all leaks — back to nominal ──")
    sim.clear_leak()
    time.sleep(10)

    print_go_nogo(engine_thread.last_decision or FusionDecision())
    print("\n[DEMO] Demo complete.")


# ─── Interactive CLI ──────────────────────────────────────────────────────────

def interactive_loop(sim: EngineSimulator):
    """Simple command loop for manual control."""
    print("\n[CLI] Commands:")
    print("  leak A <severity>   — inject Zone A leak (0.0–1.0)")
    print("  leak B <severity>   — inject Zone B leak")
    print("  leak C <severity>   — inject Zone C leak")
    print("  clear               — clear all leaks")
    print("  rpm <value>         — set RPM (600–3000)")
    print("  load <value>        — set load % (0–100)")
    print("  dpf on|off          — toggle DPF regen")
    print("  quit                — exit\n")

    while True:
        try:
            cmd = input("[CAT] > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        parts = cmd.split()
        if not parts:
            continue

        if parts[0] == "leak" and len(parts) >= 3:
            zone     = parts[1].upper()
            severity = float(parts[2])
            sim.inject_leak(zone, severity)
            print(f"  → Zone {zone} leak injected at {severity*100:.0f}% severity")

        elif parts[0] == "clear":
            sim.clear_leak()
            print("  → All leaks cleared")

        elif parts[0] == "rpm" and len(parts) >= 2:
            rpm = float(parts[1])
            sim.set_operating_point(rpm, sim.state.load_pct)
            print(f"  → RPM set to {rpm:.0f}")

        elif parts[0] == "load" and len(parts) >= 2:
            load = float(parts[1])
            sim.set_operating_point(sim.state.rpm, load)
            print(f"  → Load set to {load:.0f}%")

        elif parts[0] == "dpf" and len(parts) >= 2:
            active = parts[1] == "on"
            sim.set_engine_flags(dpf_regen=active)
            print(f"  → DPF regen {'ACTIVE' if active else 'OFF'}")

        elif parts[0] == "quit":
            break
        else:
            print(f"  Unknown command: {cmd}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Caterpillar Leak Detection System"
    )
    parser.add_argument("--demo",     action="store_true",
                        help="Run scripted demo scenario")
    parser.add_argument("--no-ml",   action="store_true",
                        help="Skip ML engine (physics-only mode)")
    parser.add_argument("--rpm",      type=float, default=2000.0)
    parser.add_argument("--load",     type=float, default=60.0)
    parser.add_argument("--quiet",    action="store_true",
                        help="Suppress console output (dashboard only)")
    args = parser.parse_args()

    print_banner()

    # ── Setup ─────────────────────────────────────────────────────────────────
    sim = EngineSimulator(initial_rpm=args.rpm, initial_load=args.load)

    ml = MLEngine()
    if not args.no_ml:
        try:
            ml.load()
        except FileNotFoundError:
            print("[MAIN] ML model not found. Run: python ml_engine.py --train")
            print("[MAIN] Continuing in physics-only mode.\n")

    out = OutputManager(log_to_csv=True, print_to_console=not args.quiet)
    engine = EngineThread(sim, ml, out, verbose=not args.quiet)

    # ── Start ─────────────────────────────────────────────────────────────────
    engine.start()
    print(f"[MAIN] Engine thread started — {args.rpm:.0f} RPM, {args.load:.0f}% load")
    print(f"[MAIN] Warming up steady-state detector (10 s)...\n")

    try:
        if args.demo:
            run_demo(sim, engine, out)
        else:
            interactive_loop(sim)
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by user.")
    finally:
        engine.stop()
        out.close()
        print("[MAIN] Shutdown complete.")


if __name__ == "__main__":
    main()

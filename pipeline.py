"""
pipeline.py
-----------
Data pipeline — sits between the simulator and the detection engines.
Handles: rolling median filter, sensor-range validation, steady-state detection.
Works in live mode (row by row from simulator) or replay mode (from CSV).
"""

import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Deque, Dict, Tuple

# ─── Sensor Valid Ranges ──────────────────────────────────────────────────────
SENSOR_RANGES = {
    "maf_gs":               (20.0,   600.0),
    "map_kpa":              (95.0,   300.0),
    "iat_c":                (-10.0,  80.0),
    "boost_temp_c":         (20.0,   220.0),
    "intercooler_outlet_c": (-5.0,   120.0),
    "ebp_kpa":              (2.0,    80.0),
    "egt_1_c":              (200.0,  900.0),
    "egt_2_c":              (200.0,  900.0),
    "fuel_rate_gs":         (0.1,    15.0),
    "rpm":                  (500.0,  3200.0),
}

# ─── Rolling-median window (10 readings = 1 s at 100 ms/step) ─────────────────
MEDIAN_WINDOW = 10

# ─── Steady-state: last N seconds must have low std-dev ──────────────────────
STEADY_STATE_WINDOW_S = 10.0      # seconds
STEADY_STATE_DT       = 0.1       # simulator timestep
STEADY_RPM_STD        = 30.0      # RPM std allowed in window
STEADY_LOAD_STD       = 2.0       # load % std allowed


@dataclass
class PipelineRow:
    raw:    Dict[str, float]     # original sensor values
    filt:   Dict[str, float]     # after median filter
    valid:  Dict[str, bool]      # per-sensor range validity
    all_valid: bool
    steady_state: bool
    timestamp: float


class DataPipeline:
    """
    Stateful pipeline that processes one sensor row at a time.

    Usage (live mode):
        pipe = DataPipeline()
        for _ in range(N):
            raw_row = simulator.step()
            processed = pipe.process(raw_row)

    Usage (replay mode):
        pipe = DataPipeline()
        df = pd.read_csv("run_log.csv")
        for _, row in df.iterrows():
            processed = pipe.process(row.to_dict())
    """

    def __init__(self, median_window: int = MEDIAN_WINDOW):
        self._window: int = median_window
        # Rolling buffers per sensor
        self._buffers: Dict[str, Deque[float]] = {
            k: deque(maxlen=median_window) for k in SENSOR_RANGES
        }
        # Steady-state buffers for RPM and load
        ss_len = int(STEADY_STATE_WINDOW_S / STEADY_STATE_DT)
        self._rpm_buf:  Deque[float] = deque(maxlen=ss_len)
        self._load_buf: Deque[float] = deque(maxlen=ss_len)
        self._n_processed: int = 0

    # ── Core Processing ────────────────────────────────────────────────────────

    def process(self, raw: dict) -> PipelineRow:
        """
        Process a single raw sensor dict.
        Returns a PipelineRow with filtered values, validity flags,
        and steady-state indicator.
        """
        # 1. Feed rolling buffers
        for key in SENSOR_RANGES:
            val = raw.get(key, np.nan)
            if val is not None and not np.isnan(val):
                self._buffers[key].append(float(val))

        # 2. Median filter
        filt: Dict[str, float] = {}
        for key in SENSOR_RANGES:
            buf = self._buffers[key]
            if len(buf) == 0:
                filt[key] = float(raw.get(key, np.nan))
            else:
                filt[key] = float(np.median(buf))

        # 3. Range validation (on filtered values)
        valid: Dict[str, bool] = {}
        for key, (lo, hi) in SENSOR_RANGES.items():
            v = filt.get(key, np.nan)
            valid[key] = bool(not np.isnan(v) and lo <= v <= hi)
        all_valid = all(valid.values())

        # 4. Steady-state detection
        self._rpm_buf.append(float(raw.get("rpm", 0.0)))
        self._load_buf.append(float(raw.get("load_pct", 0.0)))
        steady = self._is_steady_state()

        self._n_processed += 1
        return PipelineRow(
            raw=dict(raw),
            filt=filt,
            valid=valid,
            all_valid=all_valid,
            steady_state=steady,
            timestamp=float(raw.get("timestamp", self._n_processed * 0.1)),
        )

    # ── Steady-State Check ────────────────────────────────────────────────────

    def _is_steady_state(self) -> bool:
        """Return True if RPM and load have been stable for STEADY_STATE_WINDOW_S."""
        rpm_buf  = self._rpm_buf
        load_buf = self._load_buf
        ss_len = rpm_buf.maxlen
        if len(rpm_buf) < ss_len:
            return False   # not enough history yet
        rpm_std  = float(np.std(rpm_buf))
        load_std = float(np.std(load_buf))
        return rpm_std <= STEADY_RPM_STD and load_std <= STEADY_LOAD_STD

    # ── Batch Replay ──────────────────────────────────────────────────────────

    def replay_csv(self, filepath: str) -> list:
        """
        Load a saved CSV and replay through the pipeline.
        Returns list of PipelineRow objects.
        """
        df = pd.read_csv(filepath)
        results = []
        for _, row in df.iterrows():
            results.append(self.process(row.to_dict()))
        return results

    # ── Helpers ───────────────────────────────────────────────────────────────

    def reset(self):
        """Clear all internal buffers."""
        for buf in self._buffers.values():
            buf.clear()
        self._rpm_buf.clear()
        self._load_buf.clear()
        self._n_processed = 0

    @property
    def n_processed(self) -> int:
        return self._n_processed

    def summary_stats(self) -> pd.DataFrame:
        """Return current rolling-window stats as a DataFrame for debugging."""
        rows = []
        for key, (lo, hi) in SENSOR_RANGES.items():
            buf = self._buffers[key]
            rows.append({
                "sensor":  key,
                "n_samples": len(buf),
                "median":  np.median(buf) if buf else np.nan,
                "std":     np.std(buf) if buf else np.nan,
                "min_ok":  lo,
                "max_ok":  hi,
            })
        return pd.DataFrame(rows)


# ─── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from simulator import EngineSimulator

    sim  = EngineSimulator(2000, 60)
    pipe = DataPipeline()

    print("Processing 150 rows (15 s)...")
    for i in range(150):
        row = sim.step()
        pr  = pipe.process(row)
        if i == 149:
            print(f"\nTimestamp: {pr.timestamp:.1f}s")
            print(f"Steady state: {pr.steady_state}")
            print(f"All sensors valid: {pr.all_valid}")
            print(f"Filtered MAF: {pr.filt['maf_gs']:.2f} g/s")
            print(f"Filtered MAP: {pr.filt['map_kpa']:.2f} kPa")

    print("\nSummary stats:")
    print(pipe.summary_stats().to_string(index=False))

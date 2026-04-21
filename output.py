"""
output.py
---------
Formats and dispatches FusionDecision output.
  - Console: coloured, compact status line
  - CSV log: appends one row per call to run_log.csv
"""

import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from fusion import FusionDecision

LOG_DIR  = Path(__file__).parent / "logs"
LOG_FILE = LOG_DIR / "run_log.csv"

# ANSI colours (Windows 10+ supports these)
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_GREEN  = "\033[92m"
_CYAN   = "\033[96m"
_BLUE   = "\033[94m"
_RESET  = "\033[0m"
_BOLD   = "\033[1m"

STATUS_COLOUR = {
    "ALERT":      _RED,
    "MONITOR":    _YELLOW,
    "LOG":        _BLUE,
    "PASS":       _GREEN,
    "SUPPRESSED": _CYAN,
}

CSV_FIELDNAMES = [
    "wall_clock", "sim_time_s",
    "status", "leak_detected", "zone", "sub_location",
    "confidence_pct", "physics_score", "ml_score", "fused_score",
    "triggered_by", "sensor_name", "residual_pct",
    "expected_value", "actual_value",
    "ml_flag", "ml_recon_error", "ml_worst_feature",
    "drift", "suppressed", "suppression_reason", "action",
]


class OutputManager:
    """
    Singleton-friendly output manager.
    Call emit(decision) once per second.
    """

    def __init__(self, log_to_csv: bool = True, print_to_console: bool = True):
        self._log_to_csv     = log_to_csv
        self._print_console  = print_to_console
        self._csv_writer     = None
        self._csv_file       = None

        if log_to_csv:
            self._setup_csv()

    # ── CSV Setup ─────────────────────────────────────────────────────────────

    def _setup_csv(self):
        LOG_DIR.mkdir(exist_ok=True)
        file_exists = LOG_FILE.exists()
        self._csv_file   = open(LOG_FILE, "a", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=CSV_FIELDNAMES
        )
        if not file_exists:
            self._csv_writer.writeheader()

    # ── Main Emit ─────────────────────────────────────────────────────────────

    def emit(self, decision: FusionDecision):
        if self._print_console:
            self._console_print(decision)
        if self._log_to_csv and self._csv_writer:
            self._csv_write(decision)

    # ── Console Formatting ────────────────────────────────────────────────────

    def _console_print(self, d: FusionDecision):
        colour = STATUS_COLOUR.get(d.status, _RESET)
        ts     = f"[{d.timestamp:7.1f}s]"

        # Header line
        status_str = f"{colour}{_BOLD}{d.status:10s}{_RESET}"
        zone_str   = f"Zone {d.zone}" if d.zone != "none" else "       "
        conf_str   = f"{d.confidence_pct:5.1f}%" if d.leak_detected else "      "
        print(f"{ts} {status_str}  {zone_str}  conf={conf_str}", end="")

        if d.status == "ALERT" or d.status == "MONITOR":
            print(f"  {_BOLD}>>>{_RESET} {d.sensor_name}: "
                  f"expected={d.expected_value:.2f}  "
                  f"actual={d.actual_value:.2f}  "
                  f"residual={d.residual_pct:+.1f}%")
            print(f"         {'':10s}  sub-location : {d.sub_location}")
            print(f"         {'':10s}  action       : {d.action}")
            if d.drift:
                print(f"         {'':10s}  {_YELLOW}[DRIFT detected — gradual onset]{_RESET}")
        elif d.status == "LOG":
            print(f"  ML anomaly in Zone {d.zone}  "
                  f"feature={d.ml_worst_feature}  "
                  f"recon_err={d.ml_recon_error:.5f}")
        elif d.status == "SUPPRESSED":
            print(f"  [{d.suppression_reason}]")
        else:
            print()   # PASS — just newline

    # ── CSV Row ───────────────────────────────────────────────────────────────

    def _csv_write(self, d: FusionDecision):
        row = {
            "wall_clock":         datetime.now().isoformat(timespec="seconds"),
            "sim_time_s":         d.timestamp,
            "status":             d.status,
            "leak_detected":      int(d.leak_detected),
            "zone":               d.zone,
            "sub_location":       d.sub_location,
            "confidence_pct":     d.confidence_pct,
            "physics_score":      d.physics_score,
            "ml_score":           d.ml_score,
            "fused_score":        d.fused_score,
            "triggered_by":       d.triggered_by,
            "sensor_name":        d.sensor_name,
            "residual_pct":       d.residual_pct,
            "expected_value":     d.expected_value,
            "actual_value":       d.actual_value,
            "ml_flag":            int(d.ml_flag),
            "ml_recon_error":     d.ml_recon_error,
            "ml_worst_feature":   d.ml_worst_feature,
            "drift":              int(d.drift),
            "suppressed":         int(d.suppressed),
            "suppression_reason": d.suppression_reason,
            "action":             d.action,
        }
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close(self):
        if self._csv_file:
            self._csv_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def print_banner():
    """Print a startup banner."""
    print(f"""
{_BOLD}{_CYAN}
╔══════════════════════════════════════════════════════════╗
║   CATERPILLAR DIESEL ENGINE — AIR PATH LEAK DETECTOR    ║
║   Digital Twin + Physics + ML Fusion System             ║
╚══════════════════════════════════════════════════════════╝
{_RESET}""")


def print_go_nogo(decision: FusionDecision):
    """Large GO / NO-GO banner for demo presentations."""
    if decision.status in ("ALERT", "MONITOR"):
        colour = _RED
        verdict = "NO-GO"
        msg = f"LEAK DETECTED — Zone {decision.zone} — {decision.confidence_pct:.0f}% confidence"
    else:
        colour = _GREEN
        verdict = "  GO  "
        msg = "System Nominal — No Leaks Detected"

    print(f"""
{colour}{_BOLD}
  ┌─────────────────────────────────────────┐
  │  {'':>5}{verdict}{'':>5}                              │
  │  {msg:<40} │
  └─────────────────────────────────────────┘
{_RESET}""")

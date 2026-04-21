"""
dashboard.py
------------
Streamlit real-time dashboard for the Caterpillar Leak Detection System.

Run with:  streamlit run dashboard.py

The dashboard runs the simulator + pipeline + detection in a background
thread (same process). No separate main.py needed — this is self-contained.

Layout:
  ┌──────────────────────────────────────────────────────────┐
  │  HEADER  Status badge + Go/No-Go indicator               │
  ├──────────────┬─────────────────────┬────────────────────┤
  │ SENSOR PANEL │  RESIDUAL GAUGES    │  ALERT PANEL       │
  │ Live readings│  Zone A / B / C     │  Zone + action     │
  ├──────────────┴─────────────────────┴────────────────────┤
  │  HISTORY GRAPH  — 60 s rolling residuals               │
  ├──────────────────────────────────────────────────────────┤
  │  CONTROLS  — RPM / Load sliders, Leak inject buttons   │
  └──────────────────────────────────────────────────────────┘
"""

import streamlit as st
import pandas as pd
import numpy as np
import threading
import time
import json
import sys
from pathlib import Path
from collections import deque

# ── Import project modules ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from simulator      import EngineSimulator
from pipeline       import DataPipeline
from physics_engine import PhysicsEngine
from ml_engine      import MLEngine, MLResult
from fusion         import fuse, FusionDecision
from output         import OutputManager

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CAT Engine — Leak Detection",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&family=JetBrains+Mono:wght@400;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
  }

  .main .block-container { padding-top: 1rem; max-width: 100%; }

  /* ── Header ── */
  .cat-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f1f3d 100%);
    border: 1px solid #1e40af;
    border-radius: 12px;
    padding: 1.2rem 2rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .cat-title {
    font-size: 1.5rem;
    font-weight: 900;
    color: #facc15;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }
  .cat-subtitle { font-size: 0.75rem; color: #94a3b8; margin-top: 2px; }

  /* ── Status badges ── */
  .badge {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .badge-pass    { background: #052e16; color: #4ade80; border: 1px solid #16a34a; }
  .badge-monitor { background: #422006; color: #fb923c; border: 1px solid #ea580c; }
  .badge-alert   { background: #450a0a; color: #f87171; border: 1px solid #dc2626;
                   animation: pulse-red 1s ease-in-out infinite; }
  .badge-log     { background: #172554; color: #93c5fd; border: 1px solid #3b82f6; }
  .badge-warm    { background: #1a1a2e; color: #94a3b8; border: 1px solid #475569; }
  @keyframes pulse-red {
    0%,100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.4); }
    50%      { box-shadow: 0 0 0 8px rgba(239,68,68,0); }
  }

  /* ── Sensor cards ── */
  .sensor-card {
    background: #0f1929;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    transition: border-color 0.3s;
  }
  .sensor-card.warn { border-color: #dc2626; background: #1a0a0a; }
  .sensor-label { font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
  .sensor-value { font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 700; color: #e2e8f0; }
  .sensor-unit  { font-size: 0.7rem; color: #94a3b8; }

  /* ── Gauge container ── */
  .gauge-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; margin-bottom: 4px; }
  .gauge-value { font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 700; }
  .gauge-ok    { color: #4ade80; }
  .gauge-warn  { color: #fb923c; }
  .gauge-crit  { color: #f87171; }

  /* ── Alert panel ── */
  .alert-box {
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    background: #0f1929;
    border: 1px solid #1e3a5f;
  }
  .alert-box.red   { background: #1a0808; border-color: #dc2626; }
  .alert-box.orange{ background: #1a1008; border-color: #ea580c; }
  .alert-box.green { background: #080f12; border-color: #16a34a; }
  .alert-zone      { font-size: 2rem; font-weight: 900; }
  .alert-sub       { font-size: 0.75rem; color: #94a3b8; margin-top: 4px; word-break: break-word; }
  .alert-action    { font-size: 0.78rem; color: #e2e8f0; margin-top: 8px; padding-top: 8px;
                     border-top: 1px solid #1e293b; }

  /* ── Control panel ── */
  .control-section {
    background: #0d1526;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1rem 1.2rem;
  }
  .control-title { font-size: 0.75rem; font-weight: 600; color: #facc15;
                   text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; }

  /* ── Metric override ── */
  [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.2rem !important;
    color: #e2e8f0 !important;
  }
  [data-testid="stMetricLabel"] { color: #64748b !important; }

  /* scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #0a0e1a; }
  ::-webkit-scrollbar-thumb { background: #1e40af; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─── Session State Boot ────────────────────────────────────────────────────────

def _init_session():
    if "initialized" in st.session_state:
        return

    sim = EngineSimulator(initial_rpm=2000, initial_load=60)
    ml  = MLEngine()
    try:
        ml.load()
    except FileNotFoundError:
        pass   # physics-only mode

    out  = OutputManager(log_to_csv=True, print_to_console=False)
    pipe = DataPipeline()
    pe   = PhysicsEngine()

    # History deques
    HIST = 600   # 60 s at 10 samples/s
    hist = {
        "t":           deque(maxlen=HIST),
        "res_a":       deque(maxlen=HIST),
        "res_b":       deque(maxlen=HIST),
        "res_c":       deque(maxlen=HIST),
        "maf":         deque(maxlen=HIST),
        "map":         deque(maxlen=HIST),
        "ebp":         deque(maxlen=HIST),
        "confidence":  deque(maxlen=HIST),
    }

    shared = {"decision": FusionDecision(), "raw": {}, "filt": {},
              "res_a": 0.0, "res_b": 0.0, "res_c": 0.0,
              "running": True}
    lock = threading.Lock()

    def engine_loop():
        from physics_engine import ZoneResult
        step = 0
        last_ra = ZoneResult("A")
        last_rb = ZoneResult("B")
        last_rc = ZoneResult("C")
        while shared["running"]:
            t0  = time.perf_counter()
            raw = sim.step()
            pr  = pipe.process(raw)
            step += 1

            if step % 10 == 0:
                if pr.steady_state:
                    ra, rb, rc = pe.run(pr.filt, pr.raw)
                    ml_res     = ml.run(pr.filt) if ml.is_loaded else MLResult()
                    decision   = fuse(ra, rb, rc, ml_res, raw["timestamp"])
                    out.emit(decision)
                    last_ra, last_rb, last_rc = ra, rb, rc
                else:
                    decision = FusionDecision(status="WARMING_UP",
                                              timestamp=raw["timestamp"])

                with lock:
                    shared["decision"] = decision
                    shared["raw"]      = raw
                    shared["filt"]     = pr.filt
                    shared["res_a"]    = last_ra.residual_pct
                    shared["res_b"]    = last_rb.residual_pct
                    shared["res_c"]    = last_rc.residual_pct
                    hist["t"].append(raw["timestamp"])
                    hist["res_a"].append(last_ra.residual_pct)
                    hist["res_b"].append(last_rb.residual_pct)
                    hist["res_c"].append(last_rc.residual_pct)
                    hist["maf"].append(pr.filt.get("maf_gs", 0))
                    hist["map"].append(pr.filt.get("map_kpa", 0))
                    hist["ebp"].append(pr.filt.get("ebp_kpa", 0))
                    hist["confidence"].append(decision.confidence_pct)

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, 0.1 - elapsed))

    t = threading.Thread(target=engine_loop, daemon=True, name="DashEngine")
    t.start()

    st.session_state.initialized = True
    st.session_state.sim    = sim
    st.session_state.shared = shared
    st.session_state.lock   = lock
    st.session_state.hist   = hist
    st.session_state.ml_loaded = ml.is_loaded


_init_session()

sim    = st.session_state.sim
shared = st.session_state.shared
lock   = st.session_state.lock
hist   = st.session_state.hist

# ─── Helper: snapshot ─────────────────────────────────────────────────────────

def snap():
    with lock:
        d    = shared["decision"]
        raw  = dict(shared["raw"])
        filt = dict(shared["filt"])
        ra   = shared["res_a"]
        rb   = shared["res_b"]
        rc   = shared["res_c"]
        h    = {k: list(v) for k, v in hist.items()}
    return d, raw, filt, ra, rb, rc, h

# ─── Header ───────────────────────────────────────────────────────────────────

d, raw, filt, res_a, res_b, res_c, h = snap()

status = d.status or "WARMING_UP"
badge_class = {
    "PASS":       "badge-pass",
    "MONITOR":    "badge-monitor",
    "ALERT":      "badge-alert",
    "LOG":        "badge-log",
    "WARMING_UP": "badge-warm",
    "SUPPRESSED": "badge-warm",
}.get(status, "badge-warm")

ts_str = f"{d.timestamp:.1f} s" if d.timestamp else "—"

st.markdown(f"""
<div class="cat-header">
  <div>
    <div class="cat-title">🔧 Caterpillar C7 — Air Path Leak Detection System</div>
    <div class="cat-subtitle">Digital Twin · Physics Engine · ML Autoencoder · Fusion Layer</div>
  </div>
  <div style="text-align:right">
    <span class="badge {badge_class}">{status}</span>
    <div class="cat-subtitle" style="margin-top:6px">Sim time: {ts_str} &nbsp;|&nbsp;
      ML: {"✅ Loaded" if st.session_state.ml_loaded else "⚠ Physics-only"}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Main 3-column layout ─────────────────────────────────────────────────────
col_sensors, col_gauges, col_alert = st.columns([2, 2.2, 2], gap="medium")

# ── LEFT: Sensor readings ──────────────────────────────────────────────────────
with col_sensors:
    st.markdown('<div class="control-title">📡 Live Sensor Readings</div>',
                unsafe_allow_html=True)

    def sensor_card(label, value, unit, warn=False):
        cls = "sensor-card warn" if warn else "sensor-card"
        st.markdown(f"""
        <div class="{cls}">
          <div class="sensor-label">{label}</div>
          <span class="sensor-value">{value}</span>
          <span class="sensor-unit"> {unit}</span>
        </div>""", unsafe_allow_html=True)

    rpm_v   = filt.get("rpm",   raw.get("rpm",   0))
    load_v  = raw.get("load_pct", 0)
    maf_v   = filt.get("maf_gs", 0)
    map_v   = filt.get("map_kpa", 0)
    iat_v   = filt.get("iat_c", 0)
    boost_v = filt.get("boost_temp_c", 0)
    ic_v    = filt.get("intercooler_outlet_c", 0)
    ebp_v   = filt.get("ebp_kpa", 0)
    egt1_v  = filt.get("egt_1_c", 0)
    egt2_v  = filt.get("egt_2_c", 0)
    fuel_v  = filt.get("fuel_rate_gs", 0)

    sensor_card("Engine Speed (RPM)",      f"{rpm_v:,.0f}",  "RPM")
    sensor_card("Engine Load",             f"{load_v:.1f}",  "%")
    sensor_card("Mass Air Flow (MAF)",     f"{maf_v:.2f}",   "g/s",
                warn=(d.zone == "A" and d.leak_detected))
    sensor_card("Manifold Abs. Pressure",  f"{map_v:.2f}",   "kPa",
                warn=(d.zone == "B" and d.leak_detected))
    sensor_card("Intake Air Temp",         f"{iat_v:.1f}",   "°C")
    sensor_card("Boost Temp (post-Turbo)", f"{boost_v:.1f}", "°C")
    sensor_card("Intercooler Outlet Temp", f"{ic_v:.1f}",    "°C")
    sensor_card("Exhaust Back Pressure",   f"{ebp_v:.2f}",   "kPa",
                warn=(d.zone == "C" and d.leak_detected))
    sensor_card("EGT Bank 1",             f"{egt1_v:.1f}",  "°C")
    sensor_card("EGT Bank 2",             f"{egt2_v:.1f}",  "°C")
    sensor_card("Fuel Rate",              f"{fuel_v:.2f}",  "g/s")

# ── CENTRE: Residual Gauges ────────────────────────────────────────────────────
with col_gauges:
    st.markdown('<div class="control-title">📊 Zone Residuals vs Threshold</div>',
                unsafe_allow_html=True)

    THRESH = {"A": 8.0, "B": 6.0, "C": 12.0}

    def residual_gauge(zone_name, residual, threshold, sensor_label):
        pct    = min(100, abs(residual) / threshold * 100)
        colour = ("#4ade80" if pct < 50 else
                  "#fb923c" if pct < 85 else "#f87171")
        cls    = ("gauge-ok" if pct < 50 else
                  "gauge-warn" if pct < 85 else "gauge-crit")
        sign   = "+" if residual > 0 else ""
        st.markdown(f"""
        <div class="sensor-card" style="margin-bottom:0.8rem">
          <div class="gauge-label">Zone {zone_name} — {sensor_label}</div>
          <div class="{cls}" style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;font-weight:700">
            {sign}{residual:.1f}%
          </div>
          <div style="background:#1e293b;border-radius:999px;height:8px;margin:6px 0">
            <div style="background:{colour};width:{pct:.1f}%;height:8px;
                        border-radius:999px;transition:width 0.4s ease"></div>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:0.65rem;color:#475569">
            <span>0%</span><span>Threshold {threshold}%</span><span>100%</span>
          </div>
        </div>""", unsafe_allow_html=True)

    residual_gauge("A", res_a, THRESH["A"], "MAF residual (drop)")
    residual_gauge("B", res_b, THRESH["B"], "MAP residual (drop)")
    residual_gauge("C", res_c, THRESH["C"], "EBP residual (rise)")

    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="control-title">🤖 ML Engine</div>',
                unsafe_allow_html=True)

    ml_err = d.ml_recon_error or 0.0
    ml_f   = "✅ Normal" if not d.ml_flag else f"⚠ Anomaly — {d.ml_worst_feature}"
    ml_z   = f"Zone {d.anomaly_zone if hasattr(d,'anomaly_zone') else '?'}" if d.ml_flag else "—"
    st.markdown(f"""
    <div class="sensor-card">
      <div class="gauge-label">Reconstruction Error</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:1.2rem;
                  color:{'#f87171' if d.ml_flag else '#4ade80'};font-weight:700">
        {ml_err:.5f}
      </div>
      <div style="font-size:0.78rem;color:#94a3b8;margin-top:4px">{ml_f}</div>
    </div>""", unsafe_allow_html=True)

    # Fusion scores
    st.markdown(f"""
    <div class="sensor-card" style="margin-top:0.5rem">
      <div class="gauge-label">Fusion Scores (Physics 60% | ML 40%)</div>
      <div style="display:flex;gap:1rem;margin-top:6px">
        <div>
          <div style="font-size:0.65rem;color:#64748b">PHYSICS</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;
                      font-weight:700;color:#facc15">{d.physics_score:.1f}</div>
        </div>
        <div>
          <div style="font-size:0.65rem;color:#64748b">ML</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;
                      font-weight:700;color:#facc15">{d.ml_score:.1f}</div>
        </div>
        <div>
          <div style="font-size:0.65rem;color:#64748b">FUSED</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:1.2rem;
                      font-weight:700;color:#e2e8f0">{d.fused_score:.1f}</div>
        </div>
        <div>
          <div style="font-size:0.65rem;color:#64748b">CONFIDENCE</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:1.2rem;
                      font-weight:700;color:{'#f87171' if d.confidence_pct>70 else '#4ade80'}">
            {d.confidence_pct:.1f}%</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

# ── RIGHT: Alert panel ─────────────────────────────────────────────────────────
with col_alert:
    st.markdown('<div class="control-title">🚨 Detection Output</div>',
                unsafe_allow_html=True)

    if d.leak_detected:
        colour_cls = "red" if status == "ALERT" else "orange"
        zone_emoji = {"A": "💨", "B": "💨", "C": "🔥"}.get(d.zone, "⚠")
        st.markdown(f"""
        <div class="alert-box {colour_cls}">
          <div class="alert-zone">{zone_emoji} ZONE {d.zone} LEAK</div>
          <div style="font-size:0.8rem;color:#94a3b8;margin-top:2px">
            Status: <b style="color:#e2e8f0">{status}</b> &nbsp;|&nbsp;
            Triggered by: <b style="color:#e2e8f0">{d.triggered_by.upper()}</b>
          </div>
          <div class="alert-sub">📍 {d.sub_location}</div>
          <div class="alert-action">🔧 {d.action}</div>
        </div>""", unsafe_allow_html=True)

        # Confidence bar
        conf = d.confidence_pct
        cbar_col = "#dc2626" if conf > 80 else "#ea580c" if conf > 50 else "#facc15"
        st.markdown(f"""
        <div class="sensor-card">
          <div class="gauge-label">Confidence</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:1.8rem;
                      font-weight:900;color:{cbar_col}">{conf:.1f}%</div>
          <div style="background:#1e293b;border-radius:999px;height:10px;margin-top:6px">
            <div style="background:{cbar_col};width:{conf:.1f}%;height:10px;
                        border-radius:999px"></div>
          </div>
        </div>""", unsafe_allow_html=True)

        # Evidence
        st.markdown(f"""
        <div class="sensor-card">
          <div class="gauge-label">Evidence</div>
          <div style="font-size:0.8rem;line-height:1.8">
            <b>Sensor:</b> {d.sensor_name}<br>
            <b>Expected:</b> {d.expected_value:.3f}<br>
            <b>Actual:</b>   {d.actual_value:.3f}<br>
            <b>Residual:</b> <span style="color:#f87171">{d.residual_pct:+.1f}%</span>
            {"<br><b style='color:#fb923c'>⚠ DRIFT detected</b>" if d.drift else ""}
          </div>
        </div>""", unsafe_allow_html=True)

    else:
        icon = "⏳" if status == "WARMING_UP" else "✅"
        msg  = "Warming up..." if status == "WARMING_UP" else "System Nominal"
        st.markdown(f"""
        <div class="alert-box green">
          <div class="alert-zone">{icon} {msg}</div>
          <div class="alert-sub">All residuals within normal bounds.<br>
            No leaks detected in any zone.</div>
        </div>""", unsafe_allow_html=True)

    # DPF / EGR status
    dpf_on = bool(raw.get("dpf_regen", 0))
    egr_p  = raw.get("egr_pct", 15)
    cool   = raw.get("coolant_temp_c", 88)
    st.markdown(f"""
    <div class="sensor-card" style="margin-top:0.5rem">
      <div class="gauge-label">ECU Status Flags</div>
      <div style="font-size:0.8rem;line-height:1.9">
        DPF Regen: <b style="color:{'#fb923c' if dpf_on else '#4ade80'}">
          {'ACTIVE' if dpf_on else 'OFF'}</b><br>
        EGR Position: <b style="color:#e2e8f0">{egr_p:.1f}%</b><br>
        Coolant Temp: <b style="color:#e2e8f0">{cool:.1f} °C</b>
      </div>
    </div>""", unsafe_allow_html=True)

# ─── History Chart ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="control-title">📈 60-Second Residual History</div>',
            unsafe_allow_html=True)

if len(h["t"]) > 5:
    df_hist = pd.DataFrame({
        "Time (s)": h["t"],
        "Zone A residual (%)": h["res_a"],
        "Zone B residual (%)": h["res_b"],
        "Zone C residual (%)": h["res_c"],
    }).set_index("Time (s)")

    # Keep last 60 s
    if df_hist.index[-1] - df_hist.index[0] > 60:
        df_hist = df_hist[df_hist.index >= df_hist.index[-1] - 60]

    st.line_chart(df_hist, height=220, use_container_width=True)
else:
    st.info("History will appear after warm-up (~10 s).")

# ─── Controls ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="control-title">🎛️ Simulation Controls</div>',
            unsafe_allow_html=True)

ctrl_l, ctrl_m, ctrl_r = st.columns([2, 2, 3], gap="medium")

with ctrl_l:
    st.markdown('<div class="control-section">', unsafe_allow_html=True)
    st.markdown("**Engine Operating Point**")
    new_rpm  = st.slider("RPM",      600, 3000, int(sim.state.rpm),  step=50)
    new_load = st.slider("Load (%)", 0,   100,  int(sim.state.load_pct), step=5)
    if st.button("Apply", key="apply_op"):
        sim.set_operating_point(new_rpm, new_load)
    st.markdown('</div>', unsafe_allow_html=True)

with ctrl_m:
    st.markdown('<div class="control-section">', unsafe_allow_html=True)
    st.markdown("**ECU Flags**")
    dpf_toggle = st.toggle("DPF Regen Active", value=False, key="dpf_tog")
    egr_slider = st.slider("EGR Position (%)", 0, 60, 15, key="egr_sl")
    if st.button("Apply Flags", key="apply_flags"):
        sim.set_engine_flags(
            dpf_regen=dpf_toggle,
            egr_pct=egr_slider,
            coolant_temp_c=sim.state.coolant_temp_c,
        )
    st.markdown('</div>', unsafe_allow_html=True)

with ctrl_r:
    st.markdown('<div class="control-section">', unsafe_allow_html=True)
    st.markdown("**Leak Injection**")
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        sev_a = st.slider("Zone A", 0.0, 1.0, 0.0, 0.05, key="sev_a",
                          help="MAF reduction fraction")
        if st.button("Inject A", key="inj_a",
                     type="primary" if sev_a > 0 else "secondary"):
            sim.inject_leak("A", sev_a)
    with lc2:
        sev_b = st.slider("Zone B", 0.0, 1.0, 0.0, 0.05, key="sev_b",
                          help="MAP reduction fraction")
        loc_b = st.selectbox("Location", ["after_intercooler", "before_intercooler"],
                             key="loc_b")
        if st.button("Inject B", key="inj_b",
                     type="primary" if sev_b > 0 else "secondary"):
            sim.inject_leak("B", sev_b, b_location=loc_b)
    with lc3:
        sev_c = st.slider("Zone C", 0.0, 1.0, 0.0, 0.05, key="sev_c",
                          help="EBP increase fraction")
        bank_c = st.selectbox("Bank", ["upstream", "downstream"], key="bank_c")
        if st.button("Inject C", key="inj_c",
                     type="primary" if sev_c > 0 else "secondary"):
            sim.inject_leak("C", sev_c, c_bank=bank_c)

    if st.button("🔴 CLEAR ALL LEAKS", key="clear_all", use_container_width=True):
        sim.clear_leak()
    st.markdown('</div>', unsafe_allow_html=True)

# ─── Auto-refresh ─────────────────────────────────────────────────────────────
time.sleep(1)
st.rerun()

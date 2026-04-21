"""
ml_engine.py
------------
Autoencoder-based anomaly detection on 7 ratio features.

OFFLINE (train):
  Run simulator for 30 min healthy → extract features → train autoencoder
  → save model weights + anomaly threshold.

ONLINE (runtime):
  Load saved model. Every second compute 7 ratio features → reconstruction
  error → anomaly flag + worst-feature-to-zone mapping.

Uses scikit-learn's MLPRegressor as a lightweight autoencoder substitute
(encodes then decodes through a bottleneck). No CUDA needed.
"""

import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# ─── Feature Engineering ──────────────────────────────────────────────────────
#
# 7 ratio features — each is dimensionless and physically meaningful
#
#   F1  maf_gs / rpm           — air-per-revolution (intake efficiency)
#   F2  map_kpa / maf_gs       — manifold pressure per unit airflow (boost quality)
#   F3  ebp_kpa / fuel_rate_gs — back-pressure per unit fuel (exhaust restriction)
#   F4  boost_temp_c / map_kpa — charge temp per unit pressure (intercooler health)
#   F5  egt_1_c / rpm          — exhaust energy per revolution (combustion quality)
#   F6  egt_2_c / rpm          — same for bank 2
#   F7  maf_gs / map_kpa       — air-mass per pressure unit (volumetric efficiency proxy)
#
FEATURE_NAMES = [
    "maf_per_rpm",
    "map_per_maf",
    "ebp_per_fuel",
    "boost_temp_per_map",
    "egt1_per_rpm",
    "egt2_per_rpm",
    "maf_per_map",
]

# Zone association for each feature (for anomaly attribution)
FEATURE_ZONE = {
    "maf_per_rpm":        "A",
    "map_per_maf":        "B",
    "ebp_per_fuel":       "C",
    "boost_temp_per_map": "B",
    "egt1_per_rpm":       "C",
    "egt2_per_rpm":       "C",
    "maf_per_map":        "A",
}

# Model persistence paths
MODEL_DIR  = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "autoencoder.pkl"
SCALER_PATH= MODEL_DIR / "scaler.pkl"
THRESH_PATH= MODEL_DIR / "threshold.pkl"

# Anomaly threshold: set at 99th percentile of training reconstruction errors
THRESHOLD_PERCENTILE = 99

# MLP architecture — shallow encoder-decoder through a 3-node bottleneck
ENCODER_LAYERS = (7, 5, 3)
DECODER_LAYERS = (3, 5, 7)
HIDDEN_LAYERS  = ENCODER_LAYERS[1:] + DECODER_LAYERS[1:]   # (5, 3, 5)


@dataclass
class MLResult:
    flag:            bool  = False
    reconstruction_error: float = 0.0
    threshold:       float = 0.0
    anomaly_zone:    str   = "unknown"
    worst_feature:   str   = "none"
    worst_feature_error: float = 0.0
    confidence:      float = 0.0


def _extract_features(row: dict) -> Optional[np.ndarray]:
    """
    Extract the 7 ratio features from one sensor dict.
    Returns None if any denominator is zero / data is invalid.
    """
    try:
        rpm       = float(row.get("rpm",             0))
        maf       = float(row.get("maf_gs",          0))
        map_kpa   = float(row.get("map_kpa",         0))
        ebp       = float(row.get("ebp_kpa",         0))
        fuel      = float(row.get("fuel_rate_gs",    0))
        boost_t   = float(row.get("boost_temp_c",    0))
        egt1      = float(row.get("egt_1_c",         0))
        egt2      = float(row.get("egt_2_c",         0))

        if any(v <= 0 for v in [rpm, maf, map_kpa, fuel]):
            return None

        features = np.array([
            maf    / rpm,
            map_kpa / maf,
            ebp    / fuel,
            boost_t / map_kpa,
            egt1   / rpm,
            egt2   / rpm,
            maf    / map_kpa,
        ], dtype=np.float32)

        if np.any(~np.isfinite(features)):
            return None

        return features

    except (ValueError, ZeroDivisionError, TypeError):
        return None


def _build_model() -> MLPRegressor:
    """Return a fresh untrained MLPRegressor acting as an autoencoder."""
    return MLPRegressor(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation="tanh",
        solver="adam",
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.05,
        n_iter_no_change=20,
        verbose=False,
    )


def train_autoencoder(df: pd.DataFrame, save: bool = True) -> Tuple[float, "MLRegressor"]:
    """
    Train the autoencoder on healthy simulator data.

    Parameters
    ----------
    df   : DataFrame with raw sensor columns (from simulator healthy run)
    save : persist model + scaler + threshold to MODEL_DIR

    Returns
    -------
    threshold, model
    """
    # 1. Extract features
    feats = []
    for _, row in df.iterrows():
        f = _extract_features(row.to_dict())
        if f is not None:
            feats.append(f)

    if len(feats) < 50:
        raise ValueError(f"Too few valid feature rows: {len(feats)}")

    X = np.stack(feats)
    print(f"[ML] Training on {len(X)} healthy samples, {X.shape[1]} features.")

    # 2. Scale
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # 3. Train autoencoder (input == target → learns identity through bottleneck)
    model = _build_model()
    model.fit(Xs, Xs)

    # 4. Compute reconstruction errors on training set
    Xs_hat = model.predict(Xs)
    errors_per_sample = np.mean((Xs - Xs_hat) ** 2, axis=1)
    threshold = float(np.percentile(errors_per_sample, THRESHOLD_PERCENTILE))
    print(f"[ML] Anomaly threshold (p{THRESHOLD_PERCENTILE}): {threshold:.6f}")

    # 5. Persist
    if save:
        MODEL_DIR.mkdir(exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)
        with open(THRESH_PATH, "wb") as f:
            pickle.dump(threshold, f)
        print(f"[ML] Model saved to {MODEL_DIR}")

    return threshold, model


class MLEngine:
    """
    Runtime inference engine.
    Call run(row_dict) every second to get an MLResult.
    """

    def __init__(self):
        self._model:    Optional[MLPRegressor]   = None
        self._scaler:   Optional[StandardScaler] = None
        self._threshold: float                   = 0.01
        self._loaded = False

    def load(self):
        """Load persisted model, scaler, and threshold."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}.\n"
                "Run: python ml_engine.py --train"
            )
        with open(MODEL_PATH, "rb") as f:
            self._model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            self._scaler = pickle.load(f)
        with open(THRESH_PATH, "rb") as f:
            self._threshold = pickle.load(f)
        self._loaded = True
        print(f"[ML] Model loaded. Threshold = {self._threshold:.6f}")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def run(self, row: dict) -> MLResult:
        """
        Score one sensor row.
        row: filtered sensor dict from DataPipeline.
        """
        result = MLResult(threshold=self._threshold)

        if not self._loaded:
            return result

        feats = _extract_features(row)
        if feats is None:
            return result

        # Scale → predict → compute per-feature error
        Xs     = self._scaler.transform(feats.reshape(1, -1))
        Xs_hat = self._model.predict(Xs)
        per_feature_err = (Xs[0] - Xs_hat[0]) ** 2
        total_err       = float(np.mean(per_feature_err))

        # Find worst reconstructed feature
        worst_idx     = int(np.argmax(per_feature_err))
        worst_name    = FEATURE_NAMES[worst_idx]
        worst_err     = float(per_feature_err[worst_idx])

        # Confidence: how far above threshold
        ratio = total_err / max(self._threshold, 1e-9)
        conf  = float(np.clip(100.0 * (ratio ** 0.5) * 0.6, 0, 100))

        result.reconstruction_error = round(total_err, 6)
        result.worst_feature        = worst_name
        result.worst_feature_error  = round(worst_err, 6)
        result.anomaly_zone         = FEATURE_ZONE.get(worst_name, "unknown")
        result.confidence           = round(conf, 1)

        if total_err > self._threshold:
            result.flag = True

        return result


# ─── CLI: python ml_engine.py --train ─────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if "--train" in sys.argv:
        from simulator import EngineSimulator
        print("[ML] Generating robust full-envelope healthy dataset...")
        sim = EngineSimulator(2000, 60)
        # Vary RPM and load in a complete grid to map all edge cases
        rows = []
        rpm_targets = [800, 1300, 1800, 2300, 2800, 3000]
        load_targets = [10, 35, 60, 85, 100]
        
        for rpm in rpm_targets:
            for load in load_targets:
                sim.set_operating_point(rpm, load)
                df_part = sim.run_batch(duration_s=120.0)  # 2 min per combination
                rows.append(df_part)
                
        df_all = pd.concat(rows, ignore_index=True)
        print(f"[ML] Dataset: {len(df_all)} rows (Full Grid Coverage)")
        threshold, _ = train_autoencoder(df_all, save=True)
        print(f"[ML] Training complete. Threshold = {threshold:.6f}")
    else:
        print("Usage: python ml_engine.py --train")
        print("Then use MLEngine().load() at runtime.")

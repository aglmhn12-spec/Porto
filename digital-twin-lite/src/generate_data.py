# src/generate_data.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import SeparatorParams, SimConfig
from src.sim_model import simulate_separator_open_loop


@dataclass
class SensorNoiseConfig:
    # Noise standar deviasi (untuk demo)
    sigma_L: float = 0.01        # meter (misal sensor level noise ~1 cm)
    sigma_P: float = 300.0       # Pa (misal pressure noise kecil)

    # Missing data probability per sample (0.0–1.0)
    p_missing_L: float = 0.01
    p_missing_P: float = 0.01

    # Drift (opsional): penambahan bias pelan-pelan
    # drift_rate artinya bertambah per detik
    drift_L_rate: float = 0.0    # m/s
    drift_P_rate: float = 0.0    # Pa/s


def ensure_project_root() -> Path:
    """
    Mengunci root directory ke folder 'digital-twin-lite'.
    - Jika script dijalankan dari subfolder (misal notebooks/), fungsi ini tetap mencari root.
    """
    cwd = Path.cwd().resolve()
    # Cari folder bernama 'digital-twin-lite' di rantai parent
    for p in [cwd] + list(cwd.parents):
        if p.name == "digital-twin-lite":
            return p
    # fallback: kalau tidak ketemu, anggap cwd adalah root (tapi ini kurang ideal)
    return cwd


def make_dirs(root: Path) -> None:
    (root / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (root / "notebooks").mkdir(parents=True, exist_ok=True)
    (root / "src").mkdir(parents=True, exist_ok=True)


def add_sensor_effects(
    df_true: pd.DataFrame,
    noise: SensorNoiseConfig,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Mengubah 'true signals' menjadi 'measured signals' dengan:
    - Gaussian noise
    - missing values
    - drift (opsional)
    """
    rng = np.random.default_rng(seed)

    df = df_true.copy()

    t = df["t_s"].to_numpy(dtype=float)

    # Drift (bias pelan)
    drift_L = noise.drift_L_rate * t
    drift_P = noise.drift_P_rate * t

    # Noise random (Gaussian)
    eps_L = rng.normal(0.0, noise.sigma_L, size=len(df))
    eps_P = rng.normal(0.0, noise.sigma_P, size=len(df))

    # True signals
    L_true = df["L_m"].to_numpy(dtype=float)
    P_true = df["P_Pa"].to_numpy(dtype=float)

    # Measured signals
    L_meas = L_true + drift_L + eps_L
    P_meas = P_true + drift_P + eps_P

    # Missing masks
    miss_L = rng.random(len(df)) < noise.p_missing_L
    miss_P = rng.random(len(df)) < noise.p_missing_P

    L_meas = L_meas.astype(float)
    P_meas = P_meas.astype(float)
    L_meas[miss_L] = np.nan
    P_meas[miss_P] = np.nan

    # Simpan kolom rapi
    df["L_true_m"] = L_true
    df["P_true_Pa"] = P_true
    df["L_meas_m"] = L_meas
    df["P_meas_Pa"] = P_meas
    df["L_missing"] = miss_L
    df["P_missing"] = miss_P

    # Optional: drop kolom internal kalau mau, tapi untuk sekarang kita biarkan lengkap
    return df


def generate_sensor_log_csv(
    out_relpath: str = "data/raw/sensor_log.csv",
    seed: int = 42,
    make_folders: bool = True,
) -> Path:
    """
    1) Simulasi 'true' separator (open-loop)
    2) Tambahkan noise + missing + drift
    3) Simpan ke CSV di data/raw/
    """
    root = ensure_project_root()
    if make_folders:
        make_dirs(root)

    # === Parameter & config simulasi (samakan dengan Step 1) ===
    params = SeparatorParams(
        A=3.0,
        V_total=12.0,
        L_max=3.0,
        V_min=0.2,
        rho_l=850.0,
        g=9.81,
        Patm=101_325.0,
        P_out=101_325.0,
        Cv_l=2.5e-5,
        R=8.314,
        T=320.0,
        kg_n_sqrt=0.08,
        n_out_max=30.0,
    )
    sim = SimConfig(dt=0.5, t_end=600.0)

    # === Input functions (open-loop) ===
    def qin_func(t):
        return 0.004 + (0.001 if t > 200 else 0.0)

    def n_in_func(t):
        return 4.0 + (1.5 if t > 300 else 0.0)

    def u_func(t):
        return 0.55

    df_true = simulate_separator_open_loop(
        params=params,
        sim=sim,
        qin_func=qin_func,
        n_in_func=n_in_func,
        u_func=u_func,
        L0=1.0,
        n0=400.0,  # kamu sudah set ini, bagus
    )

    # === Noise config (boleh kamu tune) ===
    noise = SensorNoiseConfig(
        sigma_L=0.01,
        sigma_P=300.0,
        p_missing_L=0.01,
        p_missing_P=0.01,
        drift_L_rate=0.0,
        drift_P_rate=0.0,
    )

    df_log = add_sensor_effects(df_true, noise=noise, seed=seed)

    out_path = (root / out_relpath).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_log.to_csv(out_path, index=False)
    return out_path
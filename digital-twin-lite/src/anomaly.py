# src/anomaly.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class AnomalyConfig:
    # Rolling window (jumlah sample). dt=0.5s, 60 sample ~ 30 detik
    window: int = 60

    # Ambang z-score
    z_thresh_L: float = 3.0
    z_thresh_P: float = 3.0

    # CUSUM threshold (untuk shift kecil tapi konsisten)
    cusum_k: float = 0.5          # "reference value" dalam satuan sigma
    cusum_h: float = 8.0          # threshold

    # Berapa lama harus berturut-turut untuk dianggap alarm
    consecutive: int = 10


def ensure_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for p in [cwd] + list(cwd.parents):
        if p.name == "digital-twin-lite":
            return p
    return cwd


def make_dirs(root: Path) -> None:
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)


def rolling_zscore(x: pd.Series, window: int) -> pd.Series:
    """
    Z-score rolling: (x - mean)/std, dihitung rolling window.
    Cocok untuk residual EKF yang idealnya ~0.
    """
    mu = x.rolling(window=window, min_periods=max(10, window // 3)).mean()
    sd = x.rolling(window=window, min_periods=max(10, window // 3)).std()
    z = (x - mu) / sd
    return z


def cusum_score(z: pd.Series, k: float, h: float) -> pd.DataFrame:
    """
    CUSUM pada z-score:
    - deteksi shift mean kecil yang konsisten
    - output: C+ dan C-
    """
    z = z.fillna(0.0).to_numpy(dtype=float)
    cp = np.zeros_like(z)
    cn = np.zeros_like(z)

    for i in range(1, len(z)):
        cp[i] = max(0.0, cp[i - 1] + (z[i] - k))
        cn[i] = min(0.0, cn[i - 1] + (z[i] + k))

    alarm = (cp > h) | (cn < -h)
    return pd.DataFrame({"cusum_pos": cp, "cusum_neg": cn, "cusum_alarm": alarm})


def consecutive_alarm(flag: pd.Series, consecutive: int) -> pd.Series:
    """
    Alarm hanya jika flag True terjadi berturut-turut minimal N kali.
    Mengurangi false alarm dari spike sesaat.
    """
    flag = flag.fillna(False).to_numpy(dtype=bool)
    out = np.zeros_like(flag, dtype=bool)

    count = 0
    for i, f in enumerate(flag):
        if f:
            count += 1
        else:
            count = 0
        out[i] = count >= consecutive
    return pd.Series(out)


def detect_anomaly_from_ekf(df: pd.DataFrame, cfg: AnomalyConfig = AnomalyConfig()) -> pd.DataFrame:
    """
    Input df: hasil EKF yang punya kolom res_L dan res_P.
    Output df: tambah kolom z-score, cusum, dan flags.
    """
    out = df.copy()

    # Residual: gunakan res_* (innovation). Bisa juga pakai |res| jika mau.
    res_L = out["res_L"].astype(float)
    res_P = out["res_P"].astype(float)

    # Rolling z-score
    out["z_L"] = rolling_zscore(res_L, window=cfg.window)
    out["z_P"] = rolling_zscore(res_P, window=cfg.window)

    # Simple threshold flags
    out["flag_z_L"] = out["z_L"].abs() > cfg.z_thresh_L
    out["flag_z_P"] = out["z_P"].abs() > cfg.z_thresh_P

    # CUSUM on z-score (lebih robust untuk shift halus)
    cus_L = cusum_score(out["z_L"], k=cfg.cusum_k, h=cfg.cusum_h)
    cus_P = cusum_score(out["z_P"], k=cfg.cusum_k, h=cfg.cusum_h)

    out["cusum_L_pos"] = cus_L["cusum_pos"]
    out["cusum_L_neg"] = cus_L["cusum_neg"]
    out["flag_cusum_L"] = cus_L["cusum_alarm"]

    out["cusum_P_pos"] = cus_P["cusum_pos"]
    out["cusum_P_neg"] = cus_P["cusum_neg"]
    out["flag_cusum_P"] = cus_P["cusum_alarm"]

    # Gabungkan flags + consecutive rule
    raw_flag = out["flag_z_L"] | out["flag_z_P"] | out["flag_cusum_L"] | out["flag_cusum_P"]
    out["flag_raw"] = raw_flag

    out["alarm"] = consecutive_alarm(raw_flag, consecutive=cfg.consecutive)

    # Timestamp alarm pertama (opsional)
    if out["alarm"].any():
        first_idx = int(np.where(out["alarm"].to_numpy(dtype=bool))[0][0])
        out["alarm_first_time_s"] = np.nan
        out.loc[first_idx:, "alarm_first_time_s"] = out.loc[first_idx, "t_s"]
    else:
        out["alarm_first_time_s"] = np.nan

    return out


def save_anomaly_result(root: Path, df_out: pd.DataFrame, filename: str = "data/processed/anomaly_result.csv") -> Path:
    make_dirs(root)
    out_path = (root / filename).resolve()
    df_out.to_csv(out_path, index=False)
    return out_path
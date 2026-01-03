# src/calibrate.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from src.config import SeparatorParams, SimConfig
from src.sim_model import simulate_separator_open_loop


@dataclass
class CalibConfig:
    sigma_L: float = 0.01     # meter
    sigma_P: float = 300.0    # Pa


def ensure_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for p in [cwd] + list(cwd.parents):
        if p.name == "digital-twin-lite":
            return p
    return cwd


def make_dirs(root: Path) -> None:
    (root / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)


def _interp_func(t_arr: np.ndarray, y_arr: np.ndarray):
    """
    Membuat fungsi y(t) dengan interpolasi linear dari array logged.
    """
    t_arr = np.asarray(t_arr, dtype=float)
    y_arr = np.asarray(y_arr, dtype=float)

    def f(t: float) -> float:
        return float(np.interp(t, t_arr, y_arr))

    return f


def _initial_n_from_first_measurement(
    params: SeparatorParams,
    L0: float,
    P0: float,
) -> float:
    """
    Jika kita punya P0 (Pa) dan L0 (m), kita bisa hitung n0 dari ideal gas:
    n0 = P0*Vg(L0)/(R*T)
    """
    Vg0 = params.gas_volume(L0)
    n0 = (P0 * Vg0) / (params.R * params.T)
    return float(max(n0, 1e-9))


def simulate_from_log(
    df_log: pd.DataFrame,
    params: SeparatorParams,
    sim: SimConfig,
) -> pd.DataFrame:
    """
    Simulasi memakai input dari data log: qin, n_in, u terhadap waktu.
    """
    t = df_log["t_s"].to_numpy(dtype=float)

    qin_func = _interp_func(t, df_log["qin_m3s"].to_numpy(dtype=float))
    n_in_func = _interp_func(t, df_log["n_in_mols"].to_numpy(dtype=float))
    u_func = _interp_func(t, df_log["u"].to_numpy(dtype=float))

    # initial state: ambil dari pengukuran pertama yang tidak missing
    # Level: pakai L_meas pertama yang valid; kalau tidak ada, pakai 1.0
    L_meas = df_log["L_meas_m"].to_numpy(dtype=float)
    P_meas = df_log["P_meas_Pa"].to_numpy(dtype=float)

    idx_L = np.where(~np.isnan(L_meas))[0]
    idx_P = np.where(~np.isnan(P_meas))[0]

    L0 = float(L_meas[idx_L[0]]) if len(idx_L) else 1.0
    P0 = float(P_meas[idx_P[0]]) if len(idx_P) else params.Patm * 1.1

    # hitung n0 dari P0 dan L0 (lebih konsisten daripada menebak n0)
    n0 = _initial_n_from_first_measurement(params=params, L0=L0, P0=P0)

    df_pred = simulate_separator_open_loop(
        params=params,
        sim=sim,
        qin_func=qin_func,
        n_in_func=n_in_func,
        u_func=u_func,
        L0=L0,
        n0=n0,
    )
    return df_pred


def build_params_template() -> SeparatorParams:
    """
    Template parameter yang FIX kecuali yang akan dikalibrasi.
    Samakan dengan Step 1/2 agar konsisten.
    """
    return SeparatorParams(
        A=3.0,
        V_total=12.0,
        L_max=3.0,
        V_min=0.2,
        rho_l=850.0,
        g=9.81,
        Patm=101_325.0,
        P_out=101_325.0,
        Cv_l=2.5e-5,          # akan dikalibrasi
        R=8.314,
        T=320.0,
        kg_n_sqrt=0.08,       # akan dikalibrasi
        n_out_max=30.0,
    )


def calibrate_params(
    df_log: pd.DataFrame,
    sim: SimConfig,
    x0: np.ndarray | None = None,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    calib_cfg: CalibConfig = CalibConfig(),
) -> dict:
    """
    Kalibrasi [Cv_l, kg_n_sqrt] dengan least squares.
    Return dict berisi parameter hasil, success flag, dan df_pred sebelum/sesudah.
    """

    base = build_params_template()

    # initial guess
    if x0 is None:
        x0 = np.array([base.Cv_l, base.kg_n_sqrt], dtype=float)

    # bounds: parameter fisik harus non-negatif
    if bounds is None:
        lb = np.array([1e-8, 1e-4], dtype=float)   # batas bawah
        ub = np.array([1e-3, 5.0], dtype=float)    # batas atas (demo)
        bounds = (lb, ub)

    # masks data valid
    L_meas = df_log["L_meas_m"].to_numpy(dtype=float)
    P_meas = df_log["P_meas_Pa"].to_numpy(dtype=float)

    mask_L = ~np.isnan(L_meas)
    mask_P = ~np.isnan(P_meas)

    # Precompute weights
    wL = 1.0 / float(calib_cfg.sigma_L)
    wP = 1.0 / float(calib_cfg.sigma_P)

    # Baseline prediction (before)
    df_before = simulate_from_log(df_log, base, sim)

    def residuals(x: np.ndarray) -> np.ndarray:
        Cv_l, kg_n_sqrt = float(x[0]), float(x[1])

        params = base.model_copy(update={"Cv_l": Cv_l, "kg_n_sqrt": kg_n_sqrt})

        df_pred = simulate_from_log(df_log, params, sim)

        L_pred = df_pred["L_m"].to_numpy(dtype=float)
        P_pred = df_pred["P_Pa"].to_numpy(dtype=float)

        # residual vector (stack): only where measurements exist
        rL = (L_meas[mask_L] - L_pred[mask_L]) * wL
        rP = (P_meas[mask_P] - P_pred[mask_P]) * wP

        return np.concatenate([rL, rP])

    res = least_squares(
        residuals,
        x0=x0,
        bounds=bounds,
        method="trf",
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        max_nfev=50,  # cukup untuk pemula; bisa dinaikkan nanti
    )

    best_params = base.model_copy(update={"Cv_l": float(res.x[0]), "kg_n_sqrt": float(res.x[1])})
    df_after = simulate_from_log(df_log, best_params, sim)

    return {
        "success": bool(res.success),
        "message": str(res.message),
        "Cv_l": float(res.x[0]),
        "kg_n_sqrt": float(res.x[1]),
        "cost": float(res.cost),
        "nfev": int(res.nfev),
        "df_before": df_before,
        "df_after": df_after,
    }


def save_calibration_result(root: Path, result: dict, filename: str = "data/processed/calibration_result.csv") -> Path:
    """
    Simpan ringkasan hasil kalibrasi (parameter utama) ke CSV.
    """
    make_dirs(root)
    out_path = (root / filename).resolve()

    summary = pd.DataFrame(
        [{
            "Cv_l": result["Cv_l"],
            "kg_n_sqrt": result["kg_n_sqrt"],
            "cost": result["cost"],
            "nfev": result["nfev"],
            "success": result["success"],
            "message": result["message"],
        }]
    )
    summary.to_csv(out_path, index=False)
    return out_path
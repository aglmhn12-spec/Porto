# src/state_estimation.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import SeparatorParams, SimConfig


@dataclass
class EKFConfig:
    # Noise proses (seberapa “percaya” model)
    q_L: float = 1e-6     # var untuk L
    q_n: float = 1e-2     # var untuk n

    # Noise measurement (seberapa “percaya” sensor)
    r_L: float = 0.01**2      # var level sensor (m^2)
    r_P: float = 300.0**2     # var pressure sensor (Pa^2)


def ensure_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for p in [cwd] + list(cwd.parents):
        if p.name == "digital-twin-lite":
            return p
    return cwd


def make_dirs(root: Path) -> None:
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)


def compute_pressure(params: SeparatorParams, L: float, n: float) -> float:
    Vg = params.gas_volume(L)
    P = (n * params.R * params.T) / Vg
    return float(max(P, params.Patm))


def f_step(
    params: SeparatorParams,
    dt: float,
    x: np.ndarray,
    qin: float,      # m^3/s
    n_in: float,     # mol/s
    u: float,        # 0..1
) -> np.ndarray:
    """
    Model transisi state: x_{k+1} = f(x_k, u_k)
    x = [L, n]
    """
    L, n = float(x[0]), float(x[1])
    u = float(np.clip(u, 0.0, 1.0))

    # Hitung pressure dari state sekarang
    P = compute_pressure(params, L, n)

    # Liquid outlet ΔP
    dP_hydro = params.rho_l * params.g * max(L, 0.0)
    dP_gas = max(P - params.P_out, 0.0)
    dP_liq = dP_hydro + dP_gas

    # Liquid outflow
    qout = params.Cv_l * u * np.sqrt(max(dP_liq, 0.0))

    # Gas outflow (sqrt + limit)
    dP_gas_out = max(P - params.Patm, 0.0)
    n_out = params.kg_n_sqrt * np.sqrt(dP_gas_out)
    n_out = min(n_out, params.n_out_max)

    # Dynamics
    dL_dt = (qin - qout) / params.A
    dn_dt = (n_in - n_out)

    # Euler update
    L_new = L + dt * dL_dt
    n_new = n + dt * dn_dt

    # Constraints
    L_new = float(np.clip(L_new, 0.0, params.L_max))
    n_new = float(max(n_new, 1e-9))

    return np.array([L_new, n_new], dtype=float)


def jacobian_F_numeric(
    params: SeparatorParams,
    dt: float,
    x: np.ndarray,
    qin: float,
    n_in: float,
    u: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Jacobian F = d f(x) / d x secara numerik (finite difference).
    Ini pemula-friendly dan cukup stabil untuk demo.
    """
    x = x.astype(float)
    f0 = f_step(params, dt, x, qin, n_in, u)

    F = np.zeros((2, 2), dtype=float)
    for j in range(2):
        dx = np.zeros(2, dtype=float)
        dx[j] = eps
        f1 = f_step(params, dt, x + dx, qin, n_in, u)
        F[:, j] = (f1 - f0) / eps
    return F


def h_meas(params: SeparatorParams, x: np.ndarray) -> np.ndarray:
    """
    Measurement function: y = [L, P]
    """
    L, n = float(x[0]), float(x[1])
    P = compute_pressure(params, L, n)
    return np.array([L, P], dtype=float)


def jacobian_H_numeric(params: SeparatorParams, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Jacobian H = d h(x) / d x secara numerik.
    """
    x = x.astype(float)
    h0 = h_meas(params, x)

    H = np.zeros((2, 2), dtype=float)
    for j in range(2):
        dx = np.zeros(2, dtype=float)
        dx[j] = eps
        h1 = h_meas(params, x + dx)
        H[:, j] = (h1 - h0) / eps
    return H


def run_ekf_on_log(
    df_log: pd.DataFrame,
    params: SeparatorParams,
    sim: SimConfig,
    ekf_cfg: EKFConfig = EKFConfig(),
) -> pd.DataFrame:
    """
    Jalankan EKF pada data sensor log.
    Return DataFrame hasil estimasi per timestep.
    """

    t = df_log["t_s"].to_numpy(dtype=float)
    dt = float(sim.dt)

    # initial state dari measurement pertama yang valid
    L_meas = df_log["L_meas_m"].to_numpy(dtype=float)
    P_meas = df_log["P_meas_Pa"].to_numpy(dtype=float)

    idx_L = np.where(~np.isnan(L_meas))[0]
    idx_P = np.where(~np.isnan(P_meas))[0]

    L0 = float(L_meas[idx_L[0]]) if len(idx_L) else 1.0
    P0 = float(P_meas[idx_P[0]]) if len(idx_P) else params.Patm * 1.1

    # n0 dari ideal gas
    Vg0 = params.gas_volume(L0)
    n0 = float(max((P0 * Vg0) / (params.R * params.T), 1e-9))

    x = np.array([L0, n0], dtype=float)

    # covariance awal (ketidakpastian)
    Pcov = np.diag([0.05**2, (100.0)**2])  # tebakan awal: level ±5 cm, n ±100 mol

    # Process noise covariance Q
    Q = np.diag([ekf_cfg.q_L, ekf_cfg.q_n])

    # Measurement noise covariance R (untuk [L, P])
    R_full = np.diag([ekf_cfg.r_L, ekf_cfg.r_P])

    # logs
    x_est = np.zeros((len(t), 2), dtype=float)
    P_est = np.zeros(len(t), dtype=float)
    L_est = np.zeros(len(t), dtype=float)

    # innovation/residual logs
    res_L = np.full(len(t), np.nan)
    res_P = np.full(len(t), np.nan)

    for k in range(len(t)):
        # current inputs from log
        qin = float(df_log.loc[k, "qin_m3s"])
        n_in = float(df_log.loc[k, "n_in_mols"])
        u = float(df_log.loc[k, "u"])

        if k > 0:
            # === Predict step ===
            x_pred = f_step(params, dt, x, qin, n_in, u)
            F = jacobian_F_numeric(params, dt, x, qin, n_in, u)

            P_pred = F @ Pcov @ F.T + Q

            x = x_pred
            Pcov = P_pred

        # === Update step (pakai measurement yang tersedia) ===
        y = np.array([L_meas[k], P_meas[k]], dtype=float)
        avail = ~np.isnan(y)  # True untuk sensor yang ada

        if np.any(avail):
            # Ambil subset measurement yang tersedia (1 atau 2)
            y_av = y[avail]

            # Prediksi measurement
            h = h_meas(params, x)
            h_av = h[avail]

            # Jacobian measurement
            H = jacobian_H_numeric(params, x)
            H_av = H[avail, :]  # rows yang tersedia

            # Noise measurement subset
            R = R_full[np.ix_(avail, avail)]

            # Innovation
            innov = y_av - h_av

            # Kalman gain
            S = H_av @ Pcov @ H_av.T + R
            K = Pcov @ H_av.T @ np.linalg.inv(S)

            # Update state
            x = x + K @ innov
            Pcov = (np.eye(2) - K @ H_av) @ Pcov

            # log residual (innovation) per channel jika tersedia
            if avail[0]:
                res_L[k] = float(innov[0] if avail[0] and (avail[0] == True) else np.nan)
            if avail[1]:
                # jika dua tersedia, index innovation untuk P bisa 1; jika hanya P tersedia index 0
                res_P[k] = float(innov[-1])

            # constraints lagi setelah update
            x[0] = float(np.clip(x[0], 0.0, params.L_max))
            x[1] = float(max(x[1], 1e-9))

        # log state
        L_est[k] = float(x[0])
        P_est[k] = compute_pressure(params, float(x[0]), float(x[1]))
        x_est[k, :] = x

    df_out = df_log.copy()
    df_out["L_est_m"] = L_est
    df_out["P_est_Pa"] = P_est
    df_out["n_est_mol"] = x_est[:, 1]
    df_out["res_L"] = res_L
    df_out["res_P"] = res_P
    return df_out


def save_ekf_result(root: Path, df_out: pd.DataFrame, filename: str = "data/processed/ekf_result.csv") -> Path:
    make_dirs(root)
    out_path = (root / filename).resolve()
    df_out.to_csv(out_path, index=False)
    return out_path
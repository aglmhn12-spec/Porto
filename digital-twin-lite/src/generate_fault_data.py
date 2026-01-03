# src/generate_fault_data.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import SeparatorParams, SimConfig
from src.generate_data import SensorNoiseConfig, add_sensor_effects, ensure_project_root, make_dirs


@dataclass
class FaultConfig:
    fault_start_s: float = 350.0
    fault_type: str = "liquid_restriction"  # "liquid_restriction" atau "gas_restriction"
    severity_final: float = 0.5  # 0.5 artinya turun sampai 50% dari nilai awal


def simulate_with_fault(
    params: SeparatorParams,
    sim: SimConfig,
    qin_func,
    n_in_func,
    u_func,
    fault: FaultConfig,
    L0: float = 1.0,
    n0: float = 400.0,
) -> pd.DataFrame:
    """
    Simulasi open-loop tapi dengan parameter berubah setelah fault_start_s.
    Kita implement sederhana: parameter turun linearly sampai severity_final di akhir simulasi.
    """

    # Time grid
    t = np.arange(0.0, sim.t_end + sim.dt, sim.dt)

    # fungsi faktor degradasi 1.0 -> severity_final
    def degrade_factor(ti: float) -> float:
        if ti <= fault.fault_start_s:
            return 1.0
        # setelah fault mulai, turunkan linear sampai akhir simulasi
        frac = (ti - fault.fault_start_s) / max(sim.t_end - fault.fault_start_s, 1e-9)
        frac = float(np.clip(frac, 0.0, 1.0))
        return float(1.0 - frac * (1.0 - fault.severity_final))

    # Kita buat wrapper u_func tetap, input tetap, tapi param efektif berubah
    # Cara cepat: kita jalankan step-by-step sendiri menggunakan sim_model logika serupa.
    # Untuk menjaga pemula-friendly, kita pakai pendekatan: panggil simulator berkali-kali per-step
    # -> tapi itu lambat. Jadi kita implement loop ringkas di sini.

    L = np.zeros_like(t, dtype=float)
    n = np.zeros_like(t, dtype=float)
    P = np.zeros_like(t, dtype=float)

    qin_arr = np.zeros_like(t, dtype=float)
    qout_arr = np.zeros_like(t, dtype=float)
    n_in_arr = np.zeros_like(t, dtype=float)
    n_out_arr = np.zeros_like(t, dtype=float)
    u_arr = np.zeros_like(t, dtype=float)

    L[0] = float(L0)
    n[0] = float(max(n0, 1e-9))

    def pressure_now(L_now: float, n_now: float) -> float:
        Vg = params.gas_volume(L_now)
        P_now = (n_now * params.R * params.T) / Vg
        return float(max(P_now, params.Patm))

    P[0] = pressure_now(L[0], n[0])

    for i in range(1, len(t)):
        ti = float(t[i])
        qin = float(qin_func(ti))
        n_in = float(n_in_func(ti))
        u = float(np.clip(u_func(ti), 0.0, 1.0))

        fac = degrade_factor(ti)

        # parameter efektif saat fault
        Cv_eff = params.Cv_l
        kg_eff = params.kg_n_sqrt
        if fault.fault_type == "liquid_restriction":
            Cv_eff = params.Cv_l * fac
        elif fault.fault_type == "gas_restriction":
            kg_eff = params.kg_n_sqrt * fac

        P_now = pressure_now(L[i - 1], n[i - 1])

        # Liquid outlet
        dP_hydro = params.rho_l * params.g * max(L[i - 1], 0.0)
        dP_gas = max(P_now - params.P_out, 0.0)
        dP_liq = dP_hydro + dP_gas
        qout = Cv_eff * u * np.sqrt(max(dP_liq, 0.0))

        # Gas outlet
        dP_gas_out = max(P_now - params.Patm, 0.0)
        n_out = kg_eff * np.sqrt(dP_gas_out)
        n_out = min(n_out, params.n_out_max)

        dL_dt = (qin - qout) / params.A
        dn_dt = (n_in - n_out)

        L_new = float(np.clip(L[i - 1] + sim.dt * dL_dt, 0.0, params.L_max))
        n_new = float(max(n[i - 1] + sim.dt * dn_dt, 1e-9))

        L[i] = L_new
        n[i] = n_new
        P[i] = pressure_now(L_new, n_new)

        qin_arr[i] = qin
        qout_arr[i] = qout
        n_in_arr[i] = n_in
        n_out_arr[i] = n_out
        u_arr[i] = u

    df = pd.DataFrame(
        {
            "t_s": t,
            "qin_m3s": qin_arr,
            "qout_m3s": qout_arr,
            "u": u_arr,
            "L_m": L,
            "P_Pa": P,
            "n_in_mols": n_in_arr,
            "n_out_mols": n_out_arr,
        }
    )
    return df


def generate_fault_sensor_log_csv(
    out_relpath: str = "data/raw/sensor_log_fault.csv",
    seed: int = 7,
    make_folders: bool = True,
    fault: FaultConfig = FaultConfig(),
) -> Path:
    root = ensure_project_root()
    if make_folders:
        make_dirs(root)

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

    def qin_func(t):
        return 0.004 + (0.001 if t > 200 else 0.0)

    def n_in_func(t):
        return 4.0 + (1.5 if t > 300 else 0.0)

    def u_func(t):
        return 0.55

    df_true = simulate_with_fault(
        params=params,
        sim=sim,
        qin_func=qin_func,
        n_in_func=n_in_func,
        u_func=u_func,
        fault=fault,
        L0=1.0,
        n0=400.0,
    )

    noise = SensorNoiseConfig(
        sigma_L=0.01,
        sigma_P=300.0,
        p_missing_L=0.01,
        p_missing_P=0.01,
        drift_L_rate=0.0,
        drift_P_rate=0.0,
    )

    # Samakan nama kolom true supaya downstream konsisten
    df_true = df_true.rename(columns={"L_m": "L_m", "P_Pa": "P_Pa"})
    df_log = add_sensor_effects(df_true, noise=noise, seed=seed)

    out_path = (root / out_relpath).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_log.to_csv(out_path, index=False)
    return out_path
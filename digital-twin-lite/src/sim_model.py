# src/sim_model.py
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import SeparatorParams, SimConfig


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def simulate_separator_open_loop(
    params: SeparatorParams,
    sim: SimConfig,
    qin_func,        # m^3/s (liquid in)
    n_in_func,       # mol/s (gas in)
    u_func=None,     # valve opening 0..1 (liquid outlet), open-loop: bisa konstan
    L0: float = 1.0,         # m
    n0: float = 400.0,      # mol
) -> pd.DataFrame:
    """
    Digital Twin Lite (Open-loop) untuk Separator:
    - State: L (level cairan), n (jumlah mol gas di gas space)
    - Pressure dihitung dari ideal gas law: P = nRT / Vg(L)
    - Liquid outlet: qout = Cv_l * u * sqrt(ΔP_liq)
      dengan ΔP_liq = rho*g*L + max(P - P_out, 0)
    - Gas outlet: n_out = kg_n * max(P - Patm, 0)

    Metode integrasi: Euler.
    """

    if u_func is None:
        # default valve opening konstan 0.5 jika tidak mendefinisikan
        def u_func(t):  # noqa: F811
            return 0.5

    t = np.arange(0.0, sim.t_end + sim.dt, sim.dt)

    # state
    L = np.zeros_like(t, dtype=float)
    n = np.zeros_like(t, dtype=float)

    # logs
    P = np.zeros_like(t, dtype=float)
    Vg = np.zeros_like(t, dtype=float)

    qin_arr = np.zeros_like(t, dtype=float)
    qout_arr = np.zeros_like(t, dtype=float)

    n_in_arr = np.zeros_like(t, dtype=float)
    n_out_arr = np.zeros_like(t, dtype=float)

    u_arr = np.zeros_like(t, dtype=float)
    dP_liq_arr = np.zeros_like(t, dtype=float)

    # init
    L[0] = float(L0)
    n[0] = float(max(n0, 1e-9))

    # helper compute P,Vg at any state
    def compute_pressure_and_vg(L_now: float, n_now: float) -> tuple[float, float]:
        Vg_now = params.gas_volume(L_now)
        P_now = (n_now * params.R * params.T) / Vg_now
        # physical constraint: minimal pressure at least Patm (optional)
        P_now = max(P_now, params.Patm)
        return P_now, Vg_now

    # initial P,Vg
    P0, Vg0 = compute_pressure_and_vg(L[0], n[0])
    P[0] = P0
    Vg[0] = Vg0

    for i in range(1, len(t)):
        ti = float(t[i])

        # inputs
        qin = float(qin_func(ti))            # m^3/s
        n_in = float(n_in_func(ti))          # mol/s
        u = float(u_func(ti))                # 0..1
        u = _clip(u, 0.0, 1.0)

        # compute current pressure from state
        P_now, Vg_now = compute_pressure_and_vg(L[i - 1], n[i - 1])

        # liquid outlet ΔP (hydrostatic + gas push)
        dP_hydro = params.rho_l * params.g * max(L[i - 1], 0.0)
        dP_gas = max(P_now - params.P_out, 0.0)
        dP_liq = dP_hydro + dP_gas

        # liquid outlet flow (orifice-like)
        qout = params.Cv_l * u * np.sqrt(max(dP_liq, 0.0))

        # Gas outlet: sqrt(ΔP) + capacity limit
        dP_gas_out = max(P_now - params.Patm, 0.0)  # Pa
        n_out_unclipped = params.kg_n_sqrt * np.sqrt(dP_gas_out)  # mol/s
        n_out = min(n_out_unclipped, params.n_out_max)


        # dynamics
        dL_dt = (qin - qout) / params.A
        dn_dt = (n_in - n_out)

        # Euler update
        L_new = L[i - 1] + sim.dt * dL_dt
        n_new = n[i - 1] + sim.dt * dn_dt

        # physical constraints
        L_new = _clip(L_new, 0.0, params.L_max)
        n_new = max(n_new, 1e-9)

        # store
        L[i] = L_new
        n[i] = n_new

        # log using updated state (optional). We log using "new" state for nicer curves.
        P_new, Vg_new = compute_pressure_and_vg(L_new, n_new)
        P[i] = P_new
        Vg[i] = Vg_new

        qin_arr[i] = qin
        qout_arr[i] = qout
        n_in_arr[i] = n_in
        n_out_arr[i] = n_out
        u_arr[i] = u
        dP_liq_arr[i] = dP_liq

    df = pd.DataFrame(
        {
            "t_s": t,
            "qin_m3s": qin_arr,
            "qout_m3s": qout_arr,
            "u": u_arr,
            "L_m": L,
            "Vg_m3": Vg,
            "n_mol": n,
            "P_Pa": P,
            "n_in_mols": n_in_arr,
            "n_out_mols": n_out_arr,
            "dP_liq_Pa": dP_liq_arr,
        }
    )
    return df

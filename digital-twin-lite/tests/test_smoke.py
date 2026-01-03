import pandas as pd

from src.config import SeparatorParams, SimConfig
from src.sim_model import simulate_separator_open_loop


def test_simulation_runs():
    params = SeparatorParams(
        A=3.0, V_total=12.0, L_max=3.0, V_min=0.2,
        rho_l=850.0, g=9.81, Patm=101_325.0, P_out=101_325.0,
        Cv_l=2.5e-5, R=8.314, T=320.0, kg_n_sqrt=0.08, n_out_max=30.0
    )
    sim = SimConfig(dt=1.0, t_end=10.0)

    def qin(t): return 0.004
    def nin(t): return 4.0
    def u(t): return 0.55

    df = simulate_separator_open_loop(params, sim, qin, nin, u, L0=1.0, n0=400.0)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert (df["L_m"] >= 0).all()
    assert (df["P_Pa"] >= params.Patm).all()
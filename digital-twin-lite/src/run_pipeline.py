# src/run_pipeline.py
from pathlib import Path

import pandas as pd

from src.anomaly import AnomalyConfig, detect_anomaly_from_ekf, save_anomaly_result
from src.config import SeparatorParams, SimConfig
from src.generate_data import generate_sensor_log_csv
from src.generate_fault_data import FaultConfig, generate_fault_sensor_log_csv
from src.state_estimation import EKFConfig, run_ekf_on_log, save_ekf_result


def main():
    root = Path.cwd()
    assert root.name == "digital-twin-lite", "Jalankan dari root digital-twin-lite/"

    # 1) generate normal log
    normal_path = generate_sensor_log_csv(out_relpath="data/raw/sensor_log.csv", seed=42, make_folders=True)

    # 2) generate fault log
    fault_cfg = FaultConfig(fault_start_s=350.0, fault_type="liquid_restriction", severity_final=0.5)
    fault_path = generate_fault_sensor_log_csv(out_relpath="data/raw/sensor_log_fault.csv", seed=7, make_folders=True, fault=fault_cfg)

    # 3) load salah satu (default: fault biar demo)
    df_log = pd.read_csv(fault_path)

    params = SeparatorParams(
        A=3.0, V_total=12.0, L_max=3.0, V_min=0.2,
        rho_l=850.0, g=9.81, Patm=101_325.0, P_out=101_325.0,
        Cv_l=2.5e-5, R=8.314, T=320.0, kg_n_sqrt=0.08, n_out_max=30.0
    )
    sim = SimConfig(dt=0.5, t_end=float(df_log["t_s"].max()))

    ekf_cfg = EKFConfig(q_L=1e-6, q_n=1e-2, r_L=(0.01**2), r_P=(300.0**2))
    df_est = run_ekf_on_log(df_log, params=params, sim=sim, ekf_cfg=ekf_cfg)

    anom_cfg = AnomalyConfig(window=60, z_thresh_L=3.0, z_thresh_P=3.0, cusum_k=0.5, cusum_h=8.0, consecutive=10)
    df_anom = detect_anomaly_from_ekf(df_est, cfg=anom_cfg)

    save_ekf_result(root, df_est, filename="data/processed/ekf_result.csv")
    save_anomaly_result(root, df_anom, filename="data/processed/anomaly_result.csv")

    print("DONE. Outputs saved to data/processed/ and dashboard can be run with:")
    print("streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
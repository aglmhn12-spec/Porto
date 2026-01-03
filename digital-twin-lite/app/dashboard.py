# app/dashboard.py
from __future__ import annotations

# Import modul proyek (pastikan root ada di sys.path)
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]  # .../digital-twin-lite
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.anomaly import AnomalyConfig, detect_anomaly_from_ekf, save_anomaly_result
from src.config import SeparatorParams, SimConfig
from src.state_estimation import EKFConfig, run_ekf_on_log, save_ekf_result


def load_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    return pd.read_csv(path)


def get_default_params() -> SeparatorParams:
    # Kalau kamu sudah punya hasil kalibrasi Step 3, isi Cv_l & kg_n_sqrt dengan hasil itu.
    return SeparatorParams(
        A=3.0, V_total=12.0, L_max=3.0, V_min=0.2,
        rho_l=850.0, g=9.81, Patm=101_325.0, P_out=101_325.0,
        Cv_l=2.5e-5,
        R=8.314, T=320.0,
        kg_n_sqrt=0.08,
        n_out_max=30.0
    )


def run_pipeline(df_log: pd.DataFrame, params: SeparatorParams):
    sim = SimConfig(dt=0.5, t_end=float(df_log["t_s"].max()))
    ekf_cfg = EKFConfig(
        q_L=1e-6, q_n=1e-2,
        r_L=(0.01**2),
        r_P=(300.0**2),
    )
    df_est = run_ekf_on_log(df_log, params=params, sim=sim, ekf_cfg=ekf_cfg)

    anom_cfg = AnomalyConfig(
        window=30,
        z_thresh_L=2.0,
        z_thresh_P=2.0,
        cusum_k=0.3,
        cusum_h=4.0,
        consecutive=4
    )
    df_anom = detect_anomaly_from_ekf(df_est, cfg=anom_cfg)
    return df_est, df_anom


def main():
    st.set_page_config(page_title="Digital Twin Lite — Separator", layout="wide")
    st.title("Digital Twin Lite — Separator Level + Pressure")
    st.caption("Pipeline: Sensor Log → EKF State Estimation → Residual → Anomaly Detection")

    # Sidebar: pilih dataset
    st.sidebar.header("Data")
    data_choice = st.sidebar.radio(
        "Pilih dataset",
        options=["Normal (sensor_log.csv)", "Fault (sensor_log_fault.csv)"],
        index=0
    )

    data_file = "sensor_log.csv" if "Normal" in data_choice else "sensor_log_fault.csv"
    data_path = ROOT / "data" / "raw" / data_file

    # Sidebar: parameter (editable)
    st.sidebar.header("Model Parameters")
    params0 = get_default_params()

    Cv_l = st.sidebar.number_input("Cv_l (m^3/s/sqrt(Pa))", value=float(params0.Cv_l), format="%.6g")
    kg_n_sqrt = st.sidebar.number_input("kg_n_sqrt (mol/s/sqrt(Pa))", value=float(params0.kg_n_sqrt), format="%.6g")
    n_out_max = st.sidebar.number_input("n_out_max (mol/s)", value=float(params0.n_out_max), format="%.6g")

    # Parameter lain dibuat tetap agar UI tidak terlalu ramai
    params = params0.model_copy(update={
        "Cv_l": float(Cv_l),
        "kg_n_sqrt": float(kg_n_sqrt),
        "n_out_max": float(n_out_max),
    })

    # Load data
    try:
        df_log = load_log(data_path)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    st.sidebar.header("Run")
    run_btn = st.sidebar.button("Run EKF + Anomaly")

    # Auto-run pertama kali supaya user langsung lihat sesuatu
    if "df_est" not in st.session_state or run_btn:
        with st.spinner("Running EKF + anomaly detection..."):
            df_est, df_anom = run_pipeline(df_log, params)
        st.session_state["df_est"] = df_est
        st.session_state["df_anom"] = df_anom

        # Save outputs
        (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
        save_ekf_result(ROOT, df_est, filename="data/processed/ekf_result.csv")
        save_anomaly_result(ROOT, df_anom, filename="data/processed/anomaly_result.csv")

    df_est = st.session_state["df_est"]
    df_anom = st.session_state["df_anom"]

    # Alarm summary
    alarm_times = df_anom.loc[df_anom["alarm"] == True, "t_s"]
    if len(alarm_times) > 0:
        st.error(f"ALARM: anomaly terdeteksi. Alarm pertama pada t = {float(alarm_times.iloc[0]):.1f} s")
    else:
        st.success("Tidak ada alarm anomaly (berdasarkan konfigurasi saat ini).")

    # Layout: 2 kolom besar
    col1, col2 = st.columns(2)

    # --- Plot Level ---
    with col1:
        st.subheader("Level (L)")
        # Siapkan data long format untuk Plotly
        plot_df = pd.DataFrame({
            "t_s": df_est["t_s"],
            "L_true": df_est.get("L_true_m", pd.Series([None]*len(df_est))),
            "L_meas": df_est.get("L_meas_m", pd.Series([None]*len(df_est))),
            "L_est": df_est.get("L_est_m", pd.Series([None]*len(df_est))),
        })
        plot_long = plot_df.melt(id_vars="t_s", var_name="series", value_name="value")
        fig = px.line(plot_long, x="t_s", y="value", color="series", title="Level: true vs measured vs estimated")
        st.plotly_chart(fig, use_container_width=True)

    # --- Plot Pressure ---
    with col2:
        st.subheader("Pressure (P)")
        plot_df = pd.DataFrame({
            "t_s": df_est["t_s"],
            "P_true": df_est.get("P_true_Pa", pd.Series([None]*len(df_est))),
            "P_meas": df_est.get("P_meas_Pa", pd.Series([None]*len(df_est))),
            "P_est": df_est.get("P_est_Pa", pd.Series([None]*len(df_est))),
        })
        plot_long = plot_df.melt(id_vars="t_s", var_name="series", value_name="value")
        fig = px.line(plot_long, x="t_s", y="value", color="series", title="Pressure: true vs measured vs estimated")
        st.plotly_chart(fig, use_container_width=True)

    # Residual + Z-score
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("EKF Residual (Innovation)")
        plot_df = pd.DataFrame({
            "t_s": df_anom["t_s"],
            "res_L": df_anom["res_L"],
            "res_P_scaled": df_anom["res_P"] / 10000.0,  # skala biar kebaca bareng
        })
        plot_long = plot_df.melt(id_vars="t_s", var_name="series", value_name="value")
        fig = px.line(plot_long, x="t_s", y="value", color="series",
                      title="Residual: res_L dan res_P (scaled)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Catatan: res_P dibagi 10,000 agar satu grafik dengan res_L.")

    with col4:
        st.subheader("Z-score + Alarm Flag")
        plot_df = pd.DataFrame({
            "t_s": df_anom["t_s"],
            "z_L": df_anom["z_L"],
            "z_P": df_anom["z_P"],
        })
        plot_long = plot_df.melt(id_vars="t_s", var_name="series", value_name="value")
        fig = px.line(plot_long, x="t_s", y="value", color="series", title="Rolling Z-score")
        st.plotly_chart(fig, use_container_width=True)

        alarm_df = pd.DataFrame({"t_s": df_anom["t_s"], "alarm": df_anom["alarm"].astype(int)})
        fig2 = px.line(alarm_df, x="t_s", y="alarm", title="Alarm (0/1)")
        st.plotly_chart(fig2, use_container_width=True)

    # Data preview
    st.subheader("Data Preview")
    st.write("Log sensor (raw):")
    st.dataframe(df_log.head(20), use_container_width=True)

    st.write("Hasil EKF + anomaly (processed):")
    st.dataframe(df_anom[["t_s", "L_meas_m", "L_est_m", "P_meas_Pa", "P_est_Pa", "z_L", "z_P", "alarm"]].head(30),
                 use_container_width=True)

    st.caption("Output tersimpan ke data/processed/ekf_result.csv dan data/processed/anomaly_result.csv")


if __name__ == "__main__":
    main()
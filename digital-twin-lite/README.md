# Digital Twin Lite — Separator Level + Pressure

Mini digital twin untuk separator (vessel) yang memodelkan **liquid level** dan **gas pressure**, melakukan:
1) simulasi berbasis neraca massa + ideal gas,
2) pembuatan data sensor sintetis (noise + missing),
3) kalibrasi parameter (least squares),
4) state estimation real-time (EKF),
5) deteksi anomali berbasis residual (Z-score + CUSUM),
6) dashboard interaktif (Streamlit).

## Fitur Utama
- **Physics-based model**:
  - Level: dL/dt = (qin − qout)/A
  - Gas: dn/dt = n_in − n_out, dan P = nRT/Vg(L)
  - Liquid outlet: qout ∝ u * sqrt(ΔP)
- **Data realism**: noise, missing values, optional drift
- **Kalibrasi**: estimasi parameter `Cv_l` dan `kg_n_sqrt`
- **EKF**: menggabungkan model + sensor untuk estimasi state
- **Anomaly detection**: alarm otomatis saat fault (restriction) terjadi
- **Dashboard**: visual true/measured/estimated + residual + alarm

## Struktur Proyek
- `src/` : model, data generator, kalibrasi, EKF, anomaly
- `notebooks/` : step-by-step eksperimen
- `data/raw/` : log sensor sintetis
- `data/processed/` : hasil kalibrasi, EKF, anomaly
- `app/` : Streamlit dashboard

## Quickstart (Windows / Anaconda)
1) Buat environment dan install dependency:
```bash
conda create -n dt-lite python=3.11 -y
conda activate dt-lite
pip install -r requirements.txt
2) Generarte data (normal dan fault) dari notebook:
    - notebooks/02_generate_data.ipynb
    - notebooks/05_anomaly_detection.ipynb
3) Jalankan dashboard:
```bash
streamlit run app/dashboard.py
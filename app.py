# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import joblib

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="EcoSentinel: Forest Resilience Framework",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Data Loading (Cached) ---
@st.cache_resource
def load_assets():
    df = pd.read_parquet('Armenia_ML_Training_Data.parquet')
    border = gpd.read_file('arm_admin0.geojson')
    model = joblib.load('RF_FVS_Model.joblib')
    return df, border, model

try:
    df_forest, armenia_border, rf_model = load_assets()
except Exception as e:
    st.error(f"Error loading data files: {e}")
    st.stop()

# Constraints & Mapping
vmax_vpd = df_forest['Delta_VPD_GS'].quantile(0.99)
vmax_temp = df_forest['Delta_Tmax_GS'].quantile(0.99)
vmin_prec = df_forest['Delta_P_GS'].quantile(0.01)

ssp_mapping = {
    'ssp126': 'SSP1-2.6', 'ssp245': 'SSP2-4.5', 'ssp370': 'SSP3-7.0', 'ssp585': 'SSP5-8.5'
}

projection_matrix = {
    'Medium-Term (2041–2060)': {
        'ssp126': (1.3, 14.24, 0.058), 'ssp245': (1.5, -9.48, 0.068),
        'ssp370': (1.8, -45.81, 0.082), 'ssp585': (2.3, 1.36, 0.106)
    },
    'Long-Term (2081–2100)': {
        'ssp126': (1.3, 17.12, 0.058), 'ssp245': (2.4, 6.88, 0.111),
        'ssp370': (3.8, -81.32, 0.184), 'ssp585': (5.2, -5.11, 0.262)
    }
}

# --- 3. Academic Header ---
st.markdown("""
    <div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; border-left: 8px solid #1b5e20; margin-bottom: 20px;">
        <h1 style="color: #1b5e20; margin: 0;">Predictive Framework for Climate-Smart Reforestation</h1>
        <p style="font-size: 1.1em; color: #555; margin-bottom: 15px;"><b><i>Technical Concept Note | Validated Modeling for Armenia's Forest Resilience</i></b></p>
        <hr style="border: 0.5px solid #ddd;">
        <table style="width: 100%; border: none; font-size: 0.95em; color: #333;">
            <tr><td><b>Author:</b> Narek Ohanyan</td><td><b>Date:</b> May, 2026</td></tr>
            <tr><td><b>Institution:</b> American University of Armenia</td><td><b>Subject:</b> BSCS Capstone Project</td></tr>
        </table>
    </div>
""", unsafe_allow_html=True)

# --- 4. NEW: Interpretation & Technical Documentation ---
with st.expander("📖 User Guide: How to Interpret the Output"):
    st.markdown("""
    ### **1. Understanding the Metrics**
    * **Canopy Structural Complexity [σ(H)]:** Measured in meters. It represents the standard deviation of vegetation height within a 1km pixel. High values (Green) indicate mature, multi-layered forests. Low values (Red) indicate sparse or simplified canopy structures.
    * **Forest Vulnerability Score (FVS):** A relative index (0-100). It measures the percentage of structural loss predicted between the 2017 baseline and the future scenario. 
        * *0-20:* Low Risk (Stable)
        * *20-50:* Moderate Risk (Degradation likely)
        * *50+:* High Risk (Potential ecosystem collapse/transition)

    ### **2. The Climate Variables (Predictors)**
    All inputs are **Deltas (Δ)**—the difference between a future value and the historical baseline (1979-2018).
    * **Δ Tmax (°C):** The increase in maximum summer temperatures. Higher heat leads to leaf scorching and metabolic stress.
    * **Δ Prec (mm):** The change in annual precipitation. Negative values indicate drought stress; positive values can buffer the effects of heat.
    * **Δ VPD (kPa):** *Vapor Pressure Deficit*. This is the "atmospheric thirst." It measures how much moisture the air pulls out of the trees. It is calculated based on the gap between the air's humidity and its saturation point at a given temperature.

    ### **3. Data Sources**
    * **Vegetation Structure:** Sentinel-2 Vegetation Height Model (VHM) 2017 (WSL Institute).
    * **Climate Archives:** CHELSA High-Resolution Climatologies (V2.1).
    * **Future Scenarios:** IPCC AR6 CMIP6 Multi-Model Ensembles (SSPs).
    """)

with st.expander("🔬 Methodology & Mathematical Framework"):
    st.write("""
    The framework utilizes a **Random Forest Regressor** trained on 18 years of historical climate-vegetation interactions. 
    The core equation for vulnerability is defined as:
    """)
    st.latex(r"FVS = \left( \frac{\sigma(H)_{baseline} - \sigma(H)_{predicted}}{\sigma(H)_{baseline}} \right) \times 100")
    st.write("""
    The predictive engine applies a **spatial-clipping logic** to ensure that extreme climate forcing (outliers) does not produce mathematically 
    impossible structural values, maintaining biological realism in the 2080-2100 projections.
    """)

# --- 5. Sidebar UI ---
st.sidebar.title("🌲 Model Controls")
mode = st.sidebar.selectbox("Operation Mode", ['Historical Baseline', 'IPCC Scenarios', 'Custom Forcing'])

metric_options = [('Canopy Structure [σ(H)]', 'sigma')]
if mode != 'Historical Baseline':
    metric_options = [('Forest Vulnerability Score [FVS]', 'fvs')] + metric_options

metric = st.sidebar.selectbox("Metric", options=metric_options, format_func=lambda x: x[0])[1]

dt, dp, dv = 0.0, 0.0, 0.0
if mode == 'IPCC Scenarios':
    period = st.sidebar.selectbox("Time Horizon", options=list(projection_matrix.keys()))
    ssp_key = st.sidebar.radio("Pathway (SSP)", options=list(ssp_mapping.keys()), 
                               format_func=lambda x: ssp_mapping[x])
    dt, dp, dv = projection_matrix[period][ssp_key]
elif mode == 'Custom Forcing':
    dt = st.sidebar.slider('Δ Tmax (°C)', 0.0, 6.0, 0.0, 0.1)
    dp = st.sidebar.slider('Δ Prec (mm)', -150, 100, 0, 1)
    dv = st.sidebar.slider('Δ VPD (kPa)', 0.0, 1.0, 0.0, 0.01)

# --- 6. Predictive Engine ---
if mode == 'Historical Baseline':
    y_vals = df_forest['vhm_std']
    title, subtitle = "Historical Baseline Structure", "Current forest complexity before climate forcing."
    vmin, vmax, cmap, unit = 0, 8, 'RdYlGn', "meters"
else:
    X_in = pd.DataFrame({
        'Delta_VPD_GS': (df_forest['Delta_VPD_GS'] + dv).clip(upper=vmax_vpd),
        'Delta_Tmax_GS': (df_forest['Delta_Tmax_GS'] + dt).clip(upper=vmax_temp),
        'Delta_P_GS': (df_forest['Delta_P_GS'] + dp).clip(lower=vmin_prec)
    })
    y_pred = rf_model.predict(X_in)
    
    if metric == 'fvs':
        y_vals = ((df_forest['vhm_std'] - y_pred) / df_forest['vhm_std'] * 100).clip(0, 100)
        title, vmin, vmax, cmap, unit = "Forest Vulnerability Score (FVS)", 0, 80, 'YlOrRd', "Index"
    else:
        y_vals = y_pred
        title, vmin, vmax, cmap, unit = "Projected Canopy Structure [σ(H)]", 0, 8, 'RdYlGn', "meters"
    
    label = f"IPCC Scenario ({period} | {ssp_mapping[ssp_key]})" if mode == 'IPCC Scenarios' else "Custom Scenario"
    subtitle = f"{label} | ΔTmax: +{dt}°C | ΔPrec: {dp}mm | ΔVPD: +{dv}kPa"

# --- 7. Main Dashboard Display ---
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader(title)
    st.caption(subtitle)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    armenia_border.plot(ax=ax, color='#eeeeee', edgecolor='#bcbcbc')
    sc = ax.scatter(df_forest['x'], df_forest['y'], c=y_vals, cmap=cmap, s=15, vmin=vmin, vmax=vmax)
    plt.colorbar(sc, label=unit)
    ax.axis('off')
    st.pyplot(fig)

with col2:
    st.metric("Mean Value", f"{y_vals.mean():.2f}")
    if mode != 'Historical Baseline':
        max_vuln = y_vals.max()
        st.metric("Max Vulnerability", f"{max_vuln:.1f}%")
        st.info("💡 High vulnerability indicates areas where forest structure is predicted to simplify significantly.")

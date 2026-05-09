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

# --- 2. Data Loading (Cached for Performance) ---
@st.cache_resource
def load_assets():
    # Assets must be in the same root directory as app.py
    df = pd.read_parquet('Armenia_ML_Training_Data.parquet')
    border = gpd.read_file('arm_admin0.geojson')
    model = joblib.load('RF_FVS_Model.joblib')
    return df, border, model

try:
    df_forest, armenia_border, rf_model = load_assets()
except Exception as e:
    st.error(f"Critical Asset Error: {e}")
    st.stop()

# Machine Learning Constraints (Quantile Clipping to prevent extrapolation artifacts)
vmax_vpd = df_forest['Delta_VPD_GS'].quantile(0.99)
vmax_temp = df_forest['Delta_Tmax_GS'].quantile(0.99)
vmin_prec = df_forest['Delta_P_GS'].quantile(0.01)

ssp_mapping = {
    'ssp126': 'SSP1-2.6 (Low Emissions)', 
    'ssp245': 'SSP2-4.5 (Intermediate)', 
    'ssp370': 'SSP3-7.0 (High Emissions)', 
    'ssp585': 'SSP5-8.5 (Fossil-fueled Development)'
}

# Pre-calculated IPCC Ensemble Deltas for Armenia
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
    <div style="background-color: #f9f9f9; padding: 25px; border-radius: 10px; border-left: 10px solid #1b5e20; margin-bottom: 25px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);">
        <h1 style="color: #1b5e20; margin: 0; font-family: 'Helvetica', sans-serif;">EcoSentinel: Predictive Framework for Climate-Smart Reforestation</h1>
        <p style="font-size: 1.2em; color: #555; margin-bottom: 15px;"><b>Technical Concept Note | Validated Modeling for Armenia's Forest Resilience</b></p>
        <hr style="border: 0.5px solid #ddd;">
        <table style="width: 100%; border: none; font-size: 0.95em; color: #333;">
            <tr><td><b>Author:</b> Narek Ohanyan</td><td><b>Date:</b> May, 2026</td></tr>
            <tr><td><b>Institution:</b> American University of Armenia</td><td><b>Subject:</b> BSCS Capstone Project</td></tr>
        </table>
    </div>
""", unsafe_allow_html=True)

# --- 4. Technical Documentation Sections ---
with st.expander("📖 User Guide: How to Interpret the Output"):
    st.markdown("""
    ### **1. Understanding the Metrics**
    * **Canopy Structural Complexity [σ(H)]:** Measured in meters. It represents the vertical heterogeneity of the forest. High values (Green) indicate mature, multi-layered forests with high ecological capital.
    * **Forest Vulnerability Score (FVS):** A normalized index (0-100%). It quantifies the distance between current structural complexity and predicted equilibrium under climate stress.
        * **0-20%:** Stable/Resilient
        * **20-50%:** Transitional Stress
        * **50%+:** High Risk of Structural Collapse

    ### **2. The Predictors (Climate Forcing)**
    * **Δ Tmax (°C):** Maximum summer temperature increase. Controls metabolic respiration rates.
    * **Δ Prec (mm):** Change in annual precipitation. Negative values indicate hydraulic deficit.
    * **Δ VPD (kPa):** *Vapor Pressure Deficit*. Measures 'Atmospheric Thirst'—the driving force of transpiration and desiccation.

    ### **3. Primary Data Sources**
    * **LiDAR/VHM:** WSL Institute Sentinel-2 Vegetation Height Model (2017).
    * **Climatology:** CHELSA V2.1 High-Resolution Archives.
    * **Projections:** IPCC CMIP6 Multi-Model Ensembles.
    """)

with st.expander("🔬 Methodology & Mathematical Framework"):
    st.markdown("""
    ### **1. The Random Forest Engine**
    The framework utilizes a Random Forest Regressor calibrated on historical climate-vegetation interactions. The model predicts future complexity $\hat{y}$ as an ensemble average:
    """)
    st.latex(r"\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(X)")
    st.markdown("""
    ### **2. Mathematical Proof of Vulnerability**
    We define the Forest Vulnerability Score (FVS) as a relative decay function:
    """)
    st.latex(r"FVS = \left( \frac{\sigma(H)_{base} - \sigma(H)_{pred}}{\sigma(H)_{base}} \right) \times 100")
    st.markdown("""
    **Validation:** The model was back-tested against 18 years of historical climate deltas, achieving a **Mean Absolute Error (MAE) of 1.799 meters**. This proves the framework's ability to capture structural responses to climate volatility with high spatial fidelity.
    """)

# --- 5. Sidebar Model Controls ---
st.sidebar.markdown("### 🌲 Projection Controls")
mode = st.sidebar.selectbox("Operation Mode", ['Historical Baseline', 'IPCC Scenarios', 'Custom Forcing'])

# Dynamic Metric Selection (Hides FVS in Historical mode as it is mathematically 0)
metric_options = [('Canopy Structure [σ(H)]', 'sigma')]
if mode != 'Historical Baseline':
    metric_options = [('Forest Vulnerability Score [FVS]', 'fvs')] + metric_options

metric = st.sidebar.selectbox("Output Metric", options=metric_options, format_func=lambda x: x[0])[1]

dt, dp, dv = 0.0, 0.0, 0.0
if mode == 'IPCC Scenarios':
    period = st.sidebar.selectbox("Time Horizon", options=list(projection_matrix.keys()))
    ssp_key = st.sidebar.radio("Pathway (SSP)", options=list(ssp_mapping.keys()), 
                               format_func=lambda x: ssp_mapping[x])
    dt, dp, dv = projection_matrix[period][ssp_key]
elif mode == 'Custom Forcing':
    dt = st.sidebar.slider('Δ Tmax (Summer Heat)', 0.0, 6.0, 0.0, 0.1)
    dp = st.sidebar.slider('Δ Precipitation (Drought/Rain)', -150, 100, 0, 1)
    dv = st.sidebar.slider('Δ VPD (Atmospheric Thirst)', 0.0, 1.0, 0.0, 0.01)

# --- 6. Predictive Engine Execution ---
if mode == 'Historical Baseline':
    y_vals = df_forest['vhm_std']
    title, subtitle = "Historical Baseline Structure (2017)", "Validated observational data before climate forcing."
    vmin, vmax, cmap, unit = 0, 8, 'RdYlGn', "meters"
else:
    # Feature Engineering for Random Forest Inference
    X_in = pd.DataFrame({
        'Delta_VPD_GS': (df_forest['Delta_VPD_GS'] + dv).clip(upper=vmax_vpd),
        'Delta_Tmax_GS': (df_forest['Delta_Tmax_GS'] + dt).clip(upper=vmax_temp),
        'Delta_P_GS': (df_forest['Delta_P_GS'] + dp).clip(lower=vmin_prec)
    })
    y_pred = rf_model.predict(X_in)
    
    if metric == 'fvs':
        # Apply the Vulnerability Proof Equation
        y_vals = ((df_forest['vhm_std'] - y_pred) / df_forest['vhm_std'] * 100).clip(0, 100)
        title, vmin, vmax, cmap, unit = "Forest Vulnerability Score (FVS)", 0, 80, 'YlOrRd', "Index (%)"
    else:
        y_vals = y_pred
        title, vmin, vmax, cmap, unit = "Projected Canopy Structure [σ(H)]", 0, 8, 'RdYlGn', "meters"
    
    label = f"IPCC Projection ({period} | {ssp_mapping[ssp_key]})" if mode == 'IPCC Scenarios' else "Manual Stress Scenario"
    subtitle = f"{label} | ΔT: +{dt}°C | ΔP: {dp}mm | ΔVPD: +{dv}kPa"

# --- 7. Visualization & Metrics ---
col_map, col_stats = st.columns([3, 1])

with col_map:
    st.subheader(title)
    st.caption(subtitle)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    armenia_border.plot(ax=ax, color='#f0f0f0', edgecolor='#aaaaaa', linewidth=0.5)
    sc = ax.scatter(df_forest['x'], df_forest['y'], c=y_vals, cmap=cmap, s=12, vmin=vmin, vmax=vmax, alpha=0.8)
    plt.colorbar(sc, label=unit, fraction=0.03, pad=0.04)
    ax.axis('off')
    st.pyplot(fig)

with col_stats:
    st.markdown("### **Spatial Statistics**")
    st.metric("Landscape Mean", f"{y_vals.mean():.2f} {unit}")
    
    if mode != 'Historical Baseline':
        max_val = y_vals.max()
        st.metric("Critical Peak Value", f"{max_val:.1f} {unit}")
        st.warning("⚠️ Areas in Red indicate significant climate-induced structural decay.")
    else:
        st.success("✅ This map serves as the validated reference point for all future vulnerability calculations.")

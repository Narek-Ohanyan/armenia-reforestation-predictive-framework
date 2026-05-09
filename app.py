# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import joblib

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="EcoSentinel: Validated Predictive Framework",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Data Loading (Cached) ---
@st.cache_resource
def load_assets():
    # Assets generated in project development (Notebooks 01-05)
    # df_forest contains the 4,338 pixels identified at 1km resolution
    df = pd.read_parquet('Armenia_ML_Training_Data.parquet')
    arm_border = gpd.read_file('arm_admin0.geojson')
    model = joblib.load('RF_FVS_Model.joblib')
    return df, arm_border, model

try:
    df_forest, armenia_border, rf_model = load_assets()
except Exception as e:
    st.error(f"Error loading framework assets: {e}")
    st.stop()

# --- 3. Academic Header ---
st.markdown("""
    <div style="background-color: #f9f9f9; padding: 25px; border-radius: 10px; border-left: 10px solid #1b5e20; margin-bottom: 25px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);">
        <h1 style="color: #1b5e20; margin: 0; font-family: 'Helvetica', sans-serif;">A Validated Predictive Framework</h1>
        <p style="font-size: 1.2em; color: #555; margin-bottom: 15px;"><b><i>Climate-Smart Reforestation and Resilience Mapping in Armenia</i></b></p>
        <hr style="border: 0.5px solid #ddd;">
        <table style="width: 100%; border: none; font-size: 0.95em; color: #333;">
            <tr><td><b>Author:</b> Narek Ohanyan</td><td><b>Date:</b> May, 2026</td></tr>
            <tr><td><b>Institution:</b> American University of Armenia</td><td><b>Project:</b> BSCS Capstone</td></tr>
        </table>
    </div>
""", unsafe_allow_html=True)

# --- 4. Refined Methodology & Mathematical Framework ---
with st.expander("ℹ️ About the Project"):
    st.markdown(r"""
    This project provides a technical contribution to the **Forest Restoration and Climate Change in Armenia (FORACCA)** initiative, specifically targeting **Output 1.2**. 
    """)
with st.expander("🔬 Methodology: Computational Calibration & Bioclimatic Stacking"):
    st.markdown("""
    ### **1. Geospatial Integration & Temporal Baselines**
    The study utilizes a high-resolution remote sensing framework to quantify forest vertical structure. Initial canopy metrics were derived from Global Ecosystem Dynamics Investigation (GEDI) and secondary Vegetation Height Models (VHM). To maintain computational efficiency and align with bioclimatic datasets, all raster layers were spatially resampled to a standardized 1 km² grid using the rioxarray and rasterio libraries.
    The framework utilizes a multi-decadal supervised learning architecture to correlate bioclimatic variables with physical forest structure.
    * **Historical Training Period (1979–2000):** Baseline climate means were established to define the "stable" ecological state.
    * **Validation & Hindcasting (2000–2018):** Evaluation of forest structural response to observed climate anomalies.
    * **Structural Baseline (2017):** Derived from Sentinel-2 Vegetation Height Models (VHM) at 10m, then aggregated to a **standardized 1 km² grid**.

    ### **2. The Integrated Climate Stressor Formula**
    The vulnerability logic is governed by the cumulative climatic stressor intensity relative to the initial structural complexity ($\sigma H$). The total stress for a pixel ($p$) under a scenario ($s$) is calculated as the integral of weighted stressors across the **Growing Season (May–September)**:
    """)
    
    st.latex(r"Stress(p, s) = \int_{May}^{Sept} \frac{1}{\sigma(H)} \left( \alpha \cdot \Delta vpd(p, m, s) + \beta \cdot \Delta T_{max}(p, m, s) + \gamma \cdot \Delta P(p, m, s) \right) dm")
    
    st.markdown(r"""
    ### **3. Bioclimatic Predictors & Feature Engineering**
    * **$\Delta$ Variables:** Computed as the difference between the projection years (2041–2060 or 2081–2100) and the historical baseline.
    * **$\sigma(H)^{-1}$ Normalization:** By multiplying the integral by the inverse of the standard deviation of canopy height, the framework accounts for the higher relative vulnerability of complex vertical structures to atmospheric demand.
    * **Stability Proof:** Using the **Hansen Global Forest Change (2000–2023) dataset**, we programmatically filtered pixels to ensure the 2017 baseline represents stable forest cover devoid of anthropogenic disturbance.

    ### **4. Random Forest Calibration & Interpretability**
    A **Random Forest Regressor** (500 estimators) was trained to map these stressors to structural outcomes. 
    * **Performance Metrics:** $R^2 = 0.292$ | Mean Absolute Error (MAE) = **1.799 meters**.
    * **SHAP Interpretability:** Feature importance was derived using Shapley values to identify dominant drivers:
        - **$\alpha$ (Δ VPD):** 34.9% (Atmospheric Drying Power)
        - **$\beta$ (Δ Tmax):** 31.4% (Metabolic Respiration Cost)
        - **$\gamma$ (Δ Prec):** 33.7% (Hydraulic Stress)
    """)

with st.expander("📖 User Guide: Output Interpretation"):
    st.markdown(r"""
    ### **1. Forest Vulnerability Score [$FVS$]**
    The FVS quantifies the percentage of structural loss predicted under climate forcing compared to the validated 2017 reference state:
    """)
    st.latex(r"FVS = \left( \frac{\sigma(H)_{2017} - \sigma(H)_{predicted}}{\sigma(H)_{2017}} \right) \times 100")
    
    st.markdown(r"""
    ### **2. Canopy Structure [$\sigma(H)$]**
    Measures vertical heterogeneity within the 1 km² pixel. High values (Green) indicate mature, multi-layered forest canopies, while low values signify simpler or degraded structures.
    
    ### **3. Operation Modes**
    * **Historical Baseline:** Represents the observed 2017 state (validated across the 1979-2018 observational period).
    * **IPCC Scenarios:** Applies CMIP6 climate deltas for the **Medium-Term (2041–2060)** and **Long-Term (2081–2100)**.
    * **Custom Forcing:** Allows manual stress-testing of the ecosystem by adjusting individual bioclimatic variables.
    """)
# --- 5. Model Constants ---
vmax_vpd = df_forest['Delta_VPD_GS'].quantile(0.99)
vmax_temp = df_forest['Delta_Tmax_GS'].quantile(0.99)
vmin_prec = df_forest['Delta_P_GS'].quantile(0.01)

ssp_mapping = {'ssp126': 'SSP1-2.6', 'ssp245': 'SSP2-4.5', 'ssp370': 'SSP3-7.0', 'ssp585': 'SSP5-8.5'}

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

# --- 6. Sidebar Controls ---
st.sidebar.markdown("### 🌲 Projection Controls")
mode = st.sidebar.selectbox("Operation Mode", ['Historical Baseline', 'IPCC Scenarios', 'Custom Forcing'])

metric_options = [('Canopy Structure [σ(H)]', 'sigma')]
if mode != 'Historical Baseline':
    metric_options = [('Forest Vulnerability Score [FVS]', 'fvs')] + metric_options

metric = st.sidebar.selectbox("Metric", options=metric_options, format_func=lambda x: x[0])[1]

dt, dp, dv = 0.0, 0.0, 0.0
if mode == 'IPCC Scenarios':
    period = st.sidebar.selectbox("Time Horizon", options=list(projection_matrix.keys()))
    ssp_key = st.sidebar.radio("Pathway", options=list(ssp_mapping.keys()), format_func=lambda x: ssp_mapping[x])
    dt, dp, dv = projection_matrix[period][ssp_key]
elif mode == 'Custom Forcing':
    dt = st.sidebar.slider('Δ Tmax (K)', 0.0, 6.0, 0.0)
    dp = st.sidebar.slider('Δ Prec (kg/m²/mo)', -150, 100, 0)
    dv = st.sidebar.slider('Δ VPD (Pa)', 0.0, 1.0, 0.0)

# --- 7. Predictive Run ---
if mode == 'Historical Baseline':
    y_vals = df_forest['vhm_std']
    title, vmin, vmax, cmap, unit = "Historical Baseline Structure (2017)", 0, 8, 'RdYlGn', "m"
    subtitle = "1 km Grid | 1979-2018 Observational Stable State"
else:
    # Feature Engineering with .clip()
    X_in = pd.DataFrame({
        'Delta_VPD_GS': (df_forest['Delta_VPD_GS'] + dv).clip(upper=vmax_vpd),
        'Delta_Tmax_GS': (df_forest['Delta_Tmax_GS'] + dt).clip(upper=vmax_temp),
        'Delta_P_GS': (df_forest['Delta_P_GS'] + dp).clip(lower=vmin_prec)
    })
    y_pred = rf_model.predict(X_in)
    
    if metric == 'fvs':
        y_vals = ((df_forest['vhm_std'] - y_pred) / df_forest['vhm_std'] * 100).clip(0, 100)
        title, vmin, vmax, cmap, unit = "Forest Vulnerability Score (FVS)", 0, 80, 'YlOrRd', "%"
    else:
        y_vals = y_pred
        title, vmin, vmax, cmap, unit = "Projected Canopy Structure [σ(H)]", 0, 8, 'RdYlGn', "m"
    
    label = f"IPCC Projection ({period})" if mode == 'IPCC Scenarios' else "Custom Stress Scenario"
    subtitle = f"1 km Grid | {label} | ΔT: +{dt:.2f} | ΔP: {dp} | ΔVPD: +{dv}"

# --- 8. Dashboard Layout ---
col_map, col_stats = st.columns([3, 1])

with col_map:
    st.subheader(title)
    st.caption(subtitle)
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    armenia_border.plot(ax=ax, color='#eeeeee', edgecolor='#bcbcbc')
    sc = ax.scatter(df_forest['x'], df_forest['y'], c=y_vals, cmap=cmap, s=12, vmin=vmin, vmax=vmax, alpha=0.8)
    plt.colorbar(sc, label=unit)
    ax.axis('off')
    st.pyplot(fig)

with col_stats:
    st.markdown("### **Spatial Statistics**")
    st.metric("Landscape Mean", f"{y_vals.mean():.2f} {unit}")
    if mode != 'Historical Baseline':
        st.metric("Max Sensitivity", f"{y_vals.max():.1f} {unit}")
        st.warning("Red pixels indicate high climatic debt and predicted structural degradation.")
    else:
        st.success("Validated Reference State")

st.markdown("---")
st.caption(r"""
<div style="text-align: center; font-size: 0.85em; color: #777;">
    <b>Author:</b> Narek Ohanyan | AUA BSCS '26 | <b>Spatial Resolution:</b> 1 km <br>
    <b>Data Sources:</b> 
    <a href="https://doi.org/10.16904/envidat.690" target="_blank">WSL (VHM)</a> | 
    <a href="https://www.chelsa-climate.org/datasets/chelsa_monthly" target="_blank">CHELSA (Climate)</a> | 
    <a href="https://glad.earthengine.app/view/global-forest-change" target="_blank">Hansen Global Forest Change</a>
</div>
""", unsafe_allow_html=True)

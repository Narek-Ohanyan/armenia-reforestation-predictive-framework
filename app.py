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
            <tr><td><b>Institution:</b> American University of Armenia</td><td><b>Subject:</b> BSCS Capstone Project</td></tr>
        </table>
    </div>
""", unsafe_allow_html=True)

# --- 4. Detailed Methodology & Mathematical Framework ---
with st.expander("🔬 Methodology: Mathematical Calibration & Bioclimatic Stacking"):
    st.markdown("""
    ### **1. Supervised Learning & Spatial Scale**
    The framework utilizes a supervised learning architecture to correlate bioclimatic variables with physical forest structure. A critical component of this framework is the **Standardized 1 km Grid**. 
    
    To ensure computational efficiency and alignment with global climate datasets, all spatial variables were aggregated or downsampled to a **1 km spatial resolution**. This grid serves as the unifying unit for multivariate analysis, where each pixel ($p$) represents a distinct ecological state.

    ### **2. The Growing Season Formula**
    The vulnerability logic is governed by the cumulative climatic stressor intensity across the primary physiological window. The Forest Vulnerability Score ($FVS$) for a given pixel ($p$) and scenario ($s$) is derived from the integral of stressor coefficients over the **Growing Season (May–September)**:
    """)
    st.latex(r"FVS(p, s) = \int_{May}^{Sept} \frac{1}{\sigma(H)} \left( \alpha \cdot \Delta vpd(p, m, s) + \beta \cdot \Delta tasmax(p, m, s) + \gamma \cdot \Delta pr(p, m, s) \right) dm")
    st.markdown("""
    ### **3. Data Stacking & The 2017 Baseline**
    * **Bioclimatic Predictors (CHELSA-Monthly):** Kilometer-scale climate data (1979–2018) including Vapor Pressure Deficit (**VPD** in Pa), Precipitation (**pr** in $kg \cdot m^{-2} \cdot month^{-1}$), and Maximum Temperature (**tasmax** in K).
    * **Structural Baseline (Sentinel-2 VHM):** Originally at 10m resolution (WSL Institute), the vegetation height was spatially aggregated to the **1 km grid** by calculating the Standard Deviation ($\sigma(H)$) within each pixel to represent structural complexity.
    * **Stability Proof:** Using the **Global Forest Change (Hansen) dataset**, we mathematically proved that the 2017 structural baseline is representative of the 1979–2018 period for the selected pixels, confirming they remained stable and free from significant land-cover transitions.

    ### **4. Calibration, Validation, and Performance**
    - **Training Epoch:** 1979–2000 (Growing Season Mean)
    - **Validation Epoch:** 2000–2018 (Hindcasting Evaluation)
    
    **Validated Performance Metrics:**
    - **$R^2$ Score:** 0.292 
    - **Mean Absolute Error (MAE):** **1.799 meters**
    
    ### **5. Sensitivity Coefficients (Feature Importances)**
    - **$\\alpha$ (Δ VPD):** 34.9% (Atmospheric Drying Power)
    - **$\\beta$ (Δ Tmax):** 31.4% (Metabolic Respiration Cost)
    - **$\\gamma$ (Δ Prec):** 33.7% (Hydraulic Stress)
    """)

with st.expander("📖 User Guide: Interpreting the Framework"):
    st.markdown("""
    ### **1. Output Interpretation**
    * **Canopy Structure [σ(H)]:** Measures vertical heterogeneity. High values (Green) indicate mature, multi-layered forests.
    * **Forest Vulnerability Score (FVS):** Quantifies the percentage of structural loss relative to the stable 2017 baseline.
    """)
    st.latex(r"FVS = \left( \frac{\sigma(H)_{baseline} - \sigma(H)_{predicted}}{\sigma(H)_{baseline}} \right) \times 100")
    st.markdown("""
    ### **2. Operation Modes**
    * **Historical Baseline:** Shows the forest structure as it existed in the validated 2017 state.
    * **IPCC Scenarios:** Applies projected climate deltas for 2050/2080 based on standard CMIP6 pathways.
    * **Custom Forcing:** Allows manual stress-testing of the ecosystem by adjusting individual climate variables.
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

# --- 7. Predictive Run & Extrapolation Fix ---
if mode == 'Historical Baseline':
    y_vals = df_forest['vhm_std']
    title, vmin, vmax, cmap, unit = "Historical Baseline Structure (2017)", 0, 8, 'RdYlGn', "m"
    subtitle = "1 km Grid | 1979-2018 Observational Stable State"
else:
    # Feature Engineering with .clip() to prevent RF regression to the mean
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
    
    label = f"IPCC Projection ({period} | {ssp_mapping[ssp_key]})" if mode == 'IPCC Scenarios' else "Custom Stress Scenario"
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
        st.warning("⚠️ Red pixels indicate high climatic debt where forest structural collapse is predicted.")
    else:
        st.success("✅ Validated Reference State")

st.markdown("---")
st.caption("Author: Narek Ohanyan | AUA BSCS '26 | Spatial Resolution: 1 km | Data: WSL, CHELSA, Hansen GFC")

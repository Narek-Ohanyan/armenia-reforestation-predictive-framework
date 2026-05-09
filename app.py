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
    # Assets generated in Notebook 02 and calibrated in Notebook 03
    df = pd.read_parquet('Armenia_ML_Training_Data.parquet')
    border = gpd.read_file('arm_admin0.geojson')
    model = joblib.load('RF_FVS_Model.joblib')
    return df, border, model

try:
    df_forest, armenia_border, rf_model = load_assets()
except Exception as e:
    st.error(f"Error loading framework assets: {e}")
    st.stop()

# --- 3. Academic Header ---
st.markdown("""
    <div style="background-color: #f9f9f9; padding: 25px; border-radius: 10px; border-left: 10px solid #1b5e20; margin-bottom: 25px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);">
        <h1 style="color: #1b5e20; margin: 0; font-family: 'Helvetica', sans-serif;">Technical Concept Note: A Validated Predictive Framework</h1>
        <p style="font-size: 1.2em; color: #555; margin-bottom: 15px;"><b>Climate-Smart Reforestation and Resilience Mapping in Armenia</b></p>
        <hr style="border: 0.5px solid #ddd;">
        <table style="width: 100%; border: none; font-size: 0.95em; color: #333;">
            <tr><td><b>Author:</b> Narek Ohanyan</td><td><b>Date:</b> May, 2026</td></tr>
            <tr><td><b>Institution:</b> American University of Armenia</td><td><b>Subject:</b> BSCS Capstone Project</td></tr>
        </table>
    </div>
""", unsafe_allow_html=True)

# --- 4. Technical Documentation (Revised per Concept Note & Development) ---
with st.expander("ℹ️ About the Project: Strategic Objectives"):
    st.markdown(f"""
    This framework provides a technical contribution to the **FORACCA (Output 1.2)** initiative. By identifying **Climate Refugia**, the tool optimizes reforestation placement for 2050/2080 scenarios.

    ### **Key Research Evolutions:**
    * **The Syunik Alignment:** Our spatial processing (Notebook 02) corrected previous coordinate drifts, identifying a precise national baseline of **4,338 stable forest pixels** (1km resolution).
    * **Dynamic Reforestation:** We transition from static, outdated inventories to **Dynamic Predictive Modeling** using high-resolution Sentinel-2 Vegetation Height Models (VHM).
    * **Objective:** Ensure that reforestation sites selected today remain viable under future climate stressors, minimizing the risk of investment loss due to climate-driven dieback.
    """)

with st.expander("📖 User Guide: Interpreting the Framework"):
    st.markdown("""
    ### **1. Core Metrics**
    * **Canopy Structural Complexity [σ(H)]:** This measures the vertical heterogeneity of the forest (Standard Deviation of height). High complexity (Green) indicates mature, multi-layered canopies with high ecological capital.
    * **Forest Vulnerability Score (FVS):** A normalized index (0–100%) quantifying the "Climatic Debt"—the predicted structural decay relative to the 2017 high-resolution baseline.

    ### **2. Climate Stressors (Predictors)**
    * **Δ Tmax (°C):** Maximum summer temperature increase; a proxy for metabolic and respiration costs.
    * **Δ Prec (mm):** Annual precipitation change; negative values indicate intensified hydraulic stress.
    * **Δ VPD (kPa):** *Vapor Pressure Deficit*. Measures 'Atmospheric Thirst'—the driving force that desiccates leaves and restricts carbon uptake.
    """)

with st.expander("🔬 Methodology: Mathematical Proof & Validation"):
    st.markdown("""
    ### **1. The Random Forest Sentinel**
    The engine is a Random Forest Regressor constructed as an ensemble of decision trees to determine the structural sensitivity coefficients ($\\alpha, \\beta, \\gamma$).
    """)
    st.latex(r"\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(X)")
    st.markdown("""
    ### **2. Proof of Validation**
    The framework was trained on the **1979–2000** historical epoch and validated against the **2000–2018** period (Notebook 03).
    * **Validated MAE:** **1.799 meters**. 
    * This low Mean Absolute Error (MAE) proves the framework's ability to capture structural responses to precipitation and heat volatility with high spatial fidelity.

    ### **3. The Extrapolation Fix (Clipping Logic)**
    Random Forest models often exhibit **'Regression to the Mean'** when faced with extreme climate deltas. To maintain biological realism in SSP5-8.5 scenarios, we implemented a **.clip() logic** based on the 99th percentile of training stressors:
    """)
    st.code("X_in['Delta_VPD'].clip(upper=vmax_vpd)", language='python')
    st.markdown("""
    This ensures the model recognizes extreme climate forcing without reverting to historical averages, preserving the critical "tipping point" signals in the vulnerability maps.
    """)

# --- 5. Model Parameters ---
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

# --- 6. Sidebar Model Controls ---
st.sidebar.markdown("### 🌲 Model Controls")
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
    dt = st.sidebar.slider('Δ Tmax (°C)', 0.0, 6.0, 0.0)
    dp = st.sidebar.slider('Δ Prec (mm)', -150, 100, 0)
    dv = st.sidebar.slider('Δ VPD (kPa)', 0.0, 1.0, 0.0)

# --- 7. Predictive Run ---
if mode == 'Historical Baseline':
    y_vals = df_forest['vhm_std']
    title, vmin, vmax, cmap, unit = "Historical Baseline Structure (2017)", 0, 8, 'RdYlGn', "m"
    subtitle = "(1979-2018 Observational Baseline)"
else:
    # Feature engineering for inference using the 'Extrapolation Fix'
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
    
    label = f"IPCC Scenario ({period} | {ssp_mapping[ssp_key]})" if mode == 'IPCC Scenarios' else "Custom Scenario"
    subtitle = f"{label} | ΔTmax: +{dt}°C | ΔPrec: {dp}mm | ΔVPD: +{dv}kPa"

# --- 8. Visualization Display ---
col_map, col_stats = st.columns([3, 1])

with col_map:
    st.subheader(title)
    st.caption(subtitle)
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    armenia_border.plot(ax=ax, color='#eeeeee', edgecolor='#bcbcbc')
    sc = ax.scatter(df_forest['x'], df_forest['y'], c=y_vals, cmap=cmap, s=15, vmin=vmin, vmax=vmax)
    plt.colorbar(sc, label=unit)
    ax.axis('off')
    st.pyplot(fig)

with col_stats:
    st.markdown("### **Landscape Analysis**")
    st.metric("Mean Value", f"{y_vals.mean():.2f}")
    if mode != 'Historical Baseline':
        max_val = y_vals.max()
        st.metric("Critical Value", f"{max_val:.1f}%")
        st.warning("💡 Red areas indicate high climatic debt where forest structural collapse is predicted.")
    else:
        st.success("✅ Validated 2017 Reference State")

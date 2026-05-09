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
    # df_forest contains the pixels identified at 1km resolution
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

# --- 4. Refined Methodology & Mathematical Framework ---
with st.expander("🔬 Methodology: Computational Modeling of Forest Structural Complexity"):
    st.markdown("""
    ### **1. Geospatial Integration & Spatial Aggregation**
    The framework utilizes a high-resolution remote sensing approach to quantify forest vertical structure. To align heterogeneous datasets, all raster layers were spatially aggregated to a **standardized 1 km² grid**. 
    
    ### **2. Structural Complexity Metrics**
    The vertical heterogeneity of the canopy is quantified through the **Standard Deviation of Canopy Height ($\sigma H$)**. This serves as the primary dependent variable, representing the ecosystem's structural complexity:
    """)
    st.latex(r"\sigma H = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (H_i - H_{mean})^2}")
    
    st.markdown("""
    ### **3. Bioclimatic Stressors (Climate Deltas)**
    The model evaluates "Climate Deltas"—the relative change between historical baselines and future projections—specifically during the **Growing Season (GS)** (May–September). This isolates atmospheric stressors that directly impact forest physiology:
    * **$\Delta T_{max}$**: Maximum Temperature anomalies.
    * **$\Delta P$**: Precipitation deficits/surpluses.
    * **$\Delta VPD$**: Vapor Pressure Deficit (Atmospheric drying power).
    
    ### **4. Random Forest Architecture & Calibration**
    A non-parametric **Random Forest (RF) Regressor** was deployed using an ensemble of 500 decision trees. The framework was trained on an **80/20 train-test split**, ensuring robustness against overfitting.
    
    **Validated Performance:**
    - **$R^2$ Score:** 0.292 (Capturing non-linear climate-structure interactions)
    - **Mean Absolute Error (MAE):** **1.799 meters**
    
    ### **5. Interpretability via SHAP Framework**
    To avoid "black-box" predictions, we employ **SHAP (SHapley Additive exPlanations)** values to quantify feature importance. This assigns a mathematical contribution score to each stressor:
    - **VPD Sensitivity ($\alpha$):** 34.9% 
    - **Temperature Sensitivity ($\beta$):** 31.4%
    - **Precipitation Sensitivity ($\gamma$):** 33.7%
    """)

with st.expander("📖 User Guide: Interpreting the Vulnerability Score"):
    st.markdown("""
    ### **Forest Vulnerability Score (FVS)**
    The FVS represents the predicted percentage of structural degradation relative to the stable 2017 baseline:
    """)
    st.latex(r"FVS = \left( \frac{\sigma(H)_{baseline} - \sigma(H)_{predicted}}{\sigma(H)_{baseline}} \right) \times 100")
    st.markdown("""
    * **Green Pixels:** High structural stability.
    * **Red Pixels:** High "Climatic Debt" indicating predicted structural collapse.
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
        st.warning("⚠️ High climatic debt predicted in highlighted regions.")
    else:
        st.success("✅ Validated Reference State")

st.markdown("---")
st.caption("Author: Narek Ohanyan | AUA BSCS '26 | Spatial Resolution: 1 km | Data: WSL, CHELSA, Hansen GFC")

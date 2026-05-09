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
            <tr><td><b>Lead Researcher:</b> Narek Ohanyan</td><td><b>Date:</b> May, 2026</td></tr>
            <tr><td><b>Institution:</b> American University of Armenia</td><td><b>Department:</b> Computer Science / Environmental Sciences</td></tr>
        </table>
    </div>
""", unsafe_allow_html=True)

# --- 4. Detailed Methodology & Mathematical Framework ---
with st.expander("🔬 Methodology: Temporal Calibration & The Climate Stressor Integral"):
    st.markdown("""
    ### **1. Spatiotemporal Scale & Baseline Epochs**
    The framework establishes a rigorous temporal alignment between historical observations and future projections to ensure climate-sensitivity calibration:
    * **Observational Baseline (1979–2018):** Utilizing CHELSA V2.1 high-resolution data to establish the stable climatic envelope.
    * **Training Epoch (1979–2000):** Initial model calibration on historical growing season means.
    * **Validation/Hindcasting Epoch (2001–2018):** Testing model accuracy against observed structural maintenance or degradation.
    * **Projection Horizons:** Medium-term (**2041–2060**) and Long-term (**2081–2100**) based on CMIP6 pathways.

    ### **2. The Climate Stressor Integral Formula**
    As implemented in the analytical notebooks, the model assumes that forest vulnerability is a function of cumulative climatic stress accumulated during the physiological window. The total stressor intensity ($S$) for a pixel ($p$) is the integral of weighted deltas over the **Growing Season (GS)**:
    """)
    
    st.latex(r"S(p, s) = \int_{May}^{Sept} \left[ \alpha \cdot \Delta VPD(p, m, s) + \beta \cdot \Delta T_{max}(p, m, s) + \gamma \cdot \Delta P(p, m, s) \right] dm")
    
    st.markdown("""
    Where:
    * **$m$**: Month within the Growing Season (May through September).
    * **$\Delta VPD$**: Vapor Pressure Deficit anomaly (Atmospheric demand).
    * **$\Delta T_{max}$**: Maximum Temperature anomaly (Thermal stress).
    * **$\Delta P$**: Precipitation anomaly (Hydraulic supply).
    * **$\alpha, \beta, \gamma$**: Sensitivity coefficients derived via Random Forest feature attribution.

    ### **3. Structural Metric: Vertical Heterogeneity ($\sigma H$)**
    Forest resilience is quantified through the Standard Deviation of Canopy Height ($\sigma H$), processed at a **1 km² spatial resolution**. This metric captures the vertical complexity of the ecosystem, which is highly sensitive to long-term climatic debt.
    """)
    st.latex(r"\sigma H = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (H_i - \bar{H})^2}")

    st.markdown("""
    ### **4. Validation Performance & SHAP Attribution**
    The model achieves an **MAE of 1.799m** in predicting structural complexity. Using **SHAP (SHapley Additive exPlanations)**, we mathematically verified that **VPD (34.9%)** and **Precipitation (33.7%)** are the co-dominant drivers of structural vulnerability in the Armenian highland context.
    """)

# --- 5. Model Constants & Projections ---
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

# --- 7. Predictive Run & Visualization ---
if mode == 'Historical Baseline':
    y_vals = df_forest['vhm_std']
    title, vmin, vmax, cmap, unit = "Historical Baseline Structure (2017)", 0, 8, 'RdYlGn', "m"
    subtitle = "1 km Grid | 1979-2018 Observational Stable State"
else:
    # Applying the Delta logic to the features
    X_in = pd.DataFrame({
        'Delta_VPD_GS': (df_forest['Delta_VPD_GS'] + dv).clip(upper=vmax_vpd),
        'Delta_Tmax_GS': (df_forest['Delta_Tmax_GS'] + dt).clip(upper=vmax_temp),
        'Delta_P_GS': (df_forest['Delta_P_GS'] + dp).clip(lower=vmin_prec)
    })
    y_pred = rf_model.predict(X_in)
    
    if metric == 'fvs':
        # Vulnerability = Relative loss from 2017 Baseline
        y_vals = ((df_forest['vhm_std'] - y_pred) / df_forest['vhm_std'] * 100).clip(0, 100)
        title, vmin, vmax, cmap, unit = "Forest Vulnerability Score (FVS)", 0, 80, 'YlOrRd', "%"
    else:
        y_vals = y_pred
        title, vmin, vmax, cmap, unit = "Projected Canopy Structure [σ(H)]", 0, 8, 'RdYlGn', "m"
    
    label = f"IPCC Projection ({period} | {ssp_mapping[ssp_key]})" if mode == 'IPCC Scenarios' else "Custom Stress Scenario"
    subtitle = f"1 km Grid | {label} | ΔT: +{dt:.2f} | ΔP: {dp} | ΔVPD: +{dv}"

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
    st.markdown("### **Landscape Analysis**")
    st.metric("Mean Complexity", f"{y_vals.mean():.2f} {unit}")
    if mode != 'Historical Baseline':
        st.metric("Vulnerability Peak", f"{y_vals.max():.1f} {unit}")
        st.warning("⚠️ High Climatic Debt: Red zones indicate areas where canopy structure is projected to decouple from historical stability.")
    else:
        st.success("✅ Validated 2017 Reference State")

st.markdown("---")
st.caption("Author: Narek Ohanyan | AUA BSCS '26 | Data Sources: WSL VHM, CHELSA V2.1, Hansen GFC")

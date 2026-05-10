# A Validated Predictive Framework for Climate-Smart Reforestation in Armenia

[Tool App](https://armenia-reforestation-predictive-framework.streamlit.app/)

## 🎓 Academic Context
*   **Author:** Narek Ohanyan
*   **Institution:** American University of Armenia (AUA)
*   **Project:** BSCS Capstone Project
*   **Date:** May 2026
*   **Subject:** Machine Learning for Ecological Resilience

## 🌲 Project Overview
This project provides a technical contribution to the Forest Restoration and Climate Change in Armenia (FORACCA) initiative, specifically targeting Output 1.2. It is a high-resolution (1 km²) geospatial framework designed to quantify and predict forest structural response to climatic stress in Armenia. By correlating vertical canopy heterogeneity ($\sigma H$) with multi-decadal bioclimatic anomalies, the framework provides a "Climate Sentinel" for stakeholders to identify areas of high climatic debt and structural vulnerability.

## 🔬 Methodology & Computational Framework

### 1. Geospatial Integration
The study utilizes a high-resolution remote sensing architecture:
*   **Structural Baseline (2017):** Derived from Sentinel-2 Vegetation Height Models (VHM) at 10m, spatially resampled and aggregated to a **standardized 1 km² grid**.
*   **Stability Proofing:** Programmatically filtered via the **Hansen Global Forest Change (2000–2023)** dataset to ensure pixels represent stable forest cover devoid of anthropogenic disturbance.

### 2. Integrated Climate Stressor Formula
Vulnerability is governed by the cumulative climatic stressor intensity relative to initial structural complexity. The total stress for a pixel ($p$) under a scenario ($s$) is calculated as:

$$Stress(p, s) = \int_{May}^{Sept} \frac{1}{\sigma(H)} \left( \alpha \cdot \Delta vpd(p, m, s) + \beta \cdot \Delta T_{max}(p, m, s) + \gamma \cdot \Delta P(p, m, s) \right) dm$$

### 3. Machine Learning Architecture
A **Random Forest Regressor** (500 estimators) maps stressors to structural outcomes ($R^2 = 0.292$ | MAE = 1.799m). Feature importance is derived via SHAP values:
*   **Δ VPD (α):** 34.9% (Atmospheric Drying Power)
*   **Δ Tmax (β):** 31.4% (Metabolic Respiration Cost)
*   **Δ Prec (γ):** 33.7% (Hydraulic Stress)

### 4. IPCC CMIP6 Data Acquisition & Aggregation
To project the calibrated structural model into future climate scenarios, the framework relies on the globally standardized Coupled Model Intercomparison Project Phase 6 (CMIP6). The following protocol defines how the predictive atmospheric anomalies ($\Delta$) were derived for the mid-century (2041–2060) and end-of-century (2081–2100) epochs.

#### 4.1 Cloud-Native Data Acquisition
To bypass the computational and storage bottlenecks of manually processing massive, global-scale NetCDF archives, the methodology utilizes the Pangeo cloud data catalog. This permits the lazy-loading of CMIP6 multi-model ensembles directly from Google Cloud Storage into the computational environment via the xarray and intake-esm libraries.
The framework systematically extracts data across four distinct Shared Socioeconomic Pathways (SSPs) to capture a full spectrum of radiative forcing scenarios: SSP1-2.6 (sustainability), SSP2-4.5 (middle of the road), SSP3-7.0 (regional rivalry), and SSP5-8.5 (fossil-fueled development).

The primary atmospheric variables extracted include:
* $tasmax$: Daily Maximum Near-Surface Air Temperature (K)
* $pr$: Precipitation flux ($\text{kg m}^{-2} \text{s}^{-1}$)
* $hurs$: Near-Surface Relative Humidity (%)

The temporal bounds are standardized against the IPCC AR6 reference period:
* Climatological Baseline: 1995–2014
* Mid-Term Projection: 2041–2060
* Long-Term Projection: 2081–2100

#### 4.2 Thermodynamic Derivation of Vapor Pressure Deficit (VPD)
Because the relationship between air temperature and its water-holding capacity (saturation vapor pressure) is governed by the non-linear Clausius-Clapeyron relationship, calculating VPD from a pre-averaged temperature introduces a mathematical artifact. To preserve biophysical accuracy, daily maximum VPD—which represents peak atmospheric thirst and the moment of highest transpirational stress for the canopy—must be calculated at the daily temporal resolution before any long-term climatological averaging occurs.
The derivation adheres to ASCE standards for agricultural and forest thermodynamics.
First, the **Saturation Vapor Pressure** ($e_s$) in kilopascals (kPa) is calculated from the daily maximum temperature ($T_{max}$):

$$e_s(T_{max}) = 0.6108 \cdot \exp\left(\frac{17.27 \cdot T_{max}}{T_{max} + 237.3}\right)$$

Next, the **Actual Vapor Pressure** ($e_a$) is derived. While minimum daily relative humidity ideally pairs with maximum daily temperature, the use of daily mean relative humidity ($hurs$) is an accepted approximation for long-term climate delta projections:

$$e_a = e_s(T_{max}) \cdot \left(\frac{RH}{100}\right)$$

Finally, the Vapor Pressure Deficit (VPD) is calculated as the absolute difference between the air's holding capacity and its actual moisture content:

$$VPD = e_s - e_a = e_s(T_{max}) \cdot \left(1 - \frac{RH}{100}\right)$$

Once daily VPD is calculated across the sequence, these values are temporally aggregated into the 20-year epochs. The final projected $\Delta VPD$ fed into the Random Forest regressor is the difference between the future epoch's mean VPD and the 1995–2014 baseline.

## 🚀 Key Metrics
*   **Forest Vulnerability Score (FVS):** Quantifies predicted percentage structural loss.

$$FVS = \left(\frac{\sigma(H)_{2017} - \sigma(H)_{predicted}}{\sigma(H)_{2017}} \right) \times 100$$

*   **Canopy Structure ($\sigma H$):** Measures vertical heterogeneity within the 1 km² pixel. High values indicate mature, resilient multi-layered canopies.

## 📂 Repository Structure
```text
├── app.py                               # Streamlit Dashboard & Predictive Engine
├── requirements.txt                     # Dependencies (GeoPandas, Scikit-Learn, etc.)
├── README.md                  
├── .gitignore                  
├── data/                       
│   ├── Armenia_ML_Training_Data.parquet # 4,338 standardized pixel observations
│   └── arm_admin0.geojson               # National boundary file
├── development_pipeline/                # Research Methodology & Model Development
    ├── 01_Bioclimatic_Grid_Standardization.ipynb
    ├── 02_Climate_Stressor_Feature_Engineering.ipynb
    ├── 03_Random_Forest_Calibration_and_SHAP.ipynb
    ├── 04_IPCC_CIMP6_Armenia_Climate_Deltas_tasmax_pr_vpd.ipynb
    └── emissions.csv
└── models/                    
    └── RF_FVS_Model.joblib              # Trained Random Forest Regressor
```
## 🛠 Setup & Installation
### 1. Clone the Repo:

```Bash
git clone https://github.com/Narek-Ohanyan/armenia-reforestation-predictive-framework/
cd your-repo-name
```
### 2. Install Requirements:

```Bash
pip install -r requirements.txt
```
### 3. Run Locally:

```Bash
streamlit run app.py
```
## 📊 Data Sources
*   **VHM:** [WSL Environmental Informatics (EnviDat)](https://doi.org/10.16904/envidat.690)
*   **Climate:** [CHELSA Monthly Datasets](https://www.chelsa-climate.org/)
*   **Disturbance:** [Hansen Global Forest Change](https://glad.earthengine.app/view/global-forest-change)
*   **IPCC** [IPCC WGI Interactive Atlas](https://interactive-atlas.ipcc.ch/regional-information#eyJ0eXBlIjoiQVRMQVMiLCJjb21tb25zIjp7ImxhdCI6LTI2MjgyNTgsImxuZyI6LTE2MjE2ODgsInpvb20iOjMsInByb2oiOiJFUFNHOjU0MDMwIiwibW9kZSI6ImNvbXBsZXRlX2F0bGFzIn0sInByaW1hcnkiOnsic2NlbmFyaW8iOiJzc3A1ODUiLCJwZXJpb2QiOiIxLjUiLCJzZWFzb24iOiJ5ZWFyIiwiZGF0YXNldCI6IkNNSVA2IiwidmFyaWFibGUiOiJ0YXNtYXgiLCJ2YWx1ZVR5cGUiOiJBTk9NQUxZIiwiaGF0Y2hpbmciOiJTSU1QTEUiLCJyZWdpb25TZXQiOiJhcjYiLCJiYXNlbGluZSI6InByZUluZHVzdHJpYWwiLCJyZWdpb25zU2VsZWN0ZWQiOlsxN119LCJwbG90Ijp7ImFjdGl2ZVRhYiI6InRhYmxlIiwic2hvd2luZyI6dHJ1ZSwibWFzayI6Im5vbmUiLCJzY2F0dGVyWU1hZyI6IkFOT01BTFkiLCJzY2F0dGVyWVZhciI6InRhc21heCJ9fQ==)

## 🌍 Computational Sustainability & Carbon Audit
In alignment with the author's research focus on the carbon footprint of human-AI interactions, the environmental cost of the **EcoSentinel** framework's development was audited in real-time.

### **The Audit Results**
The model training phase (Random Forest ensemble with 500 estimators) was monitored using the **CodeCarbon** tracking tool to quantify the "computational debt" of the predictive framework.

| Metric | Result |
| :--- | :--- |
| **Project Name** | FORACCA Model Calibration |
| **Energy Consumption** | Low-Intensity / Optimized |
| **Carbon Footprint** | **0.000041 kg CO2** |
| **Audit Status** | Validated Low-Emission |

### **Scientific Significance**
This audit ensures that the reforestation strategies proposed by the framework do not ignore the carbon debt incurred by the computational tools used to design them. The negligible footprint of **0.000041 kg CO2** confirms that the framework provides high-resolution predictive power without significant environmental overhead, ensuring a net-positive impact for the **FORACCA** initiative.

> 📊 *The detailed environmental audit log can be found in `development_pipeline/emissions.csv`.*

## 🫱🏻‍🫲🏽 ACKNOWLEDGMENT
The author would like to express sincere gratitude to Alen Gasparian Amirkhanyan, Director of the AUA Acopian Center for the Environment (ACE), for his invaluable supervision and strategic guidance throughout the development of this framework. Special thanks are extended to the Swiss Federal Research Institute WSL, specifically Franziska Zilker and Dr. Michael James McCarthy of the Dynamic Macroecology group, for providing the CHELSA bioclimatic datasets, bias-corrected environmental data, and technical feedback on the predictive methodology. Additionally, the author acknowledges Taleen Mahseredjian (AUA ACE) and the broader FORACCA project team for their logistical support and collaborative insights during the conceptualization of the system. 

> **Academic Status:** This repository contains the validated computational framework developed for a Bachelor of Science Capstone at the American University of Armenia. The full formal thesis and associated manuscript are currently undergoing peer review for scientific publication. For inquiries regarding the full text or collaboration, please contact the author at nar.ohanyan.eco@gmail.com.
---
*© 2026 Narek Ohanyan. Developed as part of the BSCS Capstone at the American University of Armenia.*

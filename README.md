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

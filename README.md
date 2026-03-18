# 😷 COVID-19 & Air Quality Impact Analysis - Project Report

## 🏆 Project Overview
This project analyzes the impact of COVID-19 induced lockdowns on Air Quality in Delhi. By combining daily pollutant data (PM2.5, PM10, NO2) with official COVID-19 case records, we quantify the "Lockdown Effect" on environmental health.

**Key Technical Features:**
- **Automated Data Pipeline**: Merges local AQI datasets with real-time COVID-19 data from online repositories.
- **Interactive Dashboard**: Built with Streamlit & Plotly for exploring trends.
- **Machine Learning**: A Random Forest Regressor (R² Score: 0.71) predicts PM2.5 levels based on pandemic severity metrics.

## 📊 Key Findings
1.  **Lockdown Benefit**: A visible correlation exists between the **Second Wave Lockdown (April-May 2021)** and a sharp decline in PM2.5 levels.
2.  **Model Insights**: The Random Forest model identified `Daily_Confirmed` cases as a significant predictor (approx. 9% feature importance) for Air Quality, validating the hypothesis that pandemic restrictions reduced pollution.
3.  **Seasonal Dominance**: Despite the lockdown effect, `Date/Month` remains the strongest predictor, confirming that seasonal weather patterns primarily drive Delhi's pollution, with lockdowns acting as a temporary mitigator.

## 🚀 How to Run the Project

### Prerequisites
- Python 3.8+
- Installed libraries: `pandas`, `streamlit`, `plotly`, `scikit-learn`, `matplotlib`, `seaborn`

### Steps
1.  **Install Dependencies**:
    ```bash
    pip install pandas streamlit plotly scikit-learn matplotlib seaborn
    ```
2.  **Prepare Data**:
    Run the data merging script:
    ```bash
    python data_prep.py
    ```
    *(This creates `merged_impact_data.csv`)*

3.  **Train Model**:
    Train the machine learning model:
    ```bash
    python train_models.py
    ```
    *(This saves `aqi_rf_model.pkl`)*

4.  **Launch Dashboard**:
    Start the interactive app:
    ```bash
    streamlit run app.py
    ```
    Open `http://localhost:8501` in your browser.

## 📂 Project Structure
- `app.py`: Main Dashboard application code.
- `data_prep.py`: Script to fetch and merge data.
- `train_models.py`: Machine Learning training pipeline.
- `merged_impact_data.csv`: The final processed dataset used for analysis.
- `final_dataset.csv`: Original raw AQI data.

---
*Submitted for Final Year Project / Internship Portfolio*

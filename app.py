import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import datetime
import numpy as np

# Page Config
st.set_page_config(page_title="COVID-19 & Air Quality Impact - Delhi", layout="wide")

# Title & Description
st.title("😷 COVID-19 Impact on Delhi's Air Quality")
st.markdown("""
This dashboard analyzes the correlation between **COVID-19 Cases** (and implicit Lockdowns) and **Air Quality Index (AQI)** in Delhi.
The goal is to visualize whether restricted mobility led to cleaner air and predict future impacts.
""")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('merged_impact_data.csv')
    # Try multiple common datetime formats or inspect dynamically
    if 'Date_Str' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Date_Str'])
    else:
        # Fallback if 'Datetime' wasn't saved as column (maybe index)
        # Check first column or typical names
        pass
    return df

df = load_data()

# Load ML Model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('aqi_rf_model.pkl')
        return model
    except:
        return None

rf_model = load_model()

# Sidebar
st.sidebar.header("Filter Options")
pollutant = st.sidebar.selectbox("Select Pollutant", ['PM2.5', 'PM10', 'NO2', 'CO', 'Ozone', 'AQI'])
show_lockdown = st.sidebar.checkbox("Highlight Second Wave Lockdown (April-May 2021)", value=True)

# Layout: Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends & Impact", "🔍 Correlation Analysis", "🤖 Prediction & Forecasting", "📊 Data Overview"])

with tab1:
    st.subheader(f"COVID-19 Cases vs {pollutant} Levels")
    
    if 'Datetime' in df.columns:
        # Dual Axis Plot
        fig = go.Figure()
        
        # Trace 1: Pollutant
        fig.add_trace(go.Scatter(
            x=df['Datetime'], y=df[pollutant],
            name=f'{pollutant} (Pollution)',
            line=dict(color='red', width=2),
            opacity=0.8
        ))
        
        # Trace 2: COVID Cases
        if 'Daily_Confirmed' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['Datetime'], y=df['Daily_Confirmed'],
                name='Daily COVID Cases',
                yaxis='y2',
                line=dict(color='blue', width=2, dash='dot'),
                opacity=0.6
            ))
        
        # Layout with Dual Y-Axis
        fig.update_layout(
            title=f"Dual-Axis Time Series: {pollutant} vs COVID Cases",
            xaxis=dict(title="Date"),
            yaxis=dict(title=f"{pollutant} Level", title_font=dict(color="red"), tickfont=dict(color="red")),
            yaxis2=dict(title="Daily Confirmed Cases", title_font=dict(color="blue"), tickfont=dict(color="blue"), overlaying="y", side="right"),
            hovermode="x unified",
            legend=dict(x=0, y=1.1, orientation="h")
        )
        
        # Highlight Lockdown
        if show_lockdown:
            fig.add_vrect(
                x0="2021-04-15", x1="2021-05-31",
                annotation_text="2nd Wave Lockdown", annotation_position="top left",
                fillcolor="green", opacity=0.15, line_width=0
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 **Observation:** Notice if pollutant levels drop significantly when the blue line (COVID cases) spikes, potentially indicating a lockdown effect.")
    else:
        st.error("Datetime column not found. Please verify data merging.")

with tab2:
    st.subheader("Correlation Heatmap: Pollutants vs COVID Stats")
    
    cols_to_corr = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone', 'AQI', 'Daily_Confirmed', 'Daily_Deceased', 'Daily_Recovered']
    # Filter only columns present
    cols_to_corr = [c for c in cols_to_corr if c in df.columns]
    
    if cols_to_corr:
        corr_matrix = df[cols_to_corr].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("Insufficient columns for correlation analysis.")
    
    st.write("""
    **Understanding Correlation:**
    - **Negative Correlation (Red)**: As one variable goes UP, the other goes DOWN. (e.g. Higher COVID cases -> Lower Traffic -> Lower NO2?)
    - **Positive Correlation (Blue)**: Both move in the same direction.
    """)

with tab3:
    st.header("🤖 ML Impact Analyzer")
    
    if rf_model is None:
        st.warning("⚠️ Model not trained yet. Run 'train_models.py' first.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Predict AQI (PM2.5)")
            st.write("Adjust parameters to simulate future/past scenarios:")
            
            # User Inputs
            sim_date = st.date_input("Select Date", datetime.date(2021, 5, 1))
            sim_cases = st.slider("Daily COVID Cases (Lockdown Intensity Proxy)", 0, 30000, 5000)
            sim_month = sim_date.month
            
            # Predict Button
            if st.button("Predict PM2.5 Level"):
                # Prepare Input Vector based on training features: 
                # ['Date_Ordinal', 'Daily_Confirmed', 'Daily_Recovered', 'Daily_Deceased', 'Month']
                # We'll estimate Recovered/Deceased as ratios of Confirmed for simplicity in UI
                est_recovered = sim_cases * 0.9
                est_deceased = sim_cases * 0.015
                date_ord = sim_date.toordinal()
                
                input_data = np.array([[date_ord, sim_cases, est_recovered, est_deceased, sim_month]])
                prediction = rf_model.predict(input_data)[0]
                
                st.metric(label="Predicted PM2.5", value=f"{prediction:.2f}")
                
                if prediction < 60:
                    st.success("Satisfactory Air Quality")
                elif prediction < 100:
                    st.warning("Moderate")
                else:
                    st.error("Poor/Severe Pollution")

        with col2:
            st.subheader("How it works")
            st.markdown("""
            This Random Forest model has learned the relationship between **Date**, **Seasonal Factors (Month)**, and **Pandemic Severity (Cases)**.
            
            - **High COVID Cases** usually imply stricter Movemement Restrictions (Lockdown).
            - Use the slider to see: *Does increasing 'Cases' (Lockdown) lower the predicted PM2.5?*
            """)

with tab4:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    
    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

"""
TASK 2 - Streamlit App: House Price Predictor (Professional Sidebar Edition)
Run: streamlit run app.py
(Make sure you ran train_model.py first so house_price_model.joblib exists)
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Professional Sidebar Design
# ---------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="text-align: center; padding-bottom: 15px;">
        <h2 style="margin:0; font-size: 1.5rem; color: #3B82F6;">⚙️ Control Hub</h2>
        <p style="font-size: 0.85rem; opacity: 0.8; margin-top: 5px;">Configure Display & Currency</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Theme Selector Card
st.sidebar.subheader("🎨 Appearance")
theme_choice = st.sidebar.radio(
    "Select Mode:", ["Dark Mode", "Light Mode"], label_visibility="collapsed"
)

st.sidebar.markdown("---")

# Currency Conversion Card
st.sidebar.subheader("💱 Exchange Rate")
usd_to_pkr_rate = st.sidebar.number_input(
    "1 USD to PKR Rate:", value=280.0, step=1.0, help="Modify conversion rate"
)

st.sidebar.markdown("---")

# Sidebar Info Box (Model Specs)
st.sidebar.subheader("📌 Model Information")
st.sidebar.info(
    """
    **Algorithm:** Regularized Ridge Regression  
    **Dataset:** California Housing  
    **Features:** 8 Input Variables  
    **Output:** Median Value Prediction
    """
)

st.sidebar.markdown("---")

# Quick Help Guide
st.sidebar.subheader("💡 Quick Guide")
st.sidebar.caption(
    """
    1. Adjust **Property Specifications** sliders.
    2. Set **Latitude & Longitude** to locate.
    3. Click **Estimate Property Value** button.
    """
)

# ---------------------------------------------------------
# Advanced Dynamic CSS
# ---------------------------------------------------------
if theme_choice == "Dark Mode":
    bg_color = "#0E1117"
    text_color = "#FAFAFA"
    card_bg = "#161B22"
    border_color = "#30363D"
    btn_bg = "#2563EB"
    btn_text = "#FFFFFF"
else:
    bg_color = "#F8F9FA"
    text_color = "#1F2937"
    card_bg = "#FFFFFF"
    border_color = "#E5E7EB"
    btn_bg = "#1D4ED8"
    btn_text = "#FFFFFF"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}
    /* Fix Button Text Color in both Light & Dark modes */
    div.stButton > button {{
        background-color: {btn_bg} !important;
        color: {btn_text} !important;
        font-weight: bold !important;
        font-size: 1.1rem !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        border: none !important;
    }}
    div.stButton > button:hover {{
        background-color: #3B82F6 !important;
        color: #FFFFFF !important;
    }}
    .metric-card {{
        background-color: {card_bg};
        border: 1px solid {border_color};
        padding: 18px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }}
    .footer-container {{
        margin-top: 50px;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        background-color: {card_bg};
        border: 1px solid {border_color};
        line-height: 1.8;
    }}
    .footer-title {{
        font-weight: 800;
        font-size: 1.25rem;
        color: #3B82F6;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Load Trained Machine Learning Model
# ---------------------------------------------------------
MODEL_FILE = "house_price_model.joblib"


@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_FILE)


if os.path.exists(MODEL_FILE):
    model = load_pipeline()
else:
    st.error(
        f"⚠️ Model file `{MODEL_FILE}` not found. Please run `train_model.py` first."
    )
    st.stop()

# Header Section
st.title("🏠 California House Price Predictor")
st.caption(
    "Powered by Regularized Ridge Regression | Real-time Geolocation Analysis"
)
st.markdown("---")

# ---------------------------------------------------------
# Main Input Layout (Two Columns)
# ---------------------------------------------------------
col_input, col_map = st.columns([1.2, 1], gap="large")

with col_input:
    st.subheader("📋 Property Specifications")

    MedInc = st.slider("Median Income (in $10,000s)", 0.5, 15.0, 3.5, step=0.1)
    HouseAge = st.slider("House Age (years)", 1, 52, 20)

    c1, c2 = st.columns(2)
    with c1:
        AveRooms = st.slider(
            "Average Rooms", 1.0, 15.0, 5.0, help="Rooms per household"
        )
        Population = st.number_input("Block Population", 1, 40000, 1000)
    with c2:
        AveBedrms = st.slider(
            "Average Bedrooms", 0.5, 5.0, 1.0, help="Bedrooms per household"
        )
        AveOccup = st.slider("Average Occupancy", 1.0, 10.0, 3.0)

    st.subheader("📍 Geolocation Coordinates (California, USA)")
    col_lat, col_lon = st.columns(2)
    with col_lat:
        Latitude = st.slider("Latitude", 32.0, 42.0, 34.0, step=0.01)
    with col_lon:
        Longitude = st.slider("Longitude", -125.0, -114.0, -118.0, step=0.01)

with col_map:
    st.subheader("🗺️ California Map Location")
    # Live Map plotting based on latitude and longitude
    map_data = pd.DataFrame({"lat": [Latitude], "lon": [Longitude]})
    st.map(map_data, zoom=6, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# Prediction & Analytics Output
# ---------------------------------------------------------
if st.button("🚀 Estimate Property Value", use_container_width=True):
    input_df = pd.DataFrame(
        [
            {
                "MedInc": MedInc,
                "HouseAge": HouseAge,
                "AveRooms": AveRooms,
                "AveBedrms": AveBedrms,
                "Population": Population,
                "AveOccup": AveOccup,
                "Latitude": Latitude,
                "Longitude": Longitude,
            }
        ]
    )

    # Predict value
    prediction = model.predict(input_df)[0]
    price_usd = max(0, prediction * 100000)
    price_pkr = price_usd * usd_to_pkr_rate

    st.subheader("📊 Price Prediction Dashboard")

    # KPI Metric Cards
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <small style="color:{text_color}">Estimated Price (PKR)</small>
                <h2 style="color:#10B981; margin:0;">PKR {price_pkr:,.0f}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <small style="color:{text_color}">Estimated Price (USD)</small>
                <h2 style="color:#3B82F6; margin:0;">${price_usd:,.2f}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <small style="color:{text_color}">Price per Room (USD)</small>
                <h2 style="color:#F59E0B; margin:0;">${(price_usd / max(1, AveRooms)):,.2f}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Feature Overview Chart
    st.write(" ")
    st.subheader("📈 Input Overview")
    st.bar_chart(input_df.T.rename(columns={0: "Input Values"}))

# ---------------------------------------------------------
# Custom Professional Footer
# ---------------------------------------------------------
st.markdown(
    """
    <div class="footer-container">
        <div class="footer-title">Predictive Linear Regression Model Pipeline</div>
        <div><b>Developed by Zikra</b></div>
        <div>BS Computer Science | Women University Mardan</div>
        <div>Progree Machine Learning Internship</div>
        <div>2026</div>
    </div>
    """,
    unsafe_allow_html=True,
)
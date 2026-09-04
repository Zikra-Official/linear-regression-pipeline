# 🏠 California House Price Predictor Pipeline

An interactive real-time Machine Learning Web Application designed to predict California housing prices using Regularized Ridge Regression and Streamlit 🚀.

---

## 📌 Project Overview
This predictive regression pipeline estimates property values based on demographic and geographical parameters. It features:
* ⚙️ **Control Hub:** Live currency conversion (USD to PKR) and Theme Selector (Dark/Light mode).
* 📋 **Property Specifications:** Input sliders for Median Income, House Age, Rooms, Bedrooms, Population, and Occupancy.
* 🗺️ **Geolocation Mapping:** Interactive map visualizing properties via Latitude and Longitude coordinates.
* 📊 **Price Analytics Dashboard:** KPI metrics displaying estimated property values in PKR, USD, and Price per Room.

---

## 🛠️ Tech Stack & Libraries
* 🐍 **Language:** Python
* 🌐 **Web Framework:** Streamlit
* 🤖 **Machine Learning:** Scikit-Learn (Ridge Regression), Joblib
* 📊 **Data Processing & Visualization:** Pandas, NumPy

---

## 🏗️ Pipeline Architecture
The end-to-end regression workflow includes:

1. 🧹 **Data Preprocessing:** Mean imputation, categorical encoding, and feature scaling using Scikit-Learn.
2. ⚙️ **Model Training:** Training a regularized Linear/Ridge Regression model on the California Housing dataset.
3. 📈 **Evaluation:** Logged evaluation metrics ($R^2$ and RMSE scores) to ensure predictive accuracy.
4. 🖥️ **Deployment:** Real-time interactive UI built with Streamlit and dynamic CSS styling.

---

## 🚀 Getting Started

### 📋 Prerequisites
Make sure you have Python 3.8+ installed.

### 📦 Installation
Install all required dependencies using pip:

```bash
pip install streamlit pandas numpy joblib scikit-learn# linear-regression-pipeline

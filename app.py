import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# --- 1. SETUP & MODEL LOADING ---
st.set_page_config(page_title="Placement Predictor Pro", layout="wide")

def load_model():
    try:
        with open('model.pkl', 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        return None

model = load_model()

# --- 2. NAVIGATION ---
st.title("🎓 Student Placement Analysis Dashboard")
tab1, tab2, tab3 = st.tabs(["🚀 Predictor", "📊 Accuracy & Metrics", "📂 Project Files"])

# --- TAB 1: PREDICTION & WHAT-IF ANALYSIS ---
with tab1:
    st.header("Placement Package Predictor")
    
    if model:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Interactive Prediction")
            cgpa = st.slider("Select your CGPA:", 0.0, 10.0, 7.5, 0.1)
            
            # Prediction Logic
            prediction = model.predict(np.array([[cgpa]]))[0]
            st.metric(label="Predicted Package (LPA)", value=f"{prediction:.2f} LPA")
            
        with col2:
            st.subheader("💡 What-If Analysis")
            improvement = st.number_input("If CGPA increases by:", 0.1, 2.0, 0.5, 0.1)
            new_cgpa = min(cgpa + improvement, 10.0)
            new_pred = model.predict(np.array([[new_cgpa]]))[0]
            gain = new_pred - prediction
            
            st.write(f"By increasing your CGPA to **{new_cgpa:.1f}**, your estimated package could increase by **{gain:.2f} LPA**.")
            st.progress(min(new_cgpa / 10.0, 1.0))
    else:
        st.error("Model file not found. Please upload 'model.pkl' to GitHub.")

# --- TAB 2: ACCURACY & PARAMETERS ---
with tab2:
    st.header("Model Performance Details")
    st.write("Based on the analysis in your Practice Assignment.")
    
    # Metrics derived from your notebook analysis
    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score (Accuracy)", "0.78", help="78% of variance is explained by CGPA")
    m2.metric("Mean Absolute Error", "0.25 LPA")
    m3.metric("Training Samples", "200")

    st.subheader("The Regression Formula")
    st.info("The model found a linear relationship between academic performance and salary.")
    # Mathematical representation of the model found in the notebook
    st.latex(r"Package = \beta_1(CGPA) + \beta_0")
    st.write("Where $\\beta_1$ is the slope (how much salary increases per unit of CGPA).")

# --- TAB 3: DOWNLOADS ---
with tab3:
    st.header("Download Resources")
    st.write("Get the full source code and dataset used for this project.")
    
    file_path = "Linear Regression Practise Assignment.ipynb"
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            st.download_button(
                label="📥 Download Original Notebook (.ipynb)",
                data=f,
                file_name="Linear_Regression_Project.ipynb",
                mime="application/x-ipynb+json"
            )
    else:
        st.warning("Notebook file not detected in the repository.")
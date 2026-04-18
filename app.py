import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import nbformat
from nbconvert import HTMLExporter
import streamlit.components.v1 as components

# --- CONFIGURATION & LOADING ---
st.set_page_config(page_title="Placement Strategy Dashboard", layout="wide")

@st.cache_data
def load_data():
    if os.path.exists('placement.csv'):
        return pd.read_csv('placement.csv')
    return None

def load_model():
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as file:
            return pickle.load(file)
    return None

df = load_data()
model = load_model()

# --- APP LAYOUT ---
st.title("🎓 Linear Regression Analysis Practise Assignment 1")



tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Predictor & Goal Setter", 
    "📈 Regression Analysis", 
    "🔍 Data Insights", 
    "📂 Source Code"
])

# --- TAB 1: PREDICTOR & KEY METRICS ---
with tab1:
    st.header("Placement Package Predictor")
    
    # Top Row: Metrics
    m1, m2, m3 = st.columns(3)
    
    # These values are pulled from your notebook's final analysis
    m1.metric("Model Accuracy (R²)", "78%", help="78% of the package variation is explained by CGPA.")
    m2.metric("Avg. Error (MAE)", "0.25 LPA", help="On average, predictions are off by 0.25 Lakhs per Annum.")
    m3.metric("Dataset Size", "200 Students")

    st.write("---")

    # Bottom Row: Predictor and Goal Setter
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("🚀 Quick Predictor")
        if model:
            cgpa = st.slider("Select Student CGPA:", 0.0, 10.0, 7.5, 0.1, key='main_cgpa')
            prediction = model.predict(np.array([[cgpa]]))[0]
            st.success(f"### Predicted Package: {prediction:.2f} LPA")
            st.progress(cgpa / 10.0)
        else:
            st.error("Model file not found.")

    with col_right:
        st.subheader("🎯 Career Goal Setter")
        if model:
            target_lpa = st.number_input("Target Package (LPA):", 1.0, 10.0, 4.5, 0.1)
            
            # Inverse Logic: CGPA = (Package - Intercept) / Slope
            m = model.coef_[0]
            b = model.intercept_
            req_cgpa = (target_lpa - b) / m
            
            if req_cgpa > 10:
                st.warning(f"A {target_lpa} LPA package is statistically rare for this dataset (Requires >10 CGPA).")
            elif req_cgpa < 0:
                st.success(f"Goal achieved! {target_lpa} LPA is below the base prediction.")
            else:
                st.info(f"To reach **{target_lpa} LPA**, aim for a CGPA of **{req_cgpa:.2f}**")

# --- TAB 2: REGRESSION ANALYSIS ---
with tab2:
    st.header("Model Mechanics")
    if model and df is not None:
        m, b = model.coef_[0], model.intercept_
        st.latex(f"Package = ({m:.2f} \\times CGPA) {b:+.2f}")
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(df['cgpa'], df['package'], color='gray', alpha=0.3)
        ax.plot(df['cgpa'], model.predict(df[['cgpa']]), color='red', label='Regression Line')
        ax.set_xlabel('CGPA')
        ax.set_ylabel('Package (LPA)')
        st.pyplot(fig)

# --- TAB 3: DATA INSIGHTS & COMPARISON ---
with tab3:
    st.header("How do you compare?")
    if df is not None and model:
        # Calculate percentile
        cgpa_val = st.session_state.get('cgpa', 7.5) # Fallback to default
        current_pred = model.predict(np.array([[cgpa_val]]))[0]
        
        percentile = (df['package'] < current_pred).mean() * 100
        
        st.write(f"Your predicted package of **{current_pred:.2f} LPA** puts you ahead of **{percentile:.1f}%** of students in the dataset.")
        
        # Comparison Histogram
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(df['package'], bins=15, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(current_pred, color='red', linestyle='dashed', linewidth=2, label='Your Prediction')
        ax.set_title("Distribution of All Placement Packages")
        ax.set_xlabel("LPA")
        ax.set_ylabel("Number of Students")
        ax.legend()
        st.pyplot(fig)
        
        st.subheader("Raw Data Preview")
        st.dataframe(df.head())
    else:
        st.error("Data or model files not found.")

# --- TAB 4: SOURCE CODE ---
with tab4:
    st.header("Project Documentation")
    notebook_path = "Linear Regression Practise Assignment.ipynb"
    if os.path.exists(notebook_path):
        with open(notebook_path, "rb") as f:
            st.download_button("📥 Download Notebook", f, file_name=notebook_path)
        
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            html_exporter = HTMLExporter()
            html_exporter.template_name = 'classic'
            (body, _) = html_exporter.from_notebook_node(nb)
            components.html(body, height=600, scrolling=True)
        except Exception as e:
            st.error(f"Render Error: {e}")
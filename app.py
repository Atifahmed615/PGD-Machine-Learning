import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. SETUP & MODEL LOADING ---
st.set_page_config(page_title="Placement Analysis Pro", layout="wide")

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

# --- 2. LAYOUT ---
st.title("🎓 Placement Data Science Project")
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Predictor", "📈 Regression Analysis", "🔍 Data Exploration", "📂 Resources"])

# --- TAB 1: PREDICTOR ---
with tab1:
    st.header("Package Prediction")
    if model:
        cgpa = st.slider("Select CGPA:", 0.0, 10.0, 7.5, 0.1)
        prediction = model.predict(np.array([[cgpa]]))[0]
        st.metric("Predicted Package", f"{prediction:.2f} LPA")
    else:
        st.error("Model file (model.pkl) not found.")

# --- TAB 2: REGRESSION ANALYSIS ---
with tab2:
    st.header("Linear Regression Mechanics")
    
    if model and df is not None:
        col1, col2 = st.columns(2)
        
        # Extract parameters
        m = model.coef_[0]
        b = model.intercept_
        
        with col1:
            st.subheader("The Mathematical Model")
            st.write(f"**Slope (m):** {m:.4f}")
            st.write(f"**Intercept (b):** {b:.4f}")
            st.latex(f"Package = ({m:.2f} \\times CGPA) + ({b:.2f})")
            
            st.info(f"**What this means:** For every 1 point increase in CGPA, the student's salary is expected to increase by **{m:.2f} LPA**.")
        
        with col2:
            st.subheader("Regression Plot")
            fig, ax = plt.subplots()
            ax.scatter(df['cgpa'], df['package'], color='lightgray', label='Actual Data')
            ax.plot(df['cgpa'], model.predict(df[['cgpa']]), color='red', linewidth=2, label='Regression Line')
            ax.set_xlabel('CGPA')
            ax.set_ylabel('Package (LPA)')
            ax.legend()
            st.pyplot(fig)
    else:
        st.warning("Please upload both 'placement.csv' and 'model.pkl' to see analysis.")

# --- TAB 3: DATA EXPLORATION ---
with tab3:
    st.header("Raw Data Preview")
    if df is not None:
        st.write("First 5 rows of the dataset used for training:")
        st.dataframe(df.head())
        
        st.subheader("Data Distribution")
        st.bar_chart(df['cgpa'].head(20))
    else:
        st.error("placement.csv not found in repository.")

# --- TAB 4: RESOURCES ---
with tab4:
    st.header("Downloads")
    file_name = "Linear Regression Practise Assignment.ipynb"
    if os.path.exists(file_name):
        with open(file_name, "rb") as f:
            st.download_button("📥 Download Notebook", f, file_name=file_name)
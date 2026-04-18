import streamlit as st
import pickle
import numpy as np

# 1. Load the model you exported in Step 1
model = pickle.load(open('model.pkl', 'rb'))

# 2. Design the web interface
st.title("Student Placement Package Predictor")
st.write("This app predicts your expected package (LPA) based on your CGPA.")

# 3. Create an input field for the user
cgpa_input = st.number_input("Enter your CGPA:", min_value=0.0, max_value=10.0, value=7.0, step=0.1)

# 4. Make a prediction when the user clicks the button
if st.button("Predict Package"):
    # Reshape input for the model
    prediction = model.predict(np.array([[cgpa_input]]))
    st.success(f"Predicted Package: {prediction[0]:.2f} LPA")
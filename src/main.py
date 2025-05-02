import streamlit as st

# 🚨 MUST be first
st.set_page_config(page_title="Scholarship Predictor", page_icon="🎓", layout="centered")

import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model

# Load model once
@st.cache_resource
def load_keras_model():
    model_path = os.path.join(os.path.dirname(__file__), '../models/scholarship_model.h5')
    return load_model(model_path)

model = load_keras_model()

# Label maps
education_map = {'Undergraduate': 0, 'Postgraduate': 1, 'Doctrate': 2}
percentage_map = {'60-70': 0, '70-80': 1, '80-90': 2, '90-100': 3}
gender_map = {'Male': 1, 'Female': 0}
disability_map = {'Yes': 1, 'No': 0}
sports_map = {'Yes': 1, 'No': 0}

# Display → Internal for income
display_income_map = {
    "Below 200K": "Upto 1.5L",
    "200K - 500K": "1.5L to 3L",
    "500K - 1M": "3L to 6L",
    "Above 1M": "Above 6L"
}
income_map = {
    "Upto 1.5L": 0,
    "1.5L to 3L": 1,
    "3L to 6L": 2,
    "Above 6L": 3
}

# Reverse prediction
def reverse_result(pred):
    return "Eligible 🎯" if pred > 0.5 else "Not Eligible ❌"

# Header
st.title("🎓 ANALYZE SCHOLARSHIP")
st.markdown("Fill out the details below to analyze scholarship application.")

st.divider()

tab1, tab2 = st.tabs(["🔍 Single Prediction", "📁 Batch Upload"])

# Tab 1 — Single Prediction
with tab1:
    st.subheader("🔹 Enter Applicant Details")

    with st.form("prediction_form"):
        # gender = st.selectbox("Gender", gender_map.keys(), help="Select applicant's gender")
        gender = st.selectbox("Gender", gender_map.keys())
        # education = st.selectbox("Education Qualification", education_map.keys(), help="Current level of education")
        education = st.selectbox("Education Qualification", education_map.keys())
        # Same row for Disability and Sports
        col1, col2 = st.columns(2)
        with col1:
            # disability = st.selectbox("Disability", disability_map.keys(), help="Does the applicant have a disability?")
            disability = st.selectbox("Disability", disability_map.keys())
        with col2:
            # sports = st.selectbox("Sports", sports_map.keys(), help="Has the applicant participated in sports?")
            sports = st.selectbox("Sports", sports_map.keys())
        # percentage = st.selectbox("Annual Percentage", percentage_map.keys(), help="Academic performance")
        percentage = st.selectbox("Annual Percentage", percentage_map.keys())
        # display_income = st.selectbox("Family Income", display_income_map.keys(), help="Total family income")
        display_income = st.selectbox("Family Income", display_income_map.keys())
        income = display_income_map[display_income]

        submitted = st.form_submit_button("🚀 Start Analysis")

    if submitted:
        features = [
            gender_map[gender],
            disability_map[disability],
            sports_map[sports],
            education_map[education],
            percentage_map[percentage],
            income_map[income]
        ]
        input_array = np.array([features])
        input_array = np.expand_dims(input_array, axis=-1)
        prediction = model.predict(input_array)[0][0]
        outcome = reverse_result(prediction)

        if prediction > 0.5:
            st.success("✅ The applicant is likely to be eligible for a scholarship!")
            st.success(f"**Prediction:** {outcome}")
        else:
            st.warning("⚠️ The applicant is likely not eligible for a scholarship.")
            st.warning(f"**Prediction:** {outcome}")
        st.caption(f"🧮 Raw Score: {prediction:.4f}")

# Tab 2 — Batch Prediction
with tab2:
    st.subheader("📁 Upload CSV File")

    with st.expander("📌 CSV Format Help"):
        st.markdown("""
        Your CSV must contain the following columns:
        - `Gender`  
        - `Disability`  
        - `Sports`  
        - `Education_Qualification`  
        - `Annual-Percentage`  
        - `Income`  
        
        💡 Use these values for `Income`:
        - Below 200K  
        - 200K - 500K  
        - 500K - 1M  
        - Above 1M
        """)

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            required = ['Gender', 'Disability', 'Sports', 'Education_Qualification', 'Annual-Percentage', 'Income']

            if not all(col in df.columns for col in required):
                st.error("❗ Missing one or more required columns.")
            else:
                def encode(row):
                    try:
                        mapped_income = display_income_map[row['Income']]
                        return [
                            gender_map[row['Gender']],
                            disability_map[row['Disability']],
                            sports_map[row['Sports']],
                            education_map[row['Education_Qualification']],
                            percentage_map[row['Annual-Percentage']],
                            income_map[mapped_income]
                        ]
                    except:
                        return None

                df['encoded'] = df.apply(encode, axis=1)
                df = df[df['encoded'].notnull()].reset_index(drop=True)

                if df.empty:
                    st.warning("No valid rows found after encoding.")
                else:
                    X = np.array(df['encoded'].to_list())
                    X = np.expand_dims(X, axis=-1)
                    predictions = model.predict(X).flatten()
                    df['Raw Prediction'] = predictions
                    df['Predicted Outcome'] = [reverse_result(p) for p in predictions]
                    df.drop(columns='encoded', inplace=True)

                    st.success("✅ Predictions generated!")
                    st.dataframe(df)

                    csv_out = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv_out,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

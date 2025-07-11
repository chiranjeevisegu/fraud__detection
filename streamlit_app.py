import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load saved model
model = joblib.load("fraud_model.pkl")

# Title
st.title("💳 Bank Transaction Fraud Detection App")

# File uploader
uploaded_file = st.file_uploader("Upload Transaction CSV File", type=["csv"])

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Raw Uploaded Data")
    st.write(df.head())

    # ----------------- Preprocessing -----------------
    try:
        # Drop personal info if exists
        cols_to_drop = ['Customer_Name', 'Customer_Contact', 'Customer_Email']
        df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

        # Handle missing values
        if 'Is_Fraud' in df.columns:
            df.dropna(subset=['Is_Fraud', 'Transaction_Amount'], inplace=True)
        else:
            df.dropna(subset=['Transaction_Amount'], inplace=True)

        cat_cols = df.select_dtypes(include='object').columns
        df[cat_cols] = df[cat_cols].fillna('Unknown')

        num_cols = df.select_dtypes(include=np.number).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        # Convert date and extract features
        if 'Transaction_Date' in df.columns:
            df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'], errors='coerce')
            df['Year'] = df['Transaction_Date'].dt.year
            df['Month'] = df['Transaction_Date'].dt.month
            df['Day'] = df['Transaction_Date'].dt.day
            df.drop(columns=['Transaction_Date'], inplace=True)

        # Encode categorical features
        cat_cols = df.select_dtypes(include='object').columns
        le = LabelEncoder()
        for col in cat_cols:
            df[col] = le.fit_transform(df[col].astype(str))

        # Normalize
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df.select_dtypes(include=np.number))
        df_scaled = pd.DataFrame(scaled, columns=df.select_dtypes(include=np.number).columns)

        # ----------------- Prediction -----------------
        predictions = model.predict(df_scaled)
        df_result = df.copy()
        df_result['Prediction'] = predictions
        df_result['Prediction'] = df_result['Prediction'].apply(lambda x: "Fraud" if x == 1 else "Legit")

        # ----------------- Output -----------------
        st.subheader("📋 Prediction Results")
        st.write(df_result[['Prediction']].value_counts().rename_axis('Class').reset_index(name='Count'))

        st.subheader("🧾 Preview of Results with Predictions")
        st.dataframe(df_result)

        st.download_button(
            label="📥 Download Results as CSV",
            data=df_result.to_csv(index=False).encode('utf-8'),
            file_name="fraud_detection_results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error during processing: {e}")

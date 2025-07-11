import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the LSTM model
model = load_model("lstm_fraud_detection_model.keras")

# Title
st.title("💳 LSTM-Based Bank Transaction Fraud Detection")

# Upload CSV
uploaded_file = st.file_uploader("📤 Upload Bank Transaction CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Uploaded Data Preview")
    st.write(df.head())

    try:
        # Step 1: Drop personal info if present
        cols_to_drop = ['Customer_Name', 'Customer_Contact', 'Customer_Email']
        df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

        # Step 2: Handle missing values
        if 'Is_Fraud' in df.columns:
            df.dropna(subset=['Is_Fraud', 'Transaction_Amount'], inplace=True)
        else:
            df.dropna(subset=['Transaction_Amount'], inplace=True)

        # Fill categorical with 'Unknown'
        cat_cols = df.select_dtypes(include='object').columns
        df[cat_cols] = df[cat_cols].fillna('Unknown')

        # Fill numerical with median
        num_cols = df.select_dtypes(include=np.number).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        # Step 3: Extract features from Transaction_Date
        if 'Transaction_Date' in df.columns:
            df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'], errors='coerce')
            df['Year'] = df['Transaction_Date'].dt.year
            df['Month'] = df['Transaction_Date'].dt.month
            df['Day'] = df['Transaction_Date'].dt.day
            df.drop(columns=['Transaction_Date'], inplace=True)

        # Step 4: Label Encoding for categorical columns
        le = LabelEncoder()
        for col in df.select_dtypes(include='object').columns:
            df[col] = le.fit_transform(df[col].astype(str))

        # Step 5: Scaling
        scaler = StandardScaler()
        numeric_data = df.select_dtypes(include=np.number)
        X_scaled = scaler.fit_transform(numeric_data)

        # Step 6: Reshape input for LSTM: (samples, timesteps, features)
        X_reshaped = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

        # Step 7: Prediction
        predictions = model.predict(X_reshaped)
        predictions_binary = (predictions > 0.5).astype(int).flatten()

        # Step 8: Display results
        df_result = df.copy()
        df_result['Prediction'] = predictions_binary
        df_result['Prediction'] = df_result['Prediction'].apply(lambda x: "Fraud" if x == 1 else "Legit")

        st.subheader("✅ Prediction Summary")
        st.write(df_result['Prediction'].value_counts())

        st.subheader("📋 Data with Prediction Labels")
        st.dataframe(df_result)

        st.download_button(
            label="📥 Download Results CSV",
            data=df_result.to_csv(index=False).encode('utf-8'),
            file_name="fraud_detection_lstm_results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error: {e}")

import pandas as pd
import joblib
import streamlit as st

st.title("Hoppy Loan App Prediction")

# Load model and encoder with error handling
try:
    model = joblib.load('reg_model.joblib')
    enc = joblib.load('encoder.joblib')
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Input fields
DTI_Ratio = st.number_input('Enter DTI_Ratio', min_value=0.0, value=2.53)
Employment_Status = st.selectbox('Employment_Status', ['employed', 'unemployed'])
Credit_Score = st.number_input('Enter Credit_Score', min_value=300, value=650)
Income = st.number_input('Enter Income', min_value=0, value=50000)

# Prepare data
data = {
    "DTI_Ratio": [DTI_Ratio],
    "Employment_Status": [Employment_Status],
    "Credit_Score": [Credit_Score],
    "Income": [Income],
}

try:
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # One-hot encode employment status
    employment_encoded = enc.transform(df[['Employment_Status']]).toarray()
    employment_cols = enc.get_feature_names_out(['Employment_Status'])
    employment_df = pd.DataFrame(employment_encoded, columns=employment_cols)
    
    # Combine features
    final_df = pd.concat([df.drop(columns=['Employment_Status']), employment_df], axis=1)
    
    # Prediction
    if st.button('Predict'):
        prediction = model.predict(final_df)
        amount = max(0, prediction[0])  # Ensure non-negative
        st.success(f'Predicted Loan Amount: ${amount:0,.2f}')
        
except Exception as e:
    st.error(f"Prediction error: {str(e)}")
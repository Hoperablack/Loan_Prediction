import pandas as pd
import joblib
import streamlit as st

st.title("Hoppy Loan App Prediction")

model = joblib.load('reg_model.joblib')
enc = joblib.load('encoder.joblib')

DTI_Ratio = st.number_input('Enter DTI_Ratio', min_value=2.53)
Employment_Status = st.selectbox('Employment_Status', ['employed', 'unemployed'])
Credit_Score = st.number_input('Enter Credit_Score', min_value=300)
Income = st.number_input('Enter Income', min_value=20000)

data = {
    "DTI_Ratio": DTI_Ratio,
    "Employment_Status": Employment_Status,
    "Credit_Score": Credit_Score,
    "Income": Income,
}
df = pd.DataFrame(data, index=[0])

employment_encoded = enc.transform(df[['Employment_Status']]).toarray()
employment_cols = enc.get_feature_names_out(['Employment_Status'])
employment_df = pd.DataFrame(employment_encoded, columns=employment_cols)


df = pd.concat([df.drop(columns=['Employment_Status']), employment_df], axis=1)


Button = st.button('Predict')

if Button:
    Prediction = model.predict(df)
    if Prediction[0] < 0:
        st.info(f'Predicted Loan Amount: ${0:0,.2f}')
    else:
        st.info(f'Predicted Loan Amount: ${Prediction[0]:0,.2f}')

import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained numerical imputer, scaler, and model using joblib
num_imputer = joblib.load('numerical_imputer.joblib')
scaler = joblib.load('scaler.joblib')
model = joblib.load('Final_model.joblib')

# Define a function to preprocess the input data
def preprocess_input_data(input_data):
    input_data_df = pd.DataFrame(input_data, columns=['PRG', 'PL', 'PR', 'SK', 'TS', 'M11', 'BD2', 'Age', 'Insurance'])
    num_columns = input_data_df.select_dtypes(include='number').columns

    input_data_imputed_num = num_imputer.transform(input_data_df[num_columns])
    input_scaled_df = pd.DataFrame(scaler.transform(input_data_imputed_num), columns=num_columns)

    return input_scaled_df

# Define a function to make the sepsis prediction
def predict_sepsis(input_data):
    input_scaled_df = preprocess_input_data(input_data)
    prediction = model.predict(input_scaled_df)[0]
    probabilities = model.predict_proba(input_scaled_df)[0]
    sepsis_status = "Positive" if prediction == 1 else "Negative"

    output_df = pd.DataFrame(input_data, columns=['PRG', 'PL', 'PR', 'SK', 'TS', 'M11', 'BD2', 'Age', 'Insurance'])
    output_df['Prediction'] = sepsis_status
    output_df['Negative Probability'] = probabilities[0]
    output_df['Positive Probability'] = probabilities[1]

    return output_df

# Create a Streamlit app
def main():
    st.title('Sepsis Prediction App')

    st.sidebar.title('Input Parameters')
    PRG = st.sidebar.number_input('PRG', value=0.0)
    PL = st.sidebar.number_input('PL', value=0.0)
    PR = st.sidebar.number_input('PR', value=0.0)
    SK = st.sidebar.number_input('SK', value=0.0)
    TS = st.sidebar.number_input('TS', value=0.0)
    M11 = st.sidebar.number_input('M11', value=0.0)
    BD2 = st.sidebar.number_input('BD2', value=0.0)
    Age = st.sidebar.number_input('Age', value=0.0)
    Insurance = st.sidebar.number_input('Insurance', value=0)

    input_data = [[PRG, PL, PR, SK, TS, M11, BD2, Age, Insurance]]

    if st.sidebar.button('Predict'):
        output_df = predict_sepsis(input_data)

        st.subheader('Prediction Result')
        st.write(output_df)

if __name__ == '__main__':
    main()

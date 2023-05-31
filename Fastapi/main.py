import pandas as pd
import joblib
from fastapi import FastAPI
import uvicorn
import numpy as np

app = FastAPI()

def load_model():
    num_imputer = joblib.load('numerical_imputer.joblib')
    scaler = joblib.load('scaler.joblib')
    model = joblib.load('Final_model.joblib')
    return num_imputer, scaler, model

def preprocess_input_data(input_data, num_imputer, scaler):
    input_data_df = pd.DataFrame([input_data])
    num_columns = [col for col in input_data_df.columns if input_data_df[col].dtype != 'object']
    input_data_imputed_num = num_imputer.transform(input_data_df[num_columns])
    input_scaled_df = pd.DataFrame(scaler.transform(input_data_imputed_num), columns=num_columns)
    return input_scaled_df

@app.get("/")
def read_root():
    return "Sepsis Prediction App"

@app.get("/sepsis/predict")
def predict_sepsis_endpoint(PRG: float, PL: float, PR: float, SK: float, TS: float,
                            M11: float, BD2: float, Age: float, Insurance: int):
    num_imputer, scaler, model = load_model()

    input_data = {
        'PRG': PRG,
        'PL': PL,
        'PR': PR,
        'SK': SK,
        'TS': TS,
        'M11': M11,
        'BD2': BD2,
        'Age': Age,
        'Insurance': Insurance
    }

    input_scaled_df = preprocess_input_data(input_data, num_imputer, scaler)

    probabilities = model.predict_proba(input_scaled_df)[0]
    prediction = np.argmax(probabilities)

    sepsis_status = "Positive" if prediction == 1 else "Negative"
    probability = probabilities[1] if prediction == 1 else probabilities[0]

    statement = f"The patient is {sepsis_status}. There is a {'high' if prediction == 1 else 'low'} probability ({probability:.2f}) that the patient is susceptible to developing sepsis."

    user_input_statement = "Please note this is the user-inputted data: "

    output_df = pd.DataFrame([input_data])

    result = {'predicted_sepsis': sepsis_status, 'statement': statement, 'user_input_statement': user_input_statement, 'input_data_df': output_df.to_dict('records')}

    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)

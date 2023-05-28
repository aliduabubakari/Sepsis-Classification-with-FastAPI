import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

#Initializing the FastAPI Application:
app = FastAPI() #An instance of the FastAPI class is created to serve as the application.


# Define a Pydantic model for the input data
class SepsisPredictionRequest(BaseModel):
    PRG: float
    PL: float
    PR: float
    SK: float
    TS: float
    M11: float
    BD2: float
    Age: float
    Insurance: int

# Define a Pydantic model for the response
class SepsisPredictionResponse(BaseModel):
    predicted_sepsis: str

# Load the pre-trained numerical imputer, scaler, and model using joblib
num_imputer = joblib.load('numerical_imputer.joblib')
scaler = joblib.load('scaler.joblib')
model = joblib.load('Final_model.joblib')

#Defining the Root Endpoint
@app.get("/")
def read_root():
    return "Sepsis Prediction App" #It serves as the root endpoint of the application.

# Defining the Prediction Endpoint:
# Defining the Prediction Endpoint
@app.get("/predict/{PRG}/{PL}/{PR}/{SK}/{TS}/{M11}/{BD2}/{Age}/{Insurance}", response_model=SepsisPredictionResponse)
def predict_sepsis(
    PRG: float,
    PL: float,
    PR: float,
    SK: float,
    TS: float,
    M11: float,
    BD2: float,
    Age: float,
    Insurance: int
):
    # Create an instance of SepsisPredictionRequest from the input data
    input_data = SepsisPredictionRequest(
        PRG=PRG,
        PL=PL,
        PR=PR,
        SK=SK,
        TS=TS,
        M11=M11,
        BD2=BD2,
        Age=Age,
        Insurance=Insurance
    )

    # Convert the input data to a pandas DataFrame
    input_data_df = pd.DataFrame(input_data.dict(), index=[0])

    # Get the numerical columns from the input data
    num_columns = [col for col in input_data_df.columns if input_data_df[col].dtype != 'object']

    # Apply the numerical imputer to fill missing values in the numerical columns
    input_data_imputed_num = num_imputer.transform(input_data_df[num_columns])

    # Scale the numerical columns using the pre-trained scaler
    input_scaled_df = pd.DataFrame(scaler.transform(input_data_imputed_num), columns=num_columns)

    # Make the prediction using the pre-trained model
    prediction = model.predict(input_scaled_df)[0]
    probabilities = model.predict_proba(input_scaled_df)[0]

    # Determine the sepsis status based on the prediction
    sepsis_status = "Positive" if prediction == 1 else "Negative"

    # Create an output DataFrame with input data, prediction, and probabilities
    output_df = input_data_df.copy()
    output_df['Prediction'] = sepsis_status
    output_df['Negative Probability'] = probabilities[0]
    output_df['Positive Probability'] = probabilities[1]

    # Print the output DataFrame
    print("Output DataFrame:")
    print(output_df)

    # Return the prediction result as a response
    return SepsisPredictionResponse(predicted_sepsis=sepsis_status)

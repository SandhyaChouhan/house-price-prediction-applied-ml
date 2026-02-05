from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load trained model and encoders
model = joblib.load("models/price_model.pkl")
encoders = joblib.load("models/encoders.pkl")

# Create FastAPI app
app = FastAPI(title="House Price Prediction API")

# Define input format
class HouseInput(BaseModel):
    Location: str
    Size: float
    Bedrooms: float
    Bathrooms: float
    Year_Built: float
    Condition: str
    Type: str
    Sold_Year: int
    Sold_Month: int
    Property_Age: float

@app.post("/predict")
def predict_price(data: HouseInput):
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])

    # Encode categorical columns
    for col, encoder in encoders.items():
        df[col] = encoder.transform(df[col])

    # Rename column to match training data
    df.rename(columns={"Year_Built": "Year Built"}, inplace=True)

    # Predict price
    prediction = model.predict(df)[0]

    return {
        "predicted_price": round(float(prediction), 2)
    }

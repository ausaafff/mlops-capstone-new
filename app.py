from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")

# Request schema
class InputData(BaseModel):
    area: float

# FastAPI app
app = FastAPI()

@app.post("/predict")
def predict(data: InputData):
    X = np.array([[data.area]])
    prediction = model.predict(X)[0]
    return {"prediction": float(prediction)}
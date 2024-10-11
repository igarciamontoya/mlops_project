import os
import json
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, validator
from datetime import datetime, timedelta
import pandas as pd
from prophet import Prophet
from google.cloud import storage
from google.auth import credentials

app = FastAPI()

# Set up the templates directory for Jinja2
templates = Jinja2Templates(directory="templates")

# Use environment variables for GCS configuration
BUCKET_NAME = os.environ['GCS_BUCKET_NAME'] 
MODEL_FILE_PATH = os.environ['MODEL_FILE_PATH'] 

# Download and load the model from GCS
def load_model_from_gcs():
    # Initialize the GCS client
    client = storage.Client()

    # Get the bucket and blob (file)
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_FILE_PATH)

    # Download the model file to a temporary location
    temp_model_path = "/tmp/model.json"
    blob.download_to_filename(temp_model_path)

    # Load the model from the downloaded JSON file
    with open(temp_model_path, 'r') as f:
        model_json = json.load(f)

    # Reconstruct the Prophet model from the JSON data
    model = Prophet()
    model = model.from_json(model_json)
    
    return model

# Load the model on app startup
prophet_model = load_model_from_gcs()

# Variables to predict (assume you have multiple variables in a dataset)
available_variables = [
    'tmed', 'tmin', 'tmax', 'prec', 'sol', 'velmedia', 'presMax', 
    'presMin', 'hrMedia', 'hrMax', 'hrMin'
]

class PredictionInput(BaseModel):
    variable: str
    date: str

    @validator("date")
    def validate_date(cls, date_value):
        date_obj = datetime.strptime(date_value, "%Y-%m-%d")
        today = datetime.now()
        if not today <= date_obj <= today + timedelta(days=365):
            raise ValueError("Date must be within the next 365 days.")
        return date_value

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "variables": available_variables})

@app.post("/predict", response_class=HTMLResponse)
async def make_prediction(request: Request, variable: str = Form(...), date: str = Form(...)):
    try:
        # Validate and process the input
        prediction_input = PredictionInput(variable=variable, date=date)
        prediction_date = datetime.strptime(prediction_input.date, "%Y-%m-%d")

        # Create future dataframe for Prophet
        future = pd.DataFrame({"ds": [prediction_date]})

        # Predict with Prophet
        forecast = prophet_model.predict(future)

        prediction = forecast['yhat'].values[0]  # Get prediction

        return templates.TemplateResponse("index.html", {
            "request": request, 
            "variables": available_variables, 
            "prediction": prediction,
            "selected_variable": variable,
            "selected_date": date
        })

    except ValueError as e:
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "variables": available_variables,
            "error": str(e)
        })

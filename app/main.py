from fastapi import FastAPI
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json

# from google.cloud import storage
from pydantic import BaseModel, validator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from google.cloud import storage

app = FastAPI()

bucket_name = "mlops-igm-final-project-bucket"

available_variables = [
    'tmed', 'tmin', 'tmax', 'prec', 'sol', 'velmedia', 'presMax', 
    'presMin', 'hrMedia', 'hrMax', 'hrMin'
]

class request_body(BaseModel):
    variable: str
    date: str 
    
    @validator("date")
    def validate_date(cls, date_value):
        date_obj = datetime.strptime(date_value, "%Y-%m-%d")
        today = datetime.now()
        if not today <= date_obj <= today + timedelta(days=365):
            raise ValueError("Date must be within the next 365 days.")
        return date_value

# @app.get("/")
# async def root():
#     return {"message": "Hello World"}
@app.get("/")
async def get_form(data : request_body):
    return {"message": "Hello this is the FAST API of Irene's project"}

def download_blob_as_json(bucket_name, source_blob_name):
    """Downloads a JSON file from the GCS bucket and returns it as a dictionary."""
    try:
        # Initialize a storage client
        storage_client = storage.Client()

        # Get the bucket and blob (file) from the bucket
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)

        # Download the content of the file as text
        file_content = blob.download_as_text()

        # Parse the text content into a JSON object (dictionary)
        json_content = json.loads(file_content)
        return json_content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading or parsing JSON file: {str(e)}")
        
def load_prophet_model_from_json(json_model):
    """Reconstruct a Prophet model from the JSON data."""
    try:
        # Create a new Prophet model
        model = Prophet()
        if 'params' in json_model:
            model.params = json_model['params']
        if 'seasonality' in json_model:
            model.seasonality = json_model['seasonality']
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading Prophet model from JSON: {str(e)}")



@app.post("/predict")
async def make_prediction(data : request_body):
    """Endpoint to download the Prophet model and make a prediction."""
    json_model = download_blob_as_json(bucket_name, f"""models/{data['variable']}_best_model.json""")
    model = load_prophet_model_from_json(json_model)
    
    future = model.make_future_dataframe(periods=365)
    
    forecast = model.predict(future)
    num_days = int(abs(datetime.strptime(date_value, "%Y-%m-%d") - datetime.now())) - 365
    return {'Variable': data['variable'], 'Prediction2': forecast.yhat.iloc[num_days]}

    

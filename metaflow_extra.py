import numpy as np
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet.serialize import model_to_json, model_from_json

import os
import re
import json 

import warnings
warnings.filterwarnings("ignore")


def extract_word(file_name):
    # Regular expression pattern to match 'output_{something}.csv'
    match = re.search(r'output_(.*?).csv$', file_name)
    if match:
        return match.group(1)  # Return the extracted word
    return None

def prophet_exp(file_path, var):
    df = pd.read_csv(file_path,parse_dates=['ds'])
    train, test = df[0:-365], df[-365:]
    mlflow.set_experiment(f"""final-project-{variable}""")
    fourier_order = [1,3,5,7,10]
    best_mae, best_runid = np.inf, None
    for i in fourier_order:
        with mlflow.start_run() as run:
            mlflow.set_tags({"Model":"Prophet", "Target": var})
            mlflow.log_artifact(file_path)
            mlflow.log_params({'fourier_order':i})
    
            model = Prophet() #default include weekly and yearly seasonalities
    
            #Fit with default settings
            model.add_seasonality(name='yearly', period=365, fourier_order=i)
            model.fit(train)
    
            future = model.make_future_dataframe(periods=365) #freq='D'
            forecast = model.predict(future)
            
            mlflow.sklearn.log_model(model, artifact_path = 'final-project-models')
            # mlflow.log_artifact(model_to_json(model))
            

            fig_path = f'''figs/forecast_{var}_plot.png'''
            fig1 = model.plot(forecast)
            fig1.savefig(fig_path, dpi=300, bbox_inches='tight')
            
            mlflow.log_artifact(fig_path)
            
            y_pred = forecast['yhat'][-365:]
            y_test = test.y
            
            # Mean Absolute Error (MAE)
            mae = np.round(mean_absolute_error(y_test, y_pred),4)
            mlflow.log_metric('MAE', mae)
            
            # Mean Squared Error (MSE)
            mse = np.round(mean_squared_error(y_test, y_pred),4)
            mlflow.log_metric('MSE', mse)
    
            # Root Mean Squared Error (RMSE)
            rmse = np.round(np.sqrt(mse),4)
            mlflow.log_metric('RMSE', rmse)
    
            # R² Score (Coefficient of Determination)
            r2 = np.round(r2_score(y_test, y_pred),4)
            mlflow.log_metric('R2', r2)
            
            if mae < best_mae:
                best_mae, best_runid = mae, run.info.run_id
                with open(f'models/{var}_best_model.json', 'w') as fout:
                    fout.write(model_to_json(model))  # Save model
                print("best-run-id",best_runid)
    
        mlflow.end_run()

    
    mod_path = f'runs:/{best_runid}/artifacts/best-models'
    mlflow.register_model(model_uri = mod_path, name = f'''best-model-{var}''')
    return best_runid
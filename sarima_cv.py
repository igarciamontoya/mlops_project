from metaflow import FlowSpec, step
import pandas as pd
import yaml
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm 
import sklearn.metrics as skmetrics #For evaluation metrics

import mlflow

def evaluate_models_cv(dataset, p_values, d, q_values, P_values, D_values, Q_values, m):
    K=5
    best_mae, best_cfg, best_runid = float("inf"), None, ''
    for p in p_values:
        for q in q_values:
            for P in P_values:
                for D in D_values:
                    for Q in Q_values:
                        trend_order = (p,d,q)
                        seasonal_order = (P,D,Q,m)
                        cfg = [(p,d,q), (P,D,Q,m)]
                        validation_size=180
                        train_size=len(dataset)-validation_size*K
                        mae=0
                        mse = 0
                        r2 = 0
                        with mlflow.start_run() as run:
                            mlflow.set_tags({"Model":"Sarima", "Lab":"Lab 7"})
                            mlflow.log_params({'cfg':cfg})
                            for k in range(0,5):
                                train, test = dataset[0:train_size+k*validation_size],dataset[train_size+k*validation_size:train_size+(k+1)*validation_size]
                                model = ARIMA(train, order=trend_order, seasonal_order=seasonal_order, trend=[0,0,1]) 
                                model.initialize_approximate_diffuse() # this line
                                model_fit = model.fit()
                                predictions=model_fit.forecast(validation_size)
                                mae = mae+np.round(skmetrics.mean_absolute_error(test, predictions),4)
                                mse = mse+np.round(skmetrics.mean_squared_error(test, predictions),4)
                                r2 = r2+np.round(skmetrics.r2_score(test, predictions),4)
    
                            
    
                            mae_avg=mae/5
                            mse_avg=mse/5
                            r2_avg=r2/5
                            # Mean Absolute Error (MAE)
                            mlflow.log_metric('Avg MAE', mae)
                            # Mean Squared Error (MSE)
                            mlflow.log_metric('Avg MSE', mse)
                            # Root Mean Squared Error (RMSE)
                            mlflow.log_metric('Avg RMSE', np.sqrt(mse))
                            # R² Score (Coefficient of Determination)
                            mlflow.log_metric('Avg R2', r2)
                            mlflow.sklearn.log_model(model_fit, artifact_path = 'models_lab7')
                            
                            if mae_avg < best_mae:
                                best_mae, best_cfg, best_runid = mae_avg, cfg, run.info.run_id
                                print("best-run-id",best_runid)
                        mlflow.end_run()
    
    #register best model
    mod_path = f'runs:/{best_runid}/artifacts/models_lab7'
    mlflow.register_model(model_uri = mod_path, name = 'best_sarima_lab7')
    return best_runid
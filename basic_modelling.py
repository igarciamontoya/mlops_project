import numpy as np
import pandas as pd
import pickle
import yaml

from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def launch_prophet(fourier_order):
    data = pd.read_csv('data/preprocessed_data.csv')
    
    model = Prophet() #default trend = piece-wise linear model
                #default include weekly and yearly seasonalities

    #Fit with default settings
    model.add_seasonality(name='yearly', period=365, fourier_order=fourier_order)

    model.fit(model_data)


    #Fataframe with forecasting steps
    future = model.make_future_dataframe(periods=365) #freq='D'
    
    #Forecast
    forecast = model.predict(future)
    
    forecast.to_csv('output/forecast_1y.csv', index=False)
    fig1 = model.plot(forecast)
    fig1.savefig('fig/forecast_plot.png', dpi=300, bbox_inches='tight')
    

    with open('model/prophet.pkl','wb') as f:
        pickle.dump(model,f)

    return model

def validation(model, cv_initial,cv_period,cv_horizon):
    #cross validation
    forecast_cv = cross_validation(model, initial=cv_initial, period=cv_period, horizon = cv_horizon)
    forecast_cv.to_csv('output/forecast_cross_validation.csv', index=False)

    #performance metrics
    forecast_perf = performance_metrics(forecast_cv) #by default start with 10% of the horizon
    forecast_perf.to_csv('output/forecast_performance_metrics.csv', index=False)
    
if __name__=="__main__":
    params = yaml.safe_load(open("params.yaml"))["features"]
    fourier_order = params["fourier_order"]
    cv_initial = params["cv_initial"]
    cv_period = params["cv_period"]
    cv_horizon = params["cv_horizon"]

    
    m = launch_prophet(fourier_order)
    validation(m, cv_initial,cv_period,cv_horizon)
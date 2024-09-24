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

class project_training(FlowSpec):

    @step
    def start(self):
        print("Training flow started succesfully")
        self.next(self.data_ingestion)

    @step
    def data_ingestion(self):
        from basic_preprocessing import preprocessing 
        params = yaml.safe_load(open("params.yaml"))["features"]
        data_path = params["data_path"]
        city_name = params["city"]
        preprocessing(data_path,city_name)
        print("Data loaded and cleaned successfully")
        
        self.next(self.data_split)

    @step
    def data_split(self):
        df = pd.read_csv('data/preprocessed_data.csv')
        train, test = df[0:-365], df[-365:]
        train.to_csv('data/train_data.csv', index=False)
        test.to_csv('data/test_data.csv', index=False)
        print("Data splited successfully")

        self.next(self.arima_cv)

    @step
    def arima_cv(self):
        from sarima_cv import evaluate_models_cv 

        
        mlflow.set_tracking_uri('sqlite:///mlflow.db')
        mlflow.set_experiment('experiment-lab7')
        train = pd.read_csv('data/train_data.csv')
        
        p=[0,1]
        d=1
        q=[1,2]
        P=[0,1]
        D=[1]
        Q=[1,2]
        m=12
    
        self.best_run_id = evaluate_models_cv(train.y, p,d,q,P,D,Q,m)
        self.next(self.end)
    

    @step
    def end(self):
        print('done')

if __name__=='__main__':
    project_training()
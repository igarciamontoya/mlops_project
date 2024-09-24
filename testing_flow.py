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
from metaflow import FlowSpec, step, Flow, Parameter, JSONType

class project_test(FlowSpec):

    @step
    def start(self):
        print("Testing flow started succesfully")
        run = Flow('project_training').latest_run 
        self.train_run_id = run.pathspec 
        self.best_model_id = run['arima_cv'].task.data.best_run_id
                
        self.next(self.data_loading)

    @step
    def data_loading(self):
        self.test_data = pd.read_csv('data/test_data.csv')
        print(self.best_model_id)
        
        model_uri = f"runs:/{self.best_model_id}/artifacts/models_lab7/"
        # model_uri = f"models:/best_sarima_lab7/latest"
        self.model = mlflow.pyfunc.load_model(model_uri)
        # self.model = mlflow.sklearn.load_model(model_uri)
        print("Model and Test data loaded successfully")

        self.next(self.model_testing)

    @step
    def model_testing(self):
        final_pred = self.model.forecast(365)
        # prediction_final
        mae_final = skmetrics.mean_absolute_error(self.test_data.y, final_pred)
        print("FINAL MAE ---", mae_final)
        self.final_mae = mae_final
        self.next(self.end)
    

    @step
    def end(self):
        print('done')

if __name__=='__main__':
    project_test()
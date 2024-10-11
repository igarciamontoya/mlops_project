from metaflow import FlowSpec, step
import pandas as pd
import requests
import csv
import os
from datetime import datetime, timedelta
import time
import json
import yaml
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm 
import sklearn.metrics as skmetrics #For evaluation metrics

import mlflow

class project_training(FlowSpec):

    @step
    def start(self):
        print("Load data flow started succesfully")
        self.next(self.data_ingestion)

    @step
    def time_intervals(self):
        start_date = datetime(2000, 1, 1)
        end_date = datetime.now()
        
        fecha_ini = []
        fecha_fin = []
        
        current_date = start_date
        
        while current_date <= end_date:
            formatted_ini = current_date.strftime("%Y-%m-%dT00:00:00UTC")
            fecha_ini.append(formatted_ini)
            
            current_date += pd.DateOffset(months=6)
            
            formatted_fin = current_date.strftime("%Y-%m-%dT00:00:00UTC")
            fecha_fin.append(formatted_fin)
        
        else:
            fecha_fin[-1] = end_date.strftime("%Y-%m-%dT00:00:00UTC")

        self.fecha_fin = fecha_fin
        self.fecha_ini = fecha_ini
        self.next(self.aemet_calls)
    @step
    def aemet_calls(self):
        for ini, end in zip(list(reversed(fecha_ini)), list(reversed(fecha_fin))):
            time.sleep(2)
            url = f"""https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos/fechaini/{ini}/fechafin/{end}/estacion/8025/"""
        
            querystring = {"api_key":"eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJtb25pcmVnYXJAZ21haWwuY29tIiwianRpIjoiNzI0Njc4MTQtYmEzMC00NWMwLWE3ODMtM2NkYzkyMWRjZjRmIiwiaXNzIjoiQUVNRVQiLCJpYXQiOjE3MjgxOTUwNTMsInVzZXJJZCI6IjcyNDY3ODE0LWJhMzAtNDVjMC1hNzgzLTNjZGM5MjFkY2Y0ZiIsInJvbGUiOiIifQ.s8izGMi6TjDfTi_qik3g2Qt6qiGtJDUMf70FfZiKiMc"}
            headers = {
                'cache-control': "no-cache"
                }
        
            response = requests.request("GET", url, headers=headers, params=querystring)
            if response.status_code == 200:
                response_data = response.json()
        
                data_url = response_data.get("datos")
                if data_url:
                    time.sleep(2)
                    second_response = requests.get(data_url)
                    if second_response.status_code == 200:
                        json_data = second_response.json()
        
                        csv_file_path = 'data/output.csv'  # Define the output file path
                        file_exists = os.path.isfile(csv_file_path)
                        if file_exists:
                            with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
                                reader = csv.DictReader(csvfile)
                                existing_columns = reader.fieldnames
                        filtered_data = [{k: v for k, v in entry.items() if k in existing_columns} for entry in json_data]
                        with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                            if file_exists:
                                writer = csv.DictWriter(csvfile, fieldnames=existing_columns)
    
                        # Write the header only if the file does not exist
                            if not file_exists:
                                writer = csv.DictWriter(csvfile, fieldnames=json_data[0].keys())
                                writer.writeheader()
        
                              # Write the rows
                            writer.writerows(filtered_data)
                        print(f"Data has been written to {csv_file_path} -- {ini} -- {end}")
                    else:
                        print(f"Failed to retrieve data from {second_response}: {second_response.status_code}")
                else:
                    print("No 'data' field found in the initial response.")
            else:
                print(f"Failed to retrieve initial data: {response.json()}")

        self.csv_file_path = csv_file_path
        self.next(self.divide_data)
        
    @step
    def divide_data(self):
        df = pd.read_csv(self.csv_file_path)

        # Sort the dataframe by the 'fecha' column
        df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d')
        df.sort_values(by='fecha', inplace=True)
        
        # Columns of interest
        columns_of_interest = [
            'tmed', 'tmin', 'tmax', 'prec', 'sol', 'velmedia', 'presMax', 
            'presMin', 'hrMedia', 'hrMax', 'hrMin'
        ]
        
        for column in columns_of_interest:
            if df[column].dtype != 'float64':
                df[column] = df[column].str.replace(',', '.').str.replace('Ip', '0').astype(float)
        
        df.to_csv('data/output.csv', index=False)

        self.next(self.data_quality_check)

    @step
    def data_quality_check(self):
        import great_expectations as ge
        data = ge.dataset.PandasDataset(pd.read_csv(self.csv_file_path))
        context = ge.get_context()
        checkpoint = context.get_checkpoint("source-checkpoint")
        checkpoint.run()

        self.next(self.data_ingestion_end)

    @step
    def data_ingestion_end(self):
        for column in columns_of_interest:
            columns_of_interest = ['tmed', 'tmin', 'tmax', 'prec', 'sol', 'velmedia', 'presMax', 'presMin', 'hrMedia', 'hrMax', 'hrMin']
            df_split = df[['fecha', column]].copy()
            df_split.rename(columns={"fecha": "ds", column: "y"},inplace=True)
            file_name = f"data/output_{column}.csv"
            df_split.to_csv(file_name, index=False)
            print(f"Saved {file_name}")
        self.next(self.end_ingestion)
    

    @step
    def end_ingestion(self):
        print('done')
        

class project_training(FlowSpec):
    @step
    def start_training(self):
        print("Training process started succesfully")
        self.next(self.training)

    @step
    def training(self):
        from metaflow_extra import extract_word,prophet_exp 
        folder_path = 'data/'  # Change to your folder path
        mlflow.set_tracking_uri('https://final-cloud-run-mlflow-905728444246.us-west2.run.app')
        best_models_map = {}
        
        for file_name in os.listdir(folder_path):
            if file_name.startswith('output_') and file_name.endswith('csv'):
                file_path = os.path.join(folder_path, file_name)
                variable = extract_word(file_path)
                best_models_map[variable] = prophet_exp(file_path, variable)
            print(variable,' experiments done')
        
        with open("best_models_map.json", "w") as outfile: 
            json.dump(best_models_map, outfile)
        self.next(self.training_next)
        
    @step
    def training_end(self):
        os.system('gsutil -m cp -r . gs://mlops-igm-final-project-bucket/')

        
if __name__=='__main__':
    pull_data()
    project_training()
import numpy as np
import pandas as pd
import pickle
import yaml

import warnings
warnings.filterwarnings("ignore")

def preprocessing(data_path,city):
    df = pd.read_csv(data_path)
    
    df['dt_iso'] = pd.to_datetime(df['dt_iso'], utc=True, format='%Y-%m-%d %H:%M:%S%z')
    df['date'] = df['dt_iso'].dt.date
    df['time'] = df['dt_iso'].dt.time
    daily_average = df.groupby(['date','city_name']).agg({
                        'temp': 'mean',
                        'temp_min': 'mean',
                        'temp_max': 'mean',
                        'rain_1h': 'mean',
                        'clouds_all':'mean'
                    }).reset_index()
    daily_madrid = daily_average[daily_average['city_name']==city]
    model_data=daily_madrid.rename(columns={'date': 'ds', 'temp': 'y'}).reset_index()
    
    model_data.to_csv('data/preprocessed_data.csv' index=False)


    
if __name__=="__main__":
    params = yaml.safe_load(open("params.yaml"))["features"]
    data_path = params["data_path"]
    city_name = params["city"]
    preprocessing(data_path,city_name)
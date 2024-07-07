import numpy as np
import pandas as pd

def load_taxi_data(month, year, vehicle_type):
    '''
    Load taxi data by month, year, and vehicle type.

    inputs:
        month: int or string, month of data to load
        year: int or string, year of data to load
        vehicle_type: string, type of vehicle data to load

    outputs:
        df: pandas dataframe extracted from raw data file (parquet)
    '''
    # process inputs
    month = str(int(month)) ; year = str(int(year)) # allow for value-type inputs
    if len(month) == 1:
        month = '0' + month

    # check vehicle type is valid
    if vehicle_type not in ['yellow', 'green', 'fhv', 'fhvhv']: # error check vehicle_type
        raise ValueError('Vehicle must be either \'yellow\', \'green\', \'fhv\', or \'hvfhv\'')

    url = "https://d37ci6vzurychx.cloudfront.net/trip-data/{}_tripdata_{}-{}.parquet".format(vehicle_type, year, month)
    return pd.read_parquet(url)
import numpy as np
import pandas as pd
from tqdm import tqdm

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

def bin_data(df):
    '''
    Bin ride data by hour and pickup location. If desired, can easily be made to bin by other time intervals.
    Src: 99_tableau_processing.ipynb

    inputs:
        df: pandas dataframe with datetime column
    
    outputs:
        ts_H_y_z: pandas dataframe with datetime index, location column, and counts column
    '''
    ts_h_y_z = df.set_index('pickup_datetime').groupby(['PULocationID', pd.Grouper(freq='h')]).size() # group by hour and location
    ts_H_y_z = ts_h_y_z.to_frame(name = 'counts') # convert to dataframe 
    ts_H_y_z = ts_H_y_z.unstack(level=0).fillna(0) # unstack to obtain missing hours as NaNs, fill missing rides with 0
    ts_H_y_z.index = ts_H_y_z.index.tz_localize('America/New_York', ambiguous = True) # need to localize timezone for time to appear
    ts_H_y_z = ts_H_y_z.stack(future_stack = True) # undo the unstacking
    ts_H_y_z = ts_H_y_z.swaplevel() # swap indices so location is first
    ts_H_y_z = ts_H_y_z.reset_index() # remove the stacked structure to have standard dataframe with rows and columns only
    ts_H_y_z.sort_values(by = ['PULocationID', 'pickup_datetime'], inplace = True) # sort by location and time
    return ts_H_y_z

def load_and_process_data(month, year, vehicle_type = 'yellow'):
    '''
    Load and process taxi data by month, year, and vehicle type. Since these datasets are large, we also
    rebin the dataset into a dataframe with date as its index, location as a column, and number of rides "Counts" as another column.

    TODO: 
    - Clean the data further based on findings from 02_cleaning_data.ipynb
    - Add data on prices
    - Allow for creation and processing of other vehicle types: green, fhv, fhvhv (see above)
    - Add extraneous data (weather, holidays, etc.)

    inputs:
        month: int, month of data to load
        year: int, year of data to load
        vehicle_type: string, type of vehicle data to load
    
    outputs:
        ts_H_y_z: pandas dataframe with datetime index, location column, and counts column
    '''
    # load data
    df = load_taxi_data(month = month, year = year, vehicle_type = 'yellow')

    # processing (minimal right now)
    if vehicle_type == 'yellow':
        # split datetime into data and time
        df.rename(
            columns = {'tpep_pickup_datetime': 'pickup_datetime',
                    'tpep_dropoff_datetime': 'dropoff_datetime'}, 
            inplace = True
        )
    else:
        raise ValueError('Only yellow taxi data is supported at the moment')

    # remove rows outside of month and year
    df = df[(df.pickup_datetime.dt.month == month) & (df.pickup_datetime.dt.year == year)]

    # bin data by hour and timezone
    ts_H_y_z = bin_data(df)
    return ts_H_y_z

def generate_processed_data(month_year, vehicle_type = 'yellow'):
    '''
    Generate processed data for a list of months and years. This function is useful for generating a large dataset.
    To save memory, it processes and bins the data corresponding to each month and year.
    Takes about ~30 sec per month and year of data

    inputs:
        month_year: (N,2)-array containing N months and years to process
        vehicle_type: string, type of vehicle data to load
    
    outputs:
        ts: pandas dataframe with datetime index, location column, and counts column for all months and years
    '''
    ts_list = []
    for year, month in tqdm(month_year): # can use itertools.product, but I think better in arrays
        ts = load_and_process_data(month, year)
        ts_list.append(ts)
    ts = pd.concat(ts_list, ignore_index = False).reset_index() # cannot ignore index! reset to be ordered!
    ts = ts.drop(columns = 'index') # remove index column
    return ts

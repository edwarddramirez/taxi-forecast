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

def bin_data(df, by_value = ['PULocationID'], additional_features = False):
    '''
    Bin ride data by hour and location information. If desired, can easily be made to bin by other time intervals.
    Src: 99_tableau_processing.ipynb (by_value = ['PULocationID'])
    Src: 02_a_processed_data_dev.ipynb (by_value = ['PULocationID', 'DOLocationID'])

    inputs:
        df: pandas dataframe with datetime column
        by_value: list of strings, values to bin data by (e.g. ['PULocationID'], ['PULocationID', 'DOLocationID'])
        additional_features: boolean, whether to include additional features (e.g. fare amount, tip amount, etc.)
    
    outputs:
        ts: pandas dataframe with datetime index, location column, and counts column
    '''
    # bin data
    gb = df.set_index('pickup_datetime').groupby(by_value + [pd.Grouper(freq='h')]) # group by hour and location
    ts = gb.size().to_frame(name = 'counts') # convert to dataframe with binned counts
    if additional_features: # include additional features
        ts = (ts
            .join(gb.agg({'total_amount': 'mean'}))
            .join(gb.agg({'tip_amount': 'mean'}))
            .join(gb.agg({'fare_amount': 'mean'}))
            .join(gb.agg({'trip_distance': 'mean'}))
            .join(gb.agg({'passenger_count': 'mean'}))
            .join(gb.agg({'trip_duration': 'mean'}))
        )
    # process dataframe format
    stack_level = [i for i in range(len(by_value))] # get level of indices
    unstack_level = [i+1 for i in range(len(by_value))] # get level of indices to unstack
    ts = ts.unstack(level=stack_level).fillna(0) # unstack to obtain missing hours as NaNs, fill missing rides with 0
    ts.index = ts.index.tz_localize('America/New_York', ambiguous = True) # need to localize timezone for time to appear
    ts = ts.stack(level=unstack_level, future_stack = True) # undo the unstacking
    ts = ts.swaplevel() # swap indices so location is first
    ts = ts.reset_index() # remove the stacked structure to have standard dataframe with rows and columns only
    ts.sort_values(by = ['pickup_datetime'] + by_value, inplace = True) # sort by location and time
    
    if by_value not in [['PULocationID'], ['DOLocationID'], ['PULocationID', 'DOLocationID']]:
        raise ValueError('Binning by ride or pickup location must be specified')
    return ts

def load_and_process_data(month, year, vehicle_type = 'yellow', by_value = ['PULocationID'], additional_features = False, taxi_zones = None):
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
        by_value: list of strings, values to bin data by (e.g. ['PULocationID'], ['PULocationID', 'DOLocationID'])
        additional_features: boolean, whether to include additional features (e.g. fare amount, tip amount, etc.)
        taxi_zones: list of desired taxi zones
    
    outputs:
        ts: pandas dataframe with datetime index, location columns, and counts column
    '''
    # load data
    df = load_taxi_data(month = month, year = year, vehicle_type = 'yellow')

    # downcast data types to save time and memory (from 64 to 32 bits)
    df = df.astype({'PULocationID':'int32', 'DOLocationID':'int32'}) # downgrade data type to save memory (from 64 to 32 bits)
    df = df.astype({'total_amount':'float32', 'trip_distance':'float32', 'fare_amount':'float32', 'tip_amount':'float32', 'passenger_count':'float32'}) # downgrade data type to save memory (from 64 to 32 bits)

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

    # return only desired taxi zones
    if taxi_zones is not None:
        df = df[df.PULocationID.isin(taxi_zones)]
        df = df[df.DOLocationID.isin(taxi_zones)]
    
    # calculate trip duration (min)
    df['trip_duration'] = (df.dropoff_datetime - df.pickup_datetime).dt.total_seconds() / 60 
    df = df.astype({'trip_duration':'float32'}) # downcast data type to save memory (from 64 to 32 bits)

    # impose conditions based on features
    df = df.dropna() # drop missing values (as per 01_b_cleaning_data.ipynb)
    conditions = (
        (df.pickup_datetime.dt.month == month) & (df.pickup_datetime.dt.year == year) & # filter by month and year
        (df.trip_distance > 0.) & (df.trip_distance < 500.) & # filter by trip distance
        (df.total_amount > 0.) & (df.total_amount < 5000.) & # filter by total amount
        (df.fare_amount > 0.) & (df.fare_amount < 5000.) & # filter by fare amount
        (df.tip_amount >= 0.) & (df.tip_amount < 1000.) &  # filter by tip amount 
        (df.trip_duration > 0.) & (df.trip_duration < 600.) & # filter by trip duration
        (df.passenger_count > 0.) & (df.passenger_count < 10) # filter by passenger count
        )
    df = df[conditions] # apply conditions


    # bin data by hour and timezone
    ts = bin_data(df, by_value=by_value, additional_features=additional_features)
    ts = ts.astype({'counts':'int32'}) # downcast data type to save memory (from 64 to 32 bits)
    return ts

def generate_processed_data(month_year, vehicle_type = 'yellow', by_value = ['PULocationID'], additional_features = False, taxi_zones = None):
    '''
    Generate processed data for a list of months and years. This function is useful for generating a large dataset.
    To save memory, it processes and bins the data corresponding to each month and year.
    Takes about ~30 sec per month and year of data

    inputs:
        month_year: (N,2)-array containing N months and years to process
        vehicle_type: string, type of vehicle data to load
        by_value: list of strings, values to bin data by (e.g. ['PULocationID'], ['PULocationID', 'DOLocationID'])
        additional_features: boolean, whether to include additional features (e.g. fare amount, tip amount, etc.)
        taxi_zones: list of desired taxi zones
    
    outputs:
        ts: pandas dataframe with datetime index, location column, and counts column for all months and years
    '''
    ts_list = []
    for year, month in tqdm(month_year): # can use itertools.product, but I think better in arrays
        ts = load_and_process_data(month, year, by_value=by_value, additional_features=additional_features, taxi_zones=taxi_zones)
        ts_list.append(ts)
    ts = pd.concat(ts_list, ignore_index = False).reset_index() # cannot ignore index! reset to be ordered!
    ts = ts.drop(columns = 'index') # remove index column
    return ts

def bin_data_expanded(df, by_value = ['PULocationID'], additional_features = False):
    '''
    Expanded version of bin_data function.
    '''
    if by_value == ['PULocationID']:
        # bin data
        if ~additional_features:
            ts = df.set_index('pickup_datetime').groupby(['PULocationID', pd.Grouper(freq='h')]).size() # group by hour and location
            ts = ts.to_frame(name = 'counts') # convert to dataframe 
        else:
            # do the same but include counts and average fare, tip, etc.
            gb = df.set_index('pickup_datetime').groupby(['PULocationID', pd.Grouper(freq='h')])
            ts = gb.size().to_frame('counts')
            ts = (ts
                .join(gb.agg({'total_amount': 'mean'}))
                .join(gb.agg({'tip_amount': 'mean'}))
                .join(gb.agg({'fare_amount': 'mean'}))
                .join(gb.agg({'trip_distance': 'mean'}))
            )
        # process dataframe format
        ts = ts.unstack(level=0).fillna(0) # unstack to obtain missing hours as NaNs, fill missing rides with 0
        ts.index = ts.index.tz_localize('America/New_York', ambiguous = True) # need to localize timezone for time to appear
        ts = ts.stack(future_stack = True) # undo the unstacking
        ts = ts.swaplevel() # swap indices so location is first
        ts = ts.reset_index() # remove the stacked structure to have standard dataframe with rows and columns only
        ts.sort_values(by = ['PULocationID', 'pickup_datetime'], inplace = True) # sort by location and time
    elif by_value == ['PULocationID', 'DOLocationID']:
        # bin data
        if ~additional_features:
            ts = df.set_index('pickup_datetime').groupby(['PULocationID', 'DOLocationID', pd.Grouper(freq='h')]).size() # group by hour and location
            ts = ts.to_frame(name = 'counts') # convert to dataframe 
        else:
            # do the same but include counts and average fare, tip, etc.
            gb = df.set_index('pickup_datetime').groupby(['PULocationID', 'DOLocationID', pd.Grouper(freq='h')])
            ts = gb.size().to_frame('counts')
            ts = (ts
                .join(gb.agg({'total_amount': 'mean'}))
                .join(gb.agg({'tip_amount': 'mean'}))
                .join(gb.agg({'fare_amount': 'mean'}))
                .join(gb.agg({'trip_distance': 'mean'}))
            )
        ts = df.set_index('pickup_datetime').groupby(['PULocationID', 'DOLocationID', pd.Grouper(freq='h')]).size() # group by hour and location
        ts = ts.to_frame(name = 'counts') # convert to dataframe 
        ts = ts.unstack(level=[0,1]).fillna(0) # unstack to obtain missing hours as NaNs, fill missing rides with 0
        ts.index = ts.index.tz_localize('America/New_York', ambiguous = True) # need to localize timezone for time to appear
        ts = ts.stack(level = [1,2], future_stack = True) # undo the unstacking
        ts = ts.reset_index() # remove the stacked structure to have standard dataframe with rows and columns only
        ts.sort_values(by = ['pickup_datetime', 'PULocationID', 'DOLocationID'], inplace = True) # NOTE: order is different here
    else:
        raise ValueError('Binning by ride or pickup location must be specified')
    return ts

def postprocess_data(ts, by_value = ['DOLocationID', 'PULocationID']):
    '''
    Postprocess data to account for fare hikes and missing values in routes with no counts.
    src: scratch/02_a_processed_data_dev.ipynb, scratch/03_a_processed_data_dev.ipynb

    inputs:
        ts: pandas dataframe with datetime index, location columns, and counts column (created using load_and_process_data function)

    outputs:
        ts: pandas dataframe with datetime index, location columns, and counts column (postprocessed) 
    '''

    # 23% fare hike: this simpler implementation better than only modifying fare price and recalculating total amount
    fare_hike_date = pd.Timestamp('2022-12-19 00:00:00').tz_localize('America/New_York', ambiguous = True) # need to specify time zone to avoid ambiguity
    prices = ['fare_amount', 'total_amount', 'tip_amount']
    for price in prices:
        ts[price] = np.where(ts['pickup_datetime'] <= fare_hike_date, ts[price] * 1.23, ts[price]) 
    
    # non-count values in routes with no counts are set to the mean of the non-zero values in the route (PULocationID, DOLocationID)
    # verified correct implementation by checking that the mean of the non-zero values in the route is the same as the assigned mean with this approach
    cols = ['total_amount', 'fare_amount', 'tip_amount', 'trip_distance', 'passenger_count', 'trip_duration']
    for c in cols:
        # Calculate average price per (latitude, longitude) where average_price is not zero
        avg_c_by_route = ts[ts[c] != 0].groupby(by_value)[c].mean().reset_index()
        avg_c_by_route.rename(columns={c : 'avg_c_route'}, inplace=True)

        # Merge this average price back to the original dataframe
        ts = ts.merge(avg_c_by_route, on=by_value, how='left')

        # Replace zero average prices with the calculated average price per location
        ts.loc[ts[c] == 0, c] = ts['avg_c_route']

        # Drop the temporary column
        ts.drop(columns=['avg_c_route'], inplace=True)
    return ts

def route_to_pulocation(ts, pulocationid):
    '''
    Combine route-level data to obtain pulocation-level data for a single taxi zone.
    UPDATE: This function inputs mean values for routes with no counts, rather than zero values.

    inputs:
        ts: pandas dataframe with datetime index, location columns, and counts column (created using load_and_process_data function)
        pulocationid: int, pickup location ID to filter data by

    outputs:
        ts_p: pandas dataframe with datetime index, location columns, and counts column for a single pickup location
    '''
    ts = ts[ts['PULocationID'] == pulocationid].copy()
    gb = ts.groupby(['pickup_datetime', 'PULocationID'])

    # weighted-average function accounting for zero-weights
    def weighted_average(df, val_col, weight_col, print_statement = False):
        values = df[val_col]
        weights = df[weight_col]
        if weights.sum() == 0.:
            return df[val_col].mean()
        return np.average(values, weights=weights)

    # from groupby, sum the counts and perform weighted average of total_amount, etc. using counts as weights
    ts_p = pd.DataFrame(gb['counts'].sum())
    ts_p['total_amount'] = gb.apply(lambda x: weighted_average(x, 'total_amount', 'counts'))
    ts_p['fare_amount'] = gb.apply(lambda x: weighted_average(x, 'fare_amount', 'counts'))
    ts_p['tip_amount'] = gb.apply(lambda x: weighted_average(x, 'tip_amount', 'counts'))
    ts_p['trip_distance'] = gb.apply(lambda x: weighted_average(x, 'trip_distance', 'counts'))
    ts_p['passenger_count'] = gb.apply(lambda x: weighted_average(x, 'passenger_count', 'counts'))
    ts_p['trip_duration'] = gb.apply(lambda x: weighted_average(x, 'trip_duration', 'counts'))
    ts_p = ts_p.reset_index()

    return ts_p


def route_to_pulocation_old(ts, pulocationid):
    '''
    Combine route-level data to obtain pulocation-level data for a single taxi zone.

    inputs:
        ts: pandas dataframe with datetime index, location columns, and counts column (created using load_and_process_data function)
        pulocationid: int, pickup location ID to filter data by

    outputs:
        ts_p: pandas dataframe with datetime index, location columns, and counts column for a single pickup location
    '''
    ts = ts[ts['PULocationID'] == pulocationid].copy()
    gb = ts.groupby(['pickup_datetime', 'PULocationID'])

    # weighted-average function accounting for zero-weights
    def weighted_average(x, weights):
        if weights.sum() == 0.:
            return 0.
        return np.average(x, weights=weights)

    # from groupby, sum the counts and perform weighted average of total_amount, etc. using counts as weights
    ts_p = pd.DataFrame(gb['counts'].sum())
    ts_p['total_amount'] = gb.apply(lambda x: weighted_average(x['total_amount'], weights=x['counts']))
    ts_p['fare_amount'] = gb.apply(lambda x: weighted_average(x['fare_amount'], weights=x['counts']))
    ts_p['tip_amount'] = gb.apply(lambda x: weighted_average(x['tip_amount'], weights=x['counts']))
    ts_p['trip_distance'] = gb.apply(lambda x: weighted_average(x['trip_distance'], weights=x['counts']))
    ts_p['passenger_count'] = gb.apply(lambda x: weighted_average(x['passenger_count'], weights=x['counts']))
    ts_p['trip_duration'] = gb.apply(lambda x: weighted_average(x['trip_duration'], weights=x['counts']))
    ts_p = ts_p.reset_index()
    return ts_p

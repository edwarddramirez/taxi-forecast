import sys, os
sys.path.append('..') # add parent directory to path

import numpy as np 
import pandas as pd
from tqdm import tqdm

from utils import processing as pr

class TaxiDataset():
    def __init__(self):
        self.df = pd.DataFrame()
        self.test = 'test'

    def generate_tabular_data(month_year, vehicle_type = 'yellow'):
        print('Generating Tabular Data: \n')
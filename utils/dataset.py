import sys, os
sys.path.append('..') # add parent directory to path

import numpy as np 
import pandas as pd
from tqdm import tqdm

from utils import processing as pr

class YellowTaxiDataset():
    def __init__(self, df, graph):
        self.df = df
        self.graph = graph


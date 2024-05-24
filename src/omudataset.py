import os
import pandas as pd 
import numpy as np

from torch.utils.data import Dataset

class OMUDataSet(Dataset):
    def __init__(self, csvfilename: str = None):
        self.X = None
        self.y = None
    
        if csvfilename is not None:
            sequences, labels = data_from_csv(csvfilename)
    
    def data_from_csv(csvfilename: str):
        sequences = []
        labels = []

        df = pd.read_csv(csvfilename)
        
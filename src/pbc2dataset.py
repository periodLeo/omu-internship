import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class PBC2DataSet(Dataset):
    def __init__(self, csvfilename: str = None, parsed_data: list = []):
        self.X = None
        self.y = None

        self.X_time     = None
        self.X_series   = None

        if csvfilename is not None :
            self.X, self.y = self.tensor_from_pbc2_csv(csvfilename)
            
            # Separate timestamps and data
            self.X_time     = [x[0] for x in self.X]
            self.X_series   = [x[1] for x in self.X]

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx: int) -> tuple :
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (tensor_time, tensor_data)
        """
        tensor_time, tensor_data = self.X[idx]
        label = self.y[idx]
        return (tensor_time, tensor_data), label

    def remove_nan(self) :
        """
        Remove NaN value,
        Replace them by mean of values in column.
        """
        for i in range(len(self)):

            nan_mask = torch.isnan(self.X_series[i])
            column_means = torch.nanmean(self.X_series[i], dim=0)

            self.X_series[i] = torch.where(nan_mask, column_means, self.X_series[i])

        return

    def tensor_from_pbc2_csv(self, csvfilename: str) -> tuple:
        """
        Args:
            csvfilename (str): Name of csv file containing data

        Returns:
            tuple: (data, labels)
        """
        df = pd.read_csv(csvfilename, index_col = 0)

        # group data by patients id to split them
        grouped = df.groupby('id')
        df_list = [group_data for _, group_data in grouped]

        data = []
        labels = []

        time_stamp = 'year'
        time_features_columns = ['serBilir', 'serChol', 'albumin', 'alkaline', 'SGOT', 'platelets', 'prothrombin']
        
        for df in df_list:
            # [ array of time stamp] , [ 1-d : each row of df, 2-d : each value of row]
            data.append([
                torch.from_numpy(df['year'].values), 
                torch.from_numpy(df[time_features_columns].values)
                ])
            labels.append(df['status2'].iloc[0]) # dead or alive

        return data, labels

if __name__ == "__main__":
    filename = "./pbc2.csv"
    eto = PBC2DataSet(filename)
    eto.remove_nan()
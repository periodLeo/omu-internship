import os
import pandas as pd
import numpy as np
import torch

from torch.nn.functional import normalize
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class PBC2DataSet(Dataset):
    def __init__(self, csvfilename: str = None, tensor_data: tuple = ()):
        self.X = None
        self.y = None

        self.X_time     = None
        self.X_series   = None
        self.labels     = None

        if csvfilename is not None :
            self.X, self.labels = self.tensor_from_pbc2_csv(csvfilename)
            self.y = torch.FloatTensor(self.labels)

        if tensor_data and type(tensor_data[0][0]) == torch._tensor.Tensor :
            self.X = [normalize(single_tensor) for single_tensor in tensor_data[0]]
            self.y = tensor_data[1]
            

        self.remove_nan(global_mean=320.)

        # Separate timestamps and data
        self.X_time     = [x[0] for x in self.X]
        self.X_series   = [x[1:] for x in self.X]

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx: int, splitted_data: bool = False) -> tuple :
        """
        Args:
            idx (int): Index
            splitted_data (bool): Timestep and vals splitted or not

        Returns:
            tuple: item, label
        """
        #tensor_time, tensor_data = self.X[idx]
        if(splitted_data):
            item = (self.X_time[idx], self.X_series[idx])
        else:
            item = self.X[idx]

        label = self.y[idx]
        return item, label

    def remove_nan(self, global_mean: float) :
        """
        Remove NaN value,
        Replace them by mean of values in column.
        Args:
            global_mean (float): mean to apply if columns only contain NaN vals 
        """

        # In this dataset it seems that only serChol can have only NaN values
        serChol_global_mean = global_mean

        for i in range(len(self)):
            nan_mask = torch.isnan(self.X[i])
            column_means = torch.nanmean(self.X[i], dim=0)

            # case where column is only NaN
            all_nan_columns = torch.isnan(column_means)
            column_means[all_nan_columns] = global_mean

            self.X[i] = torch.where(nan_mask, column_means, self.X[i])

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

        time_features_columns = ['year','serBilir', 'albumin', 'alkaline', 'SGOT', 'platelets', 'prothrombin']

        for df in df_list:
            # [ 1-d : each row of df, 2-d : each value of row]
            data.append(
                torch.from_numpy(np.float32(df[time_features_columns].values))
            )
            labels.append(df['status2'].iloc[0]) # dead or alive

        return data, labels

if __name__ == "__main__":
    raise NotImplementedError
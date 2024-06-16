import pandas as pd
import numpy as np
from scipy.stats import zscore

import datetime
from dateutil.relativedelta import relativedelta

def prepadding_dataframe(df_list: list, max_len: int) -> None:
    """
    Create a Zero-padding of size: max_len - len(dataframe).
    The padding is placed before other data points.

    Args:
    -- df_list: list -> list of pandas DataFrame of different size inferior to max_len.
    -- max_len: int  -> max size of ours DataFrame.
    """
    df_cols = df_list[0][0].columns.to_list()

    for i in range(len(df_list)):
        padd_length = max_len - len(df_list[i][0]['year'])
        padding_df = pd.DataFrame(
            0., index=range(padd_length), columns = df_cols)
        df_list[i][0] = pd.concat([padding_df, df_list[i][0]])

def get_data() -> list:
    # Create dataframe with everything from csv file
    filename = "./data/pbc2.csv"
    df = pd.read_csv(filename)

    # We will use status2 as our label
    df.rename(columns = {'status2' : 'label'}, inplace=True)

    # Name of the columns we will use
    FEATURE_COLUMNS = ['year','serBilir','albumin',
                'alkaline', 'SGOT', 'platelets', 
                'prothrombin']
    FEATURE_WITHOUT_YEAR = ['serBilir','albumin',
                'alkaline', 'SGOT', 'platelets', 
                'prothrombin']

    # Sequence is our lists of (dataframe, label) splitted by patient id.
    sequences = []
    for series_id, group in df.groupby('id'):
        sequences_features = group[FEATURE_COLUMNS]
        label = group.iloc[0].label
        sequences.append([sequences_features, label])

    # We prepadd our data for every dataframe to be the same length.
    prepadding_dataframe(sequences, 16)

    return sequences

def get_data_timesfm(sequences: list = []) -> tuple:
    if not sequences:
        # If no sequences is given :
        # Run get_data() to create list of tuple (dataframe, labels) and extract df on a new list.
        sequences = [seq for seq, _ in get_data()]

    timesfm_sequences = [] # Our list of sequences ready for timesfm.
    values_to_predict = [] # Last values of sequences that we will try to forecast.
    first_date = pd.to_datetime('2000-01-01') # relative start date.
    timesfm_columns = ['unique_id','ds','data']

    for df in sequences:
        # Names of values we will take for our DataFrame
        val_names = ['date','serBilir','albumin','alkaline','SGOT','platelets','prothrombin']

        formatted = pd.DataFrame(columns = timesfm_columns)
        last_vals = pd.DataFrame(columns = val_names)

        time_array = []
        id_array = []
        val_array = []

        val_names.remove('date')
        for e in val_names:
            time_array += df['year'].iloc[0:15].to_list()
            id_array += 15*[e]
            val_array += df[e].iloc[0:15].to_list()

            last_vals.loc[0, e] = df[e].iloc[15] 
        
        last_vals.loc[0, 'date'] = first_date + relativedelta(months=int(df['year'].iloc[15]*12))
        datetime_array = [first_date + relativedelta(months=int(time*12)) for time in time_array]

        formatted.unique_id = id_array
        formatted.ds = datetime_array
        formatted.data = val_array
        timesfm_sequences.append(formatted)
        values_to_predict.append(last_vals)

    return timesfm_sequences, values_to_predict

if __name__ == "__main__":
    sequences_with_labels = get_data()

    # Make a list of sequences only when label is not needed.
    sequences = [seq for seq, _ in sequences_with_labels]
    data, expected = get_data_timesfm(sequences)
    # grouped = {}
    # for label, group in data.groupby('unique_id'):
    #     grouped[label] = group
    # print(grouped['platelets'])
    # print(expected)
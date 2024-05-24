import torch
import numpy as np

def collate_fn_pre_padding(batch):
    # Split elements from batch
    #times  = [item[0] for item, _  in batch]
    series = [item  for item, _  in batch]
    labels = [label for _, label in batch]
    
    lengths = torch.tensor([len(tv) for tv in series], dtype=torch.float32)

    # Find max length
    max_len = max(len(t) for t in series)
    
    #padded_times = []
    padded_series = []

    for s in series:
        # Pre-padding 'time' data
        """
        padding_time = [0] * (max_len - len(t))
        padded_time_tensor = torch.tensor(padding_time, dtype=torch.float32) #torch.tensor(padding_time + t.tolist(), dtype=torch.long)
        padded_time_tensor = torch.cat((padded_time_tensor, t), dim=0)
        padded_times.append(padded_time_tensor)
        """
        
        # Pre-padding 'series' data
        padding_series = torch.zeros((max_len - s.size(0), s.size(1)), dtype=torch.float32)
        padded_series_tensor = torch.cat((padding_series, s), dim=0)
        padded_series.append(padded_series_tensor)
    
    # Stack padded sequences them to return as batch forpm
    #padded_times = torch.stack(padded_times, dim=0)
    padded_series = torch.stack(padded_series, dim=0)
    labels = torch.stack(labels, dim=0).unsqueeze(1)

    return padded_series, labels

def prepadding_dataframe(df_list: list, max_len: int):
    df_cols = df_list[0][0].columns.to_list()
    for i in range(len(df_list)):
        padd_length = max_len - len(df_list[i][0]['year'])
        padding_df = pd.DataFrame(0., index=range(padd_length), columns = df_cols)
        df_list[i][0] = pd.concat([padding_df, df_list[i][0]])

if __name__ == "__main__":
    raise NotImplementedError
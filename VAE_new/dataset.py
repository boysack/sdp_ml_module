from torch.utils.data import Dataset
import pandas as pd
from datetime import timedelta
import torch
from tqdm import tqdm

class ArolDataset(Dataset):
    def __init__(self, data_path):
        df = pd.read_csv(data_path, index_col=0)
        #df = pd.read_csv(data_path, index_col=0)
        if df.isna().any().any():
            raise Exception("Data contains NaN values")

        df['sensor_time'] = pd.to_datetime(df['sensor_time'])

        sample_df = pd.DataFrame(columns=df.columns)
        n_sample_df = pd.DataFrame(columns=df.columns)
    
        for _, row in df.iterrows():
            next_timestamp = row['sensor_time'] + timedelta(seconds=1)
            next_row = df[df['sensor_time'] == next_timestamp].head(1)
            if not next_row.empty:
                sample_df = pd.concat([sample_df, row.to_frame().T], ignore_index=True)
                n_sample_df = pd.concat([n_sample_df, next_row], ignore_index=True)

        self.data = sample_df.drop(columns=['sensor_time'])
        self.n_data = n_sample_df.drop(columns=['sensor_time'])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = torch.tensor(self.data.iloc[idx], dtype=torch.float).to(self.device)
        n_data = torch.tensor(self.n_data.iloc[idx], dtype=torch.float).to(self.device)

        return sample, n_data

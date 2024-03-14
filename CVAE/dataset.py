from torch.utils.data import Dataset
import pandas as pd
from datetime import timedelta
import torch

class ArolDataset(Dataset):
    def __init__(self, data_path, mode="train", train_perc=.7, val_perc=.2):
        df = pd.read_csv(data_path, index_col=0)
        if df.isna().any().any():
            raise Exception("Data contains NaN values")

        df['sensor_time'] = pd.to_datetime(df['sensor_time'])
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)

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

        # change the last two columns to be ProdSpee and LockDegree
        columns_to_move = ['ProdSpeed', 'LockDegree']
        remaining_columns = [col for col in self.data.columns if col not in columns_to_move]
        self.data = df[remaining_columns + columns_to_move]
        self.n_data = df[remaining_columns + columns_to_move]

        if mode == "train":
            self.data = self.data.iloc[:round(len(self.data)*train_perc)]
            self.n_data = self.n_data.iloc[:round(len(self.n_data)*train_perc)]
        elif mode == "val":
            self.data = self.data.iloc[round(len(self.data)*train_perc):round(len(self.data)*(train_perc+val_perc))]
            self.n_data = self.n_data.iloc[round(len(self.n_data)*train_perc):round(len(self.n_data)*(train_perc+val_perc))]
        elif mode == "test":
            self.data = self.data.iloc[round(len(self.data)*(train_perc+val_perc)):]
            self.n_data = self.n_data.iloc[round(len(self.n_data)*(train_perc+val_perc)):]
        else:
            raise Exception("Modality error, choose between train, val and test")

        self.mode = mode
        self.columns_name = self.data.columns
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = torch.tensor(self.data.iloc[idx], dtype=torch.float).to(self.device)
        n_data = torch.tensor(self.n_data.iloc[idx], dtype=torch.float).to(self.device)

        return sample, n_data

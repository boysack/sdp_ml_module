from torch.utils.data import Dataset
import pandas as pd
import torch

class ArolDataset(Dataset):
    def __init__(self, data_path):
        df = pd.read_csv(data_path).drop(columns=['sensor_time'])

        #self.data = df.drop(columns=["ProdSpeed", "LockDegree"])
        self.data = df
        self.conditional = df[["ProdSpeed", "LockDegree"]]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = torch.tensor(self.data.iloc[idx], dtype=torch.float).to(self.device)
        conditional = torch.tensor(self.conditional.iloc[idx], dtype=torch.float).to(self.device)

        return sample, conditional

import argparse
from model_module import CVAE
import time
from dataset import ArolDataset
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*SettingWithCopyWarning.*")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Start training-generating synthetic data')
    parser.add_argument('--train', action='store_true', help='Begin training')
    parser.add_argument('--num_epochs', type=int, default=25, help='Epochs number')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--dataset_path', default="data/linear_interpolated_union.csv", type=str, help='Dataset path', metavar="")
    parser.add_argument('--checkpoint_path', type=str, default=f"checkpoints/checkpoint-{int(time.time())}", help='Checkpoint path', metavar="")
    parser.add_argument('--gen_data',  action='store_true', help='Begin generating synthetic data')
    return parser

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CVAE(seq_len=1, feat_dim=11, conditional_dim=13, enc_out_dim=5, latent_dim=3, beta=.5, learning_rate=0.005, min_std=0.025, checkpoint_path=args.checkpoint_path).to(device)
    trainer = pl.Trainer(max_epochs=args.num_epochs)

    dataset = ArolDataset(args.dataset_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    trainer.fit(model, dataloader)
    
""" optimizer = model.configure_optimizers()
    
    for epoch in range(args.num_epochs):
        total_loss = 0.0
        for batch_data in dataloader:
            # Forward pass
            loss = model.training_step(batch_data)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()  # Compute gradients
            
            # Update model parameters
            optimizer.step()
            
            total_loss += loss.item()
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        
        # Print epoch and average loss
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {avg_loss:.4f}")
        model.on_train_epoch_end()
        model.save_model(args.checkpoint_path)
 """

def generate_synthetic_data():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CVAE(seq_len=1, feat_dim=11, conditional_dim=13, enc_out_dim=5, latent_dim=3, beta=.5, learning_rate=0.005, min_std=0.025, checkpoint_path=args.checkpoint_path).to(device)
    model.load_model()
    model.eval()

    df = generate_df(model=model)
    print(df)

def generate_df(model: CVAE):
    df_means, df_stds = pd.read_csv(f"data/metadata/df_original_means.csv", index_col=0).iloc[0], pd.read_csv(f"data/metadata/df_original_stds.csv", index_col=0).iloc[0]
    prev = pd.read_csv(f"data/linear_interpolated_all.csv", index_col=0).drop(columns=['sensor_time']).iloc[0]
    
    means = df_means.values
    stds = df_stds.values

    prev_norm = torch.tensor(((prev - means)/stds).values)
    prev_norm = prev_norm.view(1, prev_norm.shape[0])

    pred = generate_sample(model=model, prev=prev_norm, num_preds=1)
    print(pred.shape)
    sample = pred.numpy()[0] * stds + means
    df_sample = pd.DataFrame(sample, columns=range(model.feat_dim))
    #??
    for column in df_sample.columns:
        df_sample[column] = df_sample[column].transform(log_revert_field)

    return df_sample

def generate_sample(model: CVAE, prev, num_preds=1):
    p = torch.distributions.Normal(torch.zeros(1, model.latent_dim), torch.ones(1, model.latent_dim))
    z = p.rsample((num_preds,)).flatten(1)

    z_cond = torch.cat((z, prev), dim=1)
    with torch.no_grad():
        pred_mean, pred_log_scale = model.decoder(z_cond.float())
    pred_scale = torch.exp(pred_log_scale) + model.min_std
    e = torch.randn(num_preds, 1, 1)
    pred = pred_mean + e * pred_scale
        
    return pred

def log_revert_field(column):
    return np.exp(column)

def main(args):
    if args.train and args.gen_data:
        print("Can't both train the model and generate data in one run!")
    if args.train:
        train(args)
    elif args.gen_data:
        generate_synthetic_data()
    else:
        parser.print_help()

if __name__=="__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    main(args)
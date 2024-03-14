import os
import argparse
import time
import torch
from torch import nn
from dataset import ArolDataset
from torch.utils.data import DataLoader
from model import CVAE
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import pandas as pd
from query_mongo import query_mongo
import json
import warnings
import matplotlib.pyplot as plt
from math import exp


warnings.filterwarnings("ignore", category=FutureWarning)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Start training-generating synthetic data')

    parser.add_argument('--train', action='store_true', help='Begin training')
    parser.add_argument('--dataset_path', default="src/ml_module_final/other_models/data/dataframe_interpolated_augmented.csv", type=str, help='Dataset path', metavar="")
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Epochs number')
    parser.add_argument('--checkpoint_path', type=str, default=f"src/ml_module_final/CVAE/checkpoints/CVAE_final_likelihood", help='Checkpoint path', metavar="")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate of the task", metavar="")
    parser.add_argument('--save_plots', action='store_true', help='Save plots about loss in validation and test phase')
    parser.add_argument('--plot_path', type=str, default="src/ml_module_final/CVAE/plots", help='Save plots about loss in train and test phase')
    parser.add_argument('--es_patience', type=int, default=10, help='Patience of the early stop policy')
    parser.add_argument('--std_path', type=str, default="src/ml_module_final/other_models/data/metadata/dataframe_interpolated_augmented_std.csv")

    parser.add_argument('--gen',  action='store_true', help='Begin generating synthetic data')
    parser.add_argument('--prod_speed', type=float, default=27380.0, help='Production speed value', metavar="")
    parser.add_argument('--lock_degree',  type=float, default=10.0, help='Lock degree value', metavar="")
    parser.add_argument('--frequency',  type=int, default=1, help='Sampling frequency', metavar="")
    parser.add_argument('--sensors', help='Sensors to be used', metavar="")

    return parser

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    min_std = pd.read_csv(args.std_path, header=None)
    min_std = torch.tensor(min_std.transpose().iloc[1].values.astype(float))
    model = CVAE(seq_len=1, feat_dim=13, enc_out_dim=10, latent_dim=8, cond_dim=2, beta=0.5, learning_rate=args.lr, min_std=0.25*min_std).to(device)
    losses = {}
    epochs_digits = len(str(args.num_epochs))

    train_dataset = ArolDataset(args.dataset_path)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    val_dataset = ArolDataset(args.dataset_path, mode="val")
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    test_dataset = ArolDataset(args.dataset_path, mode="test")
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # TRAIN
    optimizer = model.configure_optimizers()
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    best_val_loss = torch.inf

    print("\nTRAINING")
    for epoch in range(args.num_epochs):
        model.train()
        overall_loss = 0
        overall_recon_loss = 0
        overall_KDL_loss = 0
        for batch_idx, (x, y) in enumerate(train_dataloader):
            if x.size(0) != args.batch_size:
                # last batch
                batch_idx -= 1
                break
            x.to(device)
            y.to(device)

            optimizer.zero_grad()

            z, mu, std, y_hat, y_hat_mean, y_hat_log_scale = model(x)
            
            loss, kl, recon_loss = model.calculate_loss(z, mu, std, y_hat, y_hat_mean, y_hat_log_scale, y)

            overall_loss += loss

            overall_recon_loss += recon_loss
            overall_KDL_loss += kl

            loss.backward()
            optimizer.step()

        # every 5 epochs
        if epoch%5 == 0 and epoch != 0:
            val_loss, val_recon_loss, val_kdl_loss = eval(model, val_dataloader, "VALIDATION", args)
            losses['val_avg_elbo'] = val_loss
            losses['val_avg_recon_loss'] = val_recon_loss
            losses['val_avg_kdl_loss'] = val_kdl_loss

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= args.es_patience:
                    print("EARLY STOPPING")
                    break
            
            scheduler.step(val_loss)

        train_avg_loss = (overall_loss / ((batch_idx+1)*args.batch_size)).item()
        train_avg_recon_loss = (overall_recon_loss / ((batch_idx+1)*args.batch_size)).mean().item()
        train_avg_kdl_loss = (overall_KDL_loss / ((batch_idx+1)*args.batch_size)).mean().item()

        print(f"Epoch {epoch+1:0>{epochs_digits}} | Average training Loss (ELBO): {train_avg_loss:4f} | Average training Reconstruction Loss (Negative Log Likelihood): {train_avg_recon_loss:4f} | Average training KDL Loss: {train_avg_kdl_loss:4f} | Learning Rate: {optimizer.param_groups[0]['lr']}")

        losses['train_avg_elbo'] = (overall_loss / ((batch_idx+1)*args.batch_size)).item()
        losses['train_avg_recon_loss'] = (overall_loss / ((batch_idx+1)*args.batch_size)).item()
        losses['train_avg_kdl_loss'] = (overall_loss / ((batch_idx+1)*args.batch_size)).item()

    model.save_model(args.checkpoint_path)

    test_loss, test_recon_loss, test_kdl_loss = eval(model, test_dataloader, "TEST", args)
    losses['test_avg_elbo'] = test_loss
    losses['test_avg_recon_loss'] = test_recon_loss
    losses['test_avg_kdl_loss'] = test_kdl_loss
    losses['epochs'] = epoch+1

    if args.save_plots:
        plt.figure(figsize=(6,5))
        plt.bar(x=0-.15, height=exp((-losses['val_avg_recon_loss'])/len(val_dataset.data.columns)), width=.15, label="Validation")
        plt.bar(x=0+.15, height=exp((-losses['test_avg_recon_loss'])/len(test_dataset.data.columns)), width=.15, label="Test")

        plt.ylabel('Likelihood')
        plt.xticks(ticks=[0], labels=['CVAE'], rotation=45)
        plt.legend()
        plt.savefig(args.plot_path+"/"+"train_val_plot")

def eval(model: CVAE, dataloader: DataLoader, descr: str, args):
    model.eval()
    average_loss = 0
    print(f"\n{descr.upper()}")
    with torch.no_grad():
        overall_loss = 0
        overall_recon_loss = 0
        overall_kl_loss = 0
        overall_samples = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            z, mu, std, y_hat, y_hat_mean, y_hat_log_scale = model(x)

            loss, kl, recon_loss = model.calculate_loss(z, mu, std, y_hat, y_hat_mean, y_hat_log_scale, y)
            
            overall_loss += loss
            overall_recon_loss += recon_loss
            overall_kl_loss += kl

            overall_samples += x.size(0)

        average_loss = (overall_loss / overall_samples).item()
        average_recon_loss = (overall_recon_loss / overall_samples).mean().item()
        average_kl_loss = (overall_kl_loss / overall_samples).mean().item()

        print(f'Average {descr.lower()} ELBO: {average_loss:.4f}')
        print(f'Average {descr.lower()} Reconstruction Loss (Negative Log Likelihood): {average_recon_loss:.4f}')
        print(f'Average {descr.lower()} KL Loss: {average_kl_loss:.4f}')

        print()

    return average_loss, average_recon_loss, average_kl_loss

def gen(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CVAE(seq_len=1, feat_dim=13, enc_out_dim=10, latent_dim=8, cond_dim=2, beta=0.5, learning_rate=args.lr, min_std=0.025).to(device)
    if os.path.exists(args.checkpoint_path):
        model.load_model(args.checkpoint_path)
    else:
        raise Exception("Checkpoint doesn't exists!")
    model.eval()

    with torch.no_grad():
        dataset = ArolDataset(args.dataset_path)
        
        cond = torch.tensor([args.lock_degree, args.prod_speed], dtype=torch.float32)
        cond = cond.reshape((1, cond.shape[0]))

        start = dataset.__getitem__(0)[0]
        start = start[:-2]
        start = start.reshape((1, start.shape[0]))
        start = torch.cat([start, cond], dim=1)

        freq = args.frequency

        sensors = json.loads(args.sensors)
        result = {}
        for key in sensors:
            for item in sensors[key]:
                headNumber = item['headNumber']
                sensorNames = [sensor['name'] for sensor in item['sensorNames']]
                if headNumber == 0:
                    for key in result:
                        result[key].extend(sensorNames)
                else:
                    if headNumber not in result:
                        result[headNumber] = []
                    result[headNumber].extend(sensorNames)

        dataframes = []
        while True:
            for head in result.keys():
                columns_to_keep = result[head]
                df_01 = pd.DataFrame(columns=dataset.columns_name)
                # generate one minute of data
                for i in range(60):
                    z, mu, std, y_hat, y_hat_mean, y_hat_log_scale = model(start)
                    start = y_hat
                    start = start[:,:-2]
                    start = torch.cat([start, cond], dim=1)

                    df_01 = pd.concat([df_01, pd.DataFrame(y_hat, index=[i], columns=dataset.columns_name)])

                    for col in df_01.columns:
                        if col not in columns_to_keep:
                            df_01.drop(columns=col, inplace=True)

                dataframes.append((df_01, head))

            query_mongo(dataframes, freq)
            time.sleep(60)
            #wait(signal)

def main(args):
    if args.train:
        train(args)
    elif args.gen:
        gen(args)

if __name__=="__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    main(args)

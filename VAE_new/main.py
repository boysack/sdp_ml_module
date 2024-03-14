import argparse
import time
import torch
from dataset import ArolDataset
from torch.utils.data import DataLoader
from model import VAE
from model_CVAE import CVAE
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import matplotlib.pyplot as plt
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser(description='Start training-generating synthetic data')

    parser.add_argument('--train', action='store_true', help='Begin training')
    parser.add_argument('--dataset_path', default="data/linear_interpolated_union.csv", type=str, help='Dataset path', metavar="")
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=25, help='Epochs number')
    parser.add_argument('--checkpoint_path', type=str, default=f"VAE_new/checkpoints/checkpoint-{int(time.time())}", help='Checkpoint path', metavar="")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate of the task", metavar="")

    parser.add_argument('--gen',  action='store_true', help='Begin generating synthetic data')

    return parser

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ArolDataset(args.dataset_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = VAE(input_dim=13, hidden_dim=10, latent_dim=8, checkpoint_path=args.checkpoint_path, lr=args.lr).to(device)
    model = CVAE(seq_len=1, feat_dim=13, enc_out_dim=10, latent_dim=8, cond_dim=2, beta=0.5, learning_rate=args.lr, min_std=0.025).to(device)

    #optimizer = model.get_optimizer()
    optimizer = model.configure_optimizers()
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.0025, patience=10, verbose=True)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

    model.train()

    for epoch in range(args.num_epochs):
        overall_loss = 0
        overall_recon_loss = 0
        overall_KDL_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            if x.shape[0] != args.batch_size:
                continue
            x.to(device)
            y.to(device)

            optimizer.zero_grad()

            # VAE
            #y_hat, mean, log_var = model(x)
            # CVAE
            z, mu, std, y_hat, y_hat_mean, y_hat_log_scale = model(x)
            # VAE
            #loss, recon_loss, KDL_loss = model.loss_function(y, y_hat, mean, log_var)
            loss, kl, recon_loss = model.calculate_loss(z, mu, std, y_hat, y_hat_mean, y_hat_log_scale, y)

            overall_loss += loss

            overall_recon_loss += recon_loss
            overall_KDL_loss += kl

            loss.backward()
            optimizer.step()

        scheduler.step()

        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / ((batch_idx+1)*args.batch_size), "\tLearning Rate: ", {optimizer.param_groups[0]['lr']})
        print("\t\tAverage recon loss: ", overall_recon_loss / ((batch_idx+1)*args.batch_size))
        print("\t\tAverage KDL loss: ", overall_KDL_loss / ((batch_idx+1)*args.batch_size))

    #model.save_model(args.num_epochs, optimizer, overall_loss / (batch_idx*args.batch_size))
    model.save_model(args.checkpoint_path)
    
def gen(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #model = VAE(input_dim=13, hidden_dim=10, latent_dim=8, checkpoint_path=args.checkpoint_path, lr=args.lr, load=True).to(device)
    model = CVAE(seq_len=1, feat_dim=13, enc_out_dim=10, latent_dim=8, cond_dim=2, beta=0.5, learning_rate=args.lr, min_std=0.025).to(device)
    model.load_model(args.checkpoint_path)
    model.eval()

    with torch.no_grad():
        dataset = ArolDataset(args.dataset_path)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        """ for batch_idx, (x, y) in enumerate(dataloader):
            #VAE
            #y_hat, mean, log_var = model(x)
            #CVAE
            z, mu, std, y_hat, y_hat_mean, y_hat_log_scale = model(x)

            first_sensor_values = y_hat[:,0]
            array = first_sensor_values.numpy()
            print(y_hat)
            plt.plot(array)
            plt.show()

            break """
        
        cond = torch.tensor([10, 27380], dtype=torch.float32)
        cond = cond.reshape((1, cond.shape[0]))

        start = dataset.__getitem__(0)[0]
        start = start[:-2]
        start = start.reshape((1, start.shape[0]))
        start = torch.cat([start, cond], dim=1)
        
        df_01 = pd.DataFrame(columns=range(start.shape[0]))

        for i in range(1, 1000):
            z, mu, std, y_hat, y_hat_mean, y_hat_log_scale = model(start)
            start = y_hat
            start = start[:,:-2]
            start = torch.cat([start, cond], dim=1)

            df_01 = pd.concat([df_01, pd.DataFrame(y_hat, index=[i])])

        df_01.plot(subplots=True, figsize=(50,5))



        cond = torch.tensor([10, 1000], dtype=torch.float32)
        cond = cond.reshape((1, cond.shape[0]))

        start = dataset.__getitem__(0)[0]
        start = start[:-2]
        start = start.reshape((1, start.shape[0]))
        start = torch.cat([start, cond], dim=1)
        
        df_02 = pd.DataFrame(columns=range(start.shape[0]))

        for i in range(1, 1000):
            z, mu, std, y_hat, y_hat_mean, y_hat_log_scale = model(start)
            start = y_hat
            start = start[:,:-2]
            start = torch.cat([start, cond], dim=1)

            df_02 = pd.concat([df_02, pd.DataFrame(y_hat, index=[i])])

        df_02.plot(subplots=True, figsize=(50,5))
        plt.show()

        """ for batch_idx, (x, _) in enumerate(tqdm(test_loader)):
            x = x.view(batch_size, x_dim)
            x = x.to(DEVICE)
            
            x_hat, _, _ = model(x)


            break """

def main(args):
    if args.train:
        train(args)
    elif args.gen:
        gen(args)

if __name__=="__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    main(args)

import argparse
from model_module import CVAE
import time
from dataset import ArolDataset
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl

def parse_arguments():
    parser = argparse.ArgumentParser(description='Start training-generating synthetic data')
    parser.add_argument('--train', action='store_true', help='Begin training')
    parser.add_argument('--num_epochs', type=int, default=25, help='Epochs number')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--dataset_path', default="data/linear_interpolated_all.csv", type=str, help='Dataset path', metavar="")
    parser.add_argument('--checkpoint_path', type=str, default=f"checkpoints/checkpoint-{int(time.time())}", help='Checkpoint path', metavar="")
    parser.add_argument('--gen_data',  action='store_true', help='Begin generating synthetic data')
    return parser

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CVAE(seq_len=1, feat_dim=15, conditional_dim=2, enc_out_dim=5, latent_dim=3, beta=1.0, learning_rate=0.005, min_std=0.025, checkpoint_path=args.checkpoint_path).to(device)
    dataset = ArolDataset(args.dataset_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    trainer = pl.Trainer(max_epochs=args.num_epochs)
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
    pass

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
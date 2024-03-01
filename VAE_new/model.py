from torch import nn
import torch
import os
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        nn.init.xavier_normal_(self.FC_input.weight)
        nn.init.constant_(self.FC_input.bias, 0)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.FC_input2.weight)
        nn.init.constant_(self.FC_input2.bias, 0)

        ## added by me
        self.FC_input3 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.FC_input3.weight)
        nn.init.constant_(self.FC_input3.bias, 0)
        self.rnn = nn.RNN(input_dim, hidden_dim)
        nn.init.xavier_normal_(self.rnn.weight_ih_l0)
        nn.init.xavier_normal_(self.rnn.weight_hh_l0)

        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        nn.init.xavier_normal_(self.FC_mean.weight)
        nn.init.constant_(self.FC_mean.bias, 0)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        nn.init.xavier_normal_(self.FC_var.weight)
        nn.init.constant_(self.FC_var.bias, 0)
        
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        """ x = self.FC_input(x)
        x = self.bn(x)
        x = self.LeakyReLU(x) """

        x = x.view(1, x.size(0), x.size(1))
        x, h = self.rnn(x)
        x = x.view(x.size(1), x.size(2))

        x = self.FC_input2(x)
        x = self.LeakyReLU(x)
        
        x = self.FC_input3(x)
        x = self.LeakyReLU(x)

        mean = self.FC_mean(x)
        log_var = self.FC_var(x)
        
        return mean, log_var
    
    """ def forward(self, x):
        x = self.FC_input(x)
        x = self.LeakyReLU(x)

        x = self.FC_input2(x)
        x = self.LeakyReLU(x)

        mean = self.FC_mean(x)
        log_var = self.FC_var(x)
        
        return mean, log_var """
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()

        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        nn.init.xavier_normal_(self.FC_hidden.weight)
        nn.init.constant_(self.FC_hidden.bias, 0)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.FC_hidden2.weight)
        nn.init.constant_(self.FC_hidden2.bias, 0)

        ## added by me
        self.FC_hidden3 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.FC_hidden3.weight)
        nn.init.constant_(self.FC_hidden3.bias, 0)
        self.rnn = nn.RNN(hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.rnn.weight_ih_l0)
        nn.init.xavier_normal_(self.rnn.weight_hh_l0)

        self.FC_output = nn.Linear(hidden_dim, output_dim)
        nn.init.xavier_normal_(self.FC_output.weight)
        nn.init.constant_(self.FC_output.bias, 0)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.bn = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x):
        x = self.FC_hidden(x)
        #x = self.bn(x)
        x = self.LeakyReLU(x)

        x = self.FC_hidden2(x)
        x = self.LeakyReLU(x)

        """ x = self.FC_hidden3(x)
        x = self.LeakyReLU(x) """
        
        x = x.view(1, x.size(0), x.size(1))
        x, h = self.rnn(x)
        x = x.view(x.size(1), x.size(2))
        
        x_hat = self.FC_output(x)
        #x_hat = torch.sigmoid(x_hat)
        return x_hat
    
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, checkpoint_path, lr, load=False):
        super(VAE, self).__init__()
        self.i = 0
        self.encoder = Encoder(input_dim = input_dim, hidden_dim = hidden_dim, latent_dim = latent_dim)
        self.decoder = Decoder(latent_dim = latent_dim, hidden_dim = hidden_dim, output_dim = input_dim)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.lr = lr

        if os.path.exists(self.checkpoint_path) and load:
            self.load_model()
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), self.lr)

        self.i = 0

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)
        z = mean + var*epsilon
        return z
        
    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decoder(z)
        
        return x_hat, mean, log_var
    
    def loss_function(self, y, y_hat, mean, log_var):
        self.i += 1
        if self.i == 10000:
            print(y_hat)
            raise Exception()
        recon_loss = F.mse_loss(y_hat, y, reduction='mean')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        """ print("y_hat")
        print(y_hat)
        print("y")
        print(y)
        print("recon_loss")
        print(recon_loss)

        print("mean")
        print(mean)
        print("log_var")
        print(log_var)
        print("KLD")
        print(KLD)
        if self.i == 5:
            raise Exception()
        self.i += 1 """
        """ self.i += 1
        print(self.i) """
        """ print("recon_loss: ", recon_loss)
        print("KLD: ", KLD) """
        return recon_loss + KLD, recon_loss, KLD
    
    def save_model(self, epoch, optimizer, loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, self.checkpoint_path)

    def load_model(self):
        self.checkpoint = torch.load(self.checkpoint_path)

        self.load_state_dict(self.checkpoint["model_state_dict"])

        self.optimizer = torch.optim.Adam(self.parameters(), self.lr)
        self.optimizer.load_state_dict(self.checkpoint["optimizer_state_dict"])
    
    def get_optimizer(self):
        return self.optimizer

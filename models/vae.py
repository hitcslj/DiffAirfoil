import torch
from torch import nn

'''https://arxiv.org/abs/2004.12585'''
class BN_Layer(nn.Module):
    def __init__(self, dim_z, tau=0.5, mu=True):
        super(BN_Layer, self).__init__()
        self.dim_z = dim_z

        self.tau = torch.tensor(tau)  # tau: float in range (0,1)
        self.theta = torch.tensor(0.5, requires_grad=True)

        self.gamma1 = torch.sqrt(self.tau + (1 - self.tau) * torch.sigmoid(self.theta))  # for mu
        self.gamma2 = torch.sqrt((1 - self.tau) * torch.sigmoid((-1) * self.theta))  # for var

        self.bn = nn.BatchNorm1d(dim_z)
        self.bn.bias.requires_grad = False
        self.bn.weight.requires_grad = True

        if mu:
            with torch.no_grad():
                self.bn.weight.fill_(self.gamma1)
        else:
            with torch.no_grad():
                self.bn.weight.fill_(self.gamma2)

    def forward(self, x):  # x:(batch_size,dim_z)
        x = self.bn(x)
        return x


class VAE(nn.Module):
    def __init__(self, feature_size=257*2, latent_size=128):
        super(VAE, self).__init__()
        self.latent_size = latent_size
        self.feature_size = feature_size

        # encode
        self.fc1  = nn.Linear(feature_size, 512)
        self.fc21 = nn.Linear(512, latent_size)
        self.fc22 = nn.Linear(512, latent_size)
        self.bn_mu = BN_Layer(latent_size, mu=True)
        self.bn_var = BN_Layer(latent_size, mu=False)

        # decode
        self.fc3 = nn.Linear(latent_size, 512)
        self.fc4 = nn.Linear(512, feature_size)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):  
        '''
        x: (bs, feature_size)
        '''
        h1 = self.elu(self.fc1(x))
        z_mu = self.fc21(h1)
        z_var = self.fc22(h1)
        z_mu = self.bn_mu(z_mu)
        z_var = self.bn_var(z_var)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def encode2(self, x):
        mu, logvar = self.encode(x.view(-1, self.feature_size))
        z = self.reparameterize(mu, logvar)
        return z

    def decode(self, z): # P(x|z, c)
        '''
        z: (bs, latent_size)
        c: (bs, condition_size)
        '''
        h3 = self.elu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x.reshape(-1, self.feature_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
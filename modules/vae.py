import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# VAE
class VAE(nn.Module):
    def __init__(self,enc,dec,z_sz=10):
        super(VAE, self).__init__()
        self.enc = enc
        self.dec = dec

    def reperam(self,mu,logvar):
        eps = Variable(torch.randn(mu.size(0), mu.size(1)))
        z = mu + eps*torch.exp(logvar/2)
        return z

    def forward(self, x,c=None):
        h = self.enc(x)
        mu, logvar = torch.chunk(h,2,dim=1)
        z = self.reperam(mu, logvar)
        out = self.dec(z,c)
        return out, mu, logvar

    def sample(self, z,c=None):
        return self.dec(z,c)

    def loss(self, x, out, mu, logvar, normalize=1, size_average=False):
        KL = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KL /= normalize
        XEnt = F.binary_cross_entropy(out, x, size_average=size_average)
        return KL, XEnt

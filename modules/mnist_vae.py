import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Parameters
data_dir = '../data/MNIST'
z_sz = 5
learning_rate = 0.01
batch_sz = 50
im_sz = 28*28
num_epochs = 10

DEVICE = torch.device('cpu')
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')

# Load Data
dataset = torchvision.datasets.MNIST(root=data_dir,
                         train=True,
                         transform=transforms.ToTensor(),
                         download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_sz,
                                          shuffle=True)

# Encoder
class ConvEncoder(nn.Module):
    def __init__(self,z_sz=10):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 2*z_sz)

    def forward(self, x):
        # x is 50 x 1 x 28 x 28
        x = self.conv1(x)
        # x is 50 x 10 x 24 x 24
        x = F.relu(F.max_pool2d(x, 2))
        # x is 50 x 10 x 12 x 12
        x = self.conv2(x)
        # x is 50 x 20 x 8 x 8
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        # x is 50 x 20 x 4 x 4
        x = F.relu(x)
        x = x.view(-1, 320)
        # x is 50 x 320
        x = F.relu(self.fc1(x))
        # x is 50 x 50
        x = self.fc2(x)
        # x is 50 x (2 * z_sz)
        return x

# Decoder
class DeconvDecoder(nn.Module):
    def __init__(self,z_sz=10):
        super(DeconvDecoder, self).__init__()
        self.fc1 = nn.Linear(z_sz, 32)
        self.fc2 = nn.Linear(32, 32)
        self.deconv1 = nn.ConvTranspose2d(32,16,10)
        self.deconv2 = nn.ConvTranspose2d(16,3,10)
        self.deconv3 = nn.ConvTranspose2d(3,1,10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 32,1,1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.deconv3(x)
        return F.sigmoid(x)

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

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = torch.chunk(h,2,dim=1)
        z = self.reperam(mu, logvar)
        out = self.dec(z)
        return out, mu, logvar

    def sample(self, z):
        return self.decode(z)

    def loss(self, x, out, mu, logvar):
        KL = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KL /= (batch_sz*im_sz)
        XEnt = F.binary_cross_entropy(out, x)
        return KL, XEnt

enc = ConvEncoder(z_sz)
dec = DeconvDecoder(z_sz)
vae = VAE(enc,dec,z_sz)

optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
iter_per_epoch = len(data_loader)

# For debugging
data_iter = iter(data_loader)
fixed_x, _ = next(data_iter)
torchvision.utils.save_image(fixed_x.cpu(), '../results/MNIST/real_images.png')
fixed_x = Variable(fixed_x)


for epoch in range(num_epochs):
    for batch_idx, (images, _) in enumerate(data_loader):
        images = Variable(images)
        out, mu, logvar = vae(images)
        KL, XEnt = vae.loss(images, out, mu, logvar)
        L = KL + XEnt
        optimizer.zero_grad()
        L.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print ("Epoch[%d/%d], Step [%d/%d], Total Loss: %.4f, "
                   "KL Loss: %.7f, XEnt Loss: %.4f, "
                   %(epoch, num_epochs-1, batch_idx, iter_per_epoch, L.item(),
                     KL.item(), XEnt.item()))

        reconst_images, _, _ = vae(fixed_x)
        torchvision.utils.save_image(reconst_images.data.cpu(),
            '../results/MNIST/reconst_images_%d.png' %(epoch))

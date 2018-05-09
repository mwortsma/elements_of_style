import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import modules.vae
import modules.mnist_xcoders

# Parameters
data_dir = '../data/MNIST'
z_sz = 10
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

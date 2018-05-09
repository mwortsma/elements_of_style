import argparse

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

parser = argparse.ArgumentParser()
parser.add_argument('-load', help='path of model to load')
args = parser.parse_args()
print(args)

# Parameters
data_dir = '../data/MNIST'
z_sz = 10
learning_rate = 0.01
batch_sz = 100
im_sz = 28*28
num_epochs = 50

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



enc = modules.mnist_xcoders.FCEncoder(z_sz)
dec = modules.mnist_xcoders.FCDecoder(z_sz)
vae = modules.vae.VAE(enc,dec,z_sz)

optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
iter_per_epoch = len(data_loader)

# For debugging
data_iter = iter(data_loader)
fixed_x, _ = next(data_iter)
torchvision.utils.save_image(fixed_x.cpu(), 'results/MNIST/real_images.png')
fixed_x = fixed_x.view(fixed_x.size(0), im_sz)
fixed_x = Variable(fixed_x)

## TRAIN
if args.load == None:

    L_vec = []
    XEnt_vec = []
    KL_vec = []

    for epoch in range(num_epochs):
        for batch_idx, (images, _) in enumerate(data_loader):
            images = Variable(images.view(images.size(0), im_sz))
            out, mu, logvar = vae(images)
            KL, XEnt = vae.loss(images, out, mu, logvar)
            L = KL + XEnt
            optimizer.zero_grad()
            L.backward()
            optimizer.step()

            L_vec.append(L.item())
            KL_vec.append(KL.item())
            XEnt_vec.append(XEnt.item())

            if batch_idx % 100 == 0:
                print ("Epoch[%d/%d], Step [%d/%d], Total Loss: %.4f, "
                       "KL Loss: %.7f, XEnt Loss: %.4f, "
                       %(epoch, num_epochs-1, batch_idx, iter_per_epoch, L.item(),
                         KL.item(), XEnt.item()))

            reconst_images, _, _ = vae(fixed_x)
            reconst_images = reconst_images.view(reconst_images.size(0), 1, 28, 28)
            torchvision.utils.save_image(reconst_images.data.cpu(),
                'results/MNIST/reconst_images_%d.png' %(epoch))

    plt.plot(L_vec, label="Total Loss")
    plt.plot(XEnt_vec, label="XEnt Loss")
    plt.plot(KL_vec, label="KL Divergence")
    plt.legend(loc=2)
    #plt.show()

    torch.save(vae.state_dict(), "trained_models/mnist_fc_vae.model")

else:
    print(args.load)
    vae.load_state_dict(torch.load(args.load))

    reconst_images, _, _ = vae(fixed_x)
    reconst_images = reconst_images.view(reconst_images.size(0), 1, 28, 28)
    torchvision.utils.save_image(reconst_images.data.cpu(),
        'results/MNIST/reconst_images.png')

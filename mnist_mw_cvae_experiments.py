import argparse
import os
import math

import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.cm as cm
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)


import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import modules.vae
import modules.mnist_xcoders

#from sklearn import decompositioni

parser = argparse.ArgumentParser()
parser.add_argument('-load', help='path of model to load')
parser.add_argument('-save', help='path of model to save')
parser.add_argument('-res', help='path to save figures')
parser.add_argument("-batch_sz", type=int,
                    help="how many in batch", default=104)
parser.add_argument("-z_sz", type=int,
                    help="latent size", default=10)
parser.add_argument("-epochs", type=int,
                    help="how many epochs", default=10)
args = parser.parse_args()
print(args)



# Parameters
data_dir = 'data/MNIST'
z_sz = args.z_sz
learning_rate = 0.01
batch_sz = args.batch_sz
im_sz = 28*28
num_epochs = args.epochs
c_sz=10

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
dec = modules.mnist_xcoders.FCDecoder(z_sz,c_sz)
vae = modules.vae.VAE(enc,dec,z_sz)

optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
iter_per_epoch = len(data_loader)

# For debugging
data_iter = iter(data_loader)
fixed_x_save, fixed_y_save = next(data_iter)
fixed_x = fixed_x_save.view(fixed_x_save.size(0), im_sz)
fixed_x = Variable(fixed_x)

fixed_y = torch.zeros((batch_sz, c_sz))
fixed_y[np.arange(batch_sz), fixed_y_save] = 1
fixed_y = Variable(fixed_y)

print(fixed_y.size())
print(fixed_x.size())




# Helpers
def icdf(v):
    return torch.erfinv(2 * torch.Tensor([float(v)]) - 1) * math.sqrt(2)

## TRAIN
if args.load == None:

    torchvision.utils.save_image(fixed_x_save.cpu(),
        os.path.join(args.res, 'real_images.png'))

    L_vec = []
    XEnt_vec = []
    KL_vec = []

    for epoch in range(num_epochs):
        for batch_idx, (images, labels) in enumerate(data_loader):
            images = Variable(images.view(images.size(0), im_sz))
            y = torch.zeros((labels.size(0), c_sz))
            y[np.arange(labels.size(0)), labels] = 1
            labels = Variable(y).float()
            out, mu, logvar = vae(images, y)
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

                plt.clf()
                plt.plot(L_vec, label="Total Loss")
                plt.plot(XEnt_vec, label="XEnt Loss")
                plt.plot(KL_vec, label="KL Divergence")
                plt.legend(loc=2)
                plt.savefig(os.path.join(args.res, 'loss.png'))


            reconst_images, _, _ = vae(fixed_x, fixed_y)
            reconst_images = reconst_images.view(reconst_images.size(0), 1, 28, 28)
            torchvision.utils.save_image(reconst_images.data.cpu(),
                os.path.join(args.res, 'reconst_images_%d.png' %(epoch)))



            torch.save(vae.state_dict(), args.save)
            plt.savefig(os.path.join(args.res, 'loss.png'))

else:
    torchvision.utils.save_image(fixed_x_save.cpu(),
        os.path.join(args.res, 'real_images.png'))

    vae.load_state_dict(torch.load(args.load))

    # Save reconstructed image
    reconst_images, _, _ = vae(fixed_x, fixed_y)
    reconst_images = reconst_images.view(reconst_images.size(0), 1, 28, 28)
    torchvision.utils.save_image(reconst_images.data.cpu(),
        os.path.join(args.res, 'reconst_images.png'))

    x = torch.zeros(11,784)
    for i in range(0,11):
        x[i,:] = x[0,:]
    l = [fixed_y_save[0].item()]
    for i in range(10):
        l.append(i)

    b = np.zeros((11,10))
    l = np.array(l)
    b[np.arange(11),l] = 1.0
    labels = torch.from_numpy(b)
    labels = Variable(labels).float()

    _, mu, logvar = vae(x,labels)
    z = vae.reperam(mu,logvar)
    out = vae.sample(z, labels)
    out[0,:] = fixed_x[0,:]
    out = out.view(out.size(0), 1, 28, 28)

    for k in range(11):
        print(k)
        x = torch.zeros(11,784)
        for i in range(0,11):
            x[i,:] = fixed_x[k,:]
        l = [fixed_y_save[k].item()]
        for i in range(10):
            l.append(i)
        b = np.zeros((11,10))
        b[np.arange(11),l] = 1.0
        labels = torch.from_numpy(b)
        labels = Variable(labels).float()

        _, mu, logvar = vae(x,labels)
        z = vae.reperam(mu,logvar)
        out1 = vae.sample(z, labels)
        out1[0,:] = fixed_x[k,:]
        out1 = out1.view(out1.size(0), 1, 28, 28)
        out = torch.cat((out, out1), dim=0)
        if k == 0:
            out = out[11:]


    torchvision.utils.save_image(out.data.cpu(),
        os.path.join(args.res, 'cvae.png'), nrow=11)

    # Visualize the manifold v1
    if z_sz == 2:
        for k in range(10):
            num_images=144
            num_rows=12
            num_cols=12
            label = np.zeros((num_images, 10))
            for i in range(num_images):
                label[i,k] = 1
            label = Variable(torch.from_numpy(label).float())
            manifold_z = np.zeros((num_images, 2))
            for j in range(num_images):
                row = num_rows - j/num_cols
                col = j%num_cols
                row_z = -4 + 8*float(row/num_rows)
                col_z = -4 + 8*float(col/num_cols)
                # print(row,col, ":", row_z,col_z)
                manifold_z[j] = np.array([col_z, row_z ])
            manifold_z = Variable(torch.from_numpy(manifold_z).float())
            manifold_images = vae.sample(manifold_z,label)
            manifold_images = manifold_images.view(manifold_images.size(0), 1, 28, 28)
            torchvision.utils.save_image(manifold_images.data.cpu(),
                os.path.join(args.res, 'manifold%d.png' % k),nrow=int(num_cols))

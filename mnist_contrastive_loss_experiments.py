import argparse
import os
import math

import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
from matplotlib.mlab import PCA

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import modules.vae
import modules.mnist_xcoders
import modules.contrastive_loss

from sklearn import decomposition

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
cont = modules.contrastive_loss.ContrastiveLoss()

optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
iter_per_epoch = len(data_loader)

# For debugging
data_iter = iter(data_loader)
fixed_x_save, _ = next(data_iter)
fixed_x = fixed_x_save.view(fixed_x_save.size(0), im_sz)
fixed_x = Variable(fixed_x)

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
            images1, images2 = torch.chunk(images,2,dim=0)
            labels1, labels2 = torch.chunk(labels,2,dim=0)
            out1, mu1, logvar1 = vae(images1)
            z1 = vae.reperam(mu1,logvar1)
            z1, _  = torch.chunk(z1, 2, dim=1)
            out2, mu2, logvar2 = vae(images2)
            z2 = vae.reperam(mu2,logvar2)
            z2, _  = torch.chunk(z2, 2, dim=1)
            eqlab = Variable(1-labels1.eq(labels2).float())
            CLoss = cont(z1,z2,eqlab)
            KL1, XEnt1 = vae.loss(images1, out1, mu1, logvar1)
            KL2, XEnt2 = vae.loss(images2, out2, mu2, logvar2)
            KL = KL1 + KL2
            XEnt = XEnt1 + XEnt2
            L = KL1 + XEnt1 + KL2 + XEnt2 + CLoss
            optimizer.zero_grad()
            L.backward()
            optimizer.step()

            L_vec.append(L.item())
            KL_vec.append(KL.item())
            XEnt_vec.append(XEnt.item())

            if batch_idx % 100 == 0:
                print ("Epoch[%d/%d], Step [%d/%d], Total Loss: %.4f, "
                       "KL Loss: %.7f, XEnt Loss: %.4f, Contrastive Loss: %.4f"
                       %(epoch, num_epochs-1, batch_idx, iter_per_epoch, L.item(),
                         KL.item(), XEnt.item(), CLoss.item()))

            reconst_images, _, _ = vae(fixed_x)
            reconst_images = reconst_images.view(reconst_images.size(0), 1, 28, 28)
            torchvision.utils.save_image(reconst_images.data.cpu(),
                os.path.join(args.res, 'reconst_images_%d.png' %(epoch)))

    plt.plot(L_vec, label="Total Loss")
    plt.plot(XEnt_vec, label="XEnt Loss")
    plt.plot(KL_vec, label="KL Divergence")
    plt.legend(loc=2)
    plt.savefig(os.path.join(args.res, 'loss.png'))

    torch.save(vae.state_dict(), args.save)

else:
    torchvision.utils.save_image(fixed_x_save.cpu(),
        os.path.join(args.res, 'real_images.png'))

    vae.load_state_dict(torch.load(args.load))

    # Save reconstructed image
    reconst_images, _, _ = vae(fixed_x)
    reconst_images = reconst_images.view(reconst_images.size(0), 1, 28, 28)
    torchvision.utils.save_image(reconst_images.data.cpu(),
        os.path.join(args.res, 'reconst_images.png'))


    # Sample z from normal and view the results
    normal_z = np.random.normal(0,1,(batch_sz,z_sz))
    normal_z = Variable(torch.from_numpy(normal_z).float())
    sample_images = vae.sample(normal_z)
    sample_images = sample_images.view(sample_images.size(0), 1, 28, 28)
    torchvision.utils.save_image(sample_images.data.cpu(),
        os.path.join(args.res, 'sample_images.png'))

    if z_sz == 2:
        # Visualize the manifold, first using the icdf
        num_rows=14.0
        num_cols=9.0
        manifold_z = np.zeros((batch_sz, 2))
        for j in range(0,batch_sz):
            row = j/8
            col = j%8
            row_z = icdf((1.0/num_rows) + (1.0/num_rows)*row)
            col_z = icdf((1.0/num_cols) + (1.0/num_cols)*col)
            # print(row,col, ":", row_z,col_z)
            manifold_z[j] = np.array([row_z, col_z])
        manifold_z = Variable(torch.from_numpy(manifold_z).float())
        manifold_images = vae.sample(manifold_z)
        manifold_images = manifold_images.view(manifold_images.size(0), 1, 28, 28)
        torchvision.utils.save_image(manifold_images.data.cpu(),
            os.path.join(args.res, 'manifoldv0.png'))

        # Visualize the manifold v1
        num_images=144
        num_rows=12
        num_cols=12
        manifold_z = np.zeros((num_images, 2))
        for j in range(num_images):
            row = num_rows - j/num_cols
            col = j%num_cols
            row_z = -4 + 8*float(row/num_rows)
            col_z = -4 + 8*float(col/num_cols)
            # print(row,col, ":", row_z,col_z)
            manifold_z[j] = np.array([col_z, row_z ])
        manifold_z = Variable(torch.from_numpy(manifold_z).float())
        manifold_images = vae.sample(manifold_z)
        manifold_images = manifold_images.view(manifold_images.size(0), 1, 28, 28)
        torchvision.utils.save_image(manifold_images.data.cpu(),
            os.path.join(args.res, 'manifoldv1.png'),nrow=int(num_cols))




    # Visualize the embedding space part 2
    fig, ax = plt.subplots()
    ax.set_ylim([-4,4])
    ax.set_xlim([-4,4])
    bottom_left = [0.081, 0.081]
    top_right = [0.865, 0.845]
    colors = cm.rainbow(np.linspace(0, 1, 10))

    num_images=1000
    data_loader1 = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=num_images,
                                              shuffle=True)
    data_iter1 = iter(data_loader1)
    data, label = next(data_iter1)
    data = Variable(data.view(data.size(0), im_sz))
    out, mu, logvar = vae(data)
    z = vae.reperam(mu, logvar)
    z = z.detach().numpy()
    pca = decomposition.PCA(n_components=2)
    pca.fit(z)
    if z_sz > 2:
        z = pca.transform(z)
    for i in range(num_images):
        single_image = data[i].view(1,1,28,28)
        torchvision.utils.save_image(single_image.data.cpu(),
            os.path.join(args.res, 'single_image.png'))
        ax.scatter(z[i,0], z[i,1], color=colors[label[i].item()])

        if np.random.random() < 0.00:
            x_scale = 1-(4-z[i,0])/8.0
            y_scale = 1-(4-z[i,1])/8.0
            if x_scale < 0 or x_scale > 1 or y_scale < 0 or y_scale > 1: continue
            im = plt.imread(os.path.join(args.res,"single_image.png"), format='png')
            newax = fig.add_axes([bottom_left[0] + x_scale*(top_right[0]-bottom_left[0]), bottom_left[1] + y_scale*(top_right[1]-bottom_left[1]), .075, .075])
            newax.imshow(im)
            newax.axis('off')

    plt.savefig(os.path.join(args.res, 'manifoldv2.png'))

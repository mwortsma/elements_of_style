import argparse
import os

import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import modules.ss_vae as ss_vae
import visualize as vis

# Parameters
data_dir = 'data/MNIST'
learning_rate = 0.01
im_sz = 28*28

num_classes = 10

# Helpers
def icdf(v):
    return torch.erfinv(2 * torch.Tensor([float(v)]) - 1) * np.sqrt(2)

def horzArr(a):
    ''' finagles a (num_images)x1x(a)x(b) array into a (a)x(b*num_images) array '''
    a = np.squeeze(a) # get rid of extra 1d
    a = np.split(a,np.size(a,0)) # separate images
    a= np.concatenate(a,2) # put together horizontally
    a = np.squeeze(a) # get rid of extra 1d
    return a

def train(vae, data_loader, fixed_x, fixed_y):
    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
    iter_per_epoch = len(data_loader)

    L_vec = [] # store losses for plotting later
    for epoch in range(args.epochs):
        for batch_idx, (images, labels) in enumerate(data_loader):
            images = Variable(images.view(args.batch_sz, im_sz).to(DEVICE))
            labels = Variable(labels.to(DEVICE))
            out, z_params, pi = vae(images) # forward pass
            L = vae.loss(images, labels, out, z_params, pi) # compute loss
            optimizer.zero_grad()
            L.backward() # backward pass
            optimizer.step() # parameter update

            L_vec.append(L.item())
            if batch_idx % 100 == 0:
                print("Epoch[%d/%d], Step [%d/%d], Total Loss: %.4f " %(epoch, args.epochs-1, batch_idx, iter_per_epoch, L.item()))

            # visualise progress:
            reconst_images, _, _ = vae(fixed_x) # another forward pass on fixed inputs
            reconst_images = reconst_images.view(reconst_images.size(0), 1, 28, 28) # reshape
            torchvision.utils.save_image(reconst_images.data.cpu(), os.path.join(args.res, 'reconst_images_%d.png' %(epoch)))

            torch.save(vae.state_dict(), args.save) # save model to disk

    plt.plot(L_vec)
    plt.savefig(os.path.join(args.res, 'loss.png'))
    plt.show()

## TRAIN
def main():
    z_sz = args.z_sz

    # Load Data
    dataset = torchvision.datasets.MNIST(root=data_dir,
                                         train=True,
                                         transform=transforms.ToTensor(),
                                         download=True)

    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=args.batch_sz,
                                              shuffle=True)

    # For debugging
    data_iter = iter(data_loader)
    fixed_x_save, fixed_y_save = next(data_iter) # batch images and labels
    fixed_x = fixed_x_save.view(fixed_x_save.size(0), im_sz)
    fixed_x = Variable(fixed_x.to(DEVICE))

    fixed_y = Variable(fixed_y_save.to(DEVICE))
    torchvision.utils.save_image(fixed_x_save, os.path.join(args.res, 'real_images.png'))

    vae = ss_vae.SS_VAE(device=DEVICE, z_sz=z_sz)
    if args.load is None:
        train(vae, data_loader, fixed_x, fixed_y)
    else:
        vae.load_state_dict(torch.load(args.load, map_location=lambda storage, loc: storage))

        # Save reconstructed image
        reconst_images, _, _ = vae(fixed_x)
        reconst_images = reconst_images.view(-1, 1, 28, 28)
        torchvision.utils.save_image(reconst_images.data.cpu(), os.path.join(args.res, 'reconst_images.png'))

        # we want to visualise what happens when we take a batch example,
        #  which has label y, and sample from SS-VAE with label y' \in [0,9]

        # l is a vector of [y, 0, 1, 2, ..., 9]
        l = np.arange(-1,num_classes)
        l[0] = fixed_y_save[0].item()

        b = np.eye(num_classes)[l] # one-hot vectors 11 x 10 from l
        labels = torch.from_numpy(b).float().to(DEVICE)
        labels = Variable(labels) # XXX: do we need to wrap a tensor in a variable at inference?

        res = torch.zeros(num_classes+1, num_classes+1, im_sz)

        for k in range(num_classes+1):
            x = fixed_x[k].view(1, -1) # take a batch example
            x = x.expand(num_classes+1, -1) # duplicate along rows
            z_params, pi = vae.encoder(x)
            z = vae.reparam_z(z_params) # sample a latent z for x
            out = vae.sample(z, labels) # fix z, and vary labels
            out[0,:] = fixed_x[k,:]
            res[k] = out

        res = res.view(-1, 1, 28, 28)

        torchvision.utils.save_image(res.data.cpu(), os.path.join(args.res, 'cvae.png'), nrow=11)

        if args.walk:
            # set up visualizer-- see how changing z alters changes images over all labels y
            f = lambda z:horzArr(vae.sample(
                    # to get z: convert to torch, replicate 11 times
                    Variable(torch.from_numpy(z).float().to(DEVICE)).view(1,-1).expand(num_classes+1,-1),
                    # format to be 11 28x28 images horizontally arranged
                    labels).view(-1,1,28,28).data.numpy())
            v = vis.Visualizer(f,z_sz=z_sz)
            v.visualize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-load', help='path of model to load')
    parser.add_argument('-walk', action='store_true', help='displays GUI to walk embedding space')
    parser.add_argument('-save', help='path of model to save')
    parser.add_argument('-res', help='path to save figures')
    parser.add_argument("-batch_sz", type=int,
                        help="how many in batch", default=100)
    parser.add_argument("-z_sz", type=int,
                        help="latent size", default=50)
    parser.add_argument("-epochs", type=int,
                        help="how many epochs", default=10)
    parser.add_argument("-device", help="specify device to run on", default="cpu")
    args = parser.parse_args()
    DEVICE = torch.device('cpu')
    if args.device == 'cuda' and torch.cuda.is_available():
        DEVICE = torch.device('cuda')

    main()

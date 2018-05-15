import torch
import modules.conv_vae
import torchvision
from torch.autograd import Variable
import modules.mse_loss_vae
import cv2
import numpy as np
import argparse

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)


vae = modules.conv_vae.VAE()
mse_loss = modules.mse_loss_vae.Loss()

num_epochs = 100
data_dir = './datasets/facades/'
batch_size = 32
learning_rate = 0.001

# DEVICE = torch.device('cpu')
# if torch.cuda.is_available():
#     DEVICE = torch.device('cuda')

parser = argparse.ArgumentParser()
parser.add_argument('-load', help='path of model to load')
parser.add_argument('-save', help='path of model to save')
parser.add_argument('-res', help='path to save figures')
parser.add_argument("-epochs", type=int,
                    help="how many epochs", default=num_epochs)
args = parser.parse_args()
print(args)


data_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(size=(32, 64)), torchvision.transforms.ToTensor()])
input_data = torchvision.datasets.ImageFolder(root = data_dir, transform = data_transform)
data_loader = torch.utils.data.DataLoader(input_data, batch_size = batch_size, shuffle = True)

optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)


L_vec = []
MSE_vec = []
KL_vec = []

for epoch in range(num_epochs):
    for batch_idx, (images, l) in enumerate(data_loader):
        source, target = torch.chunk(images, 2, 3)
        source = Variable(source)
        target = Variable(target)

        out, mu, logvar = vae(source)

        # get loss
        MSE, KLD = mse_loss(out, source, mu, logvar)
        L = MSE + KLD
        optimizer.zero_grad()
        L.backward()
        optimizer.step()

        L_vec.append(L)
        KL_vec.append(KLD)
        MSE_vec.append(MSE)

        if batch_idx % 100 == 0:
            print ("Epoch[%d], Total Loss: %.4f, "
                   %(epoch, L))
            # print(L)
            targ = out[0].data.numpy()
            targ = np.swapaxes(targ,0,2)
            cv2.imshow('frame', targ)
            cv2.waitKey(1)

# print(L_vec)
# print(len(L_vec))
# print(MSE_vec)
# plt.plot(L_vec, label="Total Loss")
# plt.plot(MSE_vec, label="MSE Loss")
# plt.plot(KL_vec, label="KL Divergence")
# plt.legend(loc=2)
# plt.savefig(os.path.join(args.res, 'loss.png'))

torch.save(vae.state_dict(), args.save)

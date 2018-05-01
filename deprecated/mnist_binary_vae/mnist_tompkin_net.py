import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
import torchvision

import matplotlib.pyplot as plt
import numpy as np
import math
import operator

# HyperParameters
learning_rate = 0.001
num_epochs = 4
im_sz = 784 # size of the image
z_sz = 2 # z = [z_style, z_content]
enc_fc1_sz = 400
dec_fc1_sz = 400
batch_sz = 100

# MNIST dataset
dataset = datasets.MNIST(root='./data',
                         train=True,
                         transform=transforms.ToTensor(),
                         download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_sz,
                                          shuffle=True)

# Convert to variable which works if GPU
def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

# Construct the VAE
class TompkinNet(nn.Module):
    def __init__(self):
        super(TompkinNet, self).__init__()

        # Encoder Layers
        self.enc_fc1 = nn.Linear(im_sz, enc_fc1_sz)
        self.enc_fc2 = nn.Linear(enc_fc1_sz, int(2*z_sz))

        # Decoder Layers
        self.dec_fc1 = nn.Linear(z_sz, dec_fc1_sz)
        self.dec_fc2 = nn.Linear(dec_fc1_sz, im_sz)

        # Link Layers
        self.link_fc1 = nn.Linear(z_sz,2)

    # FC -> Leaky Relu -> Fc
    def encode(self,x):
        return self.enc_fc2(F.leaky_relu(self.enc_fc1(x),0.2))

    # FC -> Relu -> FC -> Sigmoid
    def decode(self,x):
        return F.sigmoid(self.dec_fc2(F.relu(self.dec_fc1(x))))

    # FC -> Sigmoid
    def linknet(self, x):
        return F.sigmoid(self.link_fc1(x))

    # Reperam Trick (See paper)
    def reperam(self,mu,logvar):
        eps = to_var(torch.randn(mu.size(0), mu.size(1)))
        z = mu + eps*torch.exp(logvar/2)
        return z

    # Vae Part of Net (See Paper)
    def forward_vae(self, x):
        h = self.encode(x)
        mu, logvar = torch.chunk(h,2,dim=1)
        z = self.reperam(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar, z

    # Forward pass of net for images x1, x2
    def forward(self, x1, x2):
        out1, mu1, logvar1, z1  = self.forward_vae(x1)
        out2, mu2, logvar2, z2  = self.forward_vae(x2)
        z1_style, _ = torch.chunk(z1, 2, dim=1)
        z2_style, _ = torch.chunk(z2, 2, dim=1)
        style = torch.cat((z1_style, z2_style), -1)
        linkout = self.linknet(style)
        return out1, mu1, logvar1, out2, mu2, logvar2, linkout

    # Given a z, get an x
    def sample(self,z):
        return self.decode(z)


def vae_loss(x, out, mu, logvar):
    KL_divergence = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KL_divergence /= (batch_sz*im_sz)
    cross_entropy = F.binary_cross_entropy(out, x.view(-1, im_sz))
    return KL_divergence, cross_entropy

def link_loss(linkout, eq_labels):
    return F.binary_cross_entropy(linkout, eq_labels)

tnet = TompkinNet()
print(tnet)

optimizer = torch.optim.Adam(tnet.parameters(), lr=learning_rate)
iter_per_epoch = len(data_loader)

# For Plotting
KL_ = []
XEnt_ = []
Linkloss_ = []
L_ = []


for epoch in range(num_epochs):
    for batch_idx, (data, labels) in enumerate(data_loader):
        images = to_var(data.view(data.size(0), -1))
        images1, images2 = torch.chunk(images, 2, dim=0)
        labels1, labels2 = torch.chunk(labels, 2, dim=0)

        eq_labels_vec = labels1.eq(labels2).float().view(50,1)
        eq_labels_onehot = torch.cat((1-eq_labels_vec, eq_labels_vec), dim=1)
        eq_labels = Variable(eq_labels_onehot)

        out1, mu1, logvar1, out2, mu2, logvar2, linkout = tnet(images1, images2)

        # Calculate Loss
        KL1, XEnt1 = vae_loss(images1, out1, mu1, logvar1)
        KL2, XEnt2 = vae_loss(images2, out2, mu2, logvar2)
        linkloss = link_loss(linkout, eq_labels)

        KL = KL1 + KL2
        XEnt = XEnt1 + XEnt2
        L = KL + XEnt + linkloss

        # Gradient Stuff
        optimizer.zero_grad()
        L.backward()
        optimizer.step()

        KL_.append(KL.item())
        XEnt_.append(XEnt.item())
        Linkloss_.append(linkloss.item())
        L_.append(L.item())

        if batch_idx % 100 == 0:
            print ("Epoch[%d/%d], Step [%d/%d], Total Loss: %.4f, "
                   "KL Loss: %.7f, XEnt Loss: %.4f, Link Loss: %.4f"
                   %(epoch+1, num_epochs, batch_idx+1, iter_per_epoch, L.item(),
                     KL.item(), XEnt.item(), linkloss.item()))


plt.plot(KL_, label="KL Loss")
plt.plot(XEnt_, label="XEnt Loss")
plt.plot(Linkloss_, label="Linked Loss")
plt.plot(L_, label="Total Loss")

plt.legend()

plt.show()

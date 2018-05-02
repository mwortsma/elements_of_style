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
im_side = 32
im_sz = im_side*im_side # size of the image
z_sz = 20 # z = [z_style, z_content]
dec_fc1_sz = 400
batch_sz = 100

# MNIST dataset
dataset = datasets.MNIST(root='./data',
                         train=True,
                         transform=transforms.Compose([torchvision.transforms.Pad(2),transforms.ToTensor()]),
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
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2*z_sz)

        # Decoder Layers
        self.dec_fc1 = nn.Linear(z_sz, dec_fc1_sz)
        self.dec_fc2 = nn.Linear(dec_fc1_sz, im_sz)

        # Link Layers
        self.link_fc1 = nn.Linear(z_sz,32)
        self.link_fc2 = nn.Linear(32,2)

    # LeNet
    def encode(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    # FC -> Relu -> FC -> Sigmoid
    def decode(self,x):
        return F.sigmoid(self.dec_fc2(F.relu(self.dec_fc1(x))))

    # FC -> Sigmoid
    def linknet(self, x):
        return F.sigmoid(self.link_fc2(F.relu(self.link_fc1(x))))

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
data_iter = iter(data_loader)

# fixed inputs for debugging
fixed_z = to_var(torch.randn(100, z_sz))
fixed_x, _ = next(data_iter)
torchvision.utils.save_image(fixed_x.cpu(), './data/real_images.png')
fixed_x = to_var(fixed_x)

# For Plotting
KL_ = []
XEnt_ = []
Linkloss_ = []
Linkaccuracy_ = []
L_ = []


for epoch in range(num_epochs):
    for batch_idx, (data, labels) in enumerate(data_loader):
        images = to_var(data)
        images1, images2 = torch.chunk(images, 2, dim=0)
        labels1, labels2 = torch.chunk(labels, 2, dim=0)

        eq_labels_vec = labels1.eq(labels2).float().view(50,1)
        eq_labels_onehot = torch.cat((1-eq_labels_vec, eq_labels_vec), dim=1)
        eq_labels = to_var(eq_labels_onehot)

        out1, mu1, logvar1, out2, mu2, logvar2, linkout = tnet(images1, images2)

        # Calculate Loss
        KL1, XEnt1 = vae_loss(images1, out1, mu1, logvar1)
        KL2, XEnt2 = vae_loss(images2, out2, mu2, logvar2)
        linkloss = link_loss(linkout, eq_labels)
        linkaccuracy = linkout.max(dim=1)[1].view((batch_sz/2, 1))
        linkaccuracy = linkaccuracy.eq(eq_labels_vec.long()).sum()

        # ERROR THIS GOES TO 0
        print(linkout.max(dim=1)[1].view((batch_sz/2, 1)).sum())
        #print(linkout.max(dim=1)[1].view((batch_sz/2, 1)))

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
        Linkaccuracy_.append(linkaccuracy)
        L_.append(L.item())

        if batch_idx % 100 == 0:
            print ("Epoch[%d/%d], Step [%d/%d], Total Loss: %.4f, "
                   "KL Loss: %.7f, XEnt Loss: %.4f, Link Loss: %.4f, "
                   "Link Accuracy %.4f"
                   %(epoch+1, num_epochs, batch_idx+1, iter_per_epoch, L.item(),
                     KL.item(), XEnt.item(), linkloss.item(), linkaccuracy.item()/(batch_sz/2.0)))


    # Save the reconstructed images
    reconst_images, _, _, _ = tnet.forward_vae(fixed_x)
    reconst_images = reconst_images.view(reconst_images.size(0), 1, im_side, im_side)
    torchvision.utils.save_image(reconst_images.data.cpu(),
        './data/reconst_images_%d.png' %(epoch+1))

    reconst_images = tnet.sample(fixed_z)
    reconst_images = reconst_images.view(reconst_images.size(0), 1, im_side, im_side)
    torchvision.utils.save_image(reconst_images.data.cpu(),
        './data/sample_%d.png' %(epoch+1))


plt.plot(KL_, label="KL Loss")
plt.plot(XEnt_, label="XEnt Loss")
plt.plot(Linkloss_, label="Linked Loss")
plt.plot(Linkaccuracy_, label="Linked Accuracy")
plt.plot(L_, label="Total Loss")

plt.legend()

plt.show()

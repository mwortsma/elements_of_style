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

# HyperParameters
learning_rate = 0.001
num_epochs = 100
im_sz = 784 # size of the image
z_sz = 4 # note that z = [z_mu, z_logvar]
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
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder Layers
        self.enc_fc1 = nn.Linear(im_sz, enc_fc1_sz)
        self.enc_fc2 = nn.Linear(enc_fc1_sz, z_sz)

        # Decoder Layers
        self.dec_fc1 = nn.Linear(int(z_sz/2), dec_fc1_sz)
        self.dec_fc2 = nn.Linear(dec_fc1_sz, im_sz)

    def encode(self,x):
        return self.enc_fc2(F.leaky_relu(self.enc_fc1(x),0.2))

    def decode(self,x):
        return F.sigmoid(self.dec_fc2(F.relu(self.dec_fc1(x))))

    def reperam(self,mu,logvar):
        eps = to_var(torch.randn(mu.size(0), mu.size(1)))
        z = mu + eps*torch.exp(logvar/2)
        return z

    def forward(self, x):
        h = self.encode(x)
        mu, logvar = torch.chunk(h,2,dim=1)
        z = self.reperam(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar

    def sample(self,z):
        return self.decode(z)

def loss(x, out, mu, logvar):
    KL_divergence = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    cross_entropy = F.binary_cross_entropy(out, x.view(-1, im_sz), size_average=False)
    return KL_divergence + cross_entropy, KL_divergence, cross_entropy

vae = VAE()
print(vae)

optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
iter_per_epoch = len(data_loader)

# For debugging
data_iter = iter(data_loader)
fixed_x, _ = next(data_iter)
torchvision.utils.save_image(fixed_x.cpu(), './data/real_images.png')
fixed_x = to_var(fixed_x.view(fixed_x.size(0), -1))

# For Plotting
KL_ = []
XEnt_ = []
L_ = []


'''
def icdf(v):
    return torch.erfinv(2 * torch.Tensor([float(v)]) - 1) * math.sqrt(2)

z_2d = np.zeros((104, 2))
for j in range(0,104):
    row = j/8
    col = j%8
    row_z = icdf((1.0/14.0) + (1.0/14.0)*row)
    col_z = icdf((1.0/9.0) + (1.0/9.0)*col)
    z_2d[j] = np.array([row_z, col_z])
z_2d_tensor = to_var(torch.from_numpy(z_2d).float())
'''


np_sample_z = np.random.normal(0,1,(batch_sz, int(z_sz/2)))
sample_z = to_var(torch.from_numpy(np_sample_z).float())
reconst_images = vae.sample(sample_z)
reconst_images = reconst_images.view(reconst_images.size(0), 1, 28, 28)
torchvision.utils.save_image(reconst_images.data.cpu(),
    './data/init_sample.png')


for epoch in range(num_epochs):
    for batch_idx, (data, _) in enumerate(data_loader):
        images = to_var(data.view(data.size(0), -1))
        out, mu, logvar = vae(images)
        L, KL, XEnt = loss(images, out, mu, logvar)
        optimizer.zero_grad()
        L.backward()
        optimizer.step()

        KL_.append(KL.item())
        XEnt_.append(XEnt.item())
        L_.append(L.item())

        np_sample_z = np.random.normal(0,1,(batch_sz, int(z_sz/2)))
        sample_z = to_var(torch.from_numpy(np_sample_z).float())

        if batch_idx % 100 == 0:
            print ("Epoch[%d/%d], Step [%d/%d], Total Loss: %.4f, "
                   "KL Loss: %.7f, XEnt Loss: %.4f, "
                   %(epoch+1, num_epochs, batch_idx+1, iter_per_epoch, L.item(),
                     KL.item(), XEnt.item()))

    reconst_images, _, _ = vae(fixed_x)
    reconst_images = reconst_images.view(reconst_images.size(0), 1, 28, 28)
    torchvision.utils.save_image(reconst_images.data.cpu(),
        './data/reconst_images_%d.png' %(epoch+1))


    '''
    np_sample_z = np.zeros((batch_sz, int(z_sz/2)))
    np_sample_z[0,:] = np.random.normal(0,1,int(z_sz/2))
    for j in range(1,batch_sz):
        np_sample_z[j,:] = np_sample_z[j-1,:] + (1/np.sqrt(batch_sz))*np.random.normal(0,1,int(z_sz/2))

    sample_z = to_var(torch.from_numpy(np_sample_z).float())

    reconst_images = vae.sample(sample_z)
    reconst_images = reconst_images.view(reconst_images.size(0), 1, 28, 28)
    torchvision.utils.save_image(reconst_images.data.cpu(),
        './data/sample_%d.png' %(epoch+1))

    np_sample_z = np.random.normal(0,1,(batch_sz, int(z_sz/2)))
    sample_z = to_var(torch.from_numpy(np_sample_z).float())

    reconst_images = vae.sample(sample_z)
    reconst_images = reconst_images.view(reconst_images.size(0), 1, 28, 28)
    torchvision.utils.save_image(reconst_images.data.cpu(),
        './data/rand_sample_%d.png' %(epoch+1))

    reconst_images = vae.sample(z_2d_tensor)
    reconst_images = reconst_images.view(reconst_images.size(0), 1, 28, 28)
    torchvision.utils.save_image(reconst_images.data.cpu(),
        './data/z2d_%d.png' %(epoch+1))
    '''

plt.plot(KL_)
plt.plot(XEnt_)
plt.plot(L_)
plt.show()

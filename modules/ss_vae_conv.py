"""

Network architecture, implementation reference:
http://bjlkeng.github.io/posts/semi-supervised-learning-with-variational-autoencoders/
"""

from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def xavier_init(net):
    for i in range(len(net)):
        layer = net[i]
        if type(layer) == nn.Module:
            nn.init.xavier_uniform(layer.weight)

def feature_size(cnn, input_size):
    bs = 1
    x = Variable(torch.rand(bs, *input_size))
    out = cnn(x)
    return out.view(bs, -1).size(1)

def LeNetFeatures():
    return nn.Sequential(OrderedDict([
        ('conv0', nn.Conv2d(1, 6, kernel_size=(5,5), padding=2)), # 1x28x28 -> 6x28x28
        ('nl0', nn.ReLU()),
        ('pool0', nn.MaxPool2d(2,2)), # 6x28x28 -> 6x14x14
        ('conv1', nn.Conv2d(6, 16, kernel_size=(5,5))), # 6x14x14 -> 16x10x10
        ('nl1', nn.ReLU()),
        ('pool1', nn.MaxPool2d(2,2))
    ])), 16*5*5 # final output is 16x5x5

def LeNetInverse():
    return nn.Sequential(OrderedDict([
        ('upsamp0', nn.Upsample(scale_factor=2, mode='bilinear')),
        ('dconv0', nn.ConvTranspose2d(16, 6, 5)),
        ('nl0', nn.ReLU()),
        ('upsamp1', nn.Upsample(scale_factor=2, mode='bilinear')),
        ('dconv1', nn.ConvTranspose2d(6, 1, 5, padding=2)),
        ('sigmoid', nn.Sigmoid())
    ]))

class ConvDecoder(nn.Module):
    def __init__(self, z_sz, y_sz, fsize=torch.Size([64, 14, 14]), device=torch.device("cpu")):
        """ fsize: torch.Size object - CxHxW - of the feature map to be "deconvolved"
            z_sz: dimensionality of latent z
            y_sz: dimensionality of latent y (number of classes)
            device: can pass torch.device("cuda")
        """
        super(ConvDecoder, self).__init__()
        num_flat_features = int(np.prod(fsize))
        self.mlp = nn.Sequential(OrderedDict([
            ('fc0', nn.Linear(z_sz+y_sz, 256)),
            ('relu0', nn.ReLU()),
            ('fc1', nn.Linear(256, num_flat_features))
        ]))
        self.deconvNet = nn.Sequential(OrderedDict([
            ('dconv1', nn.ConvTranspose2d(64, 64, 3, padding=1)),
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU()),
            ('dconv2', nn.ConvTranspose2d(64, 64, 3, padding=1)),
            ('norm2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU()),
            ('dconv3', nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)),
            ('norm3', nn.BatchNorm2d(64)),
            ('relu3', nn.ReLU()),
            ('dconv4', nn.ConvTranspose2d(64, 1, 1)),
            ('sigmoid', nn.Sigmoid())
        ]))

        xavier_init(self.deconvNet)

        self._fsize = fsize
        self.to(device)

    def forward(self, z, y):
        inp = torch.cat([z,y], 1)
        inp = self.mlp(inp)
#        c,h,w = self._fsize
        return self.deconvNet(inp.view(-1, *self._fsize))

class ConvEncoder(nn.Module):
    def __init__(self, img_sz, z_sz=20, device=torch.device("cpu")):
        """ img_sz: torch.Size object - CxHxW
            z_sz: dimensionality of latent z
            device: can pass torch.device("cuda")
        """
        super(ConvEncoder, self).__init__()
        self.fmap = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 64, 3, padding=1)), # -> 64x28x28
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(64, 64, 3, padding=1)), # -> 64x28x28
            ('norm2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU()),
            ('conv3', nn.Conv2d(64, 64, 3, stride=2, padding=15)), # -> 64x28x28
            ('norm3', nn.BatchNorm2d(64)),
            ('relu3', nn.ReLU())
        ]))
        fsize = feature_size(self.fmap, img_sz)
        self.MLP = nn.Sequential(OrderedDict([
            ('fc2', nn.Linear(fsize, 500)),
            ('nl2', nn.ReLU()),
            ('fc4', nn.Linear(500, z_sz*2)),
        ]))

        xavier_init(self.fmap)

        self._dev = device
        self.to(device)

    def forward(self, x):
        x = self.fmap(x)
        return self.MLP(x.view(x.size(0), -1))

class ConvClassifier(nn.Module):
    def __init__(self, img_sz, num_classes, device=torch.device("cpu")):
        """ img_sz: torch.Size object - CxHxW
            num_classes: number of output classes (dimensionality of latent y)
            device: can pass torch.device("cuda")
        """
        super(ConvClassifier, self).__init__()
        self.fmap = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv2d(1, 32, 3, padding=1)),
            ('relu1_1', nn.ReLU()),
            ('conv1_2', nn.Conv2d(32, 32, 3)), # -> 32x26x26
            ('relu1_2', nn.ReLU()),
            ('pool1', nn.MaxPool2d(2,2)), # -> 32x13x13
            ('drop1', nn.Dropout2d(0.25)),
            ('conv2_1', nn.Conv2d(32, 64, 3, padding=1)), # -> 64x13x13
            ('relu2_1', nn.ReLU()),
            ('conv2_2', nn.Conv2d(64, 64, 3)), # -> 64x11x11
            ('relu2_2', nn.ReLU()),
            ('pool2', nn.MaxPool2d(2, 2)), # -> 64x5x5
            ('drop2', nn.Dropout2d(0.25)),
        ]))
        fsize = feature_size(self.fmap, img_sz)
        self.MLP = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(fsize, 500)),
            ('nl1', nn.ReLU()),
            ('drop1', nn.Dropout2d(0.5)),
            ('fc4', nn.Linear(500, num_classes)),
            ('sm4', nn.Softmax(dim=1))
        ]))

        xavier_init(self.fmap)

        self.to(device)

    def forward(self, x):
        f = self.fmap(x)
        return self.MLP(f.view(f.size(0), -1))

class SS_VAE(nn.Module):
    """ Semi-Supervised VAE
        Kingma, et al. (2014). Semi-supervised Learning with Deep Generative Models. arXiv preprint arXiv:1406.5298v2
    """

    def __init__(self, batch_size=100, img_size=torch.Size([1, 28, 28]), num_classes=10, z_sz=20, device=torch.device("cpu")):
        super(SS_VAE, self).__init__()
        self.enc_z = ConvEncoder(img_sz=img_size, z_sz=z_sz, device=device) # q_phi(z|x) for now
        self.enc_y = ConvClassifier(img_sz=img_size, num_classes=num_classes, device=device)
        self.dec = ConvDecoder(z_sz=z_sz, y_sz=num_classes, device=device)

        self._dev = device
        self._alpha = 0.1 * batch_size # weighting term on classifier loss
        self._logpy = torch.tensor(np.log(1/num_classes), device=device)

        self.img_size = img_size
        self._N = int(np.prod(img_size)) # flattened image size

        self.to(device)

    def reparam_z(self, z_params):
        mu, logvar = torch.chunk(z_params,2,dim=1)
        eps = Variable(torch.randn(mu.size(0), mu.size(1), device=self._dev))
        z = mu + eps*torch.exp(logvar/2)
        return z

    def forward(self, x):
        z_params, pi = self.encoder(x)
        z = self.reparam_z(z_params)
        out = self.dec(z, pi).view(-1, self._N)

        # - out informs reconstruction loss term
        # - z_params inform KL divergence loss term
        # - pi inform cross-entropy loss (w/ logits) for true label y
        return out, z_params, pi

    def encoder(self, x):
        """ convenience function wrapping enc_z and enc_y steps """
        c,h,w = self.img_size
        x = x.view(-1, c, h, w)
        pi = self.enc_y(x)
        z_params = self.enc_z(x)
        return z_params, pi

    def sample(self, z, pi):
        return self.dec(z, pi).view(-1, self._N)

    def loss(self, x, y, out, z_params, pi, normalize=1, size_average=False):
        """ x: original image
            out: reconstructed image
            y: scalar class label
            z_params: mu,logvar (forward pass of encoder network)
            y_probs: forward pass of classifier network
        """
        mu, logvar = torch.chunk(z_params, 2, dim=1)
        KL = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KL /= normalize
        recon_XEnt = F.binary_cross_entropy(out, x, size_average=size_average)
        label_XEnt = self._alpha*F.cross_entropy(pi, y, size_average=size_average)
        return recon_XEnt + KL + label_XEnt - self._logpy


from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Encoder(nn.Module):
    def __init__(self, in_sz=784, h_sz=500, z_sz=50, device=torch.device("cpu")):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(in_sz, h_sz)
        self.fc2 = nn.Linear(h_sz, z_sz*2)

        self.to(device)

    def forward(self, x):
        x = F.softplus(self.fc1(x))
        return self.fc2(x)

class Decoder(nn.Module):
    def __init__(self, in_sz=50+10, out_sz=784, h_sz=500, device=torch.device("cpu")):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(in_sz, h_sz) # want to cat z and y together
        self.fc2 = nn.Linear(h_sz, out_sz)

        self.to(device)

    def forward(self, z, y):
        inp = torch.cat([z,y], 1)
        inp = F.softplus(self.fc1(inp))
        return F.sigmoid(self.fc2(inp))

class ConvEncoder(nn.Module):
    def __init__(self, z_sz=50, device=torch.device("cpu")):
        super(ConvEncoder, self).__init__()
        self.convNet = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, 32, kernel_size=(5,5))), # 1x28x28 -> 32x24x24
            ('nl0', nn.ReLU()),
            ('conv1', nn.Conv2d(32, 16, kernel_size=(5,5))), # 32x24x24 -> 16x20x20
            ('nl1', nn.ReLU()),
            ('conv2', nn.Conv2d(16, 16, kernel_size=(7,7))), # 16x20x20 -> 16x14x14
            ('nl2', nn.ReLU()),
            ('conv3', nn.Conv2d(16, 4, kernel_size=(7,7))), # 16x14x14 -> 4x8x8
        ]))
        self.MLP = nn.Sequential(OrderedDict([
            ('fc4', nn.Linear(4*8*8, 128)),
            ('nl4', nn.ReLU()),
            ('fc5', nn.Linear(128, z_sz*2))
        ]))

        for i in range(len(self.convNet)):
            layer = self.convNet[i]
            if type(layer) == nn.Module:
                nn.init.xavier_uniform(layer.weight)

        self._dev = device
        self.to(device)

    def forward(self, x):
        x = self.convNet(x)
        return self.MLP(x.view(x.size(0), -1))

class ConvClassifier(nn.Module):
    def __init__(self, num_classes=10, device=torch.device("cpu")):
        # let's try LeNet here...
        super(ConvClassifier, self).__init__()
        self.convNet = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, 6, kernel_size=(5,5), padding=2)), # 1x28x28 -> 6x28x28
            ('nl0', nn.ReLU()),
            ('pool0', nn.MaxPool2d(2,2)), # 6x28x28 -> 6x14x14
            ('conv1', nn.Conv2d(6, 16, kernel_size=(5,5))), # 6x14x14 -> 16x10x10
            ('nl1', nn.ReLU()),
            ('pool1', nn.MaxPool2d(2,2))
        ])) # final output is 16x5x5
        self.MLP = nn.Sequential(OrderedDict([
            ('fc2', nn.Linear(16*5*5, 64)),
            ('nl2', nn.ReLU()),
            ('fc4', nn.Linear(64, num_classes)),
            ('sm4', nn.Softmax(dim=1))
        ]))

        for i in range(len(self.convNet)):
            layer = self.convNet[i]
            if type(layer) == nn.Module:
                nn.init.xavier_uniform(layer.weight)

    def forward(self, x):
        f = self.convNet(x)
        return self.MLP(f.view(f.size(0), -1))

class Classifier(nn.Module):
    def __init__(self, in_sz=784, num_classes=10, h_sz=500, device=torch.device("cpu")):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(in_sz, h_sz)
        self.fc2 = nn.Linear(h_sz, num_classes)

        self.to(device)

    def forward(self, x):
        x = F.softplus(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1) # produce class scores over labels

class SS_VAE(nn.Module):
    """ Semi-Supervised VAE
        Kingma, et al. (2014). Semi-supervised Learning with Deep Generative Models. arXiv preprint arXiv:1406.5298v2
    """

    def __init__(self, batch_size=100, img_size=784, num_classes=10, z_sz=50, device=torch.device("cpu")):
        super(SS_VAE, self).__init__()
#        self.enc_z = Encoder(in_sz=img_size+num_classes, z_sz=z_sz, device=device) # q_phi(z|x,y) params
        self.enc_z = Encoder(in_sz=img_size, z_sz=z_sz, device=device) # q_phi(z|x) params
        self.enc_y = Classifier(in_sz=img_size, num_classes=num_classes, device=device) # q_phi(y|x) params
        self.dec = Decoder(in_sz=z_sz+num_classes, out_sz=img_size, device=device) # p(x|y,z)

        self._dev = device
        self._alpha = 0.1 * 60000 # weighting term on classifier loss
        self._logpy = torch.tensor(np.log(1/num_classes), device=device)

        self.to(device)

    def reparam_z(self, z_params):
        mu, logvar = torch.chunk(z_params,2,dim=1)
        eps = Variable(torch.randn(mu.size(0), mu.size(1), device=self._dev))
        z = mu + eps*torch.exp(logvar/2)
        return z

    def forward(self, x):
        pi = self.enc_y(x) # pi? pi_phi(x)?? idk

#        inp = torch.cat([x,pi], 1)
#        z_params = self.enc_z(inp)
        z_params = self.enc_z(x)
        z = self.reparam_z(z_params)

        out = self.dec(z, pi)

        # - out informs reconstruction loss term
        # - z_params inform KL divergence loss term
        # - pi inform cross-entropy loss (w/ logits) for true label y
        return out, z_params, pi

    def encoder(self, x):
        z_params = self.enc_z(x)
        pi = self.enc_y(x)
        return z_params, pi

    def sample(self, z, pi):
        return self.dec(z, pi)

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
        return recon_XEnt + label_XEnt + KL - self._logpy

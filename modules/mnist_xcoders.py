import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Simple Encoder
class FCEncoder(nn.Module):
    def __init__(self,z_sz=10,h_sz=400,im_sz=784):
        super(FCEncoder, self).__init__()
        self.fc1 = nn.Linear(im_sz,h_sz)
        self.fc2 = nn.Linear(h_sz,2*z_sz)

    def forward(self,x):
        return self.fc2(F.leaky_relu(self.fc1(x),0.2))

# Simple Decoder
class FCDecoder(nn.Module):
    def __init__(self,z_sz=10,h_sz=400,im_sz=784):
        super(FCDecoder, self).__init__()
        self.fc1 = nn.Linear(z_sz,h_sz)
        self.fc2 = nn.Linear(h_sz,im_sz)

    def forward(self,x):
        return F.sigmoid(self.fc2(F.relu(self.fc1(x))))

# Encoder
class ConvEncoder(nn.Module):
    def __init__(self,z_sz=10):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 2*z_sz)

    def forward(self, x):
        # x is 50 x 1 x 28 x 28
        x = self.conv1(x)
        # x is 50 x 10 x 24 x 24
        x = F.relu(F.max_pool2d(x, 2))
        # x is 50 x 10 x 12 x 12
        x = self.conv2(x)
        # x is 50 x 20 x 8 x 8
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        # x is 50 x 20 x 4 x 4
        x = F.relu(x)
        x = x.view(-1, 320)
        # x is 50 x 320
        x = F.relu(self.fc1(x))
        # x is 50 x 50
        x = self.fc2(x)
        # x is 50 x (2 * z_sz)
        return x

# Decoder
class DeconvDecoder(nn.Module):
    def __init__(self,z_sz=10):
        super(DeconvDecoder, self).__init__()
        self.fc1 = nn.Linear(z_sz, 32)
        self.fc2 = nn.Linear(32, 32)
        self.deconv1 = nn.ConvTranspose2d(32,16,10)
        self.deconv2 = nn.ConvTranspose2d(16,3,10)
        self.deconv3 = nn.ConvTranspose2d(3,1,10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 32,1,1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.deconv3(x)
        return F.sigmoid(x)

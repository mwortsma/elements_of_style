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
    def __init__(self,z_sz=10,c_sz=0,h_sz=400,im_sz=784):
        super(FCDecoder, self).__init__()
        self.fc1 = nn.Linear(c_sz + z_sz,h_sz)
        self.fc2 = nn.Linear(h_sz,im_sz)

    def forward(self,x,c=None):
        if c is not None:
            x = torch.cat((c,x), dim=1)
        print(x.size())
        out = F.sigmoid(self.fc2(F.relu(self.fc1(x))))
        return out

# Encoder
class ConvEncoder(nn.Module):
    def __init__(self,z_sz=10):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.conv5 = nn.Conv2d(64, 16, kernel_size=4, stride=1)


    def forward(self, x):
        # print(x.size())
        x = F.relu(self.conv1(x))
        # print(x.size())
        x = F.relu(self.conv2(x))
        # print(x.size())
        x = F.relu(self.conv3(x))
        # print(x.size())
        x = F.relu(self.conv4(x))
        # print(x.size())
        x = self.conv5(x)
        x = x.view(x.size(0), 16)
        return x

# Decoder
class DeconvDecoder(nn.Module):
    def __init__(self,z_sz=10):
        super(DeconvDecoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(8,32,kernel_size=4)
        self.deconv2 = nn.ConvTranspose2d(32,32,kernel_size=3,stride=2)
        self.deconv3 = nn.ConvTranspose2d(32,32,kernel_size=4,stride=2)
        self.deconv4 = nn.ConvTranspose2d(32,32,kernel_size=5)
        self.deconv5 = nn.ConvTranspose2d(32,1,kernel_size=5)

    def forward(self, x,c=None):
        x = x.view(x.size(0), 8, 1, 1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = self.deconv5(x)
        return F.sigmoid(x)

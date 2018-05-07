import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable


import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

# Helper functions
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# Parameters
data_dir = '../data/CIFAR10'
v_sz = 10
learning_rate = 0.01
batch_sz = 4
num_epochs = 1

DEVICE = torch.device('cpu')
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')

# Load Data

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_sz,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_sz,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class MLP(nn.Module):

    def __init__(self, device=torch.device("cpu")):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2*v_sz, 128)
        self.fc2 = nn.Linear(128, 2)

        self.to(device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class CIFAR10LinkedNet(nn.Module):
    def __init__(self, device=torch.device("cpu")):
        super(CIFAR10LinkedNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, v_sz)

        self.mlp = MLP(device)

        self._dev = device
        self.to(device)

    def forward_single_image(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


    def forward(self, x1, x2):
        # make forward call on each
        out1 = self.forward_single_image(x1)
        out2 = self.forward_single_image(x2)
        # concat
        x = torch.cat((out1, out2), dim=1)
        # pass through MLP
        return self.mlp(x)

net = CIFAR10LinkedNet()
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

loss_vec = []

for epoch in range(num_epochs):
    for batch_idx, (images, labels) in enumerate(trainloader):
        images1, images2 = torch.chunk(images, 2, dim=0)
        labels1, labels2 = torch.chunk(labels, 2, dim=0)

        eq_labels = Variable(labels1.eq(labels2).long())

        optimizer.zero_grad()

        out = net(images1, images2)
        loss = criterion(out, eq_labels)

        loss_vec.append(loss.item())
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print("Epoch[%d], Step [%d], Total Loss: %.4f"
                   %(epoch+1, batch_idx, loss.item()))

plt.plot(loss_vec)
plt.show()

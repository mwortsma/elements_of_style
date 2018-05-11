import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Helper functions
def imshow(img):
#    img = img / 2 + 0.5     # unnormalize
    print("\t #{}".format(img.size()))
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# Parameters
data_dir = '../data/CIFAR10'
v_sz = 10
learning_rate = 0.01
batch_sz = 50
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

        # --- init layers ---
        nn.init.xavier_uniform(self.conv1.weight)
        nn.init.xavier_uniform(self.conv2.weight)

        self._dev = device
        self.to(device)

    def forward_single_image(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        f = self.conv2(x)
        x = self.pool(F.relu(f))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return f,x

    def forward(self, x1, x2):
        # make forward call on each
        f1,out1 = self.forward_single_image(x1)
        f2,out2 = self.forward_single_image(x2)
        # concat
        x = torch.cat((out1, out2), dim=1)
        f = torch.cat((f1, f2), dim=1)
        # pass through MLP
        return f, self.mlp(x)

def vis_conv1(net):
    conv1_weights = net.conv1.weight.data.view(-1, 3, 5, 5).cpu()
    imshow(torchvision.utils.make_grid(conv1_weights, pad_value=1, normalize=True))
def vis_conv2(net):
    conv2_weights = net.conv2.weight.data.view(-1, 1, 5, 5).cpu()
    imshow(torchvision.utils.make_grid(conv2_weights, pad_value=1, normalize=True))

net = CIFAR10LinkedNet(device=DEVICE)
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

loss_vec = []

vis_conv1(net)
plt.title("Epoch[0] conv1")
plt.figure()
vis_conv2(net)
plt.title("Epoch[0] conv2")
plt.figure()

for epoch in range(num_epochs):
    for batch_idx, (images, labels) in enumerate(trainloader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        images1, images2 = torch.chunk(images, 2, dim=0)

        labels1, labels2 = torch.chunk(labels, 2, dim=0)

        eq_labels = Variable(labels1.eq(labels2).long())

        optimizer.zero_grad()

        fmap, out = net(images1, images2)
        loss = criterion(out, eq_labels)

        loss_vec.append(loss.item())
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print("Epoch[%d], Step [%d], Total Loss: %.4f"
                   %(epoch+1, batch_idx, loss.item()))
        if batch_idx == len(trainloader) - 1:
            print(images.data.size())
            print(fmap.data.size())
#            for i in fmap.data.size(0):
#                f = fmap[i,:,:,:]
#                imshow(torchvision.utils.make_grid(f.data.view(-1, 1, 10, 10).cpu(), pad_value=1))
#                plt.show()

    vis_conv1(net)
    plt.title("Epoch[{}] conv1".format(epoch+1))
    plt.figure()
    vis_conv2(net)
    plt.title("Epoch[{}] conv2".format(epoch+1))
    plt.figure()


plt.plot(loss_vec)
plt.title("Loss over {} epochs".format(num_epochs))
plt.show()

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms, datasets

from utils import my_transforms

###

data_transform = transforms.Compose([
    my_transforms.RandomRot(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[180.19, 159.39, 223.15],
                         std=[1.0, 1.0, 1.0])
])

###


data_dir = '/home/peter/datasets/ICIAR2018_BACH_Challenge/Mini_set'
dataset = datasets.ImageFolder(root=data_dir, transform=data_transform)
dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)


###

def conv3x3(inputs, outputs, pad, mp):
    if pad == 'valid':
        pad = 0
    elif pad == 'same':
        pad = 1
    layer = nn.Sequential(
        nn.Conv2d(inputs, outputs, kernel_size=3, stride=1, padding=pad),
        nn.BatchNorm2d(outputs),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=mp, stride=mp))
    return layer


def dense(inputs, outputs):
    layer = nn.Sequential(
        nn.Linear(inputs, outputs),
        nn.BatchNorm1d(outputs),
        nn.ReLU(),
        nn.Dropout())
    return layer


def final(inputs, outputs):
    return nn.Linear(inputs, outputs)


###

class NW(nn.Module):
    def __init__(self):
        super(NW, self).__init__()
        self.conv1 = conv3x3(3, 16, 'valid', 3)
        self.conv2 = conv3x3(16, 32, 'valid', 2)
        self.conv3 = conv3x3(32, 64, 'same', 2)
        self.conv4 = conv3x3(64, 64, 'same', 3)
        self.conv5 = conv3x3(64, 32, 'valid', 3)
        self.fc1 = dense(512, 256)
        self.fc2 = dense(256, 128)
        self.final = final(128, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.final(x)
        return (x)


###

network = NW()

num_epochs = 1
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(dataset_loader):
        images = Variable(images)
        x = network(images)
        print(x.size())
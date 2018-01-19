import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms, datasets

###

initial_learning_rate = 0.001
batch_size = 4

cuda = torch.cuda.is_available()

###

data_transform = transforms.Compose([
    transforms.Resize((768,1024)),
    transforms.ToTensor()
])

###

data_dir_mini = '/media/peter/HDD 1/ICIAR2018_BACH_Challenge/BACH_normalized'
dataset = datasets.ImageFolder(root=data_dir_mini, transform=data_transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

###

def conv_relu_maxpool(inputs, outputs, kernel, pad, maxpool):
    layer = nn.Sequential(
        nn.Conv2d(inputs, outputs, kernel_size=kernel, stride=1, padding=pad),
        # nn.BatchNorm2d(outputs, momentum=0.9),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=maxpool, stride=maxpool))
    return layer


def dense(inputs, outputs):
    layer = nn.Sequential(
        nn.Linear(inputs, outputs),
        # nn.BatchNorm1d(outputs, momentum=0.9),
        nn.ReLU())
    # nn.Dropout())
    return layer


###

class NW(nn.Module):

    def __init__(self):
        super(NW, self).__init__()
        self.conv1 = conv_relu_maxpool(3, 16, 3, 1, 2)
        self.conv2 = conv_relu_maxpool(16, 32, 3, 1, 2)
        self.conv3 = conv_relu_maxpool(32, 64, 3, 1, 2)
        self.conv4 = conv_relu_maxpool(64, 64, 3, 1, 2)
        self.conv5 = conv_relu_maxpool(64, 32, 3, 1, 2)
        self.fc1 = dense(2048, 512)
        self.fc2 = dense(512, 128)
        self.fc3 = nn.Linear(128, 2)
        self.salient = nn.Linear(128, 2)

    def get_features(self, x, verbose=False):
        if verbose: print(x.shape)
        x = self.conv1(x)
        if verbose: print(x.shape)
        x = self.conv2(x)
        if verbose: print(x.shape)
        x = self.conv3(x)
        if verbose: print(x.shape)
        x = self.conv4(x)
        if verbose: print(x.shape)
        x = self.conv5(x)
        if verbose: print(x.shape)
        x = x.view(x.size(0), -1)
        if verbose: print(x.shape)
        x = self.fc1(x)
        if verbose: print(x.shape)
        x = self.fc2(x)
        if verbose: print(x.shape)
        return x

    def forward(self, x):

        for i in range(3):
            for j in range(4):

                patch = x[:, :, i * 256:(i + 1) * 256, j * 256:(j + 1) * 256]
                features = self.get_features(patch, verbose=False)

                weights = torch.abs(self.salient(features))

                if i == 0 and j == 0:
                    scores = self.fc3(features) * weights
                    saliencies = weights
                else:
                    scores += self.fc3(features) * weights
                    saliencies = torch.cat((saliencies, weights), dim=1)

        return (scores, saliencies)


###

network = NW()
if cuda:
    network.cuda()

###

network.train()

for i, (images, labels) in enumerate(data_loader):
    if cuda:
        images.cuda()
        labels.cuda()



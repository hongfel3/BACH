import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms, datasets

import numpy as np

###

initial_learning_rate = 1e-3

epochs = 40
batch_size = 16

cuda = torch.cuda.is_available()

###

data_transform = transforms.Compose([
    transforms.Resize((768, 1024)),
    transforms.ToTensor()
])

###

data_dir_train = '/media/peter/HDD 1/datasets_peter/ICIAR2018_BACH_Challenge/Train_set_2class'
train_dataset = datasets.ImageFolder(root=data_dir_train, transform=data_transform)
data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

data_dir_val = '/media/peter/HDD 1/datasets_peter/ICIAR2018_BACH_Challenge/Val_set_2class'
val_dataset = datasets.ImageFolder(root=data_dir_val, transform=data_transform)
data_loader_val = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)


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
        # self.bn0 = nn.BatchNorm2d(3, momentum=0.9)
        self.conv1 = conv_relu_maxpool(3, 16, 3, 1, 2)
        self.conv2 = conv_relu_maxpool(16, 32, 3, 1, 2)
        self.conv3 = conv_relu_maxpool(32, 64, 3, 1, 2)
        self.conv4 = conv_relu_maxpool(64, 64, 3, 1, 2)
        self.conv5 = conv_relu_maxpool(64, 32, 3, 1, 2)
        self.fc1 = dense(2048, 512)
        self.fc2 = dense(512, 128)
        self.fc3 = nn.Linear(128, 2)

    def get_features(self, x, verbose=False):
        # if verbose: print(x.shape)
        # x = self.bn0(x)
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

                if i == 0 and j == 0:
                    scores = self.fc3(features)
                else:
                    scores += self.fc3(features)

        return scores


###

network = NW()
if cuda:
    network.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(network.parameters(), lr=initial_learning_rate)

###

for e in range(epochs):
    print('\nEpoch {} of {}\n'.format(e + 1, epochs))

    network.train()
    print('Training')
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(data_loader_train):
        if i % 5 == 0: print('Training batch {} of {}'.format(i + 1,len(data_loader_train)))
        if cuda:
            images = images.cuda()
            labels = labels.cuda()

        images = Variable(images)
        labels = Variable(labels.long())
        optimizer.zero_grad()

        scores = network(images)

        loss = criterion(scores, labels)

        loss.backward()
        optimizer.step()

        _, predicted = torch.max(scores.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
    print('Mean train accuracy over epoch = {}'.format(correct / total))

    network.eval()
    print('Validation')
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(data_loader_val):
        if cuda:
            images = images.cuda()
            labels = labels.cuda()

        images = Variable(images)
        labels = Variable(labels.long())

        scores = network(images)

        _, predicted = torch.max(scores.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
    print('Mean val accuracy over epoch = {}'.format(correct / total))

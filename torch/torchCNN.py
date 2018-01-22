import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms, datasets

from trash.pytorch_folder import my_transforms

###

cuda = torch.cuda.is_available()

lr = 1e-3

###

data_transform = transforms.Compose([
    my_transforms.RandomRot(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

###

data_dir_mini = '/media/peter/HDD 1/datasets_peter/ICIAR2018_BACH_Challenge/Mini_set'
dataset_mini = datasets.ImageFolder(root=data_dir_mini, transform=data_transform)
mini_loader = torch.utils.data.DataLoader(dataset_mini, batch_size=64, shuffle=True)


###

# data_dir_train = '/home/peter/datasets/ICIAR2018_BACH_Challenge/Train_set'
# data_dir_val = '/home/peter/datasets/ICIAR2018_BACH_Challenge/Val_set'

# dataset_train = datasets.ImageFolder(root=data_dir_train, transform=data_transform)
# train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=32, shuffle=True)

# dataset_val = datasets.ImageFolder(root=data_dir_val, transform=data_transform)
# val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=16, shuffle=True)


###

def conv3x3(inputs, outputs, pad, mp):
    if pad == 'valid':
        pad = 0
    elif pad == 'same':
        pad = 1
    layer = nn.Sequential(
        nn.Conv2d(inputs, outputs, kernel_size=3, stride=1, padding=pad),
        nn.BatchNorm2d(outputs, momentum=0.9),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=mp, stride=mp))
    return layer


def dense(inputs, outputs):
    layer = nn.Sequential(
        nn.Linear(inputs, outputs),
        nn.BatchNorm1d(outputs, momentum=0.9),
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
        return x


###

network = NW()
if cuda:
    network.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(network.parameters(), lr=lr)

loss_train = []
loss_eval = []
acc = []

num_epochs = 10
for epoch in range(num_epochs):

    print('Epoch {}'.format(epoch))
    network.train()
    cnt = 0
    losses = 0.0
    for i, (images, labels) in enumerate(mini_loader):
        print('Training batch {}'.format(i))
        if cuda:
            images=images.cuda()
            labels=labels.cuda()
        images = Variable(images)
        labels = Variable(labels.long())
        optimizer.zero_grad()
        output = network(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        losses += loss.data
        cnt += 1
    loss_train.append(losses / cnt)

    print('Validation Acc')
    network.eval()
    total = 0
    correct = 0
    cnt = 0
    losses = 0.0
    for images, labels in mini_loader:
        if cuda:
            images=images.cuda()
            labels=labels.cuda()
        images = Variable(images)
        labels = Variable(labels.long())
        output = network(images)

        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()

        loss = criterion(output, labels)
        losses += loss.data
        cnt += 1
    accuracy = 100 * correct / total
    print(accuracy)
    acc.append(accuracy)
    loss_eval.append(losses / cnt)

    # print('Saving model Params')
    # torch.save(network.state_dict(), 'nw' + str(epoch) + '.pkl')

print(loss_train)
print(loss_eval)
print(acc)

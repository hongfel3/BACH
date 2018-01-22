import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms, datasets

from pytorch_folder import my_transforms

###

lr = 1e-3

batch = 64
num_epochs = 10

cuda = torch.cuda.is_available()

###

data_transform = transforms.Compose([
    my_transforms.RandomRot(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

null_transform = transforms.Compose([
    transforms.ToTensor()
])

###

data_dir_mini = '/media/peter/HDD 1/datasets_peter/ICIAR2018_BACH_Challenge/Mini_set'
data_dir_train = '/media/peter/HDD 1/datasets_peter/ICIAR2018_BACH_Challenge/Train_set'
data_dir_val = '/media/peter/HDD 1/datasets_peter/ICIAR2018_BACH_Challenge/Val_set'

mini = True

if mini:
    dataset_train = datasets.ImageFolder(root=data_dir_mini, transform=data_transform)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch, shuffle=True)

    dataset_val = datasets.ImageFolder(root=data_dir_mini, transform=null_transform)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch, shuffle=True)

if not mini:
    dataset_train = datasets.ImageFolder(root=data_dir_train, transform=data_transform)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch, shuffle=True)

    dataset_val = datasets.ImageFolder(root=data_dir_val, transform=null_transform)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch, shuffle=True)


###

def conv3x3(inputs, outputs, pad, maxpool):
    if pad == 'valid':
        pad = 0
    elif pad == 'same':
        pad = 1
    layer = nn.Sequential(
        nn.Conv2d(inputs, outputs, kernel_size=3, stride=1, padding=pad),
        # nn.BatchNorm2d(outputs, momentum=0.9), # momentum!?
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=maxpool, stride=maxpool))
    return layer


def dense(inputs, outputs):
    layer = nn.Sequential(
        nn.Linear(inputs, outputs),
        # nn.BatchNorm1d(outputs, momentum=0.9), # momentum!?
        nn.ReLU())
    # nn.Dropout())
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

###

for epoch in range(num_epochs):
    print('Epoch {}'.format(epoch))

    network.train()
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(train_loader):
        if i % 10 == 0: print('Training batch {}'.format(i))
        if cuda:
            images = images.cuda()
            labels = labels.cuda()

        images = Variable(images)
        labels = Variable(labels.long())

        optimizer.zero_grad()
        output = network(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()

    accuracy = correct / total
    print('Mean train acc over epoch = {}'.format(accuracy))

    network.eval()
    print('Validation')
    total = 0
    correct = 0
    for images, labels in val_loader:
        if cuda:
            images = images.cuda()
            labels = labels.cuda()

        images = Variable(images)
        labels = Variable(labels.long())

        output = network(images)

        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()

    accuracy = correct / total
    print('Mean val acc over epoch = {}'.format(accuracy))
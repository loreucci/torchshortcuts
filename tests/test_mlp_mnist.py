from .context import torchshortcuts

from torchshortcuts.classification import MLPClassifier, validation
from torchshortcuts.training import train

import torch
import torch.nn as nn
from torchvision import datasets, transforms


# load dataset
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


model = MLPClassifier(28*28, 10, [128, 64])
optimizer = torch.optim.SGD(model.parameters(), lr=0.003)
criterion = nn.NLLLoss()


def reshape_img(img):
    return img.view(img.shape[0], -1)


train(model, trainloader, testloader, criterion, optimizer, 5, validation=validation, input_transform=reshape_img)

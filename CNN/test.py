from cmath import polar
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from model import ConvModel
import torchvision.transforms as tf
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr = 3e-4
batch_size = 4
num_epochs = 20

model = ConvModel().to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

transforms = tf.Compose([
    tf.ToTensor(),
    tf.Normalize((0.5,), (0.5,), (0.5,))
])

train_dataset = datasets.CIFAR10("dataset/", train=True, transform=transforms, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.CIFAR10("dataset/", train=False, transform=transforms, download=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

dataiter = iter(train_loader)

images, labels = next(dataiter)

conv1 = nn.Conv2d(3, 6, 4)
pool = nn.MaxPool2d(2,2)
conv2 = nn.Conv2d(6, 9, 5)

print(images.shape)

x = conv1(images)

#print (x.shape)

x = pool(x)

print (x.shape)

x = conv2(x)

#print (x.shape)

x = pool(x)

print (x.shape)


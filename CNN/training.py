import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from model import ConvModel
import torchvision.transforms as tf
import os


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr = 3e-4
    batch_size = 4
    num_epochs = 20

    model = ConvModel().to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    transforms = tf.Compose([
        tf.ToTensor(),
        tf.Normalize((0.5,), (0.5,), (0.5,))
    ])

    train_dataset = datasets.CIFAR10("dataset/", train=True, transform=transforms, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.CIFAR10("dataset/", train=False, transform=transforms, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    step = 0
    total_steps = len(train_loader) * num_epochs

    for epoch in range(num_epochs):
        for idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (idx+1) % 2000 == 0:
                print(f'Epoch: {epoch}/{num_epochs} | Step: {step+1};{total_steps} | Loss: {loss}')

            step += 1

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for _ in range(10)]
        n_class_samples = [0 for _ in range(10)]

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            _, predicted = torch.max(output, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            for i in range(batch_size):
                label = labels[i]
                pred = predicted[i]
                if (label == pred):
                    n_class_correct[label.item()] += 1
                n_class_samples[label.item()] += 1

        print(f'Network acc is:  {n_correct/n_samples}')



if __name__ == "__main__":
    train()

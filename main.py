import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from cnn_cuda import CNN
from train import train_model
from utils import plot_training_curves

def main():
    # Hyperparameters
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 20

    # Data preprocessing and augmentation
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='data/cifar10', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='data/cifar10', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Model, Loss, Optimizer
    model = CNN().cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_losses, val_losses, val_accuracies = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs)

    # Plot training curves
    plot_training_curves(train_losses, val_losses, val_accuracies)

if __name__ == '__main__':
    main()

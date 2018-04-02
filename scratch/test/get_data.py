"""
The data downloader.
"""
import torch
import torchvision
from torchvision import transforms


DATA_DIR = '../../data'


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


cifar10_train = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True,
                                             download=True, transform=transform)
cifar10_test = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False,
                                            download=True, transform=transform)

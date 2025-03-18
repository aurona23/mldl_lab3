import os
import torch
from torchvision.datasets import ImageFolder
from dataset.transform_dataset import *
from data.dataloader import *
from utils.visualization import *

if __name__ == "__main__":
    # Load the dataset
    tiny_imagenet_dataset_train = ImageFolder(root='./dataset/tiny-imagenet-200/train', transform=transform()['train'])
    tiny_imagenet_dataset_test = ImageFolder(root='./dataset/tiny-imagenet-200/test', transform=transform()['val'])
    dataloader_train, dataloader_test =  dataloader(tiny_imagenet_dataset_train,tiny_imagenet_dataset_test)
    
    # Determine the number of classes and samples
    num_classes = len(tiny_imagenet_dataset_train.classes)
    num_samples = len(tiny_imagenet_dataset_train)

    print(f'Number of classes: {num_classes}')
    print(f'Number of samples: {num_samples}')

    visualization(dataloader_train=dataloader_train)
    

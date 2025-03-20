import os
import torch
from torchvision.datasets import ImageFolder
from dataset.transform_dataset import *
from data.dataloader import *
from utils.visualization import *


def train(epoch, model, train_loader, criterion, optimizer):
  # epoch : epoca a cui sono ora
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    # batch size glielo dai fuori nel train loder
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda() # inputs, targets
        # Compute prediction and loss
        outputs = model(inputs)
        # criterion is loss
        loss = criterion(outputs, targets)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # loss totale finora per ogni batch
        running_loss += loss.item()
        # outputs contiene le predizioni del modello per il batch corrente.
        # outputs.max(1) restituisce due valori:
        #il valore massimo per ogni riga (predizione) di outputs
        #l'indice della colonna dove si trova il valore massimo (classe predetta).
        _, predicted = outputs.max(1)
        # targets.size(0): dimensione batch corrente
        #con total accumulo quelli fatti finora
        total += targets.size(0)
        # somma gli elementi corretti ( array di 0)
        correct += predicted.eq(targets).sum().item()
    # mean loss over the whole train set
    train_loss = running_loss / len(train_loader)
    # accuracy del train
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')


if __name__ == "__main__":
# LAB 0102 
    # Load the dataset
    tiny_imagenet_dataset_train = ImageFolder(root='./dataset/tiny-imagenet-200/train', transform=transform_1()['train'])
    tiny_imagenet_dataset_test = ImageFolder(root='./dataset/tiny-imagenet-200/test', transform=transform_1()['val'])
    # dataloader
    dataloader_train, dataloader_test =  dataloader(tiny_imagenet_dataset_train,tiny_imagenet_dataset_test, batch_size=64, shuffle_train=True, shuffle_test=True)
    
    # Determine the number of classes and samples
    num_classes = len(tiny_imagenet_dataset_train.classes)
    num_samples = len(tiny_imagenet_dataset_train)

    print(f'Number of classes: {num_classes}')
    print(f'Number of samples: {num_samples}')

    visualization(dataloader_train=dataloader_train)
    ##############################################################
    #LAB 2

    
    tiny_imagenet_dataset_train = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/train', transform=transform_2())
    tiny_imagenet_dataset_val = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/val', transform=transform_2())
    
    print(f"Length of train dataset: {len(tiny_imagenet_dataset_train)}")
    # print(f"Length of val dataset: {len(tiny_imagenet_dataset_val)}")

    # dataloader
    train_loader, _ =  dataloader(tiny_imagenet_dataset_train,tiny_imagenet_dataset_val, batch_size=32, shuffle_train=True, shuffle_test=False)

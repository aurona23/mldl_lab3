import os
import torch
from torchvision.datasets import ImageFolder
from dataset.transform_dataset import *
from data.dataloader import *
from models.custom_net import *
from train import *

# Validation loop
def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0

    correct, total = 0, 0

    with torch.no_grad(): #non calcolo i gradienti, non stiamo cercando i parametri perché
    # li ho già trovati prima, quindi non devo calolcare i gradietni
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            # numero di quelli corretti
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total
  # calcola loss e accuracy
    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    return val_accuracy

if __name__ == "__main__":

    #LAB 2
    tiny_imagenet_dataset_train = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/train', transform=transform_2)
    tiny_imagenet_dataset_val = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/test', transform=transform_2)
    
    print(f"Length of train dataset: {len(tiny_imagenet_dataset_train)}")
    # print(f"Length of val dataset: {len(tiny_imagenet_dataset_val)}")

    # dataloader
    train_loader, val_loader =  dataloader(tiny_imagenet_dataset_train,tiny_imagenet_dataset_val, batch_size=32, shuffle_train=True, shuffle_test=False)

    ####################################################
    # TRAIN AND EVAL

    model = CustomNet().cuda()
    # loss che scelgo
    criterion = nn.CrossEntropyLoss()
    # con cosa ottimizzo : stochastic gradiente descent
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    best_acc = 0

    # Run the training process for {num_epochs} epochs
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        train(epoch, model, train_loader, criterion, optimizer)

        # At the end of each training iteration, perform a validation step
        val_accuracy = validate(model, val_loader, criterion)

        # Best validation accuracy
        best_acc = max(best_acc, val_accuracy)


    print(f'Best validation accuracy: {best_acc:.2f}%')
    # mirgliore è quello se mi fermo a conv4 con test acc del 14.15%
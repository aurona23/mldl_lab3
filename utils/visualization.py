from matplotlib.pyplot import plt
from dataset.denormalize import *  

def visualization(dataloader_train):
# Visualize one example for each class for 10 classes
    fig, axes = plt.subplots(2, 5, figsize=(15, 6)) # anche se commento qua funziona ma più piccole le foto
    classes_sampled = []
    found_classes = 0

    for i, (inputs, classes) in enumerate(dataloader_train):
        img = inputs[i].squeeze()
        img = denormalize(img)
        label = classes[i]
        if label not in classes_sampled:
            classes_sampled.append(label)
        found_classes += 1
        plt.subplot(2, 5, found_classes)
        plt.title(f"Class: {label}")
        plt.imshow(img)
        if found_classes == 10:
            break
    plt.show()
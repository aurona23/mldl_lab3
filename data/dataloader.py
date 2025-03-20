from torch.utils.data import DataLoader

def dataloader(tiny_imagenet_dataset_train,tiny_imagenet_dataset_test, batch_size, shuffle_train, shuffle_test):
# Create a DataLoader
    dataloader_train = DataLoader(tiny_imagenet_dataset_train, batch_size = batch_size, shuffle = shuffle_train)
    dataloader_test = DataLoader(tiny_imagenet_dataset_test, batch_size = batch_size, shuffle = shuffle_test)

    return dataloader_train, dataloader_test

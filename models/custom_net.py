from torch import nn

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.flatten = nn.Flatten()
        # Define layers of the neural network
        # 3 quanti nodi di input passo nel layer
        # 64 sono quelli del prossimo layer
        #stride serve per dimezzare dimesnione spaziale
        # di volta in volta ti allontani e vedi il doppio
        # The output size after 5 convolutional layers with stride 2 and padding 1
        # can be calculated as:
        # output_size = (input_size - kernel_size + 2 * padding) / stride + 1
        # After conv1: (224 - 3 + 2 * 1) / 2 + 1 = 112
        # After conv2: (112 - 3 + 2 * 1) / 2 + 1 = 56
        # After conv3: (56 - 3 + 2 * 1) / 2 + 1 = 28
        # After conv4: (28 - 3 + 2 * 1) / 2 + 1 = 14
        # After conv5: (14 - 3 + 2 * 1) / 2 + 1 = 7
        # stampa x.shape dopo ogni conv per conferma

        # The input size for the fully connected layer is 512 * 7 * 7 = 25088
        #linear serve per ritornare 20 classi, a partire da ultimo strano con 512 nodi
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride = 2)
        # Add more layers 5 conv
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride = 2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride = 2)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride = 2)
        # Calculate the input size for the fully connected layer
        # Assuming the input image size is 224x224
        # and the stride of the first convolutional layer is 23
        # the output size of the first convolutional layer is (224-3+2*1)/23 + 1 = 10
        # the output size of the second convolutional layer is (10-3+2*1)/1 + 1 = 10
        # the input size for the fully connected layer is 128 * 10 * 10 = 12800
        #linear serve per ritornare 20 classi, a partire da ultimo strano con 512 nodi
        self.fc1 = nn.Linear(512*14*14, 200)  # 200 is the number of classes in TinyImageNet, 7 perché ho foto di dim 7*7

    def forward(self, x):
        # Define forward pass
        x = self.conv1(x).relu()
        #print(f'x shape after 1st conv {x.shape}')
        x = self.conv2(x).relu()
        #print(f'x shape after 2nd conv {x.shape}')
        x = self.conv3(x).relu()
        #print(f'x shape after 3rd conv {x.shape}')
        x = self.conv4(x).relu()
        #print(f'x shape after 4th conv {x.shape}')
        #x = self.conv5(x).relu()
        #print(f'x shape after 5th conv {x.shape}')
        # Pass the flattened output through the fully connected layer
        # flatten x
        x = self.flatten(x)
        x = self.fc1(x)
        return x
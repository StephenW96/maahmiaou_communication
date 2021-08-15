import torch
from torch import nn

## Simple CNN to be implemented in the classifier

class CNNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        #2 conv blocks / dropout/ flatten / linear / dropout/ softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.05)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.05)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.05)
        )


        self.flatten = nn.Flatten() # flatten from CNN to Linear
        # self.linear1 = nn.Linear(768, 256)  # for MFCC 1sec LSTM layer
        # self.linear1 = nn.Linear(768, 256) # for MelSpec 1/2 second
        self.linear1 = nn.Linear(64*4*3, 256)  # for MelSpec
        # # self.linear1 = nn.Linear(32 * 6 * 5, 256) # for MFCC
        self.linear2 = nn.Linear(256,8)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        logits = self.linear2(x)
        predictions = logits
        # print(predictions.shape)
        return predictions

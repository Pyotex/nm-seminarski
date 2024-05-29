from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from PIL import Image
import torch.nn as nn
import pandas as pd
import numpy as np 
import torchvision
import torch
import time
import os

start_time = time.time()


emotions = {
    0: 'Ljutnja', 
    1: 'Gađenje', 
    2: 'Strah', 
    3: 'Sreća', 
    4: 'Tuga', 
    5: 'Iznenađenje', 
    6: 'Neutralno'
}


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = x.view(-1, 512)
        # x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net()

net.load_state_dict(torch.load('facial_cnn.pth', map_location=torch.device('cpu')))

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
net.to(device)


def predict_emotion(image_tensor):
    # Pass the tensor through the model
    net.eval()
    start_time = time.time()

    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = net(image_tensor)
        _, predicted = torch.max(output.data, 1)
        print(emotions[int(predicted[0])])

    elapsed_time = time.time() - start_time
    print(f'Inference finished in {elapsed_time:.2f} seconds')

    return emotions[int(predicted[0])]


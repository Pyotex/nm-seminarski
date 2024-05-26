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


class FER2018Dataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_label = int(row['emotion'])
        img = np.array(row["pixels"].split(' '), dtype=np.uint8).reshape(48, 48)

        img = Image.fromarray(img)
        img = self.transform(img)

        return img, img_label


emotions = {
    0: 'Angry', 
    1: 'Disgust', 
    2: 'Fear', 
    3: 'Happy', 
    4: 'Sad', 
    5: 'Surprise', 
    6: 'Neutral'
}

dataset = pd.read_csv('./fer20131.csv')
dataset.info()

train_df = dataset[dataset["Usage"] == "Training"]
val_df = dataset[dataset["Usage"] == "PublicTest"]
test_df = dataset[dataset["Usage"] == "PrivateTest"]


train_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.RandomCrop(48, padding=4, padding_mode='reflect'),     
    transforms.RandomAffine(
        degrees=0,
        translate=(0.01, 0.12),
        shear=(0.01, 0.03),
    ),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=(-30, 30)),
    transforms.ColorJitter(),
    transforms.RandomHorizontalFlip()
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5], std=[0.5]),
])

train_dataset = FER2018Dataset(train_df, train_transform)
val_dataset = FER2018Dataset(val_df, test_transform)
test_dataset = FER2018Dataset(test_df, test_transform)

batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)


# Get some random test images
dataiter = iter(train_dataloader)
images, labels = next(dataiter)

# Print images
plt.imshow(torchvision.utils.make_grid(images).permute(1, 2, 0), cmap='gray')
plt.show()


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

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


for epoch in range(2):  # loop over the dataset multiple times
    start_time = time.time()

    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # inputs, labels = data
        # inputs, labels = inputs.to(device), labels.to(device)

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0
    
    elapsed_time = time.time() - start_time
    print(f'Epoch {epoch + 1} finished in {elapsed_time:.2f} seconds')

print('Finished Training')



# save the model weights
torch.save(net.state_dict(), 'facial_cnn.pth')


correct = 0
total = 0
with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Get some random test images
dataiter = iter(test_dataloader)
images, labels = next(dataiter)

# Print images
plt.imshow(torchvision.utils.make_grid(images).permute(1, 2, 0), cmap='gray')
plt.show()

print(f'Accuracy: {100 * correct / total:.2f}%')
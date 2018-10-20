import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5,padding=2,stride=1)
        # self.pool = nn.AvgPool2d(2,2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1,stride=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1,stride=1)
        # self.conv3 = nn.Conv2d(16, 24, 5, padding=1)
        # features = self.num_flat_features(x)
        self.fc1 = nn.Linear(128 * 4 * 4, 240)
        self.fc2 = nn.Linear(240, 120)
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, 10)

    def forward(self, x):
        # print(self.num_flat_features(x))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # print(self.num_flat_features(x))
        x = self.pool(F.relu(self.conv3(x)))
        # print(self.num_flat_features(x))
        # x = x.view(-1, 16 * 5 * 5)
        features = self.num_flat_features(x)
        # print(x.shape)
        x = x.view(-1, features)
        # print(x.size)
        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features




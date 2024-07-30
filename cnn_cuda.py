import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1).cuda()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1).cuda()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0).cuda()
        self.fc1 = nn.Linear(64 * 8 * 8, 512).cuda()
        self.fc2 = nn.Linear(512, 10).cuda()
        self.dropout = nn.Dropout(0.5).cuda()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

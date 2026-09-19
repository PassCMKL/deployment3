import torch.nn as nn
import torch.nn.functional as F


class StudentCNN(nn.Module):
    # A deliberately tiny CNN: two conv/pool blocks, one linear classifier.
    # ~9k parameters vs. the teacher ResNet18's ~11.2M.
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = x.flatten(1)
        return self.fc(x)

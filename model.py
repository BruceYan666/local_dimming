import torch
import torch.nn as nn

class SurrogateModel(nn.Module):
    def __init__(self):
        super(SurrogateModel, self).__init__()
        self.linear1 = nn.Linear(2376, 4096)
        self.linear2 = nn.Linear(4096, 2048)
        self.linear3 = nn.Linear(2048, 1024)
        self.linear4 = nn.Linear(1024, 256)
        self.linear5 = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, input):
        y = self.linear1(input)
        y = self.relu(y)
        y = self.linear2(y)
        y = self.relu(y)
        y = self.linear3(y)
        y = self.relu(y)
        y = self.linear4(y)
        y = self.relu(y)
        out = self.linear5(y)
        return out
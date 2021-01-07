import torch
import torch.nn as nn
import pdb


class SurrogateModel(nn.Module):
    def __init__(self, init_weights=False):
        super(SurrogateModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1, ),#（4，36，66）
            nn.ReLU(),
            nn.MaxPool2d(2),#（4，18，33）
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 8, 3, 1, 1),#（8，18，33）
            nn.ReLU(),
            nn.MaxPool2d(2),

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.linear = nn.Sequential(
            nn.Linear(16*4*8, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32,1)
        )
        if init_weights:
            self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # kaiming高斯初始化，目的是使得Conv2d卷积层反向传播的输出的方差都为1
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, input):
        m = input.size(0)
        # pdb.set_trace()
        input = input.reshape(m, 1, 36, 66)
        y = self.conv1(input)
        y = self.conv2(y)
        y = self.conv3(y)
        y = y.view(y.size(0), -1)
        out = self.linear(y)
        # pdb.set_trace()
        return out




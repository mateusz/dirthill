#%%

import terrain_set2
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%

#https://towardsdatascience.com/understanding-u-net-61276b10f360
class Test(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Conv1d(1, 4, 3, padding=1),
            nn.BatchNorm1d(4),
            nn.ReLU(inplace=True),
        )
        
        self.down = nn.Sequential(
            nn.MaxPool1d(2),
        )

        self.up = nn.Sequential(
            nn.ConvTranspose2d(4, 1, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.enc(x)

batch = 1
channels = 1
input1d = 8
output2d = 4
net = Test()
net(torch.randn(batch, channels, input1d))
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
boundl = 128

#%%

n=128
boundl=128
ts = terrain_set2.TerrainSet('data/USGS_1M_10_x43y465_OR_RogueSiskiyouNF_2019_B19.tif',
    size=n, stride=8)
t,v = torch.utils.data.random_split(ts, [0.90, 0.10])
train = DataLoader(t, batch_size=512, shuffle=True,
    num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=4)
val = DataLoader(v, batch_size=512, shuffle=True,
    num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=4)

#%%

class View(nn.Module):
    def __init__(self, dim,  shape):
        super(View, self).__init__()
        self.dim = dim
        self.shape = shape

    def forward(self, input):
        new_shape = list(input.shape)[:self.dim] + list(self.shape) + list(input.shape)[self.dim+1:]
        return input.view(*new_shape)


# https://github.com/pytorch/pytorch/issues/49538
nn.Unflatten = View

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        ch=16
        chd=16

        self.encoder = nn.Sequential(
            nn.Unflatten(1, (1, boundl)),

            nn.Conv1d(1, ch, 3, padding=1),
            nn.BatchNorm1d(ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(ch, ch*2, 3, padding=1),
            nn.BatchNorm1d(ch*2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(ch*2, ch*4, 3, padding=1),
            nn.BatchNorm1d(ch*4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(ch*4, ch*8, 3, padding=1),
            nn.BatchNorm1d(ch*8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(ch*8, ch*16, 3, padding=1),
            nn.BatchNorm1d(ch*16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(ch*16, ch*32, 3, padding=1),
            nn.BatchNorm1d(ch*32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(ch*32, ch*64, 3, padding=1),
            nn.BatchNorm1d(ch*64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Flatten(),
        )

        latentl = 256
        self.lstm1 = nn.LSTM(1*1*(ch*64), latentl)
        self.lstm1relu = nn.ReLU(True)

        self.decoder = nn.Sequential(
            nn.Linear(latentl, chd*64*1*1),
            nn.ReLU(inplace=True),

            nn.Unflatten(1, (chd*64, 1)),

            nn.ConvTranspose1d(chd*64, chd*32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(chd*32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(chd*32, chd*16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(chd*16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(chd*16, chd*8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(chd*8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(chd*8, chd*4, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(chd*4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(chd*4, chd*2, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(chd*2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(chd*2, chd, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(chd),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(chd, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Flatten(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x, (hn, cn)  = self.lstm1(x)
        x = self.lstm1relu(x)
        x = self.decoder(x)
        return x

net = Net()
inp = torch.Tensor([ts[0][0][:boundl], ts[1][0][:boundl]])
print(inp.shape)
net(inp).shape

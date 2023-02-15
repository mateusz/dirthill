#%%

import terrain_set
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

n=128
ts = terrain_set.TerrainSet('data/USGS_1M_10_x43y465_OR_RogueSiskiyouNF_2019_B19.tif',
    size=n, stride=8, local_norm=True, square_output=True)
t,v = torch.utils.data.random_split(ts, [0.9, 0.1])
train = DataLoader(t, batch_size=256, shuffle=True,
    num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=4)
val = DataLoader(v, batch_size=256, shuffle=True,
    num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=4)

#%%

ch = 16
conv1 = nn.Sequential(
        nn.Conv2d(1, ch, 3, padding=1),
        nn.BatchNorm2d(ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),

        nn.Conv2d(ch, ch*2, 3, padding=1),
        nn.BatchNorm2d(ch*2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),

        nn.Conv2d(ch*2, ch*4, 3, padding=1),
        nn.BatchNorm2d(ch*4),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),

        nn.Conv2d(ch*4, ch*8, 3, padding=1),
        nn.BatchNorm2d(ch*8),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),

        nn.Flatten(),

        nn.Linear(8*8*(ch*8), 512),
        nn.ReLU(True),
        nn.Linear(512, 128),
        nn.ReLU(True),
        nn.Linear(128, 512),
        nn.ReLU(True),
        nn.Linear(512, 8*8*(ch*8)),
        nn.ReLU(True),

        nn.Unflatten(dim=1, unflattened_size=(ch*8, 8, 8)),
        
        nn.ConvTranspose2d(ch*8, ch*4, 3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(ch*4),
        nn.ReLU(inplace=True),

        nn.ConvTranspose2d(ch*4, ch*2, 3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(ch*2),
        nn.ReLU(inplace=True),

        nn.ConvTranspose2d(ch*2, ch, 3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(ch),
        nn.ReLU(inplace=True),

        nn.ConvTranspose2d(ch, 1, 3, stride=2, padding=1, output_padding=1),
)

conv1(torch.Tensor([ts[0][1], ts[1][1]]).unsqueeze(1)).shape

#%%

net = conv1.to(device)
opt = optim.Adam(net.parameters())
#lossfn = nn.MSELoss()
lossfn = nn.L1Loss()

min_val_loss = 9999999999.0
early_stop_counter = 0
for epoch in range(999):  # loop over the dataset multiple times
    running_loss = 0.0
    net.train()

    for i, data in enumerate(train, 0):
        inputs, targets = data

        # zero the parameter gradients
        opt.zero_grad()

        # forward + backward + optimize
        outputs = net(targets.unsqueeze(1).to(device))

        loss = lossfn(outputs, targets.unsqueeze(1).to(device))
        loss.backward()
        opt.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:
            print("train: %.2f" % (running_loss/10.0))
            running_loss = 0.0

    running_loss = 0.0
    net.eval()
    with torch.no_grad():
        for i,data in enumerate(val, 0):
            inputs, targets = data
            outputs = net(targets.unsqueeze(1).to(device))
            loss = lossfn(outputs, targets.unsqueeze(1).to(device))
            running_loss += loss.item()

    vl = running_loss/len(val)
    print("val: %.2f" % (vl))

    if vl<min_val_loss:
        min_val_loss = vl
        early_stop_counter = 0
        print('saving...')
        torch.save(net, 'models/08')
    else:
        early_stop_counter += 1

    if early_stop_counter>=3:
        break
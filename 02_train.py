#%%

import terrain_set
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%

n=128
ts = terrain_set.TerrainSet('data/USGS_1M_10_x43y465_OR_RogueSiskiyouNF_2019_B19.tif',
    size=n, stride=8, local_norm=True, single_boundary=True)
t,v = torch.utils.data.random_split(ts, [0.90, 0.10])
train = DataLoader(t, batch_size=256, shuffle=True,
    num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=4)
val = DataLoader(v, batch_size=256, shuffle=True,
    num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=4)

#%%

print("%d,%d" % (len(train), len(val)))

#%%

net = nn.Sequential(
    nn.Linear(n,1024),
    nn.Dropout(p=0.5),
    nn.Linear(1024, n*n),
)

net = net.to(device)
opt = optim.Adam(net.parameters())
lossfn = nn.MSELoss()

min_val_loss = 9999999999.0
early_stop_counter = 0
for epoch in range(9999):  # loop over the dataset multiple times
    running_loss = 0.0
    net.train()

    for i, data in enumerate(train, 0):
        inputs, targets = data

        # zero the parameter gradients
        opt.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.to(device))

        loss = lossfn(outputs, targets.to(device))
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
            outputs = net(inputs.to(device))
            loss = lossfn(outputs, targets.to(device))
            running_loss += loss.item()

    vl = running_loss/len(val)
    print("val: %.2f" % (vl))

    if vl<min_val_loss:
        min_val_loss = vl
        early_stop_counter = 0
        print('saving...')
        torch.save(net, 'models/02')
    else:
        early_stop_counter += 1

    if early_stop_counter>=3:
        break

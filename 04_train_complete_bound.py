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
    size=n, stride=8, local_norm=True, full_boundary=True)
t,v = torch.utils.data.random_split(ts, [0.95, 0.05])
train = DataLoader(t, batch_size=1024, shuffle=True,
    num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=4)
val = DataLoader(v, batch_size=1024, shuffle=True,
    num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=4)

#%%

print("%d,%d" % (len(train), len(val)))

#%%

# 2048, 0.001 = val loss 9
# 4096, 0.001 = bad, loss 25, tons of noise
# 128->4096, 0.001 = vl 8, produces some interesting noise, seemingly better - fits boundaries better
# 128->64->512->4096, 0.001 = vl 6.4, although the result is quite smooth again.
# 128->1024->2048->8192, 0.001 = vl 8, smooth, doesn't fit edges
# 128->4096, 0.2 = vl 80, overfit
# 128->4096, 0.0 = vl 5.5 and it learned the texture!
# todo: will a simple net without dropout also learn texture?
class Net(nn.Module):
    def __init__(self):
        h = 128
        h2 = 4096
        dropout = 0.0
        super().__init__()
        self.l1 = nn.Linear(4*n,h)
        self.d1 = nn.Dropout(p=dropout)
        self.l2 = nn.Linear(h,h2)
        self.d2 = nn.Dropout(p=dropout)

        self.l5 = nn.Linear(h2, (n-2)*(n-2))
        self.d4 = nn.Dropout(p=dropout)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.d1(x)
        x = F.relu(self.l2(x))
        x = self.d2(x)
        x = F.relu(self.l5(x))
        #x = self.d2(x)
        return x

net = Net().to(device)
opt = optim.Adam(net.parameters())
lossfn = nn.MSELoss()

def area_mse(output, target):
    loss = torch.mean((output - target)**2)
    return loss

min_val_loss = 9999999999.0
early_stop_counter = 0
for epoch in range(32):  # loop over the dataset multiple times
    running_loss = 0.0
    net.train()

    for i, data in enumerate(train, 0):
        inputs, targets = data

        # zero the parameter gradients
        opt.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.to(device))

        loss = area_mse(outputs, targets.to(device))
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
        torch.save(net, 'models/04')
    else:
        early_stop_counter += 1

    if early_stop_counter>=3:
        break

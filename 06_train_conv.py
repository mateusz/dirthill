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
    size=n, stride=8, local_norm=True, full_boundary=True, ordered_boundary=True, boundary_overflow=3)
t,v = torch.utils.data.random_split(ts, [0.95, 0.05])
train = DataLoader(t, batch_size=1024, shuffle=True,
    num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=4)
val = DataLoader(v, batch_size=1024, shuffle=True,
    num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=4)

#%%

class MultiDimLinear(torch.nn.Linear):
    def __init__(self, in_features, out_shape, **kwargs):
        self.out_shape = out_shape
        out_features = np.prod(out_shape)
        super().__init__(in_features, out_features, **kwargs)

    def forward(self, x):
        out = super().forward(x)
        return out.reshape((len(x), *self.out_shape))

conv1 = torch.nn.Sequential(
        torch.nn.Conv1d(1, 32, 4, stride=4),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.01),

        torch.nn.Conv1d(32, 32, 4, stride=4, padding=2),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.01),

        torch.nn.Flatten(),

        torch.nn.Linear(1024, 4096),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.01),

        MultiDimLinear(4096, (16,16,16)),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.01),

        torch.nn.ConvTranspose2d(16,16,4,stride=2,padding=1),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.01),

        torch.nn.ConvTranspose2d(16,1,4,stride=4,padding=1),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.01),

        torch.nn.Flatten(),
)

conv1(torch.Tensor([ts[0][0], ts[1][0]]).unsqueeze(1)).shape

#%%

net = conv1.to(device)
opt = optim.Adam(net.parameters())
lossfn = nn.MSELoss()

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
        outputs = net(inputs.unsqueeze(1).to(device))

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
            outputs = net(inputs.unsqueeze(1).to(device))
            loss = lossfn(outputs, targets.to(device))
            running_loss += loss.item()

    vl = running_loss/len(val)
    print("val: %.2f" % (vl))

    if vl<min_val_loss:
        min_val_loss = vl
        early_stop_counter = 0
        print('saving...')
        torch.save(net, 'models/06')
    else:
        early_stop_counter += 1

    if early_stop_counter>=3:
        break
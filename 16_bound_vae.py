#%%

import terrain_set2
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib import cm

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%

n=128
boundl = 128
ts = terrain_set2.TerrainSet('data/USGS_1M_10_x43y465_OR_RogueSiskiyouNF_2019_B19.tif',
    size=n, stride=8)
t,v = torch.utils.data.random_split(ts, [0.9, 0.1])
train = DataLoader(t, batch_size=256, shuffle=True,
    num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=4)
val = DataLoader(v, batch_size=256, shuffle=True,
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

class VaeFull(nn.Module):
    def __init__(self):
        super().__init__()

        ch=16
        chd=16

        self.encoder = nn.Sequential(
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

            nn.Conv2d(ch*8, ch*16, 3, padding=1),
            nn.BatchNorm2d(ch*16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(ch*16, ch*32, 3, padding=1),
            nn.BatchNorm2d(ch*32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Flatten(),
        )

        latentl = 256
        self.mu1 = nn.Linear(ch*32*2*int(boundl/128), latentl)
        self.muR = nn.ReLU(inplace=True)
        self.mu2 = nn.Linear(latentl, latentl)
        self.logvar1 = nn.Linear(ch*32*2*int(boundl/128), latentl)
        self.logvarR = nn.ReLU(inplace=True)
        self.logvar2 = nn.Linear(latentl, latentl)

        self.decoder = nn.Sequential(
            nn.Linear(latentl, chd*32*2*2),
            nn.ReLU(inplace=True),

            nn.Unflatten(1, (chd*32, 2, 2)),
            
            nn.ConvTranspose2d(chd*32, chd*16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(chd*16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(chd*16, chd*8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(chd*8),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(chd*8, chd*4, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(chd*4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(chd*4, chd*2, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(chd*2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(chd*2, chd, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(chd),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(chd, 1, 3, stride=2, padding=1, output_padding=1),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        v = self.encoder(x)
        mu, logvar = self.mu2(self.muR(self.mu1(v))), self.logvar2(self.logvarR(self.logvar1(v)))
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar, z

net = torch.load('models/15-256')
decoder = net.decoder
decoder.eval()
size = 128

for i, param in enumerate(decoder.parameters()):
    param.requires_grad = False

print(decoder)
#%%

ch = 16
net = nn.Sequential(
    nn.Linear(boundl, 1024),
    nn.ReLU(True),
    nn.Linear(1024, 256),
    nn.ReLU(True),
    decoder,
).to(device)

net(torch.Tensor([ts[0][0][0:boundl], ts[1][0][0:boundl]]).to(device)).shape

#%%

opt = optim.Adam(net.parameters())
lossfn = nn.MSELoss()

min_val_loss = 9999999999.0
early_stop_counter = 0
for epoch in range(999):  # loop over the dataset multiple times
    running_loss = 0.0
    net.train()

    for i, data in enumerate(train, 0):
        inputs, targets = data
        inputs = inputs[:,0:boundl]

        # zero the parameter gradients
        opt.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.to(device))

        loss = lossfn(outputs, targets.unsqueeze(1).to(device))
        loss.backward()
        opt.step()

        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:
            print("train: %.2f" % (running_loss/10.0))
            running_loss = 0.0

    running_loss = 0.0
    net.eval()
    with torch.no_grad():
        for i,data in enumerate(val, 0):
            inputs, targets = data
            inputs = inputs[:,0:boundl]
            outputs = net(inputs.to(device))
            loss = lossfn(outputs, targets.unsqueeze(1).to(device))
            running_loss += loss.item()

    vl = running_loss/len(val)
    print("val: %.2f" % (vl))

    if vl<min_val_loss:
        min_val_loss = vl
        early_stop_counter = 0
        print('saving...')
        torch.save(net, 'models/16-%d'%boundl)
    else:
        early_stop_counter += 1

    if early_stop_counter>=3:
        break




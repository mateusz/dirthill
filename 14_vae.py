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

n=128
boundl=256
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

net = Net()
inp = torch.Tensor([ts[0][0][:boundl], ts[1][0][:boundl]])
print(inp.shape)
net(inp)[2].shape

#%%

net = net.to(device)
opt = optim.Adam(net.parameters())
mse = nn.MSELoss()

def kld(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def vaeloss(epoch, criterion, kld):
    if epoch==0:
        kld_weight = 0
    elif epoch==1:
        kld_weight = 0.0001
    else:
        kld_weight = 0.001

    return kld_weight*kld + criterion

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
        outputs, mu, logvar, z = net(inputs.to(device))

        criterion_loss = mse(outputs, targets.unsqueeze(1).to(device))
        kld_loss = kld(mu, logvar)
        loss = vaeloss(epoch, criterion_loss, kld_loss)
        loss.backward()
        opt.step()

        # print statistics
        running_loss += loss.item()
        running_kld += kld_loss.item()
        running_criterion += criterion_loss.item()
        if i % 10 == 9:
            print("train: l=%.3f, crit=%.3f, kld=%.3f" % (running_loss/10.0, running_criterion/10.0, kld_loss/10.0))
            running_loss = 0.0
            running_criterion = 0.0
            running_kld = 0.0

    running_loss = 0.0
    running_criterion = 0.0
    running_kld = 0.0
    net.eval()
    with torch.no_grad():
        for i,data in enumerate(val, 0):
            inputs, targets = data
            inputs = inputs[:,0:boundl]
            outputs, mu, logvar, z = net(inputs.to(device))
            criterion_loss = mse(outputs, targets.unsqueeze(1).to(device))
            kld_loss = kld(mu, logvar)
            loss = vaeloss(epoch, criterion_loss, kld_loss)

            running_criterion += criterion_loss.item()
            running_kld += kld_loss.item()
            running_loss += loss.item()

    vl = running_loss/len(val)
    print("val: l=%.3f, crit=%.3f, kld=%.3f" % (vl, running_criterion/len(val), kld_loss/len(val)))

    if vl<min_val_loss:
        min_val_loss = vl
        early_stop_counter = 0
        print('saving...')
        torch.save(net, 'models/14-%s' % boundl )
    else:
        early_stop_counter += 1

    if early_stop_counter>=3:
        break

# lv=256, 2-bound val: 8

#%%

input,target = ts[1000]
input = input[0:boundl]
out,mu,logvar,z = net(torch.Tensor([input]).to(device))

print("out    %.3f	%.3f	%.3f" % (torch.min(out),	torch.mean(out),	torch.max(out)))
print("mu     %.3f	%.3f	%.3f" % (torch.min(mu),	torch.mean(mu),	torch.max(mu)))
print("var    %.3f	%.3f	%.3f" % (torch.min(logvar.exp()),	torch.mean(logvar.exp()),	torch.max(logvar.exp())))
print("z      %.3f	%.3f	%.3f" % (torch.min(z),	torch.mean(z),	torch.max(z)))
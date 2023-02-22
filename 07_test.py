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
n=128
size=128
boundl = 256

def plot_surface(ax, data, cmap, alpha):
    meshx, meshy = np.meshgrid(np.linspace(0, size, size), np.linspace(0, size, size))

    ls = LightSource(270, 45)
    rgb = ls.shade(data, cmap=cmap, vert_exag=0.1, blend_mode='soft')
    _ = ax.plot_surface(meshx, meshy, data,
        facecolors=rgb, linewidth=0, antialiased=False, shade=False, alpha=alpha)

def plot_boundary(ax, data):
    ax.plot(
        np.full(size, 0),
        np.linspace(0, size-1, size),
        data[:,0],
        color="red", linewidth=2, zorder=100
    )

    if boundl<=128:
        return

    ax.plot(
        np.linspace(0, size-1, size),
        np.full(size, size),
        data[size-1, :],
        color="purple", linewidth=2, zorder=100
    )
    return
    ax.plot(
        np.full(size, size),
        np.linspace(0, size-1, size),
        data[:, size-1],
        color="purple", linewidth=2, zorder=100
    )
    ax.plot(
        np.linspace(0, size-1, size),
        np.full(size, 0),
        data[0,:],
        color="red", linewidth=2, zorder=100
    )

def show(target, out, r=35):
    _, ax = plt.subplots(2,2, subplot_kw=dict(projection='3d'), figsize=(10, 10))
    ax1, ax2, ax3, ax4 = ax.flatten()

    plot_surface(ax1, target, cm.gist_earth, 1.0)
    plot_surface(ax2, out, cm.gist_earth, 1.0)
    plot_boundary(ax1, target)
    plot_boundary(ax2, target)

    plot_surface(ax3, target, cm.gist_earth, 1.0)
    plot_surface(ax4, out, cm.gist_earth, 1.0)
    plot_boundary(ax3, target)
    plot_boundary(ax4, target)

    ax1.azim = 180+r
    ax2.azim = 180+r
    ax1.elev= 35
    ax2.elev= 35
    ax1.set_title('Truth')
    ax2.set_title('Model')

    ax3.azim = r
    ax4.azim = r
    ax3.elev= 35
    ax4.elev= 35
    ax3.set_title('Truth (back)')
    ax4.set_title('Model (back)')

    plt.show()

#%%

ts = terrain_set2.TerrainSet('data/USGS_1M_10_x43y465_OR_RogueSiskiyouNF_2019_B19.tif',
    size=n, stride=8)
tt = terrain_set2.TerrainSet('data/USGS_1M_10_x43y466_OR_RogueSiskiyouNF_2019_B19.tif',
    size=n, stride=8)
test = DataLoader(tt, batch_size=256, shuffle=True,
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

net = torch.load('models/06-%d'%boundl)
net.eval()
net = net.to(device)

#%%

running_loss = 0.0
lossfn = nn.MSELoss()
with torch.no_grad():
    for i,data in enumerate(test, 0):
        inputs, targets = data
        inputs = inputs[:,0:boundl]
        outputs = net(inputs.unsqueeze(1).to(device))
        loss = lossfn(outputs, targets.unsqueeze(1).to(device))
        running_loss += loss.item()

l = running_loss/len(test)
print("test: %.2f" % (l))

# onnx-ready 1 bound: 
# onnx-ready 2 bound: 45

#%%

with torch.no_grad():
    # 2800
    # 2000
    # 1700
    # 1400
    # 25000 - two rivers

    # second file
    # 1400
    # 2500
    # 2700
    # 2900
    # 3300
    # 3700
    # 4500
    # 4700
    # 5200 saddle
    # 5300 island
    # 5500 multiple rivers
    # 8300

    #input,target = ts[1400]
    #input = input[0:boundl]
    #out = net(torch.Tensor([input]).to(device)).cpu().squeeze(1)
    #show(target, out[0].numpy(), r=45)

    input,target = tt[2500]
    input = input[0:boundl]
    out = net(torch.Tensor([input]).unsqueeze(1).to(device)).cpu().squeeze(1)
    show(target, out[0].numpy(), r=45)
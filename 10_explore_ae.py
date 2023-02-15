#%%

import terrain_set
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import terrain_set
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib import cm

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

net = torch.load('models/08-full-ae-vl1.36')
encoder = net[:27]
decoder = net[27:]
encoder.eval()
decoder.eval()
size = 128

#%%

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
    ax.plot(
        np.linspace(0, size-1, size),
        np.full(size, 0),
        data[0,:],
        color="red", linewidth=2, zorder=100
    )

    ax.plot(
        np.full(size, size),
        np.linspace(0, size-1, size),
        data[:, size-1],
        color="purple", linewidth=2, zorder=100
    )
    ax.plot(
        np.linspace(0, size-1, size),
        np.full(size, size),
        data[size-1, :],
        color="purple", linewidth=2, zorder=100
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

_,target = ts[25001]

inp = torch.Tensor([target]).unsqueeze(1).to(device)
noise = torch.randn(1, 1, 128, 128).to(device)

with torch.no_grad():
    noisy_inp = inp+3*noise
    v = encoder(noisy_inp)
    out = decoder(v).cpu().squeeze(1)
    show(noisy_inp[0][0].cpu().numpy(), out[0].numpy(), r=45)


#%%

import terrain_set
import numpy as np
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib import cm
import torch
from torch import nn
import torch.nn.functional as F

size = 128
n = 128
stride = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%

ts = terrain_set.TerrainSet('data/USGS_1M_10_x43y466_OR_RogueSiskiyouNF_2019_B19.tif',
    size=size, stride=stride, local_norm=True, full_boundary=True, ordered_boundary=True, boundary_overflow=3)

#%%


def plot_surface(ax, data, cmap, alpha):
    meshx, meshy = np.meshgrid(np.linspace(1, size-1, size-2), np.linspace(1, size-1, size-2))

    ls = LightSource(270, 45)
    rgb = ls.shade(data, cmap=cmap, vert_exag=0.1, blend_mode='soft')
    _ = ax.plot_surface(meshx, meshy, data,
        facecolors=rgb, linewidth=0, antialiased=False, shade=False, alpha=alpha)

def plot_input(ax, data):
    ax.plot(
        np.full(size, 0),
        np.linspace(0, size-1, size),
        data[0:size],
        color="red", linewidth=2, zorder=100
    )
    ax.plot(
        np.linspace(0, size-1, size),
        np.full(size, 0),
        data[size:size*2],
        color="red", linewidth=2, zorder=100
    )

    ax.plot(
        np.full(size, size),
        np.linspace(0, size-1, size),
        data[size*2:size*3],
        color="purple", linewidth=2, zorder=100
    )
    ax.plot(
        np.linspace(0, size-1, size),
        np.full(size, size),
        data[size*3:size*4],
        color="purple", linewidth=2, zorder=100
    )

def show(input, target, out, r=45):
    _, ax = plt.subplots(2,2, subplot_kw=dict(projection='3d'), figsize=(10, 10))
    ax1, ax2, ax3, ax4 = ax.flatten()

    plot_surface(ax1, target, cm.gist_earth, 1.0)
    plot_surface(ax2, out, cm.gist_earth, 1.0)
    plot_input(ax1, input)
    plot_input(ax2, input)

    plot_surface(ax3, target, cm.gist_earth, 1.0)
    plot_surface(ax4, out, cm.gist_earth, 1.0)
    plot_input(ax3, input)
    plot_input(ax4, input)

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


net = torch.load('models/06')
net.eval()

#%%
with torch.no_grad():
    # 2800
    # 2000
    # 1700
    # 1400
    # 25001

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

    input,target = ts[5200]
    out = net(torch.Tensor([input]).unsqueeze(1).to(device)).cpu()

    sinput = np.concatenate((
        input[:size],
        np.flip(input[size*3-3:size*4-3]),
        np.flip(input[size*2-2:size*3-2]),
        input[size-1:size*2-1],
    ))
    show(sinput, target.reshape(size-2,size-2), out.reshape(size-2,size-2).numpy(), r=100)

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

ts = terrain_set.TerrainSet('data/USGS_1M_10_x43y465_OR_RogueSiskiyouNF_2019_B19.tif',
    size=size, stride=stride, local_norm=True, square_output=True)

#%%


def plot_surface(ax, data, cmap, alpha):
    meshx, meshy = np.meshgrid(np.linspace(0, size, size), np.linspace(0, size, size))

    ls = LightSource(270, 45)
    rgb = ls.shade(data, cmap=cmap, vert_exag=0.1, blend_mode='soft')
    _ = ax.plot_surface(meshx, meshy, data,
        facecolors=rgb, linewidth=0, antialiased=False, shade=False, alpha=alpha)

def show(target, out, r=45):
    _, ax = plt.subplots(2,2, subplot_kw=dict(projection='3d'), figsize=(10, 10))
    ax1, ax2, ax3, ax4 = ax.flatten()

    plot_surface(ax1, target, cm.gist_earth, 1.0)
    plot_surface(ax2, out, cm.gist_earth, 1.0)

    plot_surface(ax3, target, cm.gist_earth, 1.0)
    plot_surface(ax4, out, cm.gist_earth, 1.0)

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

#net = torch.load('models/08')
net = torch.load('models/08-full-ae-vl1.36')
net.eval()

#%%
with torch.no_grad():
    # 2800
    # 2000
    # 1700
    # 1400
    # 25001 - two rivers

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

    input,target = ts[25001]
    out = net(torch.Tensor([target]).unsqueeze(1).to(device)).cpu().squeeze(1)

    show(target, out[0].numpy(), r=45)

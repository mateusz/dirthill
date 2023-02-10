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
stride = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%

ts = terrain_set.TerrainSet('data/USGS_1M_10_x43y465_OR_RogueSiskiyouNF_2019_B19.tif',
    size=size, stride=stride, local_norm=True)

#%%
class Net(nn.Module):
    def __init__(self):
        h = 256
        dropout = 0.1
        super().__init__()
        self.l1 = nn.Linear(2*n-1,h)
        self.d1 = nn.Dropout(p=dropout)
        self.l2 = nn.Linear(h, n*n)
        self.d2 = nn.Dropout(p=dropout)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.d1(x)
        x = F.relu(self.l2(x))
        #x = self.d2(x)
        return x

def plot_surface(ax, data, cmap, alpha):
    # Assume square
    size = data.shape[0]

    meshx, meshy = np.meshgrid(np.linspace(0, size-1, size), np.linspace(0, size-1, size))

    ls = LightSource(270, 45)
    rgb = ls.shade(data, cmap=cmap, vert_exag=0.1, blend_mode='soft')
    _ = ax.plot_surface(meshx, meshy, data,
        facecolors=rgb, linewidth=0, antialiased=False, shade=False, alpha=alpha)

def plot_input(ax, data):
    edge1 = data[0:size]
    edge1x = np.full(size, 0)
    edge1y = np.linspace(0, size-1, size)

    edge2 = np.concatenate(([edge1[0]], data[size:2*size-1]))
    edge2x = np.linspace(0, size-1, size)
    edge2y = np.full(size, 0)

    ax.plot(edge1x, edge1y, edge1, color="red", linewidth=2)
    ax.plot(edge2x, edge2y, edge2, color="red", linewidth=2)

def show(input, target, out):

    _, (ax1, ax2) = plt.subplots(1,2, subplot_kw=dict(projection='3d'), figsize=(10, 5))
    plot_surface(ax1, target, cm.gist_earth, 0.7)
    plot_surface(ax2, out, cm.gist_earth, 0.7)
    plot_input(ax1, input)
    plot_input(ax2, input)

    ax1.azim = 225
    ax2.azim = 225
    ax1.elev= 35
    ax2.elev= 35
    ax1.set_title('Truth')
    ax2.set_title('Model')
    plt.show()


net = torch.load('models/simple')
net.eval()
with torch.no_grad():
    input,target = ts[720]
    out = net(torch.Tensor(input).to(device)).cpu()

    show(input, target.reshape(size,size), out.reshape(size,size).numpy())
    #show(target.reshape(size,size))
    #show(out.cpu().reshape(size,size).numpy())

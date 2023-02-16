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
from torch.utils.data import DataLoader

size = 128
n = 128
stride = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%

ts = terrain_set.TerrainSet('data/USGS_1M_10_x43y466_OR_RogueSiskiyouNF_2019_B19.tif',
    size=n, stride=8, local_norm=True, single_boundary=True)
test = DataLoader(ts, batch_size=1024, shuffle=True,
    num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=4)

#%%

net = torch.load('models/02')
net.eval()
lossfn = nn.MSELoss()

running_loss = 0.0
with torch.no_grad():
    for i,data in enumerate(test, 0):
        inputs, targets = data
        outputs = net(inputs.to(device))
        loss = lossfn(outputs, targets.to(device))
        running_loss += loss.item()

l = running_loss/len(test)
print("test: %.2f" % (l))

#%%
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

    #edge2 = np.concatenate(([edge1[0]], data[size:2*size-1]))
    #edge2x = np.linspace(0, size-1, size)
    #edge2y = np.full(size, 0)

    ax.plot(edge1x, edge1y, edge1, color="red", linewidth=2, zorder=100)
    #ax.plot(edge2x, edge2y, edge2, color="red", linewidth=2, zorder=100)

def show(input, target, out):

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

    ax1.azim = 225
    ax2.azim = 225
    ax1.elev= 35
    ax2.elev= 35
    ax1.set_title('Truth')
    ax2.set_title('Model')

    ax3.azim = 45
    ax4.azim = 45
    ax3.elev= 35
    ax4.elev= 35
    ax3.set_title('Truth (back)')
    ax4.set_title('Model (back)')
    plt.show()


#%%
with torch.no_grad():
    # 2800
    # 2000
    # 1700
    # 1400
    # 25000

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

    input,target = ts[2500]
    out = net(torch.Tensor(input).to(device)).cpu()

    show(input, target.reshape(size,size), out.reshape(size,size).numpy())
    #show(target.reshape(size,size))
    #show(out.cpu().reshape(size,size).numpy())


# notes:
# single-edge loss is 200
# it's pretty bad! - smooths everything into a line.
# How do we prevent it? one suggestion is to first predict opposite edge, or all edges, then fill in
# The other idea is to use pretrained AE decoder.
# TODO can we predict opposite edge, and then fill in the middle?
# TODO can we use LSTM to produce slices one by one, like in the paper?
# TODO can we predict slices iteratively starting from opposite, then the middle...
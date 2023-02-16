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
    size=size, stride=stride, local_norm=True, full_boundary=True)
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
class Net2(nn.Module):
    def __init__(self):
        h = 128
        h2 = 1024
        h3 = 2048
        h4 = 8192
        super().__init__()

        self.l1 = nn.Linear(4*n,h)
        self.l2 = nn.Linear(h,h2)
        self.l3 = nn.Linear(h2,h3)
        self.l4 = nn.Linear(h3,h4)

        self.l5 = nn.Linear(h4, (n-2)*(n-2))

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        return x

class Net(nn.Module):
    def __init__(self):
        h = 128
        h2 = 4096
        super().__init__()
        self.l1 = nn.Linear(4*n,h)
        self.l2 = nn.Linear(h,h2)
        self.l5 = nn.Linear(h2, (n-2)*(n-2))

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l5(x))
        #x = self.d2(x)
        return x

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


net = torch.load('models/04')
#net = torch.load('models/04-128_64_512_4096-d0')
#net = torch.load('models/04-128_4096-d0')
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
    out = net(torch.Tensor(input).to(device)).cpu()

    show(input, target.reshape(size-2,size-2), out.reshape(size-2,size-2).numpy())
    #show(target.reshape(size,size))
    #show(out.cpu().reshape(size,size).numpy())

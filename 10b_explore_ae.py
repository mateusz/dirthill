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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from matplotlib.colors import SymLogNorm

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%

n=128
ts = terrain_set.TerrainSet('data/USGS_1M_10_x43y465_OR_RogueSiskiyouNF_2019_B19.tif',
    size=n, stride=8, local_norm=True, square_output=True)

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
    #plot_boundary(ax1, target)
    #plot_boundary(ax2, target)

    plot_surface(ax3, target, cm.gist_earth, 1.0)
    plot_surface(ax4, out, cm.gist_earth, 1.0)
    #plot_boundary(ax3, target)
    #plot_boundary(ax4, target)

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

df = pd.read_parquet('data/ea-embeds.parquet')
pca = PCA(n_components=50)
p = pca.fit_transform(np.vstack(df['v'].to_numpy()))

#%%

pdf = pd.DataFrame({'pca': p.tolist()})
pdf.to_parquet('data/pca50.parquet')

#%%

pdf = pd.read_parquet('data/pca50.parquet')
p = np.vstack(pdf['pca'].to_numpy())
#%%

pr = p.reshape((50,len(p)))
plt.hist2d(pr[49], pr[48], bins=1000)
plt.ylim(-50,50)
plt.xlim(-50,50)
plt.show()

#%%

pr = p.reshape((50,len(p)))

hists = np.apply_along_axis(lambda a: np.histogram(a, bins=np.linspace(-25,25,101)), 1, pr)
x = np.vstack(hists[:,1])[:,:-1]
y = np.vstack(hists[:,0])

plt.imshow(y)
plt.xticks(np.arange(0,100,1)[0::16], x[0][0::16])
plt.title("First 50 PCA dimensions")
plt.show()

#%%

df = pd.read_parquet('data/ea-embeds.parquet')
v = np.vstack(df['v'].to_numpy())

#%%

vr = v.reshape((256,len(v)))

hists = np.apply_along_axis(lambda a: np.histogram(a, bins=np.linspace(-10,150,161)), 1, vr)
x = np.vstack(hists[:,1])[:,:-1]
y = np.log(np.vstack(hists[:,0]))

plt.imshow(y)
plt.xticks(np.arange(0,160,1)[0::32], x[0][0::32])
plt.title("All embedding dims")
plt.show()

#%%

np.count_nonzero(v)/len(v.flatten())

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

    input,target = ts[25000]
    wiped = np.random.rand(128,128)*5+25
    res = 2
    wiped[0::res,0::res] = target[0::res,0::res]
    out = net(torch.Tensor([wiped]).unsqueeze(1).to(device)).cpu().squeeze(1)
    show(wiped, out[0].numpy(), r=45)

#%%

df = pd.read_parquet('data/ea-embeds.parquet').sample(n=256)
p = np.vstack(df['v'].to_numpy())

#%%

tsne = TSNE(n_components=2)
t = tsne.fit_transform(p)

tr = t.reshape((2,len(t)))
plt.scatter(tr[0], tr[1], cmap='hot')
plt.show()

#%%
_,ax = plt.subplots(figsize=(10,10))
ax = sns.heatmap(np.vstack(p), linewidth=0.0, cmap=sns.color_palette("dark:salmon", as_cmap=True))#, norm=SymLogNorm(10.0))
plt.ylabel('samples')
plt.xlabel('latent channels')
plt.title('AE latent space')
plt.show()
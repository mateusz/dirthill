#%%
import numpy as np
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib import cm

#%%

img = rasterio.open('data/USGS_1M_10_x43y465_OR_RogueSiskiyouNF_2019_B19.tif')
data = img.read(1)

#%%
print("%d x %d" % (img.width, img.height))

flat = data.flatten()
m = np.ma.MaskedArray(flat, flat<-10000)
global_min = np.ma.min(m)
print(global_min)

normed = data - global_min
#%%

n = 128
plt.hist(normed[0:n,9000:9000+n].flatten())
plt.show()

show(normed[0:n,9000:9000+n])
plt.show()
show(normed[0:n,9000:9000+n], contour=True)
plt.show()

#%%

side = 512
stride = 128
nan_threshold = -10000
meshx, meshy = np.meshgrid(np.linspace(0, side-1, side), np.linspace(0, side-1, side))
for x in range(0, img.width, stride):
    for y in range(0, img.height, stride):
        square = normed[x:x+side, y:y+side]
        if np.min(square)<nan_threshold:
            continue

        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(5, 5))
        ls = LightSource(270, 45)
        rgb = ls.shade(square, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
        surf = ax.plot_surface(meshx, meshy, square,
            facecolors=rgb, linewidth=0, antialiased=False, shade=False)
        ax.azim = -45
        ax.elev= 45
        plt.show()
    break

#%%
import numpy as np
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt

#%%

img = rasterio.open('data/USGS_1M_10_x43y465_OR_RogueSiskiyouNF_2019_B19.tif')
data = img.read(1)

#%%
print("%d x %d" % (img.width, img.height))

flat = data.flatten()
m = np.ma.MaskedArray(flat, flat<-10000)
global_min = np.ma.min(m)
print(global_min)

#%%

n = 128
normed = data - global_min
plt.hist(normed[0:n,9000:9000+n].flatten())
plt.show()

show(normed[0:n,9000:9000+n])
plt.show()
show(normed[0:n,9000:9000+n], contour=True)
plt.show()
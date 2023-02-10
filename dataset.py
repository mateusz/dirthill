#%%
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import numpy as np
import rasterio
import pandas as pd

#%%
class TerrainSet(Dataset):
    def __init__(self, file, size, stride, nan_threshold=-10000, global_norm=False, local_norm=False):
        self.size = size
        self.local_norm = local_norm

        img = rasterio.open(file)
        data = img.read(1)

        if global_norm:
            flat = data.flatten()
            m = np.ma.MaskedArray(flat, flat<-10000)
            global_min = np.ma.min(m)

            self.data = data - global_min
        else:
            self.data = data

        index = []
        for x in range(0, img.width-size, stride):
            for y in range(0, img.height-size, stride):
                square = self.data[x:x+size, y:y+size]
                if np.min(square)<nan_threshold:
                    continue

                index.append([x,y])

        self.index = pd.DataFrame(index, columns=['x','y'])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        item = self.index.iloc[idx]

        d = self.data[item.x:item.x+self.size, item.y:item.y+self.size]
        print(d.shape)

        if self.local_norm:
            # Slower by 50%, could be optimised by preprocessing
            return d - np.min(d)
        else:
            return d

ts = TerrainSet('data/USGS_1M_10_x43y465_OR_RogueSiskiyouNF_2019_B19.tif', 128, 32, local_norm=True)

tsdl = DataLoader(ts, batch_size=32, shuffle=True)

elapseds = []
start = datetime.now()
steps = 0
for x in tsdl:
    steps += 1
    elapsed = datetime.now()-start
    start = datetime.now()
    elapseds.append(elapsed.total_seconds()*1000.0)

print('%.4fms' % (np.mean(elapseds)))
# 4ms for globally normed, 6ms for locally normed

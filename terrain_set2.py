#%%
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import numpy as np
import rasterio
import pandas as pd
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

#%%
class TerrainSet(Dataset):
    def __init__(self, file, size, stride,
        nan_threshold=-10000, n=0, return_base=False,
        rescale=1,
    ):
        self.size = size
        self.stride = stride
        self.nan_threshold = nan_threshold
        self.n = n
        self.return_base = return_base

        img = rasterio.open(file)
        data = img.read(1)

        if rescale>1:
            data = F.max_pool2d(torch.Tensor(data).unsqueeze(0), rescale).squeeze(0).numpy()

        self.data = data

        index = []
        lim = 0
        if n>0:
            lim = n

        width = self.data.shape[1]
        height = self.data.shape[0]
        for x in range(0, width-size, stride):
            for y in range(0, height-size, stride):
                square = self.data[x:x+size, y:y+size]
                if np.min(square)<nan_threshold:
                    continue

                index.append([x,y])
                lim -= 1
                if n>0 and lim<1:
                    break

            if n>0 and lim<1:
                break

        self.index = pd.DataFrame(index, columns=['x','y'])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        item = self.index.iloc[idx]

        d = self.data[item.x:item.x+self.size, item.y:item.y+self.size]

        # Normalise to -1.0-1.0
        base = np.min(d)
        span = (np.max(d) - base)/2.0
        d = (d - base)/span - 1.0

        b = np.concatenate((
            d[:, 0],
            d[-1, :],
            np.flip(d[:, -1]),
            np.flip(d[0, :]),
        ))
        t = d

        if self.return_base:
            return [
                b,
                t,
                base,
            ]
        else:
            return [
                b,
                t,
            ]

"""
ts = TerrainSet('data/USGS_1M_10_x43y465_OR_RogueSiskiyouNF_2019_B19.tif',
    size=128, stride=8, rescale=4)

d = ts[0][1]
plt.imshow(d)
plt.show()

"""
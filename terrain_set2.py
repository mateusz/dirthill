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
        rescale=1, min_elev_diff=10.0,
    ):
        self.size = size
        self.stride = stride
        self.nan_threshold = nan_threshold
        self.n = n
        self.return_base = return_base
        self.rescale = rescale
        self.min_elev_diff = min_elev_diff
        self.data = {}
        self.index = pd.DataFrame([], columns=['file','x','y'])

        for f in file:
            self.preproc(f)

    def preproc(self, file):
        img = rasterio.open(file)
        data = img.read(1)

        if self.rescale>1:
            data = F.max_pool2d(torch.Tensor(data).unsqueeze(0), self.rescale).squeeze(0).numpy()

        self.data[file] = data

        index = []
        lim = 0
        if self.n>0:
            lim = self.n

        width = self.data[file].shape[1]
        height = self.data[file].shape[0]
        for x in range(0, width-self.size, self.stride):
            for y in range(0, height-self.size, self.stride):
                square = self.data[file][x:x+self.size, y:y+self.size]

                # Check for missing data
                if np.min(square)<self.nan_threshold:
                    continue

                # Check for boring areas (too flat):
                if np.max(square) - np.min(square) < self.min_elev_diff:
                    continue

                index.append([file,x,y])
                lim -= 1
                if self.n>0 and lim<1:
                    break

            if self.n>0 and lim<1:
                break

        self.index = pd.concat([self.index, pd.DataFrame(index, columns=['file','x','y'])])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        item = self.index.iloc[idx]

        d = self.data[item.file][item.x:item.x+self.size, item.y:item.y+self.size]

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
ts = TerrainSet([
        'data/USGS_1M_10_x43y465_OR_RogueSiskiyouNF_2019_B19.tif', # good at 4, maybe 6
        #'data/USGS_1M_10_x49y452_CA_CarrHirzDeltaFires_2019_B19.tif',
        #'data/USGS_1M_10_x51y489_OR_McKenzieRiver_2021_B21.tif',
        #'data/USGS_1M_10_x51y524_WA_PierceCounty_2020_A20.tif',
        #'data/USGS_1M_10_x58y418_CA_AlamedaCounty_2021_B21.tif',
        #'data/USGS_1M_10_x67y517_WA_EasternCascades_2019_B19.tif',
        #'data/USGS_1M_11_x22y417_CA_SouthernSierra_2020_B20.tif', # good at 4
    ],
    size=128, stride=8, rescale=6)
#%%
d = ts[3300][1]
plt.imshow(d)
plt.show()
"""
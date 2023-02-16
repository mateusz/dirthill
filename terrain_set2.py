#%%
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import numpy as np
import rasterio
import pandas as pd

#%%
class TerrainSet(Dataset):
    def __init__(self, file, size, stride,
        nan_threshold=-10000, n=0,
    ):
        self.size = size
        self.stride = stride
        self.nan_threshold = nan_threshold
        self.n = n

        img = rasterio.open(file)
        data = img.read(1)

        self.data = data

        index = []
        lim = 0
        if n>0:
            lim = n
        for x in range(0, img.width-size, stride):
            for y in range(0, img.height-size, stride):
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
        d = d - np.min(d)

        b = np.concatenate((
            d[:, 0],
            d[-1, :],
            np.flip(d[:, -1]),
            np.flip(d[0, :]),
        ))
        t = d

        return [
            b,
            t,
        ]
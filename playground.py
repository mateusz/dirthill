#%%

import torch

#%%
x = torch.tensor([1,2,1])
y = torch.tensor([1,2,1])

gx, gy = torch.meshgrid(x, y, indexing='ij')
(gx*gy).shape

#%% 
batch = 1
m = torch.nn.Conv1d(2, 1, kernel_size=1)
input = torch.randn(batch, 2, 4)
m(input)

#%%


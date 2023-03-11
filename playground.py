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

t = torch.tensor([
        [
            [1,1,1,-1,-3,-1],
            [1,2,1,1,4,1],
        ],
        [
            [1,5,1,1,7,1],
            [1,6,1,1,8,1],
        ]
])

d = t.shape[2]
x = t[:,:,:d//2].exp()
y = t[:,:,d//2:].exp()

torch.einsum('bci,bcj->bcij', x, y).log()


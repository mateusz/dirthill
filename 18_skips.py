#%%

import terrain_set2
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n=128
boundl=256
rescale=4
mname='18-%d-%d' % (boundl, rescale)

report_steps = 500

batch=128
#%%
ts = terrain_set2.TerrainSet([
        'data/USGS_1M_10_x43y465_OR_RogueSiskiyouNF_2019_B19.tif',
        'data/USGS_1M_10_x43y466_OR_RogueSiskiyouNF_2019_B19.tif',
        'data/USGS_1M_10_x44y465_OR_RogueSiskiyouNF_2019_B19.tif',
        'data/USGS_1M_10_x45y466_OR_RogueSiskiyouNF_2019_B19.tif',
        'data/USGS_1M_10_x46y466_OR_RogueSiskiyouNF_2019_B19.tif',
        'data/USGS_1M_10_x46y467_OR_RogueSiskiyouNF_2019_B19.tif',
        'data/USGS_1M_10_x47y465_OR_RogueSiskiyouNF_2019_B19.tif',
        'data/USGS_1M_10_x47y466_OR_RogueSiskiyouNF_2019_B19.tif',
        'data/USGS_1M_10_x49y465_OR_RogueSiskiyouNF_2019_B19.tif',
        'data/USGS_1M_10_x49y466_OR_RogueSiskiyouNF_2019_B19.tif',
    ],
    size=n, stride=8, rescale=rescale, min_elev_diff=20.0)
t,v = torch.utils.data.random_split(ts, [0.9, 0.1])
train = DataLoader(t, batch_size=batch, shuffle=True,
    num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=4)
val = DataLoader(v, batch_size=batch, shuffle=True,
    num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=4)

print("%d" % len(ts))
print("%d, %d" % (len(train)*batch, len(val)*batch))

#%%

class View(nn.Module):
    def __init__(self, dim,  shape):
        super(View, self).__init__()
        self.dim = dim
        self.shape = shape

    def forward(self, input):
        new_shape = list(input.shape)[:self.dim] + list(self.shape) + list(input.shape)[self.dim+1:]
        return input.view(*new_shape)


# https://github.com/pytorch/pytorch/issues/49538
nn.Unflatten = View

#https://towardsdatascience.com/understanding-u-net-61276b10f360

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, layers=3):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.layers = layers

        ch = in_ch
        self.enc = nn.ModuleList()
        for _ in range(0, layers, 1):
            self.enc.append(nn.Conv1d(ch, out_ch, 3, padding=1))
            self.enc.append(nn.BatchNorm1d(out_ch))
            self.enc.append(nn.ReLU(inplace=True))
            ch = out_ch
        
        self.down = nn.Sequential(
            nn.MaxPool1d(2),
        )

    def forward(self, x):
        for l in self.enc:
            x = l(x)

        d = self.down(x)
        return x, d

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, layers=3, skip=True):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.skip = skip

        if skip:
            ch = in_ch*2
        else:
            ch = in_ch

        self.dec = nn.ModuleList()
        for _ in range(0, layers-1, 1):
            self.dec.append(nn.Conv2d(ch, out_ch, 3, padding=1))
            self.dec.append(nn.BatchNorm2d(out_ch))
            self.dec.append(nn.ReLU(inplace=True))
            ch = out_ch

        if layers==1:
            ch = in_ch
            if skip:
                ch = ch*2
        else:
            ch = out_ch

        self.dec.append(nn.ConvTranspose2d(ch, out_ch, 3, stride=2, padding=1, output_padding=1))
        self.dec.append(nn.BatchNorm2d(out_ch))
        self.dec.append(nn.ReLU(inplace=True))

    def forward(self, x, s=False):

        if self.skip:
            # We can't simply apply skip connectivity, because shapes don't match.
            # On the input side, we have two cross-sections (2*n) but we need (n*n).
            # This increases dimensionality by multiplying out relevant terms
            d = s.shape[2]
            sx = s[:,:,:d//2].exp()
            sy = s[:,:,d//2:].exp()
            up_d = torch.einsum('bci,bcj->bcij', sx, sy).log()

            print(up_d.shape, x.shape)

            c = torch.cat((up_d, x), dim=1)
        else:
            c = x

        x = self.dec[0](c)
        for l in self.dec[1:]:
            x = l(x)

        return x

class Model(nn.Module):
    def __init__(self, layers=6, block_layers=3, channels=16, boundl=256, outl=128, bottleneck=512, dropout=0.1, skip=True):
        super().__init__()

        self.layers = layers
        self.channels = channels
        self.boundl = boundl

        self.pre = nn.Unflatten(1, (1, boundl))

        size = boundl
        self.enc = nn.ModuleList()
        self.enc.append(EncoderBlock(1, channels, layers=block_layers))
        size = size//2

        for i in range(0, layers-1, 1):
            channels = channels*2
            size = size//2
            self.enc.append(EncoderBlock(channels//2, channels, layers=block_layers))

        print('inner shape: %dx%d -> %dx%dx%d (%d->%d->%d)' % (channels,size,channels,size//2,size//2,channels*size,bottleneck,channels*size))
        self.bottleneck = nn.Sequential(
            nn.Flatten(),

            nn.Linear(channels*size, bottleneck),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(bottleneck, channels*size),
            nn.Unflatten(1, (channels, size//2, size//2)),

            # No skip available
            # Note perhaps we can increase above Linear to channels*size*size and skip this,
            # but historically this hasn't worked well.
            DecoderBlock(channels, channels, layers=block_layers, skip=False),
        )

        self.dec = nn.ModuleList()
        # Note due to varying number of cross-sections, we might need to do less upscaling
        # For example input two boundaries (256), we need to upscale one less time to get to 128.
        for _ in range(0, layers-(boundl//outl), 1):
            self.dec.append(DecoderBlock(channels, channels//2, layers=block_layers, skip=skip))
            channels = channels//2
        
        self.post = nn.ConvTranspose2d(channels, 1, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.pre(x)
        skips = []

        for l in self.enc:
            s,x = l(x)
            skips.append(s)

        x = self.bottleneck(x)

        # We don't want the deepest one, which is really the a part of the bottleneck
        for l in self.dec:
            s = skips.pop()
            x = l(x,s)

        x = self.post(x)

        return x


#batch = 4
#channels = 16
#input1d = 256
net = Model(layers=6, channels=16, boundl=boundl, bottleneck=512, block_layers=1, skip=False)
#net(torch.randn(batch, input1d)).shape
inp = torch.Tensor([ts[0][0][:boundl], ts[1][0][:boundl]])
print(net(inp).shape)


#%%

net = net.to(device)
opt = optim.Adam(net.parameters())
#lossfn = nn.MSELoss()
lossfn = nn.HuberLoss(delta=0.25)

min_val_loss = 9999999999.0
early_stop_counter = 0
for epoch in range(999):  # loop over the dataset multiple times
    running_loss = 0.0
    net.train()

    for i, data in enumerate(train, 0):
        inputs, targets = data
        inputs = inputs[:,0:boundl]

        # zero the parameter gradients
        opt.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.to(device))

        loss = lossfn(outputs, targets.unsqueeze(1).to(device))
        loss.backward()
        opt.step()

        # print statistics
        running_loss += loss.item()
        if i % report_steps == report_steps-1:
            print("train: %.4f" % (running_loss/float(report_steps)))
            running_loss = 0.0

    running_loss = 0.0
    net.eval()
    with torch.no_grad():
        for i,data in enumerate(val, 0):
            inputs, targets = data
            inputs = inputs[:,0:boundl]
            outputs = net(inputs.to(device))
            loss = lossfn(outputs, targets.unsqueeze(1).to(device))
            running_loss += loss.item()

    vl = running_loss/len(val)
    print("val: %.4f" % (vl))

    if vl<min_val_loss:
        min_val_loss = vl
        early_stop_counter = 0
        print('saving and exporting model...')
        torch.save(net, 'models/%s' % (mname) )

        evalnet = torch.load('models/%s' % (mname)).eval()
        dummy_input = torch.randn(1, boundl, device="cuda")
        input_names = [ "edge" ]
        output_names = [ "tile" ]
        torch.onnx.export(
            evalnet, dummy_input, "ui/dist/%s.onnx" % (mname),
            verbose=False, input_names=input_names, output_names=output_names)
    else:
        early_stop_counter += 1

    if early_stop_counter>=3:
        break

#%%

tt = terrain_set2.TerrainSet([
        # https://www.sciencebase.gov/catalog/item/60d5632cd34ef0ccfc0c8583
        'data/USGS_1M_10_x50y466_OR_RogueSiskiyouNF_2019_B19.tif',
    ],
    size=n, stride=8, rescale=rescale)
test = DataLoader(tt, batch_size=batch, shuffle=True,
    num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=4)

running_loss = 0.0
lossfn = nn.MSELoss()
with torch.no_grad():
    for i,data in enumerate(test, 0):
        inputs, targets = data
        inputs = inputs[:,0:boundl]
        outputs = net(inputs.to(device))
        loss = lossfn(outputs, targets.unsqueeze(1).to(device))
        running_loss += loss.item()

l = running_loss/len(test)
print("test: %.4f" % (l))

# data/USGS_1M_10_x43y465_OR_RogueSiskiyouNF_2019_B19.tif - what is so special about this file that the network converges?
# is it simply small?

# 20k batch 64 delta 0.25
# Model(layers=6, channels=16, boundl=boundl, bottleneck=512)
# val 0.0022 test3 0.0852

# 20k batch 64 delta 0.25
# Model(layers=8, channels=4, boundl=boundl, bottleneck=512)
# val 0.0528 test3 0.1815

# 20k batch 64 delta 0.25
# Model(layers=8, channels=4, boundl=boundl, bottleneck=512, block_layers=1)
# also stuck on 0.05 loss, not converging further

# 20k batch 64 delta 0.25
# Model(layers=6, channels=16, boundl=boundl, bottleneck=1024, dropout=0.5)
# val 0.0036 test3 0.0866 as 18-256-4-1.onnx this is actualy quite good, and requires less data...

# 20k batch 128 delta 0.25
# Model(layers=6, channels=16, boundl=boundl, bottleneck=512)
# 10 files converge very slowly and seem stuck at 0.05

# how about back to 10 files, but stride 20?
# 100k batch 64
# does not converge

# 87k 7 various files batch 64, stride 20
# val 0.0290 test 0.0642 bad

# 88k, 1 file, stride 8, but multiply skip by 0.1
# not converging

# 88k, 1 file, stride 8 
# Model(layers=6, channels=8, boundl=boundl, bottleneck=512) - less channels
# val 0.0055 test3 0.0620 as dist/18-256-4-2.onnx kinda ok but still sad

# I think this is a lost cause :) Move on!
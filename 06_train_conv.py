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
#%%

n=128
boundl=256
rescale=4
mname='06-%d-%d' % (boundl, rescale)

report_steps = 500

#batch=256//rescale
batch=128
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
        #'data/USGS_1M_10_x49y467_OR_RogueSiskiyouNF_2019_B19.tif',
        #'data/USGS_1M_10_x50y465_OR_RogueSiskiyouNF_2019_B19.tif',
    ],
    size=n, stride=8, rescale=rescale, min_elev_diff=20.0)
# Random sampling to reduce the amount of training tiles and prevent excessive smoothing
#sampled,_ = torch.utils.data.random_split(ts, [0.2, 0.8])
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

# Todo:
# - maybe just go back to convolving and then using linear (to not muddle the results with upscaler issues)
# - try bigger kernel (8 didnt change anything), but maybe 32 or something?
ch=16
chd=16
conv1 = nn.Sequential(
        nn.Unflatten(1, (1, boundl)),

        nn.Conv1d(1, ch, 3, padding=1),
        nn.BatchNorm1d(ch),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(2),

        nn.Conv1d(ch, ch*2, 3, padding=1),
        nn.BatchNorm1d(ch*2),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(2),

        nn.Conv1d(ch*2, ch*4, 3, padding=1),
        nn.BatchNorm1d(ch*4),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(2),

        nn.Conv1d(ch*4, ch*8, 3, padding=1),
        nn.BatchNorm1d(ch*8),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(2),

        nn.Conv1d(ch*8, ch*16, 3, padding=1),
        nn.BatchNorm1d(ch*16),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(2),

        nn.Conv1d(ch*16, ch*32, 3, padding=1),
        nn.BatchNorm1d(ch*32),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(2),

        nn.Flatten(),

        nn.Linear(ch*32*2*int(boundl/128), 512),
        nn.ReLU(inplace=True),
        # This prevents instability in UI usage (otherwise single value changes blow up the output!)
        nn.Dropout(0.1),

        nn.Linear(512, chd*32*2*2),
        nn.Unflatten(1, (chd*32, 2, 2)),
        
        nn.ConvTranspose2d(chd*32, chd*16, 3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(chd*16),
        nn.ReLU(inplace=True),

        nn.ConvTranspose2d(chd*16, chd*8, 3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(chd*8),
        nn.ReLU(inplace=True),
        
        nn.ConvTranspose2d(chd*8, chd*4, 3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(chd*4),
        nn.ReLU(inplace=True),

        nn.ConvTranspose2d(chd*4, chd*2, 3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(chd*2),
        nn.ReLU(inplace=True),

        nn.ConvTranspose2d(chd*2, chd, 3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(chd),
        nn.ReLU(inplace=True),

        nn.ConvTranspose2d(chd, 1, 3, stride=2, padding=1, output_padding=1),

)

inp = torch.Tensor([ts[0][0][:boundl], ts[1][0][:boundl]])
conv1(inp).shape

print(conv1)

#%%

net = conv1.to(device)
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
        print('saving...')
        torch.save(net, 'models/%s' % (mname) )
    else:
        early_stop_counter += 1

    if early_stop_counter>=3:
        break

# 2-bound val: 8


net = torch.load('models/%s' % (mname)).eval()
print(net)

dummy_input = torch.randn(1, boundl, device="cuda")
input_names = [ "edge" ]
output_names = [ "tile" ]

torch.onnx.export(
    net, dummy_input, "ui/dist/%s.onnx" % (mname),
    verbose=True, input_names=input_names, output_names=output_names)

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

# boundl 2 rescale 4 latent 1024 val: 0.0085, test: 0.1168, 25MB
# boundl 2 rescale 4 latent 2048 val: 0.0086, test: 0.1307, 40MB
# the models above are able to "simulate" rivers - how the water flows from high to low ground. Up to two rivers per side

# On 7 files:
# min_elev_diff 10, ch 16, latent 2048, dropout 0.5: val 0.0322, test ?, result flat and boring - maybe too much flatness in the data?
# min_elev_diff 10, ch 32, latent 1024, dropout 0.0: val 0.05, test 0.13, same flat

# min_elev_diff 20, ch 32, latent 1024, dropout 0.5, rescale 8, batch 32: val 0.0328
# squares 64k min_elev_diff 20, ch 32, latent 1024, dropout 0.5, rescale 10, batch 64: val 0.0184
# squares 40k min_elev_diff 20, ch 32, latent 1024, dropout 0.5, rescale 12, batch 64: val 0.0228

# The idea here I think is that the normalised boundary of 256 can only represent a limited amount of squares.
# If too many squares are in the training set, system starts to average things out.
# We do want to capture some of the input idiosyncracies, so the set has to be curbed either via upscaling, or by sampling.
# At the same time, rescale of 8-10 gives good results, while 12 starts losing detail
# Single-file model used 20k squares. 65k with rescale 10 seemed ok

# squares 60k min_elev_diff 20, ch 32, latent 1024, dropout 0.5, rescale 8, batch 64: val 0.0446
# squares 25k min_elev_diff 20, ch 32, latent 1024, dropout 0.5, rescale 8, batch 64: val 0.0571
# squares 100k min_elev_diff 20, ch 32, latent 1024, dropout 0.5, rescale 8, batch 32: val 0.0275, test2 0.0854

# Hm so maybe it's the batches? Potentially together with dropout, we can learn more variety, but is this even deep learning?
# Or just remembering the dataset in a fancy way? If we just wanted to upscale, but not increase variety, we could instead use
# more data for the same region, maybe this would help generalisation?

# Maybe it's around 512 neurons per 100k tiles at 4 rescale. Batch 32 seems ok.
# So, 1024 neurons with 0.5 dropout is ok for 100k and 4096 neurons with 0.875 dropout is ok, something like that.
# Note test file change

# squares 500k min_elev_diff 20, ch 32, latent 4k, dropout 0.875, rescale 4, batch 32: val 0.0309, test2 0.0525. But effect is boring.

# Back to roguesiskiyou-only, 10x files. Note test file change.
# squares 250k min_elev_diff 20, ch 32, latent 2k, dropout 0.75, rescale 6, batch 32: val 0.0183, test3 0.0587, kind of ok
# squares 600k min_elev_diff 20, ch 16, latent 2k, dropout 0.85, rescale 4, batch 32: val 0.0305, test3 0.0439, too flat

# squares 600k min_elev_diff 20, ch 16, latent 512, dropout 0.0, rescale 4, batch 32: val 0.0160, test3 0.0523, quite good edge matching, comparable with 1x model visually, best so far I think, but a few square artifacts and spikes saved as ui/dist/06-256-4-10x-good.onnx
# to try: huber loss, dropout 0.01 to smoothen slightly and avoid spikes, smaller batches
# squares 600k min_elev_diff 20, ch 16, latent 512, dropout 0.0, rescale 4, batch 32, huber loss 0.25: val 0.0076, test3 0.0526, ok but quite spiky. saved as ui/dist/06-256-4-10x-huber.onnx

# So it looks like increasing the dataset 10x allowed us to remove dropout and train directly. Smaller batch sizes, removal of dropout and huber loss
# are all designed how opinionated the model is, which seems to be working. Model is able to fit edges better and shapes seem more interesting.
# I think at this point, we can conclude that 1x dataset with 1024 neurons and 0.5 dropout is good for experimentation with architectures, and the
# results can then be improved by using 10x dataset and removal of dropout.

# Fitting bounds better but too many artifacts. Let's re-add a bit of dropout.
# squares 600k min_elev_diff 20, ch 16, latent 512, dropout 0.1, rescale 4, batch 64, huber loss 0.25: val 0.0085, test 0.0488, better, ui/dist/06-256-4-1.onnx
# squares 600k min_elev_diff 20, ch 16, latent 512, dropout 0.1, rescale 4, batch 128, huber loss 0.25: val 0.0087, test 0.0499, better, ui/dist/06-256-4-2.onnx good stuff
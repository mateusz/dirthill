#%%

import terrain_set2
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%

n=128
boundl=256
rescale=4
mname='14-%d-%d' % (boundl, rescale)
report_steps = 50

batch=256//rescale
ts = terrain_set2.TerrainSet(['data/USGS_1M_10_x46y466_OR_RogueSiskiyouNF_2019_B19.tif'],
    size=n, stride=8, rescale=rescale)
t,v = torch.utils.data.random_split(ts, [0.90, 0.10])
train = DataLoader(t, batch_size=batch, shuffle=True,
    num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=4)
val = DataLoader(v, batch_size=batch, shuffle=True,
    num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=4)

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

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        ch=16
        chd=16

        self.encoder = nn.Sequential(
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
        )

        latentl = 512
        self.mu1 = nn.Linear(ch*32*2*int(boundl/128), latentl)
        self.muR = nn.ReLU(inplace=True)
        self.mu2 = nn.Linear(latentl, latentl)
        self.logvar1 = nn.Linear(ch*32*2*int(boundl/128), latentl)
        self.logvarR = nn.ReLU(inplace=True)
        self.logvar2 = nn.Linear(latentl, latentl)

        self.decoder = nn.Sequential(
            nn.Linear(latentl, chd*32*2*2),
            nn.ReLU(inplace=True),

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

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        v = self.encoder(x)
        mu, logvar = self.mu2(self.muR(self.mu1(v))), self.logvar2(self.logvarR(self.logvar1(v)))
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar, z

net = Net()
inp = torch.Tensor([ts[0][0][:boundl], ts[1][0][:boundl]])
print(inp.shape)
net(inp)[2].shape

#%%

net = net.to(device)
opt = optim.Adam(net.parameters())

def kld(mu, logvar):
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)

def vaeloss(epoch, criterion, kld, annealing=[0.000001, 0.00001, 0.0001]):#[0.0, 0.0001, 0.001]):
    if epoch<len(annealing)-1:
        kld_weight = annealing[epoch]
    else:
        kld_weight = annealing[-1]

    return kld_weight*kld + criterion

ssim_module = SSIM(data_range=1, size_average=True, channel=1)
#crit_module = nn.HuberLoss(delta=0.25)
crit_module = nn.L1Loss()

min_val_loss = 9999999999.0
early_stop_counter = 0
for epoch in range(999):  # loop over the dataset multiple times
    running_loss = 0.0
    running_criterion = 0.0
    running_kld = 0.0
    running_perc = 0.0
    net.train()

    for i, data in enumerate(train, 0):
        inputs, targets = data
        inputs = inputs[:,0:boundl]

        # zero the parameter gradients
        opt.zero_grad()

        # forward + backward + optimize
        outputs, mu, logvar, z = net(inputs.to(device))

        perc_loss = 1.0 - ssim_module((outputs+1.0)/2.0, (targets.unsqueeze(1).to(device)+1.0)/2.0)
        criterion_loss = crit_module(outputs, targets.unsqueeze(1).to(device))
        kld_loss = kld(mu, logvar)
        loss = vaeloss(epoch, criterion_loss, kld_loss)# + perc_loss
        loss.backward()
        opt.step()

        # print statistics
        running_loss += loss.item()
        running_kld += kld_loss.item()
        running_criterion += criterion_loss.item()
        running_perc += perc_loss.item()
        if i % report_steps == report_steps-1:
            print("train: l=%.4f, crit=%.4f, kld=%.4f, perc=%.4f" % (running_loss/report_steps, running_criterion/report_steps, running_kld/report_steps, running_perc/report_steps))
            running_loss = 0.0
            running_criterion = 0.0
            running_kld = 0.0
            running_perc = 0.0

    running_loss = 0.0
    running_criterion = 0.0
    running_kld = 0.0
    running_perc = 0.0
    net.eval()
    with torch.no_grad():
        for i,data in enumerate(val, 0):
            inputs, targets = data
            inputs = inputs[:,0:boundl]
            outputs, mu, logvar, z = net(inputs.to(device))

            perc_loss = 1.0 - ssim_module((outputs+1.0)/2.0, (targets.unsqueeze(1).to(device)+1.0)/2.0)
            criterion_loss = crit_module(outputs, targets.unsqueeze(1).to(device))
            kld_loss = kld(mu, logvar)
            loss = vaeloss(epoch, criterion_loss, kld_loss)# + perc_loss

            running_criterion += criterion_loss.item()
            running_kld += kld_loss.item()
            running_loss += loss.item()
            running_perc += perc_loss.item()

    vl = running_loss/len(val)
    print("val: l=%.4f, crit=%.4f, kld=%.4f, perc=%.4f" % (vl, running_criterion/len(val), running_kld/len(val), running_perc/len(val)))

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

# lv=256, 2-bound val: 8

#%%

input,target = ts[0]
input = input[0:boundl]
out,mu,logvar,z = net(torch.Tensor([input]).to(device))

print("out    %.3f	%.3f	%.3f" % (torch.min(out),	torch.mean(out),	torch.max(out)))
print("mu     %.3f	%.3f	%.3f" % (torch.min(mu),	torch.mean(mu),	torch.max(mu)))
print("var    %.3f	%.3f	%.3f" % (torch.min(logvar.exp()),	torch.mean(logvar.exp()),	torch.max(logvar.exp())))
print("z      %.3f	%.3f	%.3f" % (torch.min(z),	torch.mean(z),	torch.max(z)))

#%%

tt = terrain_set2.TerrainSet(['data/USGS_1M_10_x50y466_OR_RogueSiskiyouNF_2019_B19.tif'],
    size=n, stride=8, rescale=rescale)
test = DataLoader(tt, batch_size=256, shuffle=True,
    num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=4)

running_loss = 0.0
running_criterion = 0.0
running_kld = 0.0
running_perc = 0.0
with torch.no_grad():
    for i,data in enumerate(test, 0):
        inputs, targets = data
        inputs = inputs[:,0:boundl]

        outputs, mu, logvar, z = net(inputs.to(device))

        perc_loss = 1.0 - ssim_module((outputs+1.0)/2.0, (targets.unsqueeze(1).to(device)+1.0)/2.0)
        criterion_loss = crit_module(outputs, targets.unsqueeze(1).to(device))
        kld_loss = kld(mu, logvar)
        loss = vaeloss(epoch, criterion_loss, kld_loss)# + perc_loss

        running_criterion += criterion_loss.item()
        running_kld += kld_loss.item()
        running_loss += loss.item()
        running_perc += perc_loss.item()

print("test: l=%.4f, crit=%.4f, kld=%.4f, perc=%.4f" % (running_loss/len(test), running_criterion/len(test), running_kld/len(test), running_perc/len(test)))

# boundl 256 rescale 4 latent 2048 val: l=0.0272, crit=0.0200, kld=2.3492 test: crit=0.1223

# annealing [0.000001, 0.00001, 0.0001], 88k tiles
# Added ssim, changed to MAE - interesting result with cool shapes, and clear rivers. Still with 2048 latent space.
# test: l=0.1495, crit=0.0225, kld=64.1948, perc=0.1205

# Back to Huber 0.25 + SSIM:
# latent=256, batch 64, rescale=4, val: l=0.0499, crit=0.0037, kld=59.7516, perc=0.0402
# test: l=0.1482, crit=0.0220, kld=58.7581, perc=0.1203 - smooth and uninteresting. Maybe MAE worked in this case? Latent 2048 is impractical. as 14-256-4-1.onnx

# Back to mae
# val: l=0.1116, crit=0.0587, kld=119.4586, perc=0.0410, test: l=0.3031, crit=0.1683, kld=118.2967, perc=0.1229 - bad loss

# latent=512
# val: l=0.1134, crit=0.0605, kld=108.7220, perc=0.0420
# test: l=0.2992, crit=0.1687, kld=99.7888, perc=0.1205 - loss looks bad BUT got the interesting terrain shapes again as with initial ssim
# Saved as 14-256-4-2.onnx. Note counter-edges are a bit bad so would be hard to infini-extrude, but same with the initial.
# So KL+MAE+SSIM gives these interesting shapes (this doesn't quite work in 18 with MAE+SSIM, just too smooth).

# Let's try without SSIM, but still with MAE.
# val: l=0.0738, crit=0.0669, kld=68.9668, perc=0.0531
# test: l=0.1719, crit=0.1655, kld=64.1079, perc=0.1230
# Similarly good rivers, but very jagged. Pretty good profile replication. I think with SSIM was better.  ui/dist/14-256-4-3.onnx

# Looks like ssim + mae is for keeps (since earlier try with just huber loss gave too-smooth result). But actually, does that
# just mean that we just need to tweak huber delta down to achieve good shape vs spikiness tradeoff? b/c maybe SSIM is just a proxy for gaussian smoothing hm.
# I really like the shape of the rivers!

# Try huber with lower delta, without ssim


# latent = 1024
#%%
import torch
import torch.nn as nn

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
#%%

# https://www.aaron-powell.com/posts/2019-02-06-golang-wasm-3-interacting-with-js-from-go/

net = torch.load('models/02-128').eval()
dummy_input = torch.randn(1, 128, device="cuda")
input_names = [ "edge" ]
output_names = [ "tile" ]

torch.onnx.export(
    net, dummy_input, "ui/dist/02-128.onnx",
    verbose=True, input_names=input_names, output_names=output_names)

#%%

net = torch.load('models/11-128').eval()

dummy_input = torch.randn(1, 128, device="cuda")
input_names = [ "edge" ]
output_names = [ "tile" ]

torch.onnx.export(
    net, dummy_input, "ui/dist/11-128.onnx",
    verbose=True, input_names=input_names, output_names=output_names)

net = torch.load('models/11-256').eval()

dummy_input = torch.randn(1, 256, device="cuda")
input_names = [ "edge" ]
output_names = [ "tile" ]

torch.onnx.export(
    net, dummy_input, "ui/dist/11-256.onnx",
    verbose=True, input_names=input_names, output_names=output_names)

#%%

net = torch.load('models/06-256').eval()
print(net)

dummy_input = torch.randn(1, 256, device="cuda")
input_names = [ "edge" ]
output_names = [ "tile" ]

torch.onnx.export(
    net, dummy_input, "ui/dist/06-256.onnx",
    verbose=True, input_names=input_names, output_names=output_names)

#%%

boundl=256
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

        latentl = 256
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

net = torch.load('models/14-256').eval()
print(net)

dummy_input = torch.randn(1, 256, device="cuda")
input_names = [ "edge" ]
output_names = [ "tile" ]

torch.onnx.export(
    net, dummy_input, "ui/dist/14-256.onnx",
    verbose=True, input_names=input_names, output_names=output_names)

net = torch.load('models/14-128').eval()
print(net)

dummy_input = torch.randn(1, 128, device="cuda")
input_names = [ "edge" ]
output_names = [ "tile" ]

torch.onnx.export(
    net, dummy_input, "ui/dist/14-128.onnx",
    verbose=True, input_names=input_names, output_names=output_names)

#%%

class VaeFull(nn.Module):
    def __init__(self):
        super().__init__()

        ch=16
        chd=16

        self.encoder = nn.Sequential(
            nn.Conv2d(1, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(ch, ch*2, 3, padding=1),
            nn.BatchNorm2d(ch*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(ch*2, ch*4, 3, padding=1),
            nn.BatchNorm2d(ch*4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(ch*4, ch*8, 3, padding=1),
            nn.BatchNorm2d(ch*8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(ch*8, ch*16, 3, padding=1),
            nn.BatchNorm2d(ch*16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(ch*16, ch*32, 3, padding=1),
            nn.BatchNorm2d(ch*32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Flatten(),
        )

        latentl = 256
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

net = torch.load('models/16-256').eval()
print(net)

dummy_input = torch.randn(1, 256, device="cuda")
input_names = [ "edge" ]
output_names = [ "tile" ]

torch.onnx.export(
    net, dummy_input, "ui/dist/16-256.onnx",
    verbose=True, input_names=input_names, output_names=output_names)

net = torch.load('models/16-128').eval()
print(net)

dummy_input = torch.randn(1, 128, device="cuda")
input_names = [ "edge" ]
output_names = [ "tile" ]

torch.onnx.export(
    net, dummy_input, "ui/dist/16-128.onnx",
    verbose=True, input_names=input_names, output_names=output_names)
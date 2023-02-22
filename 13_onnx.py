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
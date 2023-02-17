#%%
import torch
import torch.nn as nn

# https://www.aaron-powell.com/posts/2019-02-06-golang-wasm-3-interacting-with-js-from-go/

net = torch.load('models/02-128').eval()
dummy_input = torch.randn(1, 128, device="cuda")
input_names = [ "edge" ]
output_names = [ "tile" ]

torch.onnx.export(
    net, dummy_input, "ui/02-128.onnx",
    verbose=True, input_names=input_names, output_names=output_names)
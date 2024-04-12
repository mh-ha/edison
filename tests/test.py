from typing import Iterable

import torch
from torch import nn, Tensor
from torchinfo import summary

def arr(shape):
    return torch.randn(shape)

def test_layer(layer:nn.Module, input_shape:Tensor=None, input_data:Tensor=None, device=None):
    if input_shape is None and input_data is None:
        raise ValueError("Please provide either input_shape or input_data")
    if input_data is not None:
        inputs = input_data
    elif input_shape is not None:
        inputs = torch.randn(input_shape)
    if device is not None:
        inputs = inputs.to(device)
        layer = layer.to(device)
    outputs = layer(inputs)
    print(f"input shape: {inputs.shape}")
    print(f"output type: {type(outputs)}")
    if type(outputs) == Tensor:
        print(f"output shape: {outputs.shape}")
    elif type(outputs) == dict:
        for key, value in outputs.items():
            print(f"{key} shape: {value.shape}")
    elif type(outputs) == tuple or type(outputs) == list:
        for i, output in enumerate(outputs):
            try:
                print(f"output {i} shape: {output.shape}")
            except:
                print(f"output {i} type: {type(output)}")
    else:
        raise ValueError("Output type not recognized")
    return outputs

def summary_layer(layer:nn.Module, input_shape:Tensor=None, input_data:Tensor=None, depth:int=10):
    if input_shape is None and input_data is None:
        raise ValueError("Please provide either input_shape or input_data")
    if input_data is not None:
        inputs = input_data
        print(summary(layer, input_data=inputs, depth=depth))
    elif input_shape is not None:
        print(summary(layer, input_size=input_shape, depth=depth))
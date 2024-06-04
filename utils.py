
import torch
import torch.nn as nn

# Rescale the output of the model to [0, 255] range
class Rescale(nn.Module):
    def __init__(self):
        super(Rescale, self).__init__()

    # https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/2
    def forward(self, tensor):
        tensor = tensor / tensor.max()                  # Rescale to [0, 1]
        tensor = tensor * 255.0
        return tensor

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

# Rescale the output of the model to [0, 255] range
class Rescale(nn.Module):
    def __init__(self):
        super(Rescale, self).__init__()

    # https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/2
    def forward(self, tensor):
        tensor = tensor / tensor.max()                  # Rescale to [0, 1]
        tensor = tensor * 255.0
        return tensor
    

# Same learning rate scheduler as in "Attention is All You Need" paper
# Referencing https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Optim.py 
class DynamicLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        super().__init__(optimizer)
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.curr_step = 0

    def get_lr(self):
        return (self.d_model**(-0.5)) * min(self.curr_step**(-0.5), self.curr_step * self.warmup_steps**(-1.5))

    def step(self):
        # call on get_lr
        self.curr_step += 1

        pass
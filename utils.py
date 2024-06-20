
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
    

# Same learning rate scheduler as in "Attention is All You Need" paper. Not very good for small datasets
# Referencing https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Optim.py 
class TransformerScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        
        # else clause has "step + 1" to cover the step = 0 issue (cann't divide by 0)
        lr = lambda step : self.d_model**(-0.5) * min(step**(-0.5), step * self.warmup_steps**(-1.5)) if step != 0 else self.d_model**(-0.5) * min((step + 1)**(-0.5), (step + 1) * self.warmup_steps**(-1.5))      
        super(TransformerScheduler, self).__init__(optimizer, lr)
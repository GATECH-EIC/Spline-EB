import torch
import torch.nn as nn

# additional subgradient descent on the sparsity-induced penalty term
def updateBN(model, s):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(s*torch.sign(m.weight.data))  # L1
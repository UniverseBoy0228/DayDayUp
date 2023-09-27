import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from activations import *

def non_linear_layer_backward_torch():
    in_dim = 4
    out_dim = 8
    SEED = 520
    np.random.seed(SEED)
    W = torch.tensor(np.random.randn(in_dim, out_dim), dtype=torch.float32, requires_grad=True)
    b = torch.tensor(np.random.randn(1, out_dim), dtype=torch.float32, requires_grad=True)

    x = torch.tensor([[0, 1, 1, 5], [0, 9, 4, 2], [6, 6, 4, 8]], dtype=torch.float32)

    y = x @ W + b
    y.retain_grad()
    y2 = 1 / (1 + torch.exp(-y))
    y2.retain_grad()
    z = y2.sum()
    z.backward()

    grads = {
        "dLdZ": y.grad.numpy(), 
        "dLdB": b.grad.numpy(), 
        "dLdW": W.grad.numpy()
    }

    print(grads["dLdW"])

non_linear_layer_backward_torch()
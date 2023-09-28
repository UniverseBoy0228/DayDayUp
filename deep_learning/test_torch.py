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

def fully_connected_network_backward_torch():
    SEED = 1314
    l1_in_dim = 16
    l1_out_dim = 32
    l2_in_dim = 32
    l2_out_dim = 8
    l3_in_dim = 8
    l3_out_dim = 16
    np.random.seed(SEED)
    W1 = torch.tensor(np.random.randn(l1_in_dim, l1_out_dim), dtype=torch.float32, requires_grad=True)
    b1 = torch.tensor(np.random.randn(1, l1_out_dim), dtype=torch.float32, requires_grad=True)
    np.random.seed(SEED)
    W2 = torch.tensor(np.random.randn(l2_in_dim, l2_out_dim), dtype=torch.float32, requires_grad=True)
    b2 = torch.tensor(np.random.randn(1, l2_out_dim), dtype=torch.float32, requires_grad=True)
    np.random.seed(SEED)
    W3 = torch.tensor(np.random.randn(l3_in_dim, l3_out_dim), dtype=torch.float32, requires_grad=True)
    b3 = torch.tensor(np.random.randn(1, l3_out_dim), dtype=torch.float32, requires_grad=True)

    SEED = 228
    np.random.seed(SEED)
    x = torch.tensor(np.random.randint(10, size=(64, 16)), dtype=torch.float32, requires_grad=True)

    z1 = x @ W1 + b1
    z1.retain_grad()
    y1 = torch.sigmoid(z1)
    y1.retain_grad()
    z2 = y1 @ W2 + b2
    z2.retain_grad()
    y2 = torch.sigmoid(z2)
    y2.retain_grad()
    z3 = y2 @ W3 + b3
    z3.retain_grad()
    y3 = torch.sigmoid(z3)
    y3.retain_grad()

    L = y3.sum()
    print(L.detach().numpy())
    L.backward()

    grads = {
        "dLdZ1": z1.grad.numpy(), 
        "dLdZ2": z2.grad.numpy(), 
        "dLdZ3": z3.grad.numpy(), 
        "dLdY1": y1.grad.numpy(), 
        "dLdY2": y2.grad.numpy(), 
        "dLdY3": y3.grad.numpy(), 
        "dLdW1": W1.grad.numpy(),
        "dLdB1": b1.grad.numpy(),
        "dLdW2": W2.grad.numpy(),
        "dLdB2": b2.grad.numpy(),
        "dLdW3": W3.grad.numpy(),
        "dLdB3": b3.grad.numpy(),
    }

    return grads["dLdW1"], grads["dLdW2"], grads["dLdW3"], grads["dLdB1"], grads["dLdB2"], grads["dLdB3"]


# non_linear_layer_backward_torch()
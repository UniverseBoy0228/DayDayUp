from abc import ABC
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class Identity(ABC):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Identity"

    def __call__(self, z):
        return self.forward(z)

    def forward(self, z):
        return z

    def grad(self, z):
        return np.ones_like(z, dtype=np.float64)

class Sigmoid(ABC):
    def __init__(self, **kwargs):
        super().__init__()

    def __str__(self):
        return "Sigmoid"

    def __call__(self, z):
        assert isinstance(z, np.ndarray)
        if z.ndim == 1:
            z = z.reshape(1, -1)
        return self.forward(z)

    def forward(self, z):
        # exp(-z)中z是负数的情况很容易引起exp()的值很大导致溢出
        # 对Sigmoid函数优化，避免出现大数溢出的问题
        y = z.copy()
        y[z >= 0] = 1.0 / (1 + np.exp(-z[z >= 0]))
        y[z < 0] = np.exp(z[z < 0] / (1 + np.exp(z[z < 0])))
        return y

    def forward_basic(self, z):
        if z.ndim == 1:
            z = z.reshape(1, -1)
        batch, dim = z.shape
        result = np.zeros_like(z, dtype=np.float64)
        for b in range(batch):
            z_b = z[b, :]
            for i in range(dim):
                result[b, i] = self.forward(z_b[i])
        return result
    
    def forward_torch(self, z):
        if z.ndim == 1:
            z = z.reshape(1, -1)
        return torch.sigmoid(torch.tensor(z, dtype=torch.float64))

    def grad(self, z):
        f_z = self.forward(z)
        return f_z * (1 - f_z)


class Softmax(ABC):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Softmax"

    def __call__(self, z):
        assert isinstance(z, np.ndarray)
        if z.ndim == 1:
            z = z.reshape(1, -1)
        return self.forward(z)

    def forward(self, z):
        batch, _ = z.shape
        result = np.zeros_like(z, dtype=np.float64)
        for b in range(batch):
            z_b = z[b, :]
            exp = np.exp(z_b - z_b.max())
            result[b, :] = exp / np.sum(exp)
        return result

    def forward_basic(self, z):
        if z.ndim == 1:
            z = z.reshape(1, -1)
        batch, dim = z.shape
        result = np.zeros_like(z, dtype=np.float64)
        for b in range(batch):
            z_b = z[b, :]
            exp_sum = 0
            r = np.zeros_like(z_b, dtype=np.float64)
            for i in range(dim):
                exp_tmp = np.exp(z_b[i] - z_b.max())
                r[i] = exp_tmp
                exp_sum += exp_tmp
            result[b, :] = r / exp_sum
        return result

    def forward_torch(self, z):
        if z.ndim == 1:
            z = z.reshape(1, -1)
        return F.softmax(torch.tensor(z, dtype=torch.float64), dim=1)

    def grad(self, z):
        return
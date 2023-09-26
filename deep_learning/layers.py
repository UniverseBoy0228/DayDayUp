from abc import ABC
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F



class LinearLayer(ABC):
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.parameters = {"W": None, "b": None}
        self.is_initialized = False

    def _init_parameters(self):
        W = np.ones((self.in_dim, self.out_dim), dtype=np.float64)
        b = np.zeros((1, self.out_dim), dtype=np.float64)
        self.parameters = {"W": W, "b": b}
        self.is_initialized = True

    def __call__(self, X):
        assert isinstance(X, np.ndarray)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.forward(X)

    def forward(self, X):
        '''
        X: np.ndarray
        '''
        batch, in_dim = X.shape
        assert in_dim == self.in_dim

        if not self.is_initialized:
            self._init_parameters()

        self.X = X # 保存当前层的输入
        W = self.parameters["W"]
        b = self.parameters["b"]
        self.Z = X @ W + b # 保存当前层输出

        return self.Z

    def backword(self):
        return
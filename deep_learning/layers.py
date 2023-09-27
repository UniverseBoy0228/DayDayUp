from abc import ABC
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from activations import *

SEED = 520

class LinearLayer(ABC):
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.parameters = {"W": None, "b": None}
        self.is_initialized = False

    def _init_parameters(self):
        np.random.seed(SEED)
        W = np.random.randn(self.in_dim, self.out_dim)
        b = np.random.randn(1, self.out_dim)
        self.parameters = {"W": W, "b": b}
        self.is_initialized = True

    def __call__(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.forward(X)

    def forward(self, X):
        '''
        X: np.ndarray
        '''
        if not self.is_initialized:
            self._init_parameters()
        
        self.X = X # 保存当前层的输入

        W = self.parameters["W"]
        b = self.parameters["b"]
        self.Z = X @ W + b # 保存当前层输出

        return self.Z

    def backward(self, dLdY, dLdZ, act_func=None):
        '''
        dLdY: 前一层额dLdY
        dLdZ: 前一层的dLdZ
        '''
        X = self.X
        Z = self.Z

        batch, _ = X.shape

        if act_func is None:
            act_func = Identity()

        dW = []
        db = []
        dZ = np.zeros((batch, self.out_dim), dtype=np.float64)
        dY = np.zeros((batch, self.in_dim), dtype=np.float64)
        W = self.parameters["W"]
        b = self.parameters["b"]
        for ba in range(batch):
            dW.append(dLdY[ba,:].reshape(1, -1) * np.tile(X[ba,:].reshape(-1, 1), self.out_dim))
            db.append(dLdY[ba,:].reshape(1, -1) * np.ones_like(b, dtype=np.float64))

            dZ[ba, :] = dLdY[ba, :].reshape(1, -1) * act_func.grad(Z[ba, :].reshape(1, -1)) # 当前层的dLdZ
            dY[ba, :] = (dZ[ba, :] @ W.T).reshape(1, -1) # 当前层的dLdY
        return dW, db, dZ, dY
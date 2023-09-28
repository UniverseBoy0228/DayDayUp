from abc import ABC
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from activations import *

SEED = 1314

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

    def backward(self, dLdY, act_func=None):
        '''
        dLdY : np.ndarry
            前一层额dLdY
            由后一层计算得到
        act_func : Activation Function
            激活函数
        '''
        X = self.X
        Z = self.Z

        batch, _ = X.shape

        if act_func is None:
            act_func = Identity()

        W = self.parameters["W"]
        b = self.parameters["b"]
        dW = [] # 当前层的Weight梯度
        db = [] # 当前层的Bias梯度
        dY = [] # 前一层的dLdY
        dZ = [] # 当前层的dLdZ
        for ba in range(batch):
            dLdZ = dLdY[ba, :] * act_func.grad(Z[ba, :])
            dZ.append(dLdZ)
            dW.append(dLdZ * np.tile(X[ba,:].reshape(-1, 1), self.out_dim))
            db.append(dLdZ * np.ones_like(b, dtype=np.float64))
            dY.append(dLdZ @ W.T)
        return dW, db, np.array(dY, dtype=np.float64), dZ
from abc import ABC
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from activations import *
from model import *
from parameter import *

SEED = 1314

class LinearLayer(ModuleBase):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.gradients = {
            "dLdZ": None, # Loss Function对当前层的Z的偏导数
            "dLdW": None, # Loss Function对当前层的W的偏导数
            "dLdb": None, # Loss Function对当前层的b的偏导数
            "dLdY": None, # Loss Function对前一层的Y的偏导数，放在这一层的原因是因为计算需要用到当前层的梯度
        }

        self.is_initialized = False
        self._init_parameters()

    def _init_parameters(self):
        np.random.seed(SEED)
        self.W = ParameterBasic(np.random.randn(self.in_dim, self.out_dim))
        self.b = ParameterBasic(np.random.randn(1, self.out_dim))
        # self._parameters["W"] = self.W.parameter
        # self._parameters["b"] = self.b.parameter
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

        W = self._parameters["W"]
        b = self._parameters["b"]
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

        W = self._parameters["W"]
        b = self._parameters["b"]
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

        self.gradients["dLdZ"] = np.array(dZ, dtype=np.float32)
        self.gradients["dLdW"] = np.array(dW, dtype=np.float32)
        self.gradients["dLdb"] = np.array(db, dtype=np.float32)
        self.gradients["dLdY"] = np.array(dY, dtype=np.float32)

        return dW, db, np.array(dY, dtype=np.float64), dZ
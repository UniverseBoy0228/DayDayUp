from abc import ABC
import numpy as np
from numpy.testing import assert_almost_equal
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



class ConvolutionalLayer2D(ModuleBase):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self._hyperparameters = {
            "in_channel": in_channels, 
            "out_channel": out_channels, 
            "kernel_size": kernel_size, 
            "stride": stride, 
            "padding": padding, 
        }

        self.is_initialized = False
        self._init_parameters()

    def _init_parameters(self):
        self.W = ParameterBasic(np.random.randn(self._hyperparameters.get("out_channel"), 
                                                self._hyperparameters.get("in_channel"), 
                                                self._hyperparameters.get("kernel_size")[0], 
                                                self._hyperparameters.get("kernel_size")[1]))
        self.b = ParameterBasic(np.random.randn(self._hyperparameters.get("out_channel")))
        self._parameters["W"] = self.W.parameter
        self._parameters["b"] = self.b.parameter
        self.is_initialized = True

    def __str__(self):
        return "ConvolutionalLayer2D"

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        '''
        X : 尺寸为(batch_size, channel, height, width)
        out : 尺寸为(batch_size, channel, height, width)
        '''
        if not self.is_initialized:
            self._init_parameters()

        self.X = X

        batch, channel, height, width = X.shape
        kernel_size = self._hyperparameters.get("kernel_size")
        stride = self._hyperparameters.get("stride")
        # 根据kernel_size和stride等参数计算当前卷积层输出Feature Map的尺寸
        out_x = int((width - kernel_size[1]) / stride) + 1
        out_y = int((height - kernel_size[0]) / stride) + 1
        out_channel = self._hyperparameters.get("out_channel")

        out = np.zeros((batch, out_channel, out_y, out_x), dtype=np.float32)
        for ba in range(batch):
            for dx in range(out_x):
                for dy in range(out_y):
                    x = X[ba, :, dy*stride:dy*stride+kernel_size[0], dx*stride:dx*stride+kernel_size[1]]
                    for c in range(out_channel):
                        kernel = self._parameters.get("W")[c, :, :, :]
                        bias = self._parameters.get("b")[c]
                        o = np.sum(kernel * x) + bias
                        out[ba, c, dy, dx] = o

        self.Z = out
        return self.Z


    def backward(self, dLdY, act_func=None):
        X = self.X
        Z = self.Z
        # 计算当前卷积层输出Feature Map大小
        batch, channel, height, width = X.shape
        kernel_size = self._hyperparameters.get("kernel_size")
        stride = self._hyperparameters.get("stride")

        out_x = int((width - kernel_size[1]) / stride) + 1  # 多少列
        out_y = int((height - kernel_size[0]) / stride) + 1 # 多少行
        out_channel = self._hyperparameters.get("out_channel")

        # 如果当前卷积层后未接激活函数，则返回单位函数
        if act_func is None:
            act_func = Identity()

        # 将dLdY排列成当前卷积层输出的Feature Map形状
        # Case 1，后一层也是卷积层，则dLdY的尺寸应该与当前层输出尺寸一致
        # Case 2，后一层是全连接层，则dLdY会被拉成向量，需要重新排列成当前层输出尺寸
        dLdY = dLdY.reshape(batch, out_channel, out_y, out_x)

        # 计算梯度
        dY = []
        dZ = []
        dW = []
        db = []
        for ba in range(batch):
            dLdZ = dLdY[ba, :, :, :] * act_func.grad(Z[ba, :, :, :])
            dZ.append(dLdZ)
            dLdW = np.zeros_like(self._parameters.get("W"), dtype=np.float32)
            dLdY_previous_layer = np.zeros_like(X[ba, :, :, :], dtype=np.float32)
            for dx in range(out_x):
                for dy in range(out_y):
                    yy = dy * stride
                    xx = dx * stride
                    x = X[ba, :, yy:yy+kernel_size[0], xx:xx+kernel_size[1]]
                    dLdW = dLdW + x

                    kernel = self._parameters.get("W")
                    for c in range(out_channel):
                        dLdY_previous_layer[:, yy:yy+kernel_size[0], xx:xx+kernel_size[1]] += (dLdZ[c, dy, dx] * kernel[c, :, :, :])
            dLdb = np.ones_like(self._parameters.get("b"), dtype=np.float32) * (out_x * out_y)
            dW.append(dLdW)
            db.append(dLdb)
            dY.append(dLdY_previous_layer)

        return np.array(dW), np.array(db), np.array(dY), np.array(dZ)



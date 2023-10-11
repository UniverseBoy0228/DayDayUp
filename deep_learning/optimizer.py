from abc import ABC
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientDescentBasic(ABC):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    def step(self, parameters, gradients):
        for k in parameters:
            parameters[k] = parameters[k] - self.learning_rate * gradients[k]
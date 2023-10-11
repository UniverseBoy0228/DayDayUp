from abc import ABC
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class MeanSquaredError(ABC):
    def __init__(self, reduction="mead", **kwargs):
        super().__init__()
        self.reduction = reduction

    def __str__(self):
        return "MeanSquaredError"

    def __call__(self, y_pred, y_actual):
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(1, -1)
        if y_actual.ndim == 1:
            y_actual = y_actual.reshape(1, -1)
        return self.forward(y_pred, y_actual)

    def forward(self, y_pred, y_actual):
        batch = y_pred.shape[0]
        if self.reduction == "mean":
            loss_list = []
            for b in range(batch):
                loss_list.append(((y_pred[b] - y_actual[b]) ** 2).mean())
            return np.mean(loss_list)
        elif self.reduction == "sum":
            loss_list = []
            for b in range(batch):
                loss_list.append(((y_pred[b] - y_actual[b]) ** 2).sum())
            return np.sum(loss_list)

    def grad(self, y_pred, y_actual):
        batch, dim = y_pred.shape
        if self.reduction == "mean":
            loss_list = []
            for b in range(batch):
                loss_each_sample = []
                for d in range(dim):
                    loss_each_sample.append(2.0 / dim * (y_pred[b][d] - y_actual[b][d]))
                loss_list.append(loss_each_sample)
            losses = np.array(loss_list, dtype=np.float32)
            return 1.0 / batch * losses
        elif self.reduction == "sum":
            loss_list = []
            for b in range(batch):
                loss_each_sample = []
                for d in range(dim):
                    loss_each_sample.append(2.0 * (y_pred[b][d] - y_actual[b][d]))
                loss_list.append(loss_each_sample)
            losses = np.array(loss_list, dtype=np.float32)
            return losses
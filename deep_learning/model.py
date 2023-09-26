from abc import ABC
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F





# class FullyConnectedNetwork(object):
#     def __init__(self):
#         self.linear_lyer_1 = LinearLayer(in_dim=4, out_dim=128)
#         self.act_1 = Sigmoid()
#         self.linear_lyer_2 = LinearLayer(in_dim=128, out_dim=32)
#         self.act_2 = Sigmoid()
#         self.linear_lyer_3 = LinearLayer(in_dim=32, out_dim=10)
#         self.act_3 = Sigmoid()

#     def forward(self, X):
#         output = self.linear_lyer_1(X)
#         output = self.act_1(output)
#         output = self.linear_lyer_2(output)
#         output = self.act_2(output)
#         output = self.linear_lyer_3(output)
#         output = self.act_3(output)
#         return output


# if __name__ == '__main__':
#     X = np.random.randint(10, size=(8, 4))

#     fully_connected_network = FullyConnectedNetwork()
#     o = fully_connected_network.forward(X)
#     print(o.shape)

#     # print(np.linspace(2.5, 9.8, num=12))
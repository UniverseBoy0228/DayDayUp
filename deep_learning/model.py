from abc import ABC
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from parameter import *

class ModuleBase(ABC):
    def __init__(self):
        super().__init__()
        # super().__setattr__()才是class中赋值操作符的真实底层操作，使用这种方式可以减少赋值操作的开销
        super().__setattr__('_parameters', dict()) # 用于记录Module中的可学习参数
        super().__setattr__('_modules', dict()) # 用于记录Module组件
        super().__setattr__('_module_names', None)
        super().__setattr__('_graph', list()) # 记录前向传播计算图，TODO

    def register_parameter(self, name, value):
        '''
        value : 参数value是一个Module实例，目的是在传入后拿到该Module参数并存入到当前class的_parameters属性中
        最终目的是实现将模型中所有参数都放入当前class的_parameters属性中
        '''
        for param in value._parameters.keys():
            self._parameters[name + '.' + param] = value._parameters[param]

    def __setattr__(self, name, value): # 重载class中的赋值操作符，使其在赋值过程中做如下事情
        if isinstance(value, ModuleBase): # 判断value是否是实例化Module组件，比如Linear层、Conv层
            modules = self.__dict__['_modules']
            modules[name] = value # 在实例化Module组件的过程中，会进入到这个函数，将Module组件存入到当前class的_modules属性中记录下来
            self._module_names = {v: k for k, v in self._modules.items()}
            self.register_parameter(name, value) # 记录完实例化的Module组件后，将Module中的参数记录到当前class的_parameters属性中
        elif isinstance(value, ParameterBasic): # 判断value是否是ParameterBasic类型，如果是，则表示当前参数是可学习参数，需要记录到_parameters属性中
            params = self.__dict__['_parameters']
            params[name] = value
        elif isinstance(value, list):
            for i, v in enumerate(value):
                if isinstance(v, ModuleBase):
                    modules = self.__dict__.get('_modules')
                    modules[name + f'_{i}'] = v
                    self._module_names = {v: k for k, v in self._modules.items()}
                    self.register_parameter(name + f'_{i}', v)
        super().__setattr__(name, value) # 做完上述两种情况的操作后再进行真正的赋值操作

    def state_dict(self):
        '''
        返回当前Module的参数列表
        实现过程中创建一个新的字典，将参数存入后返回
        目的是使此方法得到的参数列表在外部修改后不会影响Module的参数
        '''
        output = dict()
        for p in self._parameters.keys():
            output[p] = self._parameters[p]
        return output

    def __call__(self, X):
        return self.forward(X)

    def backward(self, predict, target, loss_func):
        '''
        predict : 前向传播的预测值
        target : 真实标签
        loss_func : 损失函数

        Pytorch中是对损失函数计算出的损失做backward，也就是torch.Tensor.backward()
        这里没有将numpy.ndarray包装成tensor类，所以numpy.ndarray本身没有backward方法
        所以将backward放在这里实现

        反向传播需要按照前向传播的顺序倒退，所以需要在前向传播时记录计算图 TODO
        '''
        dLdY = loss_func.grad(predict, target)
        print(dLdY)

        graph = self.__dict__['_graph'][::-1] # _graph中是前向传播顺序，所以反向传播时需要倒序排列
        for layer in graph:
            [linear, actvation] = layer
            dLdW, dLdb, dLdY, dLdZ = linear.backward(dLdY, actvation)
            print(dLdY)


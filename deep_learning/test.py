import numpy as np
from numpy.testing import assert_almost_equal
import copy

from activations import *
from layers import *
from test_torch import *
from losses import *
from optimizer import *
from model import *
from parameter import *

def test_softmax():
    softmax = Softmax()
    print(str(softmax))

    z = np.random.randn(16, 512)
    actual1 = softmax(z)
    actual2 = softmax.forward_basic(z)
    desired = softmax.forward_torch(z)

    assert_almost_equal(actual1, actual2)
    assert_almost_equal(actual1, desired)
    assert_almost_equal(actual2, desired)

def test_sigmoid():
    sigmoid = Sigmoid()
    print(str(sigmoid))

    z = np.random.randint(10, size=(2, 4))
    actual1 = sigmoid(z)
    actual2 = sigmoid.forward_basic(z)
    desired = sigmoid.forward_torch(z)

    assert_almost_equal(actual1, actual2)
    assert_almost_equal(actual1, desired)
    assert_almost_equal(actual2, desired)


def test_linear_layer():
    linear_layer = LinearLayer(4, 8)

    X = np.random.randint(10, size=(8, 4))
    print(X)
    Z = linear_layer.forward(X)
    print(Z)

def test_linear_layer_backward():
    linear_layer = LinearLayer(4, 8)

    X = np.random.randint(10, size=(3, 4))
    print(X)
    Z = linear_layer.forward(X)
    print(Z.shape)

    dLdY = np.ones_like(Z, dtype=np.float64)
    dLdW, dLdb, dLdY = linear_layer.backward(dLdY)
    print(dLdW)
    print(dLdb)
    print(dLdY)

def test_non_linear_layer_backward():
    linear_layer = LinearLayer(4, 8)
    activation = Sigmoid()

    X = np.array([[0, 1, 1, 5], [0, 9, 4, 2], [6, 6, 4, 8]], dtype=np.float64)
    # X = np.array([[0, 1, 1, 5]], dtype=np.float64)
    Z = linear_layer.forward(X)
    Y = activation(Z)

    dLdY = dLdZ = np.ones_like(Y, dtype=np.float64)
    dLdW, dLdb, dLdZ, dLdY = linear_layer.backward(dLdY, dLdZ, act_func=activation)

    sum_dLdW = dLdW[0]
    for i in range(1, len(dLdW)):
        sum_dLdW += dLdW[i]
    print(sum_dLdW)

def test_fully_connected_network():
    dLdW1, dLdW2, dLdW3, dLdB1, dLdB2, dLdB3 = fully_connected_network_backward_torch()

    linear_layer_1 = LinearLayer(16, 32)
    linear_layer_2 = LinearLayer(32, 8)
    linear_layer_3 = LinearLayer(8, 16)

    activation_1 = Sigmoid()
    activation_2 = Sigmoid()
    activation_3 = Sigmoid()

    # X = np.array([[0, 1, 1, 5], [0, 9, 4, 2], [6, 6, 4, 8]], dtype=np.float64)
    # X = np.array([[0, 1, 1, 5]], dtype=np.float64)
    SEED = 228
    np.random.seed(SEED)
    X = np.random.randint(10, size=(64, 16)).astype(np.float64)
    Z1 = linear_layer_1.forward(X)
    Y1 = activation_1(Z1)
    Z2 = linear_layer_2.forward(Y1)
    Y2 = activation_1(Z2)
    Z3 = linear_layer_3.forward(Y2)
    Y3 = activation_1(Z3)

    print(Y3.sum())

    dLdY = np.ones_like(Y3, dtype=np.float64)
    dLdW, dLdb, dLdY, dLdZ = linear_layer_3.backward(dLdY, activation_3)
    sum_dLdW3 = copy.deepcopy(dLdW[0])
    sum_dLdb3 = copy.deepcopy(dLdb[0])
    sum_dLdY3 = copy.deepcopy(dLdY[0])
    sum_dLdZ3 = copy.deepcopy(dLdZ[0])
    for i in range(1, len(dLdW)):
        sum_dLdW3 += dLdW[i]
        sum_dLdb3 += dLdb[i]
        sum_dLdY3 += dLdY[i]
        sum_dLdZ3 += dLdZ[i]

    dLdW, dLdb, dLdY, dLdZ = linear_layer_2.backward(dLdY, activation_3)
    sum_dLdW2 = copy.deepcopy(dLdW[0])
    sum_dLdb2 = copy.deepcopy(dLdb[0])
    sum_dLdY2 = copy.deepcopy(dLdY[0])
    sum_dLdZ2 = copy.deepcopy(dLdZ[0])
    for i in range(1, len(dLdW)):
        sum_dLdW2 += dLdW[i]
        sum_dLdb2 += dLdb[i]
        sum_dLdY2 += dLdY[i]
        sum_dLdZ2 += dLdZ[i]

    dLdW, dLdb, dLdY, dLdZ = linear_layer_1.backward(dLdY, activation_3)
    sum_dLdW1 = copy.deepcopy(dLdW[0])
    sum_dLdb1 = copy.deepcopy(dLdb[0])
    sum_dLdY1 = copy.deepcopy(dLdY[0])
    sum_dLdZ1 = copy.deepcopy(dLdZ[0])
    for i in range(1, len(dLdW)):
        sum_dLdW1 += dLdW[i]
        sum_dLdb1 += dLdb[i]
        sum_dLdY1 += dLdY[i]
        sum_dLdZ1 += dLdZ[i]

    decimal = 5
    assert_almost_equal(actual=sum_dLdW1, desired=dLdW1, decimal=decimal)
    assert_almost_equal(actual=sum_dLdb1.reshape(1, -1), desired=dLdB1, decimal=decimal)
    assert_almost_equal(actual=sum_dLdW2, desired=dLdW2, decimal=decimal)
    assert_almost_equal(actual=sum_dLdb2.reshape(1, -1), desired=dLdB2, decimal=decimal)
    assert_almost_equal(actual=sum_dLdW3, desired=dLdW3, decimal=decimal)
    assert_almost_equal(actual=sum_dLdb3.reshape(1, -1), desired=dLdB3, decimal=decimal)

def test_gradient_descent():
    linear_layer_1 = LinearLayer(16, 32)
    activation_1 = Sigmoid()
    loss_function = MeanSquaredError("mean")

    for i in range(10000):
        SEED = 100
        np.random.seed(SEED)
        X = np.random.randint(10, size=(64, 16)).astype(np.float32)
        Z1 = linear_layer_1.forward(X)
        Y1 = activation_1(Z1)

        Y1_actual = np.ones_like(Y1, dtype=np.float32)
        L = loss_function(Y1, Y1_actual)
        print(f"Iter {i} Loss: {L}")

        dLdY1 = loss_function.grad(Y1, Y1_actual)

        dLdW, dLdb, dLdY, dLdZ = linear_layer_1.backward(dLdY1, activation_1)
        sum_dLdW1 = copy.deepcopy(dLdW[0])
        sum_dLdb1 = copy.deepcopy(dLdb[0])
        sum_dLdY1 = copy.deepcopy(dLdY[0])
        sum_dLdZ1 = copy.deepcopy(dLdZ[0])
        for i in range(1, len(dLdW)):
            sum_dLdW1 += dLdW[i]
            sum_dLdb1 += dLdb[i]
            sum_dLdY1 += dLdY[i]
            sum_dLdZ1 += dLdZ[i]

        gradients = {
            "W": sum_dLdW1, 
            "b": sum_dLdb1
        }

        gradient_descent = GradientDescentBasic(learning_rate=0.1)
        gradient_descent.step(linear_layer_1.parameters, gradients)

def test_module_parameter_register():
    class TestModule(ModuleBase):
        def __init__(self):
            super().__init__()
            self.linear_layer_1 = LinearLayer(4, 3)
            self.linear_layer_2 = LinearLayer(3, 5)
            self.linear_layer_3 = LinearLayer(5, 4)

            self.sigmoid_1 = Sigmoid()
            self.sigmoid_2 = Sigmoid()
            self.sigmoid_3 = Sigmoid()

            self.weight = ParameterBasic(np.random.randn(4, 4))
            self.bias = ParameterBasic(np.random.randn(1, 4))

        def forward(self, X):
            # out = X @ self.weight.parameter + self.bias.parameter
            out = self.linear_layer_1(X)
            out = self.sigmoid_1(out)
            print(type(out))
            self._graph.append([self.linear_layer_1, self.sigmoid_1])

            out = self.linear_layer_2(out)
            out = self.sigmoid_2(out)
            print(type(out))
            self._graph.append([self.linear_layer_2, self.sigmoid_2])

            out = self.linear_layer_3(out)
            out = self.sigmoid_3(out)
            print(type(out))
            self._graph.append([self.linear_layer_3, self.sigmoid_3])
            return out

    test_module = TestModule()
    X = np.random.randint(10, size=(3, 4)).astype(np.float32)
    Y = test_module(X)
    print("Y", Y.shape)

    modules = test_module._modules
    print("modules", modules)
    print("---------------------------")

    for m in modules.keys():
        m_param = modules[m]._parameters
        for p in m_param.keys():
            print(p, id(m_param[p]), type(m_param[p]), m_param[p].shape)
    print("---------------------------")

    all_param = test_module._parameters
    for p in all_param.keys():
        print(p, id(all_param[p]), type(all_param[p]), all_param[p].shape)
    print("---------------------------")

    print("graph", test_module._graph)

    loss_function = MeanSquaredError("mean")
    Y_actual = np.ones_like(Y, dtype=np.float32)
    L = loss_function(Y, Y_actual)
    print("Loss", L)

    test_module.backward(Y, Y_actual, loss_function)

def test_conv2d_forward_backward():
    in_channels = 3
    out_channels = 8
    kernel_size = (3, 5)
    stride = 5
    padding = None
    conv2d = ConvolutionalLayer2D(in_channels=in_channels, 
                                out_channels=out_channels, 
                                kernel_size=kernel_size, 
                                stride=stride, 
                                padding=padding)

    X = np.random.randint(10, size=(32, in_channels, 256, 128))
    out_1 = conv2d(X)
    print("out_1", out_1.shape)

    for k in conv2d._parameters.keys():
        print(k, conv2d._parameters.get(k).shape)

    L_1 = out_1.sum()
    print(f"L_1 {L_1}")
    dLdY = np.ones_like(out_1)
    dLdW, dLdb, dLdY, dLdZ = conv2d.backward(dLdY, None)
    dLdW = np.sum(dLdW, axis=0)
    dLdb = np.sum(dLdb, axis=0)
    # 设置torch模型中的参数与我们的模型一致
    weight = torch.tensor(conv2d._parameters.get("W"), dtype=torch.float32)
    bias = torch.tensor(conv2d._parameters.get("b"), dtype=torch.float32)
    out_2, grads = test_torch_COV2D(X, in_channels, out_channels, kernel_size, stride, padding, weight, bias)
    # Forward
    assert_almost_equal(actual=out_1, desired=out_2.detach().numpy(), decimal=4)
    # Backward
    assert_almost_equal(actual=dLdY, desired=grads["X"].detach().numpy(), decimal=4)
    assert_almost_equal(actual=dLdZ, desired=grads["Z"].detach().numpy(), decimal=4)
    assert_almost_equal(actual=dLdW, desired=grads["conv2d.weight"].detach().numpy(), decimal=4)
    assert_almost_equal(actual=dLdb, desired=grads["conv2d.bias"].detach().numpy(), decimal=4)

if __name__ == '__main__':
    # test_softmax()
    # test_sigmoid()
    # test_linear_layer()
    # test_linear_layer_backward()
    # test_non_linear_layer_backward()
    # test_fully_connected_network()
    # test_gradient_descent()
    # test_module_parameter_register()
    test_conv2d_forward_backward()



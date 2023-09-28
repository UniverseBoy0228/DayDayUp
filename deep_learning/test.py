import numpy as np
from numpy.testing import assert_almost_equal
import copy

from activations import *
from layers import *
from test_torch import *

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

if __name__ == '__main__':
    # test_softmax()
    # test_sigmoid()
    # test_linear_layer()
    # test_linear_layer_backward()
    # test_non_linear_layer_backward()
    test_fully_connected_network()



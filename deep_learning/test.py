import numpy as np
from numpy.testing import assert_almost_equal

from activations import *
from layers import *

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
    linear_layer_1 = LinearLayer(4, 8)
    linear_layer_2 = LinearLayer(8, 16)

    X = np.array([[0, 1, 1, 5], [0, 9, 4, 2], [6, 6, 4, 8]], dtype=np.float64)
    # X = np.array([[0, 1, 1, 5]], dtype=np.float64)
    Z1 = linear_layer_1.forward(X)
    Z2 = linear_layer_2.forward(Z1)

    dLdY = dLdZ = np.ones_like(Z2, dtype=np.float64)
    dLdW, dLdb, dLdZ, dLdY = linear_layer_2.backward(dLdY, dLdZ)
    sum_dLdW = dLdW[0]
    for i in range(1, len(dLdW)):
        sum_dLdW += dLdW[i]
    print(sum_dLdW)
    print(sum_dLdW.shape)

    dLdW, dLdb, dLdZ, dLdY = linear_layer_1.backward(dLdY, dLdZ)
    sum_dLdW = dLdW[0]
    for i in range(1, len(dLdW)):
        sum_dLdW += dLdW[i]
    print(sum_dLdW)
    print(sum_dLdW.shape)

if __name__ == '__main__':
    # test_softmax()
    # test_sigmoid()
    # test_linear_layer()
    # test_linear_layer_backward()
    test_non_linear_layer_backward()
    # test_fully_connected_network()



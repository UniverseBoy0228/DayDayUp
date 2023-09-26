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




if __name__ == '__main__':
    test_softmax()
    test_sigmoid()

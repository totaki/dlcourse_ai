import numpy as np
from gradient_check import check_gradient
import linear_classifer


def square(x):
    return float(x*x), 2*x


check_gradient(square, np.array([3.0]))


def array_sum(x):
    assert x.shape == (2,), x.shape
    return np.sum(x), np.ones_like(x)


check_gradient(array_sum, np.array([3.0, 2.0]))


def array_2d_sum(x):
    assert x.shape == (2, 2)
    return np.sum(x), np.ones_like(x)


check_gradient(array_2d_sum, np.array([[3.0, 2.0], [1.0, 0.0]]))


# TODO Implement softmax and cross-entropy for single sample
probs = linear_classifer.softmax(np.array([-10, 0, 10]))

# Make sure it works for big numbers too!
probs = linear_classifer.softmax(np.array([1000, 0, 0]))
assert np.isclose(probs[0], 1.0)


probs = linear_classifer.softmax(np.array([-5, 0, 5]))
print(linear_classifer.cross_entropy_loss(probs, 1))


loss, grad = linear_classifer.softmax_with_cross_entropy(
    np.array([1, 0, 0]), 1
)
check_gradient(
    lambda x: linear_classifer.softmax_with_cross_entropy(x, 1),
    np.array([1, 0, 0], np.float)
)


# TODO Extend combined function so it can receive a 2d array with batch of samples
np.random.seed(42)
# Test batch_size = 1
num_classes = 4
batch_size = 1
predictions = np.random.randint(-1, 3, size=(num_classes, batch_size)).astype(np.float)
target_index = np.random.randint(0, num_classes, size=(batch_size, 1)).astype(np.int)
check_gradient(lambda x: linear_classifer.softmax_with_cross_entropy(x, target_index), predictions)

# Test batch_size = 3
num_classes = 4
batch_size = 3
predictions = np.random.randint(-1, 3, size=(num_classes, batch_size)).astype(np.float)
target_index = np.random.randint(0, num_classes, size=(batch_size, 1)).astype(np.int)
check_gradient(lambda x: linear_classifer.softmax_with_cross_entropy(x, target_index), predictions)


if False:
    batch_size = 2
    num_classes = 2
    num_features = 3
    np.random.seed(42)
    W = np.random.randint(-1, 3, size=(num_features, num_classes)).astype(np.float)
    X = np.random.randint(-1, 3, size=(batch_size, num_features)).astype(np.float)
    target_index = np.ones(batch_size, dtype=np.int)

    loss, dW = linear_classifer.linear_softmax(X, W, target_index)
    check_gradient(lambda w: linear_classifer.linear_softmax(X, w, target_index), W)


if False:
    # TODO Implement l2_regularization function that implements loss for L2 regularization
    linear_classifer.l2_regularization(W, 0.01)
    check_gradient(lambda w: linear_classifer.l2_regularization(w, 0.01), W)

if False:
    classifier = linear_classifer.LinearSoftmaxClassifier()
    loss_history = classifier.fit(train_X, train_y, epochs=10, learning_rate=1e-3, batch_size=300, reg=1e1)

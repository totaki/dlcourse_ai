import numpy as np
from dataset import load_svhn
from knn import KNN
from metrics import binary_classification_metrics, multiclass_accuracy


train_X, train_y, test_X, test_y = load_svhn("data", max_train=1000, max_test=100)

binary_train_mask = (train_y == 0) | (train_y == 9)
binary_train_X = train_X[binary_train_mask]
binary_train_y = train_y[binary_train_mask] == 0

binary_test_mask = (test_y == 0) | (test_y == 9)
binary_test_X = test_X[binary_test_mask]
binary_test_y = test_y[binary_test_mask] == 0

# Reshape to 1-dimensional array [num_samples, 3*3*32]
binary_train_X = binary_train_X.reshape(binary_train_X.shape[0], -1)
binary_test_X = binary_test_X.reshape(binary_test_X.shape[0], -1)

knn_classifier = KNN(k=1)
knn_classifier.fit(binary_train_X, binary_train_y)

if False:
    dists = knn_classifier.compute_distances_two_loops(binary_test_X)
    assert np.isclose(dists[0, 10], np.sum(np.abs(binary_test_X[0] - binary_train_X[10])))

    dists = knn_classifier.compute_distances_one_loop(binary_test_X)
    assert np.isclose(dists[0, 10], np.sum(np.abs(binary_test_X[0] - binary_train_X[10])))

    dists = knn_classifier.compute_distances_no_loops(binary_test_X)
    assert np.isclose(dists[0, 10], np.sum(np.abs(binary_test_X[0] - binary_train_X[10])))


# Короче считаем дистанции получаем наименьшие расстояния для ка ждого примера и смотрет метки
prediction = knn_classifier.predict(binary_test_X, num_loops=1)

accuracy, precision, recall, f1 = binary_classification_metrics(prediction, binary_test_y)
print("KNN with k = %s" % knn_classifier.k)
print("Accuracy: %4.2f, Precision: %4.2f, Recall: %4.2f, F1: %4.2f" % (accuracy, precision, recall, f1))

knn_classifier_3 = KNN(k=3)
knn_classifier_3.fit(binary_train_X, binary_train_y)
prediction = knn_classifier_3.predict(binary_test_X, num_loops=1)

accuracy, precision, recall, f1 = binary_classification_metrics(prediction, binary_test_y)
print("KNN with k = %s" % knn_classifier_3.k)
print("Accuracy: %4.2f, Precision: %4.2f, Recall: %4.2f, F1: %4.2f" % (accuracy, precision, recall, f1))
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

dists = knn_classifier.compute_distances_two_loops(binary_test_X)
assert np.isclose(dists[0, 10], np.sum(np.abs(binary_test_X[0] - binary_train_X[10])))

dists = knn_classifier.compute_distances_one_loop(binary_test_X)
assert np.isclose(dists[0, 10], np.sum(np.abs(binary_test_X[0] - binary_train_X[10])))

dists = knn_classifier.compute_distances_no_loops(binary_test_X)
assert np.isclose(dists[0, 10], np.sum(np.abs(binary_test_X[0] - binary_train_X[10])))


if False:
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

if False:

    num_folds = 5

    fold_step = int(binary_train_X.shape[0] // num_folds)
    ranges = list(range(0, binary_train_X.shape[0], fold_step))
    train_folds_X = [binary_train_X[ranges[i]: ranges[i+1]] for i in range(0, len(ranges) - 1)]
    train_folds_y = [binary_train_y[ranges[i]: ranges[i+1]] for i in range(0, len(ranges) - 1)]

    # TODO: split the training data in 5 folds and store them in train_folds_X/train_folds_y

    k_choices = [1, 2, 3, 5, 8, 10, 15, 20, 25, 50]
    k_to_f1 = {k: 0.0 for k in k_choices}  # dict mapping k values to mean F1 scores (int -> float)

    for k in k_choices:
        knn_classifier = KNN(k=k)
        for X, y in zip(train_folds_X, train_folds_y):
            knn_classifier.fit(X, y)
            prediction = knn_classifier.predict(binary_test_X, num_loops=1)
            accuracy, precision, recall, f1 = binary_classification_metrics(prediction, binary_test_y)
            k_to_f1[k] += f1
        k_to_f1[k] = k_to_f1[k]/len(k_choices)

        # TODO: perform cross-validation
        # Go through every fold and use it for testing and all other folds for training
        # Perform training and produce F1 score metric on the validation dataset
        # Average F1 from all the folds and write it into k_to_f1

        pass

    for k in sorted(k_to_f1):
        print('k = %d, f1 = %f' % (k, k_to_f1[k]))


    best_k = 1
    for k in k_to_f1:
        if k_to_f1[k] > k_to_f1[best_k]:
            best_k = k


    best_knn_classifier = KNN(k=best_k)
    best_knn_classifier.fit(binary_train_X, binary_train_y)
    prediction = best_knn_classifier.predict(binary_test_X, num_loops=1)

    precision, recall, f1, accuracy = binary_classification_metrics(prediction, binary_test_y)
    print("Best KNN with k = %s" % best_k)
    print("Accuracy: %4.2f, Precision: %4.2f, Recall: %4.2f, F1: %4.2f" % (accuracy, precision, recall, f1))


train_X = train_X.reshape(train_X.shape[0], -1)
test_X = test_X.reshape(test_X.shape[0], -1)

knn_classifier = KNN(k=1)
knn_classifier.fit(train_X, train_y)


predict = knn_classifier.predict(test_X, num_loops=1)
accuracy = multiclass_accuracy(predict, test_y)
print("Accuracy: %4.2f" % accuracy)

num_folds = 5

fold_step = int(train_X.shape[0] // num_folds)
ranges = list(range(0, train_X.shape[0], fold_step))
train_folds_X = [train_X[ranges[i]: ranges[i + 1]] for i in range(0, len(ranges) - 1)]
train_folds_y = [train_y[ranges[i]: ranges[i + 1]] for i in range(0, len(ranges) - 1)]

# TODO: split the training data in 5 folds and store them in train_folds_X/train_folds_y

k_choices = [1, 2, 3, 5, 8, 10, 15, 20, 25, 50]
k_to_f1 = {k: 0.0 for k in k_choices}  # dict mapping k values to mean F1 scores (int -> float)

for k in k_choices:
    knn_classifier = KNN(k=k)
    for X, y in zip(train_folds_X, train_folds_y):
        knn_classifier.fit(X, y)
        prediction = knn_classifier.predict(test_X, num_loops=1)
        accuracy = multiclass_accuracy(prediction, test_y)
        k_to_f1[k] += accuracy
    k_to_f1[k] = k_to_f1[k] / len(k_choices)

    # TODO: perform cross-validation
    # Go through every fold and use it for testing and all other folds for training
    # Perform training and produce F1 score metric on the validation dataset
    # Average F1 from all the folds and write it into k_to_f1

    pass

for k in sorted(k_to_f1):
    print('k = %d, f1 = %f' % (k, k_to_f1[k]))

best_k = 1
for k in k_to_f1:
    if k_to_f1[k] > k_to_f1[best_k]:
        best_k = k

best_knn_classifier = KNN(k=best_k)
best_knn_classifier.fit(train_X, train_y)
prediction = best_knn_classifier.predict(test_X, num_loops=1)

accuracy = multiclass_accuracy(prediction, test_y)
print("Best KNN with k = %s" % best_k)
print("Accuracy: %4.2f" % (accuracy,))

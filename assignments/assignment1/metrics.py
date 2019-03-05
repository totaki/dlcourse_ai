def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, accuracy, f1 - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0
    accuracy = (prediction == ground_truth).nonzero()[0].shape[0] / prediction.shape[0]

    true_positive = ground_truth[prediction == True].nonzero()[0].shape[0]
    all_positive = ground_truth[prediction == True].shape[0]
    false_negative = (ground_truth[prediction == False]).nonzero()[0].shape[0]

    precision = true_positive / all_positive
    recall = true_positive / (true_positive + false_negative)

    f1 = (2 * precision * recall) / (precision + recall)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    return 0

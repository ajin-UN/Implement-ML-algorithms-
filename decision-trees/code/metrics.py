import numpy as np

def confusion_matrix(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the confusion matrix. The confusion 
    matrix for a binary classifier would be a 2x2 matrix as follows:
    [
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ]
    YOU DO NOT NEED TO IMPLEMENT CONFUSION MATRICES THAT ARE FOR MORE THAN TWO 
    CLASSES (binary).
    
    Compute and return the confusion matrix.
    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N
    Output:
        confusion_matrix (np.array): 2x2 confusion matrix between predicted and actual labels
    """

    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    result = np.zeros((2, 2))
    
    for a, b in zip(predictions, actual):
        if b:
            if a:
                result[1][1] += 1
            else:
                result[1][0] += 1
        else:
            if a:
                result[0][1] += 1
            else:
                result[0][0] += 1

    return result


def accuracy(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the accuracy:
    Hint: implement and use the confusion_matrix function!
    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N
    Output:
        accuracy (float): accuracy
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    cm = confusion_matrix(actual, predictions)
    result = (cm[0][0] + cm[1][1]) / len(actual)
    return result

def precision_and_recall(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the precision and recall:
    https://en.wikipedia.org/wiki/Precision_and_recall
    Hint: implement and use the confusion_matrix function!
    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N
    Output:
        precision (float): precision
        recall (float): recall
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    cm = confusion_matrix(actual, predictions)
    precision = cm[1][1] / (cm[1][1] + cm[0][1])
    recall = cm[1][1] / (cm[1][1] + cm[1][0])
    return precision, recall


def f1_measure(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the F1-measure:
   https://en.wikipedia.org/wiki/Precision_and_recall#F-measure
    Hint: implement and use the precision_and_recall function!
    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N
    Output:
        f1_measure (float): F1 measure of dataset (harmonic mean of precision and 
        recall)
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    precision, recall = precision_and_recall(actual, predictions)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC

def get_classifier(x, y):
    """
    Create and fit a linear classifier on the provided training data.
    Args:
        x (numpy.ndarray): Inputs for training data.
        y (numpy.ndadday): Target outputs for training data.
    Returns:
        svc (sklearn.svm.LinearSVC): Linear classifier fit to training data.
    """
    svc = LinearSVC()
    svc.fit(x, y.ravel())
    return svc

def run_classifier(svc, x, y_test=None):
    """
    Run a pre-trained classifier on test data.
    Args:
        svc (sklearn.svm.LinearSVC): Linear classifier fit to training data.
        x (numpy.ndarray): Inputs for test data.
        y_test (Optional, numpy.ndarray): Target outputs for test data.
    Returns:
        y_prediction (numpy.ndarray): Predicted classes for each test datapoint.
        score (float): Classifier performance score on test data, computed if y_test is provided.
    """
    y_prediction = svc.predict(x)

    if not y_test is None:
        score = svc.score(x, y_test)
        return y_prediction, score
    return y_prediction


import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC

def fit_classifier(x, y):
    svc = LinearSVC()
    svc.fit(x.reshape(-1, 1), y.ravel())
    return svc

def run_classifier(svc, x, y_test=None):
    x_test = x.reshape(-1, 1)
    y_prediction = svc.predict(x_test)

    if not y_test is None:
        score = svc.score(x_test, y_test)
        return y_prediction, score
    return y_prediction


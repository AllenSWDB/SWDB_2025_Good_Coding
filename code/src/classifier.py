import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC

def fit_classifier(x, y):
    svc = LinearSVC()
    svc.fit(x, y.ravel())
    return svc

def run_classifier(svc, x, y_test=None):
    y_prediction = svc.predict(x)

    if not y_test is None:
        score = svc.score(x, y_test)
        return y_prediction, score
    return y_prediction


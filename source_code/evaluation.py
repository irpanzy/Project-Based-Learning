import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return acc, cm


def print_evaluation(y_true, y_pred):
    acc, cm = evaluate(y_true, y_pred)
    print("Akurasi:", round(acc * 100, 2), "%")
    print("Confusion Matrix:\n", cm)

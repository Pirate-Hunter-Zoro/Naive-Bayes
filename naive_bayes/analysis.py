"""
Code for running the 5-fold cross-validation, training the models, generating predictions, and calculating accuracy and confusion matrices
"""
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from .classifiers import BaseNBClassifier

def run_analysis(X: list[str], y: list[int], classifier: BaseNBClassifier) -> tuple[float, np.ndarray]:
    kfolder = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    confusion_matrices = []
    X = np.array(X, dtype=str)
    y = np.array(y, dtype=int)
    for (train_indices, test_indices) in kfolder.split(X):
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]
        # Fit the classifier on the training set
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        predictions = np.array(predictions, dtype=int)
        accuracy = accuracy_score(y_test, predictions)
        confusion = confusion_matrix(y_test, predictions)
        accuracies.append(accuracy)
        confusion_matrices.append(confusion)

    # Turn the lists into arrays
    accuracies = np.array(accuracies, dtype=float)
    confusion_matrices = np.array(confusion_matrices, dtype=np.ndarray)
    
    mean_accuracy = np.mean(accuracies)
    aggregated_confusion_matrix = np.sum(confusion_matrices, axis=0)
    
    return (mean_accuracy.item(), aggregated_confusion_matrix)
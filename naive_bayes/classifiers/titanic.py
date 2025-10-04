from .base import BaseNBClassifier
from .categorical import CategoricalNB
from .gaussian import GaussianNB
import numpy as np

class TitanicNB(BaseNBClassifier):
    
    def __init__(self, k:float=1.0):
        """Initialize CategoricalNB classifier

        Args:
            k (float, optional): Laplace smoothing factor. Defaults to 1.0.
        """
        self.categorical_scorer = CategoricalNB(k=k)
        self.gaussian_scorer = GaussianNB()
        
    def fit(self, X: list[tuple[list[int],list[float]]], y: list[int]):
        """Fit the Titanic classifier on the input data with its predicted outputs

        Args:
            X (list[tuple[list[int],list[float]]]): Over all observations, categorical features and numeric features
            y (list[int]): Output classifications for each lable
        """
        categorical_X = [x[0] for x in X]
        numeric_X = [x[1] for x in X]
        y_array = np.array(y, dtype=int)
        self.classes = np.unique(y_array).tolist()
        self.priors = {}
        for c in self.classes:
            self.priors[c] = np.log(np.sum(y_array==c)/len(y))
        
        # Now we get train our scorers
        self.categorical_scorer.fit(categorical_X, y)
        self.gaussian_scorer.fit(numeric_X, y)
                    
    
    def predict(self, X: list[tuple[list[int],list[float]]]) -> list[int]:
        """Prediction method to predict classes for given inputs

        Args:
            X (list[tuple[list[int],list[float]]]): Over all observations, categorical features and numeric features

        Returns:
            list[int]: Predicted classes for each observation
        """
        categorical_X = [x[0] for x in X]
        numeric_X = [x[1] for x in X]
        n = len(X)
        
        # Get the class scores from both the categorical and gaussian classifiers
        categorical_scores = self.categorical_scorer.predict_scores(categorical_X)
        numeric_scores = self.gaussian_scorer.predict_scores(numeric_X)
        
        # Loop through the observations and predict their best classes with input from both classifiers
        predictions = []
        for i in range(n):
            record_score = float('-inf')
            record_class = None
            for c in self.classes:
                cat_score = categorical_scores[i][c]
                num_score = numeric_scores[i][c]
                score = self.priors[c] + cat_score + num_score
                if score > record_score:
                    record_score = score
                    record_class = c
            predictions.append(record_class)
        
        return predictions
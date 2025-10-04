from .base import BaseNBClassifier, BaseNBScorer
import numpy as np
import math

class GaussianNB(BaseNBClassifier, BaseNBScorer):
    
    def __init__(self):
        pass
    
    def fit(self, X:list[list[float]], y:list[int]):
        """Fit the model according to the given input data and labels

        Args:
            X (list[list[float]]): Input data
            y (list[int]): corresponding labels
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)
        self.classes = np.unique(y)
        self.priors = {}
        print("     - Computing prior probabilities for each class...")
        for c in self.classes:
            # Compute log probability
            self.priors[c] = np.log(np.sum(y==c)/len(y))
            
        # We need means and variances for each feature for each class
        self.means = {}
        self.variances = {}
        for c in self.classes:
            X_subset = X[y==c]
            means = np.mean(X_subset, axis=0) # Mean across all rows - so the mean of a column - giving the mean value of an attribute over all observations
            self.means[c] = means
            variances = np.var(X_subset, axis=0)
            variances += 1e-10*np.ones(variances.shape) # Avoid division by zero in case a pixel is always empty for instance
            self.variances[c] = variances # Same except for variance
    
    def predict(self, X:list[list[float]]) -> list[int]:
        """Generate model predictions for each given class

        Args:
            X (list[list[float]]): Input data

        Returns:
            list[int]: output predictions
        """
        X = np.array(X, dtype=float)
        predictions = []
        for x in X:
            record_class_score = float('-inf')
            record_class = None
            for c in self.classes:
                # Formula for a single feature's log propbability given a class's mean and variance of said feature is -(x_i-mu_i)^2/(2sigma_i^2)-1/2log(2pi*sigma_i^2)
                score = self.priors[c] - np.sum((x-self.means[c])**2/(2*self.variances[c]))-1/2*np.sum(np.log(2*math.pi*self.variances[c]))
                if score > record_class_score:
                    record_class_score = score
                    record_class = c
            predictions.append(record_class)
        return predictions
    

    def predict_scores(self, X:list[list[float]]) -> list[dict[int,float]]:
        """Generate model prediction scores for each given class for all observations

        Args:
            X (list[list[float]]): Input data

        Returns:
            list[dict[int,float]]: output scores for each class for each observation
        """
        X = np.array(X, dtype=float)
        class_scores_for_observations = []
        for x in X:
            class_scores = {}
            for c in self.classes:
                # Formula for a single feature's log propbability given a class's mean and variance of said feature is -(x_i-mu_i)^2/(2sigma_i^2)-1/2log(2pi*sigma_i^2)
                score = - np.sum((x-self.means[c])**2/(2*self.variances[c]))-1/2*np.sum(np.log(2*math.pi*self.variances[c]))
                class_scores[c] = score
            class_scores_for_observations.append(class_scores)
        return class_scores_for_observations
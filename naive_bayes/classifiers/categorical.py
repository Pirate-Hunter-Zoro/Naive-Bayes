from .base import BaseNBScorer
import numpy as np

class CategoricalNB(BaseNBScorer):
    
    def __init__(self, k: float=1.0):
        self.k = k
        
    def fit(self, X:list[list[int]], y:list[int]):
        """Fit the categorical classifier to the input data

        Args:
            X (list[list[int]]): List of observations with their respective lists of categorical features (as numbers)
            y (list[int]): List of output classifications
        """
        X = np.array(X, dtype=int)
        y = np.array(y, dtype=int)
        self.classes = np.unique(y)
        self.conditionals = {}
        for feature_index in range(X.shape[1]):
            unique_feature_categories = np.unique(X[:,feature_index])
            num_unique_feature_categories = len(unique_feature_categories)
            self.conditionals[feature_index] = {}
            for c in self.classes:
                self.conditionals[feature_index][c] = {}
                X_class_subset = X[y==c]
                num_in_class = len(X_class_subset)
                for feature_category in unique_feature_categories:
                    # Find the observations in this class which have this label for this feature
                    X_class_label_subset = X_class_subset[X_class_subset[:,feature_index]==feature_category]
                    num_class_with_label = len(X_class_label_subset)
                    # We can now compute the log probability
                    numerator = num_class_with_label + self.k
                    denominator = num_in_class + self.k*num_unique_feature_categories
                    self.conditionals[feature_index][c][feature_category] = np.log(numerator/denominator)
    
    def predict_scores(self, X:list[list[int]]) -> list[dict[int,float]]:
        """Predict classes for a given set of inputs with categorical features

        Args:
            X (list[list[int]]): List of observations with their respective lists of categorical features

        Returns:
            list[dict[int,float]]: Predicted class scores for all classes for each input observation
        """
        # Calculate the probability for each class for each observation
        classifications_scores = []
        for x in X:
            class_scores = {}
            for c in self.classes:
                score = 0
                for feature_idx, class_feature_value_probs in self.conditionals.items():
                    # Add the probability that this feature takes on the given value in x for the given class
                    if x[feature_idx] in class_feature_value_probs[c]:
                        # We have seen this value for this feature for this class in our training
                        score += class_feature_value_probs[c][x[feature_idx]]
                class_scores[c] = score
            classifications_scores.append(class_scores)
        
        return classifications_scores
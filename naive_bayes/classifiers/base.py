from abc import ABC, abstractmethod
from enum import Enum

class BaseNBClassifier(ABC):
    
    @abstractmethod
    def fit(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X, y):
        pass


class BaseNBScorer(ABC): # To be used by the Titanic classifier
    
    @abstractmethod
    def fit(self, X, y):
        pass
    
    @abstractmethod
    def predict_scores(self, X, y): # Predicts class scores with NO prior probability considered
        pass


class NBModelName(Enum):
    
    MULTINOMIAL = "Multinomial Naive Bayes"
    BERNOULLI = "Bernoulli Naive Bayes"
    GAUSSIAN = "Gaussian Naive Bayes"
    CATEGORICAL = "Categorical Naive Bayes"
    TITANIC = "Titanic Naive Bayes"
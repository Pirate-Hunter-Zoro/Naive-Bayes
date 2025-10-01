"""
Four Naïve Bayes algorithm variants as separate classes
"""
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum

class BaseNBClassifier(ABC):
    
    @abstractmethod
    def fit(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X, y):
        pass


class NBModelName(Enum):
    
    MULTINOMIAL = "Multinomial Naive Bayes"
    BERNOULLI = "Bernoulli Naive Bayes"


class MultinomialNB(BaseNBClassifier):
    
    def __init__(self, k:float=1.0):
        """Initialize multinomial Naive Bayes model with its smoothing parameter

        Args:
            k (float): laplace smoothing parameter
        """
        self.k = k
        self.priors = None # To eventually store the prior probability of each class
        self.conditionals = None # To eventually store the conditional probability of each word given each class
        self.classes = None # To eventually store the unique class labels found in the training data
    
    def fit(self, X:list[str], y:list[int]):
        """Fit model to given training data with its labels

        Args:
            X (list[str]): List of input observations (e.g. emails)
            y (list[int]): List of corresponding classifications
        """
        print("Fitting MultinomialNB model...")
        # First calculate p(y=l) for each class
        y = np.array(y, dtype=int)
        X = np.array(X, dtype=str)
        self.classes = np.unique(y)
        self.priors = {}
        print("     - Computing prior probabilities for each class...")
        for c in self.classes:
            # Compute log probability
            self.priors[c] = np.log(np.sum(y==c)/len(y))
            
        # Then calculate p(w|y=l) for each word, for each label
        word_counts_per_class = {} # {class -> {word -> int}}
        total_words_per_class = {}
        vocabulary = set()
        print("     - Calculating word counts...")
        for i, c in enumerate(self.classes):
            word_counts_per_class[c] = {}
            total_words_per_class[c] = 0
            relevant_inputs = X[np.where(y==c)]
            for input in relevant_inputs:
                words = input.split()
                total_words_per_class[c] += len(words)
                for word in words:
                    vocabulary.add(word)
                    if word not in word_counts_per_class[c].keys():
                        word_counts_per_class[c][word]=1
                    else:
                        word_counts_per_class[c][word]+=1
                        
        # Make sure all classes that didn't have certain words have a zero count for said words
        for word in vocabulary:
            for c in self.classes:
                if word not in word_counts_per_class[c].keys():
                    word_counts_per_class[c][word]=0
        
        self.vocabulary = vocabulary
        vocab_size = len(self.vocabulary)
        self.conditionals = {}
        print(f"    - Calculating final conditional probabilities...")
        for c in self.classes:
            self.conditionals[c] = {}
            # For this class, we want the probability of each word
            # So our denominator needs to be the total number of words seen in this class 
            # This includes repeats plus our smoothing term
            denominator = total_words_per_class[c] + self.k*vocab_size
            for word in self.vocabulary:
                numerator = word_counts_per_class[c][word] + self.k
                prob = numerator / denominator
                self.conditionals[c][word]=np.log(prob)
        print("...Fit complete")
    
    def predict(self, X:list[str]) -> list[int]:
        """Predict classes of given inputs

        Args:
            X (list[str]): Inputs

        Returns:
            list[int]: Predicted Outputs
        """
        print(f"Predicting {len(X)} documents")
        predictions = []
        for i,x in enumerate(X):
            # Herein lies the Naive Bayes assumption - calculating the maximum probability label
            record_sum = float('-inf')
            record_class = None
            words = x.split()
            for c in self.classes:
                prob_sum = self.priors[c]
                for word in words:
                    if word in self.vocabulary:
                        prob_sum += self.conditionals[c][word]
                if prob_sum > record_sum:
                    record_sum = prob_sum
                    record_class = c
            predictions.append(record_class)
        return predictions
    
    
class BernoulliNB(BaseNBClassifier):
    
    def __init__(self, k:float=1):
        """Constructor for the Bernoulli Naive Bayes classifier

        Args:
            k (float, optional): Laplace smoothing parameter. Defaults to 1.
        """
        self.k = k
        self.priors = None
        self.conditionals = None
        
    def fit(self, X:list[str], y: list[int]):
        """Fit method for the Bernoulli Naive Bayes classifier on the given data

        Args:
            X (list[str]): List of input strings
            y (list[int]): Classes of each observation
        """
        print("Fitting BernouliNB model...")
        # First calculate p(y=l) for each class
        y = np.array(y, dtype=int)
        X = np.array(X, dtype=str)
        self.classes = np.unique(y)
        self.priors = {}
        print("     - Computing prior probabilities for each class...")
        for c in self.classes:
            # Compute log probability
            self.priors[c] = np.log(np.sum(y==c)/len(y))
        
        # Keep track of total vocabulary
        print("     - Amassing entire vocabulary of training observations...")
        self.vocabulary = set()
        for doc in X:
            words = doc.split(" ")
            for word in words:
                self.vocabulary.add(word)
        
        # Find the number of documents of a specific class that contain a specific word
        print("     - Calculating - for each class - for each word, the number of documents of said class containing said word...")
        doc_counts_per_class_with_word = {c: {} for c in self.classes}
        for doc, c in zip(X,y):
            doc_words = set(doc.split(" ")) # For the same document, we don't want to count a word twice
            for word in doc_words:
                if word not in doc_counts_per_class_with_word[c].keys():
                    doc_counts_per_class_with_word[c][word] = 1
                else:
                    doc_counts_per_class_with_word[c][word] += 1
        for c in doc_counts_per_class_with_word.keys():
            for word in self.vocabulary:
                if word not in doc_counts_per_class_with_word[c].keys():
                    doc_counts_per_class_with_word[c][word] = 0
                    
        # Find the conditional probabilities
        print("     - Calculating conditional probabilities")
        doc_counts_per_class = {}
        for c in self.classes:
            doc_counts_per_class[c] = np.sum(y==c)
        self.conditionals = {}
        for c in self.classes:
            self.conditionals[c] = {}
            for w in self.vocabulary:
                # Find the probability that the word is present given the class label
                numerator = doc_counts_per_class_with_word[c][w]+self.k
                denominator = doc_counts_per_class[c]+2*self.k
                log_prob_present = np.log(numerator/denominator)
                log_prob_absent = np.log(1-numerator/denominator)
                self.conditionals[c][w] = (log_prob_present, log_prob_absent)
    
    def predict(self, X:list[str]) -> list[int]:
        """Prediction method for the Bernoulli Naive Bayes classifier given a list of data

        Args:
            X (list[float]): Input data of strings

        Returns:
            list[int]: Predicted classes for said data
        """
        print(f"Predicting {len(X)} documents")
        predictions = []
        for doc in X:
            doc_words = set(doc.split(" "))
            record_score = float('-inf')
            record_class = None
            for c in self.classes:
                # What's the probability score for this class being the one?
                score = self.priors[c]
                for word in self.vocabulary:
                    if word in doc_words:
                        # Add the log probability that this word IS present in an instance of the given class
                        score += self.conditionals[c][word][0]
                    else:
                        # Add the log probability that this word is NOT present in an instance of the given class
                        score += self.conditionals[c][word][1]
                if score > record_score:
                    record_score = score
                    record_class = c
            predictions.append(record_class)
        return predictions
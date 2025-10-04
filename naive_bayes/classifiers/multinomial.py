from .base import BaseNBClassifier
import numpy as np

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
            if i % (len(X)//10) == 0:
                print(f"    - {int(100 * i/len(X))}% finished with predictions...")
            predictions.append(record_class)
        return predictions
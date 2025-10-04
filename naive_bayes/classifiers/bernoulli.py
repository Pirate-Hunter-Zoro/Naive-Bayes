from .base import BaseNBClassifier
import numpy as np

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
        for i, doc in enumerate(X):
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
            if i % (len(X)//10) == 0:
                print(f"    - {int(100 * i/len(X))}% finished with predictions...")
            predictions.append(record_class)
        return predictions
    
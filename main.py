"""
This is the entry point. 
A simple script to execute the entire analysis for each part of the project.
"""

from naive_bayes.data_loader import load_spam_ham_data
from naive_bayes.analysis import run_analysis
from naive_bayes.classifiers import MultinomialNB
import os
from pathlib import Path

if __name__=="__main__":
    data_path = "data/emails.csv"
    X, y = load_spam_ham_data(data_path)
    
    multinomial_nb = MultinomialNB()
    accuracy, confusion_matrix = run_analysis(X, y, multinomial_nb)
    
    results_mn_spam_ham = f"""Multinomial Naive Bayes on Spam/Ham Data Set:
    Mean Accuracy: {accuracy}, 
    Confusion Matrix: {confusion_matrix[0]}
                      {confusion_matrix[1]}
    """
    
    os.makedirs("results", exist_ok=True)
    with open(Path("results/performance.txt"), 'w') as f:
        f.write(results_mn_spam_ham)
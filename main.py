"""
This is the entry point. 
A simple script to execute the entire analysis for each part of the project.
"""

from naive_bayes.data_loader import *
from naive_bayes.analysis import run_analysis
from naive_bayes.classifiers import *
import os
from pathlib import Path

emails_path = "data/emails.csv"
titanic_path = "data/Titanic-Dataset.csv"

def make_model(model_name: NBModelName) -> BaseNBClassifier:
    """Return the corresponding Naive Bayes variant

    Args:
        model_name (NBModelName): variant specification

    Returns:
        BaseNBClassifier: whatever variant classifier is specified
    """
    if model_name == NBModelName.MULTINOMIAL:
        return MultinomialNB()
    elif model_name == NBModelName.BERNOULLI:
        return BernoulliNB()

def create_confusion_matrix_str(confusion_matrix: np.array) -> str:
    """Print out confusion matrix in a visually appealing form

    Args:
        confusion_matrix (np.array): Input confusion matrix

    Returns:
        str: Resulting string
    """
    return "\n      ".join([f"{confusion_matrix[i]}" for i in range(len(confusion_matrix))])

def run_tests(model_name: NBModelName):
    """Run all four datasets on the given classifier to see how it does

    Args:
        model_name (NBModelName): Some Naive Bayes classifier variant enum specification
    """
    X, y = load_spam_ham_data(emails_path)
    accuracy, confusion_matrix = run_analysis(X, y, make_model(model_name))
    results_mn_spam_ham = f"""{model_name.value} on Spam/Ham Data Set:
    Mean Accuracy: 
        {accuracy}, 
    Confusion Matrix: 
      {create_confusion_matrix_str(confusion_matrix)}
    """
    
    # News groups
    X, y = load_news_groups_data()
    accuracy, confusion_matrix = run_analysis(X, y, make_model(model_name))
    results_news_groups = f"""{model_name.value} on News Group Data Set:
    Mean Accuracy:
        {accuracy},
    Confusion Matrix:
      {create_confusion_matrix_str(confusion_matrix)}
    """
    
    os.makedirs("results", exist_ok=True)
    with open(Path(f"results/{model_name.value} Performance.txt"), 'w') as f:
        f.write(results_mn_spam_ham)
        f.write("\n\n")
        f.write(results_news_groups)

if __name__=="__main__":
    run_tests(NBModelName.MULTINOMIAL)
    run_tests(NBModelName.BERNOULLI)
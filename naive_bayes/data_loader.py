"""
Functions responsible for loading and preprocessing each of the four datasets 
"""
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

def load_spam_ham_data(file_path: str) -> tuple[list[str],list[int]]:
    """Load the email data set

    Args:
        file_path (str): path to the email data

    Returns:
        tuple[list[str],list[int]]: X and y (emails and labels)
    """
    df = pd.read_csv(file_path, dtype={'text': str, 'spam': int})
    return (df['text'].tolist(), df['spam'].tolist())

def load_news_groups_data() -> tuple[list[str], list[int]]:
    """Load the newsgroups data set from sklearn

    Returns:
        tuple[list[str], list[int]]: X and y (articles and lables)
    """
    newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))
    return(newsgroups_data.data, newsgroups_data.target)
"""
Functions responsible for loading and preprocessing each of the four datasets 
"""
import pandas as pd

def load_spam_ham_data(file_path: str) -> tuple[list[str],list[int]]:
    """Load the email data set

    Args:
        file_path (str): path to the email data

    Returns:
        tuple[list[str],list[int]]: X and y (emails and labels)
    """
    df = pd.read_csv(file_path, dtype={'text': str, 'spam': int})
    return (df['text'].tolist(), df['spam'].tolist())
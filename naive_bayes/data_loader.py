"""
Functions responsible for loading and preprocessing each of the four datasets 
"""
import pandas as pd
from sklearn.datasets import fetch_20newsgroups, load_digits
import re

stop_words = set([
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 
    'can', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 
    'each', 'few', 'for', 'from', 'further', 
    'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 
    'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', 
    "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 
    'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 
    'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 
    'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 
    'under', 'until', 'up', 
    'very', 
    'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 
    'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'
])

def _preprocess_text(text: str) -> str:
    """Helper method to preprocess test so that it does not contain a bunch of useless words/characters that will confuse our model

    Args:
        text (str): raw input string

    Returns:
        str: resulting cleaned string
    """
    text = text.lower()
    punctuation_pattern = "[^a-z ]" # Not a letter or white space
    text = re.sub(punctuation_pattern, "", text)
    words = text.split()
    filtered_words = []
    for word in words:
        if word not in stop_words:
            filtered_words.append(word)
    
    return " ".join(filtered_words)

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
    raw_text_data = newsgroups_data.data
    filtered_data = []
    for article in raw_text_data:
        # Remove all punctuation, capitalization, useless 'stop' words, etc.
        filtered_data.append(_preprocess_text(article))
    return (filtered_data, newsgroups_data.target)

def load_handwritten_digits() -> tuple[list[list[float]], list[int]]:
    """Load the handwritten digits dataset from sklearn

    Returns:
        tuple[list[list[float]], list[int]]: Image pixel arrays along with their classifications
    """
    data = load_digits()
    return (data.data, data.target)

def load_titanic_data(file_path: str) -> tuple[tuple[list[list[int]],list[list[float]]],list[int]]:
    """Load the titanic dataset which has both numeric and categorical variables

    Args:
        file_path (str): path to the titanic data

    Returns:
        tuple[list[list[int]],list[list[float]],list[int]]: Categorical attributes, numeric attributes, classifications
    """
    df = pd.read_csv(file_path)
    df = df.drop(labels=['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    
    # Handle missing embarking values
    embarked_mode = df['Embarked'].mode()[0]
    df['Embarked'] = df['Embarked'].fillna(embarked_mode)
    
    # Handle missing age values
    median_age = df['Age'].median()
    df['Age'] = df['Age'].fillna(median_age)
    
    # Convert sex and embark categories to 0/1 binary values
    df['Sex'] = df['Sex'].replace({'male':0, 'female':1})
    df['Embarked'] = df['Embarked'].replace({'S':0, 'C':1, 'Q':2})
    
    # Split up the data
    categorical_cols = ['Pclass', 'Sex', 'Embarked']
    numeric_cols = ['Age', 'SibSp', 'Parch', 'Fare']
    
    X_categorical = df[categorical_cols].to_numpy().tolist()
    X_numeric = df[numeric_cols].to_numpy().tolist()
    X = [(first, second) for first, second in zip(X_categorical,X_numeric)]
    y = df['Survived'].tolist()
    
    return (X, y)
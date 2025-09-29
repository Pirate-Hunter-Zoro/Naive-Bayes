# CS 5333/7333 Machine Learning - Project 2: Naïve Bayes

## Project Overview

This project involves the implementation of four variants of the Naïve Bayes learning algorithm from scratch in Python. The goal is to build, train, and evaluate these classifiers on a variety of datasets to understand their underlying assumptions and performance characteristics.

The models are evaluated on two text datasets and two tabular datasets.  Performance is measured using 5-fold cross-validation, with results presented as average accuracy and confusion matrices for each model-dataset combination.

## Algorithms Implemented

The following four Naïve Bayes variants are implemented:

* **Multinomial Naïve Bayes**: A probabilistic model for text classification based on word frequency.
* **Multi-Variate Bernoulli Naïve Bayes**: A model for text classification where features represent the presence or absence of words from the vocabulary.
* **Categorical Naïve Bayes**: A variant designed for datasets with discrete, categorical features.
* **Gaussian Naïve Bayes**: A variant that assumes features follow a Gaussian (normal) distribution, suitable for continuous numeric data.

## Datasets

The classifiers are tested on the following four datasets:

1. **Spam Ham**: A text dataset of emails to be classified as spam or legitimate.
2. **Newsgroups**: A text dataset of news articles to be classified into one of 20 topics.
3. **Handwritten Digits**: A tabular dataset where pixel values are treated as numeric features for digit classification.
4. **Titanic**: A tabular dataset with both categorical and numeric features used to predict passenger survival.

## Project Structure

The project is organized into the following directory structure:

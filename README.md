# Naïve Bayes

## Project Overview

This project involves the implementation of four variants of the Naïve Bayes learning algorithm from scratch in Python. The goal is to build, train, and evaluate these classifiers on a variety of datasets to understand their underlying assumptions and performance characteristics.

The models are evaluated on two text datasets and two tabular datasets. Performance is measured using 5-fold cross-validation, with results presented as average accuracy and confusion matrices for each model-dataset combination.

## Algorithms Implemented

The following four Naïve Bayes variants are implemented:

* **Multinomial Naïve Bayes**: A probabilistic model for text classification based on word frequency.
* **Multi-Variate Bernoulli Naïve Bayes**: A model for text classification where features represent the presence or absence of words from the vocabulary.
* **Gaussian Naïve Bayes**: A variant that assumes features follow a Gaussian (normal) distribution, suitable for continuous numeric data.
* **Categorical Naïve Bayes**: A variant designed for datasets with discrete, categorical features.
* **Hybrid Titanic Classifier**: A combined model that uses Gaussian NB for numeric features and Categorical NB for categorical features.

## Datasets

The classifiers are tested on the following four datasets:

1. **Spam Ham**: A text dataset of emails to be classified as spam or legitimate.
2. **Newsgroups**: A text dataset of news articles to be classified into one of 20 topics.
3. **Handwritten Digits**: A tabular dataset where pixel values are treated as numeric features for digit classification.
4. **Titanic**: A tabular dataset with both categorical and numeric features used to predict passenger survival.

## Project Structure

The project is organized into a main package `naive_bayes` with a sub-package for the classifiers.

```text
project\_root/
├── data/
│   ├── emails.csv
│   └── Titanic-Dataset.csv
├── results/
│   ├── (Performance reports are saved here)
├── naive\_bayes/
│   ├── **init**.py
│   ├── classifiers/
│   │   ├── **init**.py
│   │   ├── base.py
│   │   ├── bernoulli.py
│   │   ├── categorical.py
│   │   ├── gaussian.py
│   │   ├── multinomial.py
│   │   └── titanic.py
│   ├── analysis.py
│   └── data\_loader.py
├── main.py
└── requirements.txt
```

## Setup and Execution

1. **Create Environment:** It is recommended to use a virtual environment.

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

2. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run Analysis:** The `main.py` script is configured to run all tests.

    ```bash
    python main.py
    ```

    The results for each model are saved as `.txt` files in the `results/` directory.

## Performance Summary

The following table summarizes the final 5-fold cross-validation accuracy for each model on its designated dataset(s).

| Classifier | Dataset | 5-Fold Mean Accuracy |
| :--- | :--- | :---: |
| **Multinomial Naïve Bayes** | Spam/Ham | 98.74% |
| | Newsgroups | 63.93% |
| **Bernoulli Naïve Bayes** | Spam/Ham | 98.67% |
| | Newsgroups | 42.83% |
| **Gaussian Naïve Bayes** | Handwritten Digits | 79.08% |
| **Hybrid (Titanic) NB** | Titanic | 78.57% |

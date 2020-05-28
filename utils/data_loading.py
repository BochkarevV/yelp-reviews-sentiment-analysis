import os
import pandas as pd

from sklearn.model_selection import train_test_split

# Path to the compressed JSON
PATH_DATA = 'data', 'yelp_reviews.json.gz'


def load_and_split_dataset(path=os.path.join(*PATH_DATA), train_ratio=0.6, random_state=0):
    """
    Load dataset and split into training, validation and test sets.
    Validation and test set sizes are of equal sizes: (1 - train_ratio)/2.

    :param path: string, optional (default='data/yelp_reviews.json.gz')
        Path to the data.
    :param train_ratio: float, optional (default=0.6)
        Proportion of the dataset to include in the training set.
        Has to be between 0.0 and 1.0.
    :param random_state: int, optional (default=0)
        Seed used by the random number generator.

    :return: Tuple of three tuples containing pairs of
             predictive and target variables.

    Example
    -------
    >>> from utils.data_loading import load_and_split_dataset
    >>> (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_split_dataset()
    """

    # Load the data into a DataFrame
    df = pd.read_json(path, compression='gzip', lines=True)

    # Train, val, test split
    X = df['text']
    y = df['stars']

    # First, obtain training set
    X_train, X_rest, y_train, y_rest = train_test_split(X, y,
                                                        train_size=train_ratio,
                                                        random_state=random_state)

    # Split remaining part of the data equally into validation and test sets
    # Calculate another random state for repeatable splits
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest,
                                                    test_size=0.5,
                                                    random_state=random_state * 7 + 3511)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

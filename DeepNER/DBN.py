"""Deep Belief Network module."""

from warnings import filterwarnings
filterwarnings("ignore")

import numpy as np
from nolearn.dbn import DBN
from sklearn.cross_validation import train_test_split

def make_data(filename, test_size=0.33):
    """
    Read data from csv file and convert words, tags, and vectors into
    numpy ndarrays.
    """

    #load DataFrame
    from Preprocess import get_data
    df = get_data(filename)

    #split DataFrame into train and test sets
    train_data, test_data = train_test_split(df, test_size=test_size)

    return train_data.values, test_data.values

def split_data_matrix(matrix):
    """
    Split a matrix of form seq. no, word, tag, vector into its
    word, tag, and vector componenets.
    """
    
    #remove sequence number column
    from copy import deepcopy
    matrix = np.delete(matrix, 0, 1)

    words = matrix[:,0]
    tags = matrix[:,1]
    vectors = matrix[:,2:]

    return words, tags, vectors

def main():
    """."""

    train_data, test_data = make_data('./vectors/vectors10.csv')

    train_words, train_labels, train_vectors = split_data_matrix(train_data)
    test_words, test_labels, test_vectors = split_data_matrix(test_data)

    input_units = train_vectors.shape[1]

    dbn = DBN(
        [input_units, 100, 10],
        learn_rates = 0.3,
        learn_rate_decays = 0.9,
        epochs = 10,
        verbose = 1)
    
    #dbn.fit(trainX, trainY)

if __name__ == "__main__":
    main()
"""Deep Belief Network module."""

from warnings import filterwarnings
filterwarnings("ignore")

import numpy as np
from sklearn import datasets
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
    vectors = matrix[:,2:].astype(np.float64)

    return words, tags, vectors

def map_labels(labels, all_tags):
    """Map unicode labels to numpy.int64 values."""
    mapping = {}
    for i, label in enumerate(all_tags):
        labels[labels == label] = np.int64(i)
        mapping[i] = label

    mapped_labels = labels.astype(np.int64)

    return mapped_labels, mapping

def init_dbn(topology, learn_rates=0.3, learn_rate_decays=0.9, epochs=10, verbose=1):
    """Initialize a DBN object."""

    dbn = DBN(
        topology,
        learn_rates = 0.3,
        learn_rate_decays = 0.9,
        epochs = 10,
        verbose = 1)

    return dbn

def main():
    """."""

    #make training and test sets
    train_data, test_data = make_data('./vectors/vectors100.csv')


    #split sets into their components
    train_words, train_labels, train_vectors = split_data_matrix(train_data)
    test_words, test_labels, test_vectors = split_data_matrix(test_data)


    #map tags to np.int64's
    train_tags, test_tags = np.unique(train_labels), np.unique(test_labels)
    all_tags = list(set(train_tags) | set(test_tags))

    train_labels, mapping = map_labels(train_labels, all_tags)
    test_labels, mapping = map_labels(test_labels, all_tags)


    #initialize a DBN
    topology = [train_vectors.shape[1], 300, 300, 300, 300, 300, len(all_tags)]
    dbn = init_dbn(topology)

    #fit DBN
    dbn.fit(train_vectors, train_labels)

if __name__ == "__main__":
    main()
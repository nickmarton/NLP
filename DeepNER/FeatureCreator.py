"""Deep Belief Network module."""

from warnings import filterwarnings
filterwarnings("ignore")

import sys
import os
import logging
import timeit
import theano
import numpy as np
import theano.tensor as T
from sklearn.cross_validation import train_test_split
from lisa.DBN import DBN

def set_verbosity(verbose_level=3):
    """Set the level of verbosity of the Preprocessing."""
    if not type(verbose_level) == int:
        raise TypeError("verbose_level must be an int")

    if verbose_level < 0 or verbose_level > 4:
        raise ValueError("verbose_level must be between 0 and 4")

    verbosity = [
        logging.CRITICAL,
        logging.ERROR,
        logging.WARNING,
        logging.INFO,
        logging.DEBUG]

    logging.basicConfig(
        format='%(asctime)s:\t %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=verbosity[verbose_level])

def cast_to_Theano(dataset):
    """."""
    data_x, data_y = dataset
    shared_x = theano.shared(np.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=True)
    shared_y = theano.shared(np.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=True)
    
    return shared_x, T.cast(shared_y, 'int32')

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

    train_data, valid_data = train_test_split(train_data, test_size=test_size)

    return train_data.values, valid_data.values, test_data.values

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

def dump_model(model):
    """Dump DBN object into pickled file."""
    import pickle
    logging.info("Dumping model into model.pkl")
    with open('model.pkl', 'w') as dump_file:
        pickle.dump(model, dump_file)

def main():
    """."""

    set_verbosity(0)
    #make training and test sets
    #train_data, valid_data, test_data = make_data('./vectors/vectors300.csv', test_size=0.20)
    train_data, valid_data, test_data = make_data('./vectors/googlevectors300.csv', test_size=0.20)
    #train_data, valid_data, test_data = make_data('./vectors/subset_googlevectors.csv', test_size=0.20)
    logging.info("Created raw training and test sets")


    #split sets into their components
    train_words, train_labels, train_vectors = split_data_matrix(train_data)
    valid_words, valid_labels, valid_vectors = split_data_matrix(valid_data)
    test_words, test_labels, test_vectors = split_data_matrix(test_data)


    #map tags to np.int64's
    train_tags, valid_tags, test_tags = np.unique(train_labels), np.unique(valid_labels), np.unique(test_labels)
    all_tags = list(set(train_tags) | set(valid_tags) | set(test_tags))

    #map labels in each set to ints
    train_labels, mapping = map_labels(train_labels, all_tags)
    valid_labels, mapping = map_labels(valid_labels, all_tags)
    test_labels, mapping = map_labels(test_labels, all_tags)
    logging.info('Created parsed training and test data')


    #prep data for lisa-labs DBN; convert to Theano
    train_set = (train_vectors, train_labels)
    valid_set = (valid_vectors, valid_labels)
    test_set = (test_vectors, test_labels)

    train_set_x, train_set_y = cast_to_Theano(train_set)
    valid_set_x, valid_set_y = cast_to_Theano(valid_set)
    test_set_x, test_set_y = cast_to_Theano(test_set)

    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)
    test_set = (test_set_x, test_set_y)
    datasets = (train_set, valid_set, test_set)


    finetune_lr=0.1
    pretraining_epochs=10
    pretrain_lr=0.01
    k=1
    training_epochs=1000
    batch_size=10


    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # numpy random generator
    numpy_rng = np.random.RandomState(123)
    print '... building the model'
    # construct the Deep Belief Network

    dbn = DBN(
        numpy_rng=numpy_rng,
        n_ins=train_vectors.shape[1],
        hidden_layers_sizes=[200, 200],
        n_outs=len(all_tags))

    # start-snippet-2
    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size,
                                                k=k)

    print '... pre-training the model'
    start_time = timeit.default_timer()
    ## Pre-train layer-wise
    for i in xrange(dbn.n_layers):
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print np.mean(c)

    end_time = timeit.default_timer()
    # end-snippet-2
    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = dbn.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print '... finetuning the model'
    # early-stopping parameters
    patience = 4 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.    # wait this much longer when a new best is
                              # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatches before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:

                validation_losses = validate_model()
                this_validation_loss = np.mean(validation_losses)
                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%'
                    % (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = np.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'obtained at iteration %i, '
            'with test performance %f %%'
        ) % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    )
    print >> sys.stderr, ('The fine tuning code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time)
                                              / 60.))


if __name__ == "__main__":
    main()
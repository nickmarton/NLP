"""Deep Belief Network module."""

from warnings import filterwarnings
filterwarnings("ignore")

import os
import sys
import logging
import numpy as np
from nolearn.dbn import DBN
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from Preprocess import get_data, get_overlap, map_labels, save_overlap

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

def dump_model(model, filename):
    """Dump DBN object into pickled file."""
    import pickle
    logging.info("Dumping model into model.pkl")
    with open(filename, 'w') as dump_file:
        pickle.dump(model, dump_file)

def parse_data(data, label_map):
    """."""
    X, y = [], []
    for datum in data.values:
        X.append(datum[3:])
        y.append(label_map[datum[2]])

    X = np.array(X).astype(np.float)
    y = np.array(y)

    return X, y

def main():
    """."""

    from sklearn.cross_validation import KFold

    set_verbosity(3)


    overlap_df = get_data("./vectors/google_overlap.csv")
    #overlap_df = get_data("./vectors/freebase_overlap.csv")

    overlap_df = overlap_df[overlap_df.NER != 'O']
    overlap_df = overlap_df.groupby("NER").filter(lambda x: len(x) > 50)

    label_map, labels = map_labels(overlap_df)

    X, y = parse_data(overlap_df, label_map)
    trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.00)

    #'''
    count, n_folds, scores = 0, 30, []
    logging.info("Beginning Cross Validation with " + str(n_folds) + " folds")    
    
    kf = KFold(len(trainX), n_folds=n_folds)
    lrs = list(np.linspace(0.1, 0.4, num=n_folds))
    for train, test in kf:
        logging.debug("TRAIN:" + str(len(train)) + " TEST:" + str(len(test)))
        trainX_fold, validX_fold = trainX[train], trainX[test]
        trainY_fold, validY_fold = trainY[train], trainY[test]
    
        google_topology = [trainX_fold.shape[1], 300, 200, 100, len(labels)]
        #freebase_topology = [trainX_fold.shape[1], 750, 500, 250, len(labels)]

        dbn = DBN(
            #freebase_topology,
            google_topology,
            learn_rates=float(lrs[count]),
            learn_rate_decays=0.9,
            epochs=50,
            verbose=0)

        dbn.fit(trainX_fold, trainY_fold)
        score = dbn.score(validX_fold, validY_fold)
        scores.append((score, float(lrs[count])))

        count += 1
        logging.info(
            "Learning rate: " + str(float(lrs[count-1])) + " score:" + \
            str(score) + " " + str(float(count)/float(n_folds) * 100) + "% done")

    best_lr = max(scores, key=lambda x: x[0])[1]
    logging.info("Best CV score: " + str(best_lr))

    google_topology = [trainX.shape[1], 300, 200, 100, len(labels)]
    #freebase_topology = [trainX.shape[1], 750, 500, 250, len(labels)]

    dbn = DBN(
        #freebase_topology,
        google_topology,
        learn_rates=best_lr,
        learn_rate_decays=0.9,
        epochs=100,
        verbose=1)

    dbn.fit(trainX, trainY)

    #preds = dbn.predict(testX)
    #print classification_report(testY, preds)

    model_and_data = (dbn, label_map)
    dump_model(model_and_data, './google_model.pkl')

    #'''

    '''
    from sklearn.svm import SVC
    clf = SVC()
    clf.fit(trainX, trainY)
    preds = clf.predict(testX)
    print classification_report(testY, preds)
    '''
    

if __name__ == "__main__":
    main()
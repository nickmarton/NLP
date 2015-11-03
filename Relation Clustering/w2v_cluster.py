"""
Nicholas Marton

To build dependencies, use
1. pip install word2vec
2. pip install bcubed
"""

import os
import codecs
import bcubed
import logging
import word2vec
from collections import defaultdict

logging.basicConfig(
    format='%(asctime)s:\t %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO)

def aggregate_input_and_ground_truths(path='./triggerdata/'):
    """Go into ./triggerdata/ and concatenate all lists into single list."""
    data = []
    ldict = defaultdict(set)

    for filename in os.listdir(path):
        with open(path + filename, "r") as f:
            for line in f:
                entry = unicode(line.strip().replace(" ", "_"), errors='ignore')
                entry = str(entry)

                ldict[entry].add(filename[0:filename.find(".")])
                data.append(entry)


    with open("gt_input.txt", "w") as f:
        for datum in data:
            f.write(datum + '\n')

    return ldict

def preprocess():
    """Preprocess given data and build vocabulary file."""

    def extend_vocab():
        """."""
        ground_truth_vocab = ''
        with open('gt_input.txt', 'r') as f:
            for line in f:
                ground_truth_vocab += line.strip()
                ground_truth_vocab += " "
        ground_truth_vocab = ground_truth_vocab[:-1]

        text8_vocab = ''
        with open('./text8-phrases', 'r') as f:
            for line in f:
                text8_vocab += line

        return text8_vocab + " " + ground_truth_vocab

    logging.info('Beginning preprocessing:')

    #covert text8 vocabulary to include bigram phrases
    word2vec.word2phrase('./text8', './text8-phrases', verbose=True, min_count=1)

    logging.info('Done creating test8-phrases.')

    #extend text8-phrases vocab with ground truth vocab then write to file
    full_vocab = extend_vocab()
    with open ('./text8-phrases-extra', 'w') as f:
        f.write(full_vocab)

    logging.info('Done creating test8-phrases-extra')
    logging.info('Done preprocessing')

def main():
    """Main method."""
    k = 35

    #write ground truth vocabulary to gt_input.txt and get ground truth
    #dictionary
    ldict = aggregate_input_and_ground_truths()
    logging.info('Done generating ldict and ground truth text file.')

    #if file containing clusters hasn't already been created, create it
    if not os.path.isfile('./clusters.txt'):

        preprocess()

        #train word2vec and cluster output from the full vocab
        word2vec.word2clusters('./text8-phrases-extra', './clusters.txt',
                                            k, verbose=True, min_count=1)

        logging.info('Done training.')
        logging.info('Done creating clusters.')

    #load clusters
    clusters = word2vec.load_clusters('./clusters.txt')

    #build cluster dictionary from full vocabulary
    cdict = {}
    for i in range(0, k):
        for word in clusters.get_words_on_cluster(i):
            cdict[word] = set([i])

    logging.info('Done generating cdict.')

    #trim cluster dictionary down to only keys included in ground truths
    trimmed_cdict = {}
    for key in ldict.keys():
        try:
            trimmed_cdict[key] = cdict[key]
        except:
            pass

    logging.info('done trimming cdict; begining scoring\n')

    #compute bcubed score
    precision = bcubed.precision(trimmed_cdict, ldict)
    recall = bcubed.recall(trimmed_cdict, ldict)
    fscore = bcubed.fscore(precision, recall)

    print "precision: {p}, \t recall: {r}, \t fscore: {f}".format(p=precision, r=recall, f=fscore)

    logging.info('done scoring\n')

if __name__ == "__main__":
    main()
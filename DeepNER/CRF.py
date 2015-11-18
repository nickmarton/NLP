"""
Nicholas Marton
Conditional Random Field model for Entity-Detection.

Note:
the crfsuite dependency can be built hassle-free with the shell code in the
URL http://bl.ocks.org/madelfio/2693682 when running linux in administrative
mode.

pycrfsuite can be build using pip install python-crfsuite
I modeled some code after an example found in 
https://github.com/tpeng/python-crfsuite however I produced all the final code.

To implement pmi clusters, I used an external file from 
https://github.com/mheilman/tan-clustering; the cluster information is
contained in cluster_output.txt
"""

import logging
from itertools import chain
import nltk
import pycrfsuite
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from Preprocess import set_verbosity

cluster = None
window_number = 2

def get_clusters(cut, filename):
    """Generate clusters by cutting tree at cut."""
    data = []

    #get word and its bitstring, ignore count
    with open(filename) as f:
        for line in f:
            word, bitstring = line.strip().split('\t')[0:2]
            data.append((word, bitstring))

    #generate cluster map; bitstring for keys,
    #list of words with same bitstring cuts for values
    from collections import defaultdict
    clusters = defaultdict(list)
    
    #for each bitstring-word pair
    for datum in data:
        word, bitstring = datum
        #cut the bitstring at specified cut and save it in clusters dict
        if len(bitstring) >= cut:
            tup = tuple(word[2:-2].split("', '"))
            #correct the words, tag, label tuples that didn't parse right
            if len(tup) == 2:
                bad_word, bad_tag = tup[0].split(",")
                word, tag, label = bad_word[:-1], bad_tag[2:], tup[1]
                tup = (word, tag, label)
            if len(tup) == 1:
                word, tag, label = word.split(",")
                word, tag, label = word[2:-1], word[2:-1], label[2:-2]
                tup = (word, tag, label)
            #save formatted tuple
            clusters[bitstring[0:cut]].append(tup)

    return clusters

def word2features(sent, i):
    """Convert a given word to its set of features for CRF."""
    def get_base_features():
        """Obtain the base feature set."""
        base_features = [
            'bias',
            'word.lower=' + word.lower(),
            'word[-3:]=' + word[-3:],
            'word[-2:]=' + word[-2:],
            'word.isupper=%s' % word.isupper(),
            'word.isdigit=%s' % word.isdigit(),
            'postag=' + postag,
            'postag[:2]=' + postag[:2],
        ]

        if i == 0:
            base_features.append('BOS')

        if i == len(sent) - 1:
            base_features.append('EOS')

        return base_features

    def get_window_features(window_number):
        """Get sliding window features."""

        #get unfiltered window indicies
        unf_indicies = [j for j in range(i - window_number, i + window_number + 1)]

        #filter the indicies to get the valid window indicies
        window_indicies = []
        for j in unf_indicies:
            #eliminate i, and indicies below 0 or above the last token index
            i_cond = j == i
            lb_cond = j < 0
            ub_cond = j > len(sent) - 1

            if not i_cond and not lb_cond and not ub_cond:
                window_indicies.append(j)

        window_features = []

        for w in window_indicies:

            word_n = sent[w][0]
            postag_n = sent[w][1]

            #get prefix for feature
            if w < i:
                prefix = str(w - i)
            if i < w:
                prefix = str(w - i)

            window_features.extend([
                prefix + ':word.lower=' + word_n.lower(),
                prefix + ':word[-3:]=' + word_n[-3:],
                prefix + ':word[-2:]=' + word_n[-2:],
                prefix + ':word.isupper=%s' % word_n.isupper(),
                prefix + ':word.isdigit=%s' % word_n.isdigit(),
                prefix + ':postag=' + postag_n,
                prefix + ':postag[:2]=' + postag_n[:2],
            ])

        return window_features

    def get_cluster_features(clusters, restricted=True):
        """Add features to each datum based on the clusters they appear in."""
        
        cluster_words = []
        word = sent[i][0]
        postag = sent[i][1]
        label = sent[i][2]

        #get words in this word's cluster
        for bitstring, tuples in clusters.iteritems():
            words = [tup[0] for tup in tuples]
            if word in words and len(words) != 1:
                cluster_words = tuples
                break

        cluster_features = []

        #Add features for each word in cluster if they share the same label
        for ind, c_tuple in enumerate(cluster_words):
            c_word, c_postag, c_label = c_tuple[0], c_tuple[1], c_tuple[2]
            if restricted:
                if label == c_label:
                    cluster_features.extend([
                        "c" + str(ind) + ':word.lower=' + c_word.lower(),
                        "c" + str(ind) + ':word[-3:]=' + c_word[-3:],
                        "c" + str(ind) + ':word[-2:]=' + c_word[-2:],
                        "c" + str(ind) + ':word.isupper=%s' % c_word.isupper(),
                        "c" + str(ind) + ':word.isdigit=%s' % c_word.isdigit(),
                        "c" + str(ind) + ':postag=' + c_postag,
                        "c" + str(ind) + ':postag[:2]=' + c_postag[:2],
                    ])
            else:
                cluster_features.extend([
                    "c" + str(ind) + ':word.lower=' + c_word.lower(),
                    "c" + str(ind) + ':word[-3:]=' + c_word[-3:],
                    "c" + str(ind) + ':word[-2:]=' + c_word[-2:],
                    "c" + str(ind) + ':word.isupper=%s' % c_word.isupper(),
                    "c" + str(ind) + ':word.isdigit=%s' % c_word.isdigit(),
                    "c" + str(ind) + ':postag=' + c_postag,
                    "c" + str(ind) + ':postag[:2]=' + c_postag[:2],
                ])

        return cluster_features

    features = []

    word = sent[i][0]
    postag = sent[i][1]
    dbn_prediction = sent[i][2]
    
    #if type of prediction is unicode, a prediction was made
    if type(dbn_prediction) == unicode:
        features.append("dbn_prediction=" + str(dbn_prediction))
    else:
        features.append('dbn_prediction=O')

    features.extend(get_base_features())

    #add sliding window features if not set to default
    if window_number != 0:
        features.extend(get_window_features(window_number))

    if clusters:
        features.extend(get_cluster_features(clusters))
    
    return list(set(features))

def sent2features(sent):
    """Convert all words in a sentence to their features."""
    return [word2features(sent, i) for i in range(len(sent))]

def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.
    """

    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
        
    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )

def add_predictions(filename, overlap_df, dbn, label_map):
    """Parse CRF data into training/testing ready format."""
    df = pd.read_csv(filename, encoding='utf-8')
    logging.info(str(len(df)) + " total words")
    data = df.values

    new_data = []
    for i, row in enumerate(data):
        try:
            vec = overlap_df[overlap_df.word == row[0]].values[0][3:]
            dbn_prediction = dbn.predict(np.array([vec]).astype(np.float))
            dbn_prediction = label_map[int(dbn_prediction[0])]
            new_data.append(list(row) + [dbn_prediction])
        except IndexError:
            new_data.append(list(row) + ['NaN'])

        if i % 1000 == 0:
            logging.info(str(i) + " words processed")

    df = pd.DataFrame(new_data, columns=['word', 'POS', 'NER', 'Prediction'])
    df.to_csv('./CRF_data/' + 'pred_' + filename[11:], encoding='utf-8', index=False)

def parse_data(filename, overlap_df, dbn, label_map):
    """Parse CRF data into training/testing ready format."""
    df = pd.read_csv(filename, encoding='utf-8')
    raw_X, raw_y = df[['word', 'POS', 'Prediction']].values, df['NER'].values 
    logging.info(str(len(raw_X)) + " total words")

    sent_X, sent_y = [], []
    X, y = [], []
    for i, row in enumerate(raw_X):
        if row[0] == "END" and row[1] == "END" and raw_y[i] == "END":
            sent_X.append(X)
            sent_y.append(y)
            X, y = [], []
        else:
            X.append(list(row))
            y.append(raw_y[i])

    logging.info("Total sentences: " + str(len(sent_X)))
    count = 0

    X = []
    for sentence in sent_X:
        count += 1
        if count % 1000 == 0:
            logging.info("Sentences processed: " + str(count))
        X.append(sent2features(sentence))

    y = sent_y

    return X, y

def main():
    """Main function."""

    set_verbosity(0)

    import warnings
    warnings.filterwarnings("ignore")

    overlap_df = pd.read_csv("./vectors/google_overlap.csv", encoding='utf-8')
    #overlap_df = get_data("./vectors/freebase_overlap.csv")

    overlap_df = overlap_df[overlap_df.NER != 'O']
    overlap_df = overlap_df[overlap_df.NER != 'I-FAC']
    overlap_df = overlap_df[overlap_df.NER != 'B-FAC']
    overlap_df = overlap_df[overlap_df.NER != 'I-LOC']
    overlap_df = overlap_df[overlap_df.NER != 'B-LOC']
    overlap_df = overlap_df[overlap_df.NER != 'I-WEA']
    overlap_df = overlap_df[overlap_df.NER != 'B-WEA']
    overlap_df = overlap_df[overlap_df.NER != 'I-VEH']
    overlap_df = overlap_df[overlap_df.NER != 'B-VEH']
    overlap_df = overlap_df[overlap_df.NER != 'I-TTL']
    overlap_df = overlap_df[overlap_df.NER != 'B-TTL']

    import pickle
    with open('./google_model.pkl', 'r') as f:
        dbn, label_map = pickle.load(f)
        label_map = {value: label for label, value in label_map.items()}

    #add_predictions('./CRF_data/crf_train.csv', overlap_df, dbn, label_map)
    #add_predictions('./CRF_data/crf_test.csv', overlap_df, dbn, label_map)
    cut = 25
    global clusters
    clusters = get_clusters(cut, "./clusters.txt")


    train_X, train_y = parse_data('./CRF_data/pred_crf_train.csv', overlap_df, dbn, label_map)
    #train_X, train_y = parse_data('./CRF_data/crf_train.csv', overlap_df, dbn, label_map)
    test_X, test_y = parse_data('./CRF_data/pred_crf_test.csv', overlap_df, dbn, label_map)
    #test_X, test_y = parse_data('./CRF_data/crf_test.csv', overlap_df, dbn, label_map)


    #create trainer and append each row+label 
    trainer = pycrfsuite.Trainer(verbose=False)
    for xseq, yseq in zip(train_X, train_y):
        trainer.append(xseq, yseq)

    #set extra parameters of trainer object
    trainer.set_params({
        'c1': 1.0,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier
    })


    #Train model then load it back into mem
    trainer.train('model.crfsuite')
    tagger = pycrfsuite.Tagger()
    tagger.open('model.crfsuite')


    predictions = []
    for i in range(len(test_X)):
        #tag sentence with model to obtain prediction and pull actual labels
        predictions.append(tagger.tag(test_X[i]))

    print(bio_classification_report(test_y, predictions))
    #'''

if __name__ == "__main__":
    main()
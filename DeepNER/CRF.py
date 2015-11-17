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

I included {test,training}_data.txt (they add POS tags to the data), however
the script will automatically build these files if they are not detected in the
current working directory.
"""

import nltk
import pycrfsuite
import pandas as pd
from itertools import chain
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

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

    features = []

    word = sent[i][0]
    postag = sent[i][1]

    features.extend(get_base_features())
    
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

def parse_data(filename):
    """Parse CRF data into training/testing ready format."""
    df = pd.read_csv(filename, encoding='utf-8')
    raw_X, raw_y = df[['word', 'POS']].values, df['NER'].values
    
    sent_X, sent_y = [], []
    X, y = [], []
    for i, row in enumerate(raw_X):
        if all([col == "END" for col in row]) and raw_y[i] == "END":
            sent_X.append(X)
            sent_y.append(y)
            X, y = [], []
        else:
            X.append(list(row))
            y.append(raw_y[i])

    X = [sent2features(sentence) for sentence in sent_X]
    y = sent_y

    return X, y

def main():
    """Main function."""

    import warnings
    warnings.filterwarnings("ignore")

    #import pickle
    #with open('./google_model.pkl', 'r') as f:
    #    dbn, label_map = pickle.load(f)

    train_X, train_y = parse_data('./CRF_data/crf_train.csv')
    test_X, test_y = parse_data('./CRF_data/crf_test.csv')


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

    print predictions

    #print(bio_classification_report(test_y, predictions))

if __name__ == "__main__":
    main()
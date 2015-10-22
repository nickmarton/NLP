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

import pycrfsuite
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from itertools import chain
from collections import defaultdict

def get_clusters(cut, filename, path="./"):
    """Generate clusters by cutting tree at cut."""
    data = []

    #get word and its bitstring, ignore count
    with open(path + filename) as f:
        for line in f:
            word, bitstring = line.strip().split('\t')[0:2]
            data.append((word, bitstring))

    #generate cluster map; bitstring for keys,
    #list of words with same bitstring cuts for values
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

def write_cluster_input(data):
    """
    Write input (sentences where each token is separated by a space) for
    pmi clusters.
    """

    with open("./cluster_input.txt", "w") as f:
        for datum in data:
            sent_str = ""
            for trip in datum:
                sent_str += trip[0] + "  " + trip[1] + "  " + trip[2] + "    "
            sent_str = sent_str[:-4]
            f.write(sent_str + "\n")

def add_pos_tags(sent):
    """
    Add part of speech tags to 2-tuples word-label pairs in sentence parameter.
    """

    tokens = [token for token, label in sent]
    tagged = nltk.pos_tag(tokens)
    
    return [(sent[i][0], tagged[i][1], sent[i][1]) for i in range(len(sent))]

def write_data(data, filename, path="./"):
    """Write parsed data to file so we don't have to keep parsing each test."""

    with open(path + filename, "w") as f:
        for datum in data:
            #convert each tuple in each sentence to a string and write to file
            for tup in datum:
                tup_str = '    '.join([val for val in tup]) + '\n'
                f.write(tup_str)
            #add a newline after each sentence is completely written
            f.write('\n')

def read_data(filename, path="./"):
    """Read data from filename and return as list of 3-tuples."""
    data = []
    
    with open(path + filename, "r") as f:
        
        sentence_tokens = []
        
        for line in f:
            
            #strip each line in file of its ending newline character without
            #altering the line (i.e., creating copy of it)
            line = str(line.strip())
            #if we're currently on a line in the data that is part of a
            #sentence, save the line in sentence_token list
            if line:
                sentence_tokens.append(tuple(line.split('    ')))
            #otherwise, we have a full sentence of sentence_tokens, 
            #record it in data and reset sentence_tokens
            else:
                data.append(sentence_tokens)
                sentence_tokens = list()

    return data

def parse_data(filename, path="./English_data/"):
    """Parse data from train_nwire and test_nwire."""
    print "Beginning parsing of: " + filename

    data = []

    #open filename located at path 
    #files are assumed to be within English_data directory(located in cwd)
    with open(path + filename) as f:
        
        sentence_tokens = []

        for line in f:

            #strip each line in file of its ending newline character without
            #altering the line (i.e., creating copy of it)
            line = str(line.strip())
            #if we're currently on a line in the data that is part of a
            #sentence, save the line in sentence_token list
            if line:
                sentence_tokens.append(tuple(line.split("\t")))
            #otherwise, we have a full sentence of sentence_tokens, 
            #record it in data and reset sentence_tokens
            else:
                data.append(sentence_tokens)
                sentence_tokens = list()

    #add part of speech tag to word-label tuples in sentences
    for i in range(len(data)):
        data[i] = add_pos_tags(data[i])

    #write input for pmi clustering algorithm computed by class_lm_cluster.py
    write_cluster_input(training_data)

    print "Done parsing of: " + filename
    return data

def word2features(sent, i, ngram_number, window_number, clusters):
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

    def get_ngram_features(ngram_number):
        """Get features based on Ngram number."""
        n = ngram_number

        #if there's a full N-gram for this current word
        if i < len(sent) - n + 1:

            ngram_features = []

            #add features for each word contained in the N-gram
            for j in range(i, i + n):

                word_n = sent[j][0]
                postag_n = sent[j][1]

                ngram_features.extend([
                    '+' + str(j - i + 1) + ':word.lower=' + word_n.lower(),
                    '+' + str(j - i + 1) + ':word[-3:]=' + word_n[-3:],
                    '+' + str(j - i + 1) + ':word[-2:]=' + word_n[-2:],
                    '+' + str(j - i + 1) + ':word.isupper=%s' % word_n.isupper(),
                    '+' + str(j - i + 1) + ':word.isdigit=%s' % word_n.isdigit(),
                    '+' + str(j - i + 1) + ':postag=' + postag_n,
                    '+' + str(j - i + 1) + ':postag[:2]=' + postag_n[:2],
                ])

            return ngram_features
        else:
            return []

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

    features.extend(get_base_features())
    
    #add Ngram features if not set to default
    if ngram_number != 1:
        features.extend(get_ngram_features(ngram_number))
    
    #add sliding window features if not set to default
    if window_number != 0:
        features.extend(get_window_features(window_number))

    #add pmi cluster features if not set to default
    if clusters:
        features.extend(get_cluster_features(clusters))

    return list(set(features))

def sent2features(sent, ngram_number=1, window_number=0, cluster_cut=0, 
                                        cluster_file="cluster_output.txt"):
    """Convert all words in a sentence to their features."""
    #if this model includes pmi clusters, generate them with by cutting
    #bitstrings at index cluster_cut
    if cluster_cut != 0:
        clusters = get_clusters(cluster_cut, cluster_file)
    else:
        clusters = None

    return [word2features(
        sent, i, 
        ngram_number, 
        window_number, 
        clusters) 
    for i in range(len(sent))]

def sent2labels(sent):
    """Extract labels from sentence."""
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    """Extract words from sentence."""
    return [token for token, postag, label in sent]

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

def test(tagger, test_data, ngram_number, window_number, cluster_cut):
    """Evaluate performance of tagger on test_data using mean squared error."""
    #get test matrix and corresponding label vector
    X_test = [sent2features(s, ngram_number, window_number, cluster_cut) for s in test_data]
    y_test = [sent2labels(s) for s in test_data]

    errors = []
    predictions = []

    for i in range(len(test_data)):

        #tag sentence with model to obtain prediction and pull actual labels
        prediction = tagger.tag(X_test[i])
        predictions.append(prediction)

    print(bio_classification_report(y_test, predictions))

def main():
    """Main function."""

    #set constants for training and testing file names
    TRAINING_FILE = "train_nwire"
    TEST_FILE = "test_nwire"
    ngram_number = 1
    window_number = 2
    cluster_cut = 25

    import warnings
    warnings.filterwarnings("ignore")
    
    #try to read in data from pre-parsed files, otherwise get it from
    #{train,test}_nwire files and write them to {training,test}_data.txt
    try:
        training_data = read_data("training_data.txt")
        test_data = read_data("test_data.txt")
    except:
        print "failed to load pre-parsed files, parsing:"
        print
        #download NLTK package dependencies for tokenizing and pos tagging.
        #We only need to do this here as if reads are successful pos tags
        #are assumed to already be included
        nltk.download('punkt', 'maxent_treebank_pos_tagger')
        #get training data from train_nwire file
        training_data = parse_data(TRAINING_FILE)
        test_data = parse_data(TEST_FILE)

        #write data to file so we can avoid parsing each test
        write_data(training_data, "training_data.txt")
        write_data(test_data, "test_data.txt")

    #create training matrix and corresponding label vector
    X_train = [sent2features(s, ngram_number, window_number, cluster_cut) for s in training_data]
    y_train = [sent2labels(s) for s in training_data]

    #create trainer and append each row+label 
    trainer = pycrfsuite.Trainer(verbose=False)
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)

    #set extra parameters of trainer object
    trainer.set_params({
        'c1': 1.0,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier
    })

    trainer.train('model.crfsuite')

    tagger = pycrfsuite.Tagger()
    tagger.open('model.crfsuite')

    test(tagger, test_data, ngram_number, window_number, cluster_cut)
    print "Window size:\t" + str(window_number)
    print "Ngram:\t" + str(ngram_number)
    print "pmi cluster size:\t" + str(cluster_cut)

if __name__ == "__main__":
    main()
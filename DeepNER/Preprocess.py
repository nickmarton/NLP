"""Preprocessing module."""

import os
import sys
import nltk
import codecs
import logging
import gensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nolearn.dbn import DBN

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

def get_data(filename='./word_and_tags.csv'):
    """
    Try to read data from filename. 
    
    If reading fails create csv file with that filename from private data
    and then read it.
    """

    def add_pos_tags(sent):
        """
        Add part of speech tags to 2-tuples word-label pairs in sentence parameter.
        """

        tokens = [token for token, label in sent]
        tagged = nltk.pos_tag(tokens)

    def make_csv(root="./Data/", output_file=filename):
        """
        Recursively scan directory and concatenate files into a single csv file if
        output_file provided.
        
        Function assumes values in files are separated by tabs and only 2 relevant
        columns.
        """

        #type checking
        if not isinstance(output_file, str) and output_file != None:
            raise TypeError("output_file parameter must be a string or None")

        data = []

        #use counter to log every 10000 words
        counter = 0
        bad_lines = 0
        good_lines = 0
        line_count = 0
        #walk from root provided getting every file along the way
        for path, subdirs, files in os.walk(root):
            for name in files:
                with codecs.open(os.path.join(path, name), 'r', encoding='utf-8',
                                                            errors='ignore') as f:
                    #try to append each word tag pair if possible
                    for line in f:
                        try:
                            counter += 1
                            word, tag = line.strip('\n').split('\t')[0:2]
                            good_lines += 1
                            data.append([word, tag])
                        except ValueError:
                            bad_lines += 1
                        finally:
                            line_count += 1
                        
                        #write to log every 1000 words
                        if counter % 100000 == 0:
                            logging.info(str(counter) + " words processed.")

        logging.info("TOTAL AMOUNT OF MALFORMED LINES: " + str(bad_lines))
        logging.info("TOTAL AMOUNT OF GOOD LINES: " + str(good_lines))
        logging.info("TOTAL LINES: " + str(line_count))


        #Add pos tag to data
        logging.info("Adding POS tags to data of size: " + str(len(data)))

        nltk.download('punkt', 'maxent_treebank_pos_tagger')
        tokens = [d[0] for d in data]
        logging.info("Beginning POS tagging process")
        tagged = nltk.pos_tag(tokens)
        logging.info("Done POS tagging process; beginning appending")
        
        new_data = []
        for i in range(len(data)):
            word, ner_tag = data[i]
            pos_tag = tagged[i][1]
            if len(ner_tag) < 8:
                #new_data.append([word, ner_tag])
                new_data.append([word, pos_tag, ner_tag])

            if i % 10000 == 0:
                logging.info(str(i) + " words processed")

        #make DataFrame object from data string
        pd_data = pd.DataFrame(new_data, columns=['word', 'POS', 'NER'])
        #pd_data = pd.DataFrame(new_data, columns=['word', 'NER'])
        logging.info("Done creating DataFrame")

        pd_data = pd_data.drop_duplicates()

        logging.info("Droping duplicates; new size: " + str(len(pd_data)))

        #if output file provided, write csv, otherwise return DataFrame
        pd_data.to_csv(output_file, encoding='utf-8')
        logging.info("Done writing csv\n")

    def load_csv(input_file=filename):
        """Load csv data file into pandas DataFrame object."""
        #load DataFrame and drop any columns
        pd_data = pd.read_csv(input_file, encoding='utf-8')
        pd_data = pd_data.dropna(how='any')

        return pd_data

    #if file with filename exists
    if os.path.exists(filename):
        return load_csv(input_file=filename)
    else:
        make_csv(output_file=filename)
        return load_csv(input_file=filename)

def make_vectors(data_frame, size=100, wt_sep='~~~'):
    """Buils model from DataFrame object and save vectors."""
    if not type(wt_sep) == str:
        raise TypeError('wt_sep parameter must be of type string')

    words = [word for word in data_frame['word'].tolist()]
    pos_tags = [tag for tag in list(data_frame['POS'])]
    ner_tags = [tag for tag in list(data_frame['NER'])]

    #ensure word, tag separator is not in any word within the data so we can
    #recover words and tags later without ambiguity
    for word in words:
        if wt_sep in word:
            logging.critical(
                'ERROR: ' + wt_sep + ' appears in \'' + word + '\' in data')

    sentences = [[words[i] + '~~~' + pos_tags[i] + '~~~' + ner_tags[i]] for i in range(len(words))]
    #sentences = [[words[i]] for i in range(len(words))]


    #build model and save vectors into txt file
    logging.info("Building Word2Vec model:")
    model = gensim.models.Word2Vec(sentences, min_count=1, size=size)
    logging.info(
        "Saving vectors from model in ./vectors/vectors" + str(size) + ".txt")
    model.save_word2vec_format('./vectors/vectors' + str(size) + '.txt')


    #convert .txt file into list of lists of word-vector pairs and drop
    #dimensional summary
    word_tags_and_vectors = []
    with open('./vectors/vectors' + str(size) + '.txt', 'r') as f:
        for line in f:
            word_tags_and_vectors.append(line.split())

    word_tags_and_vectors = word_tags_and_vectors[1:]


    #split word-tag pairs from vectors and rewrite data with words and tags
    #in their own respective columns
    words_tag_triples = [p[0] for p in word_tags_and_vectors]
    vectors = [v[1:] for v in word_tags_and_vectors]
    data = []
    
    for i in range(len(words_tag_triples)):
        word, pos_tag, ner_tag = words_tag_triples[i].split(wt_sep)
        data.append([word, pos_tag, ner_tag] + vectors[i])


    #create names for columns holding vector features in csv
    vec_columns = ['vec[' + str(i) + ']' for i in range(size)]

    #make pandas DataFrame object with words and vectors and save to csv
    pd_data = pd.DataFrame(data, columns=['word', 'POS', 'NER'] + vec_columns)
    pd_data.to_csv('./vectors/vectors' + str(size) + '.csv', encoding='utf-8')

def filter_google_vectors():
    """
    Filter out vectors of words in google vector set not in provided NER
    annotated data. Other vectors are irrelevant for training purposes.

    Function assumes that data is stored in the same format used by the
    original C word2vec-tool.
    """

    _filter_vectors("./vectors/raw_googlevectors.txt", size=300, skip_index=0)

def filter_freebase_vectors():
    """
    Filter out vectors of words in freebase vector set not in provided NER
    annotated data. Other vectors are irrelevant for training purposes.

    Function assumes that data is stored in the same format used by the
    original C word2vec-tool.
    """

    if not os.path.exists('./vectors/freebase-vectors-skipgram1000-en.txt'):
        model = gensim.models.Word2Vec.load_word2vec_format(
            './vectors/freebase-vectors-skipgram1000-en.bin', binary=True)
        model.save_word2vec_format(
            './vectors/freebase-vectors-skipgram1000-en.txt')
    
    _filter_vectors("./vectors/freebase-vectors-skipgram1000-en.txt", size=1000, skip_index=4)

def _filter_vectors(filename, size, skip_index):
    """Filter vectors from filename."""
    word_map = {}

    df = get_data("./word_and_tags.csv")
    words = df['word'].tolist()
    for word in words:
        word_map[word] = True

    logging.info("Build word map")

    count = 0
    matches = {}
    with open(filename , "r") as f:
        for line in f:
            count += 1
            
            datum = line.strip().split()
            word, vec = datum[0][skip_index:], ' '.join(datum[1:])

            try:
                word_map[word]
                matches[word] = vec
            except KeyError:
                pass
            
            if count % 100000 == 0:
                logging.info(str(count) + " words processed")

    logging.info("Done processing words with " + str(len(matches)) + ", building data")

    data = [[word] + vec.split() for word, vec in matches.items()]

    #create names for columns holding vector features in csv
    vec_columns = ['vec[' + str(i) + ']' for i in range(size)]

    #make pandas DataFrame object with words and vectors and save to csv
    pd_data = pd.DataFrame(data, columns=['word'] + vec_columns)
    pd_data.to_csv(filename[:-3] + 'csv', encoding='utf-8')

def get_overlap(data_frame_1, data_frame_2):
    """."""
    word_map = {}
    words_1 = data_frame_1.values
    words_2 = data_frame_2.values
    
    for series in words_1:
        lis =  series.tolist()[1:]
        word_map[lis[0]] = lis
    
    overlap = []
    for word in words_2:
        try:
            word_map[word[1]]
            overlap.append(word_map[word[1]] + list(word[2:]))
        except KeyError:
            pass

    size = len(overlap[0]) - 3

    #create names for columns holding vector features in csv
    vec_columns = ['vec[' + str(i) + ']' for i in range(size)]
    #make pandas DataFrame object with words and vectors and save to csv
    overlap_df = pd.DataFrame(overlap, columns=['word', 'POS', 'NER'] + vec_columns)
    return overlap_df

def map_labels(data_frame):
    """Map unicode labels to numpy.int64 values."""
    labels = data_frame['NER'].unique()
    label_map = {}
    for i in range(len(labels)):
        label_map[labels[i]] = np.int64(i)
    return label_map, labels

def save_overlap(data_frame, filename):
    """."""
    #create names for columns holding vector features in csv
    vec_columns = ['vec[' + str(i) + ']' for i in range(data_frame.values.shape[1] - 3)]
    #make pandas DataFrame object with words and vectors and save to csv
    data_frame = pd.DataFrame(data_frame, columns=['word', 'POS', 'NER'] + vec_columns)
    data_frame.to_csv(filename, encoding='utf-8', index=False)

def main():
    """Quick tests."""
    set_verbosity(3)
    
    try:
        size = int(sys.argv[1])
    except ValueError:
        logging.error('Invalid vector size provided; defaulting to 100')
        size = 100
    except IndexError:
        logging.error('No vector size provided; defaulting to 100')
        size = 100

    '''
    #filter_google_vectors()
    #filter_freebase_vectors()
    '''

    #'''
    from sklearn.cross_validation import train_test_split

    df1 = get_data()
    df2 = get_data("./vectors/raw_googlevectors.csv")
    #df2 = get_data("./vectors/freebase-vectors-skipgram1000-en.csv")
    overlap_df = get_overlap(df1, df2)
    label_map, labels = map_labels(overlap_df)

    train_data, test_data = train_test_split(overlap_df, test_size=0.20)

    trainX, trainY, testX, testY = [], [], [], []
    for datum in train_data.values:
        trainX.append(datum[3:])
        trainY.append(label_map[datum[2]])

    trainX = np.array(trainX).astype(np.float)
    trainY = np.array(trainY)

    for datum in test_data.values:
        testX.append(datum[3:])
        testY.append(label_map[datum[2]])

    testX = np.array(testX).astype(np.float)
    testY = np.array(testY)

    dbn = DBN(
        [trainX.shape[1], 300, 200, 100, len(labels)],
        learn_rates = 0.3,
        learn_rate_decays = 0.9,
        epochs = 50,
        verbose = 1)

    dbn.fit(trainX, trainY)

    preds = dbn.predict(testX)
    from sklearn.metrics import classification_report
    print classification_report(testY, preds)
    #'''
    #df = subset_data(df, count=1000000)

if __name__ == "__main__":
    main()
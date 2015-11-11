"""Preprocessing module."""

import os
import sys
import nltk
import codecs
import logging
import gensim
import pandas as pd

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
            new_data.append([word, pos_tag, ner_tag])

            if i % 10000 == 0:
                logging.info(str(i) + " words processed")


        #make DataFrame object from data string
        pd_data = pd.DataFrame(new_data, columns=['word', 'POS', 'NER'])
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

def extract_matches(data_frame, raw_file='./vectors/raw_googlevectors.txt'):
    """Extract matches of word + vectors in raw file also in DataFrame."""
    words = data_frame['word'].tolist()

    from collections import defaultdict
    m = defaultdict(list)
    for word in words:
        m[word[0:1]].append(word)

    f_write = open("./vectors/subset" + raw_file[13:], 'w')    

    counter = 0
    with open(raw_file, 'r') as f:
        for line in f:

            counter += 1
            
            word = line.split()[0]
            if word in m[word[0:1]]:
                f_write.write(line)

            if counter % 10000 == 0:
                logging.info(str(counter) + " words processed")

def main():
    """Quick tests."""

    #FOR COMPARISON:
    #text8
    #Vocab size: 428554
    #Words in train file: 15772268

    set_verbosity()
    
    try:
        size = int(sys.argv[1])
    except ValueError:
        logging.error('Invalid vector size provided; defaulting to 100')
        size = 100
    except IndexError:
        logging.error('No vector size provided; defaulting to 100')
        size = 100

    df = get_data()
    make_vectors(df, size=size)
    
    #extract_matches(df, raw_file='./vectors/raw_googlevectors.txt')
    #extract_matches(df, raw_file='./vectors/raw_knowledge-vectors-skipgram1000.txt')

if __name__ == "__main__":
    main()
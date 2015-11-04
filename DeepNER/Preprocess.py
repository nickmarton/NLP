"""Preprocessing module."""

import os
import codecs
import logging
import gensim
import pandas as pd

def set_verbosity(verbose_level=3):
    """Set the level of verbosity of the Preprocessing."""
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

def get_data(filename='./data.csv'):
    """
    Try to read data from filename. 
    
    If reading fails create csv file with that filename from private data
    and then read it.
    """

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

        #make DataFrame object from data string
        pd_data = pd.DataFrame(data, columns=['word', 'tag'])
        logging.info("Done creating DataFrame")

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

def make_vectors(data_frame, size=100):
    """Buils model from DataFrame object and save vectors."""
    sentences = [[word] for word in list(data_frame['word'])]
    logging.info("Building Word2Vec model:")
    #build model and save vectors into txt file
    model = gensim.models.Word2Vec(sentences, min_count=1, size=size)
    model.save_word2vec_format('./vectors/vectors' + str(size) + '.txt')

    #convert .txt file into list of lists of word-vector pairs
    word_and_vectors = []
    with open('./vectors/vectors' + str(size) + '.txt', 'r') as f:
        for line in f:
            word_and_vectors.append(line.split())

    #create names for columns holding vector features in csv
    vec_columns = ['vec[' + str(i) + ']' for i in range(size)]

    #make pandas DataFrame object with words and vectors and save to csv
    pd_data = pd.DataFrame(word_and_vectors[1:], columns=['word'] + vec_columns)
    pd_data.to_csv('./vectors/vectors' + str(size) + '.csv', encoding='utf-8')

def main():
    """."""

    #text8
    #Vocab size: 428554
    #Words in train file: 15772268

    set_verbosity()

    df = get_data()
    make_vectors(df, size=300)

if __name__ == "__main__":
    main()
"""Preprocessing module."""

import os
import codecs
import logging
import gensim
import pandas as pd

logging.basicConfig(
    format='%(asctime)s:\t %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO)

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
                        """
                        if counter == 64006:
                            lll = line.strip('\n').split('\t')
                            print lll
                        if counter == 222501:
                            lll = line.strip('\n').split('\t')
                            print lll
                        if counter == 308807:
                            lll = line.strip('\n').split('\t')
                            print lll
                        if counter == 416281:
                            lll = line.strip('\n').split('\t')
                            print lll
                        if counter == 416282:
                            lll = line.strip('\n').split('\t')
                            print lll
                        if counter == 416284:
                            lll = line.strip('\n').split('\t')
                            print lll
                        if counter == 717334:
                            lll = line.strip('\n').split('\t')
                            print lll
                        if counter == 750377:
                            lll = line.strip('\n').split('\t')
                            print lll
                        if counter == 750378:
                            lll = line.strip('\n').split('\t')
                            print lll
                        if counter == 750380:
                            lll = line.strip('\n').split('\t')
                            print lll
                        if counter == 750381:
                            lll = line.strip('\n').split('\t')
                            print lll
                        if counter == 750382:
                            lll = line.strip('\n').split('\t')
                            print lll
                        if counter == 750383:
                            lll = line.strip('\n').split('\t')
                            print lll
                        if counter == 750384:
                            lll = line.strip('\n').split('\t')
                            print lll
                        if counter == 750386:
                            lll = line.strip('\n').split('\t')
                            print lll
                        """


        logging.info("TOTAL AMOUNT OF MALFORMED LINES: " + str(bad_lines))
        logging.info("TOTAL AMOUNT OF GOOD LINES: " + str(good_lines))
        logging.info("TOTAL LINES: " + str(line_count))


        #make DataFrame object from data string
        pd_data = pd.DataFrame(data, columns=['word', 'tag'])

        #if output file provided, write csv, otherwise return DataFrame
        pd_data.to_csv(output_file, encoding='utf-8')

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
    """."""
    sentences = [[word] for word in list(data_frame['word'])]
    model = gensim.models.Word2Vec(sentences, min_count=1, size=size)
    model.save_word2vec_format('./vectors' + str(size) + '.txt')

def main():
    """."""
    df = get_data()
    make_vectors(df, size=100)



if __name__ == "__main__":
    main()
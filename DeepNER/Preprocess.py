"""Preprocessing module."""

import os
import codecs
import logging
import pandas as pd

logging.basicConfig(
    format='%(asctime)s:\t %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO)

def concatenate(root="./Data/", output_file='./data.csv'):
    """
    Recursively scan directory and concatenate files into a single csv file if
    output_file provided, otherwise, return data as a pandas DataFrame object.
    
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
                        data.append([word, tag])
                    except ValueError:
                        bad_lines += 1
                    
                    #write to log every 1000 words
                    if counter % 10000 == 0:
                        logging.info(str(counter) + " words processed.")

    logging.info("TOTAL AMOUNT OF MALFORMED LINES: " + str(bad_lines))

    #make DataFrame object from data string
    pd_data = pd.DataFrame(data, columns=['word', 'tag'])

    #if output file provided, write csv, otherwise return DataFrame
    if output_file:
        pd_data.to_csv(output_file, encoding='utf-8')
    else:
        return pd_data

def main():
    """."""
    concatenate()
    #print data[0]
    #print pd.DataFrame(data, columns=['word', 'tag', 'offset1', 'offset2'])

if __name__ == "__main__":
    main()
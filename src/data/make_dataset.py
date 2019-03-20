"""

make_dataset.py  (author: Louis de thanhoffer de Volcsey / git: ldethanhoffer)

Import a corpus of documents together with 1-hot encoded labels

example:
    "001", "hey there how are you", 0,0,0,0,0

options:
    header = specify if the dataset has a header
    id = specify if each document has an id label

returns:
    documents: an array of tokenized tuples for each document
    labels: a numpy array of tuples for each label associated to a documents
    data: a dictionary associating each document to its label

"""


import csv
import numpy as np


def load_corpus(filename=" ", header=None, id=None):

    corpus = []
    labels = []

    with open(filename) as datafile:
        csvreader = csv.reader(datafile)

        # initialize the counter:
        progress = 0

        # determine if the features have an id tag:
        if id:
            s = 1
        else:
            s = 0

        # determine the number of categories for the labels:
        row_1 = next(csvreader)
        num_categories = len(row_1) - 1 - s

        # iterate over each datapoint:
        for row in csvreader:
            # skip over the header if necessary :
            if header:
                header = False
                continue
            # check the progress:
            if progress % 10000 == 0:
                print("reading document number: {}".format(progress))
            # read the document:
            document = row[s].split()
            # add the document to the corpus as a tuple:
            document = tuple(document)
            corpus.append(document)

            # read the labels:
            categories = []
            for i in range(1 + s, 1 + s + num_categories):
                categories.append(row[i])
            # add the label:
            labels.append(categories)

            # increase the progress counter:
            progress = progress + 1

        # turn the list of labels into a numpy array:
        labels = np.asarray(labels, dtype=int)

    # print the progress:
    print("\ndone reading {}".format(filename))
    # get the data:
    data = dict(zip(corpus, labels))

    return corpus, labels, data

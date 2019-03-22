"""

make_dataset.py  (author: Louis de thanhoffer de Volcsey / git: ldethanhoffer)

Import a corpus of documents together with 1-hot encoded labels

example:
    "001", "hey there how are you", 0,0,0,0,0

options:
    header = specify if the dataset has a header
    id = specify if each document has an id label

returns:

    data: a list of the form [document, label]
    where each entry consists of a tokenized document together with its associated label as a np array.

"""


import csv
import numpy as np
import tqdm


def load_corpus(filename=" ", header=None, id=None):

    corpus = []
    labels = []

    with open(filename) as datafile:
        csvreader = csv.reader(datafile)
        # determine if the features have an id tag:
        if id:
            s = 1
        else:
            s = 0
        # determine the number of categories for the labels:
        row_1 = next(csvreader)
        num_categories = len(row_1) - 1 - s
        # iterate over each datapoint:
        for row in tqdm.tqdm(csvreader):
            # skip over the header if necessary :
            if header:
                header = False
                continue

            # read the document:
            document = row[s].split()
            # add the document to the corpus:
            corpus.append(document)

            # read the labels:
            categories = []
            for i in range(1 + s, 1 + s + num_categories):
                categories.append(row[i])
            # add the label:
            labels.append(categories)

        # turn the list of labels into a numpy array:
        labels = np.asarray(labels, dtype=int)

    # print the progress:
    print("\ndone reading {}".format(filename))
    # get the data:
    data = [[corpus[i], labels[i]] for i in range(0, len(corpus))]

    return data

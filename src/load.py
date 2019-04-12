"""
author: Louis de thanhoffer de Volcsey / git:ldethanhoffer

the necessary functionalities to import a dataset of documents together with one-hot-encoded labels

example:
    "id_tag", "hey there how are you", 0,0,0,0,0
"""

import csv
import numpy as np
import pandas as pd

"""
load_data(): loads the data and returns a list of the form [document, label],
where each entry consists of a tokenized document together with its associated label as a np array.

options:
    header: specify if the dataset has a header
    id_tag: specify if each document has an id label

"""


def load_data(filename=" ", header=None, id=None):

    corpus = []
    labels = []

    with open(filename) as f:
        csvreader = csv.reader(f)
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
    # get the data:
    data = [[corpus[i], labels[i]] for i in range(0, len(corpus))]

    return data


"""
load_dataframe(): loads the dataset in the form of a a dataframe:

options:
    header: specify if the dataset has a header
    id_tag: specify if each document has an id label
"""


def load_dataframe(filename=" ", header=None, id_tag=None):

    if header is True:
        df = pd.read_csv(filename)
    else:
        df = df.read_csv(filename, header=None)

    if id_tag is True:
        df = df.drop(df.columns[0], axis=1)
    return df

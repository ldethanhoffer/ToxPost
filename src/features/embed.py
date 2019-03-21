"""
embed.py  (author: Louis de thanhoffer de Volcsey / git: ldethanhoffer)

A module designed to handle word embeddings:

"""
import numpy as np


"""

Obtain an embedding in the form of a dictionary

"""


def load_embedding(file):
    with open(file, 'r') as f:
        words = set()
        embedding = {}

        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            embedding[curr_word] = np.array(line[1:], dtype=np.float64)

        print('Embedding loaded... \n')

    embeddable_words = embedding.keys()
    return embedding, embeddable_words

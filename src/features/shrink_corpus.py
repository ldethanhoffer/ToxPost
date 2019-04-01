"""
TfIdf: obtain the TfIdf information of a corpus:
input: a corpus of tokenized documents
output:
    TfIdf['scores']: the matrix of TfIdf scores
    TfIdf["column_to_vocab"]: a list assicating to column j of scores the corresponding word.
    By default, scores is a sparse matrix, return as dense can be done optionally
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import csr_matrix
import tqdm


def TfIdf(corpus, dense=None):

    def untok_corpus(corpus):
        new_corpus = []
        for i in range(0, len(corpus)):
            new_corpus.append(" ".join(corpus[i]))
        return new_corpus

    # transform each tokenized comment into one single string in order to use TfIdf
    corpus = untok_corpus(corpus)
    # compute Tfidf scores of the corpus:
    tfidf = TfidfVectorizer()
    scores = tfidf.fit_transform(corpus)
    # (optional) transform to nonsparse matrix-format:
    if dense:
        scores = scores.todense()
    # TfIdf also assigns an index to each individual word. The following retrieves the word assigned to the index
    column_to_vocab = tfidf.get_feature_names()
    TfIdf_info = {'scores': scores, 'column_to_vocab': column_to_vocab}
    return TfIdf_info


"""
shrink_corpus: shrink the length of each document in the corpus by keeping only the top TfIdf-scores
input:
    corpus: a corpus of documents
    length: the maximal length of any document
output:
    new_corpus: a corpus of documents of length at most length.
"""


def shrink_corpus(corpus, length):
    TfIdf_info = TfIdf(corpus)
    scores = TfIdf_info['scores']
    column_to_vocab = TfIdf_info['column_to_vocab']
    new_corpus = []
    for i in tqdm.tqdm(range(0, len(corpus))):
        # obtain nonzero columns:
        row = scores.getrow(i)  # use getrow since scores is csr matrix
        indices = csr_matrix.nonzero(row)[1]
        # obtain the Tfidf-score of each word-index
        entries = [scores[i, j] for j in indices]
        # sort those scores and keep only the top ones
        top_entries = np.argsort(entries)[-length:]
        # get the corresponding column-index for each score
        top_word_indices = [indices[j] for j in top_entries]
        # get the corresponding word for each column index:
        top_words = [column_to_vocab[j] for j in top_word_indices]
        # shrink the commentt
        shrunken_comment = [word for word in corpus[i] if word in top_words]
        new_corpus.append(shrunken_comment)
    return new_corpus

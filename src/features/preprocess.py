"""
prepocess.py  (author: Louis de thanhoffer de Volcsey / git: ldethanhoffer)

A module that preprocesses a tokenized document. It contains the functions:



"""

import re
import tqdm
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import csr_matrix
from embed import load_embedding
from sklearn.decomposition import PCA


"""
clean_document
input:
    a tokenized document
    (optional) a list of admissible words
returns:
    a cleaned document (optionally only containing admissible words)
"""


# Declare global variables:
stopwords = stopwords.words("english")

# Declare a custom list of words we'll replace as we clean up each comment:
replacement_list = [
    ('f u c k', 'fuck'),
    ('f uck', 'fuck'),
    ('fuckyou', 'fuck you'),
    ('fuckmother', 'fuck mother'),
    ('WTF', 'what the fuck'),
    ('OMFG', 'oh my fucking god'),
    ('RTFM', 'read the fucking manual'),
    ('ASAFP', 'as soon as fucking possible'),
    ('FYVM', 'fuck you very much'),
    ('whatever TF', 'whatever the fuck'),
    ('G AY', 'GAY'),

    # the following characters need to be removed by hand:
    ('•', ''),
    ('·', '')
]


# The functions below cleans up each comment in the corpus.
def clean_document(document, admissible_words=None):

    # The functions below replaces a word according to the custom replacement list defined above:
    def replace(word, replacement_list):
        for a, b in replacement_list:
            word = word.replace(a, b)
        return word

    # We begin by defining a function that cleans up each indiviual word:
    def clean_word(word, admissible_words=None):
        cleaned_word = word
        # strip extra whitespaces:
        cleaned_word = cleaned_word.lstrip().rstrip()
        # Remove if hyperlink
        cleaned_word = re.sub(r"(https?:\/\/)(\s)?(www\.)?(\s?)(\w+\.)*([\w\-\s]+\/)*([\w-]+)\/?", "", cleaned_word)
        # Remove hashtags:
        cleaned_word = re.sub(r"#", "", cleaned_word)
        # remove numbers:
        cleaned_word = re.sub(r"$\d+\W+|\b\d+\b|\W+\d+$", "", cleaned_word)
        # Remove apostrophes:
        cleaned_word = re.sub(r"’", "", cleaned_word)
        # remove Punctuation and split 's, 't, 've with a space for filter:
        cleaned_word = re.sub(r"['+string.punctuation+']+", "", cleaned_word)
        # Remove if 2 or fewer letters:
        cleaned_word = re.sub(r"^\w\w?$", "", cleaned_word)
        # remove if an article:
        cleaned_word = re.sub("(\s+)(a|an|and|the)(\s+)", " ", cleaned_word)
        # replace words according to the custom list of replacements defined above:
        cleaned_word = replace(cleaned_word, replacement_list)
        # lowercase the word:
        cleaned_word = cleaned_word.lower()
        # remove if stopword:
        if cleaned_word in stopwords:
            cleaned_word = ""
        # remove any words that are not admissible
        if admissible_words != None and cleaned_word not in admissible_words:
            cleaned_word = ""
        return cleaned_word

    # clean each word in the document:
    cleaned_document = [clean_word(word, admissible_words) for word in document]
    # remove empty words:
    cleaned_document = [word for word in cleaned_document if word != ""]
    return cleaned_document




"""
TfIdf: obtain the TfIdf information of a corpus:
input: a corpus of tokenized documents
output:
    TfIdf['scores']: the matrix of TfIdf scores
    TfIdf["column_to_vocab"]: a list assicating to column j of scores the corresponding word.
    By default, scores is a sparse matrix, return as dense can be done optionally
"""


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
    print("TfIdf scores calculated...\n")
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
        shrunken_comment = [column_to_vocab[j] for j in top_word_indices]
        new_corpus.append(shrunken_comment)
    return new_corpus


"""
embed a corus of documents in a vector space
input:
    corpus = a corpus of documents
    embedding an embedding given as a word-vector dictionary

"""


def embed(corpus, embedding):
    new_corpus = []
    for doc in corpus:
        emb_doc = [embedding[word] for word in doc]
        new_corpus.append(emb_doc)
    return new_corpus


"""
pca_reduce_features

Applies pca to reduce the dimension of a set of n-dimensional features:

"""


def pca_reduce_features(features, dim, verbose=None):
    pca = PCA(n_components=dim)
    pca.fit(features)
    reduced_features = pca.transform(features)
    if verbose:
        return reduced_features


def preprocess():
    pass


# Driver
if __name__ == "__main__":
    pass

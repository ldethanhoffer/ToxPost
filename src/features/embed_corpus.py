"""
embed a corpus of documents in a vector space
input:
    corpus: a corpus of documents
    embedding: an embedding given as a word-vector dictionary

"""
from sklearn.decomposition import PCA
import tqdm

"""
pca_reduce_features

Applies pca to reduce the dimension of a set of n-dimensional features:

"""


def get_vocab(corpus):
    vocab = []
    for comment in corpus:
        for word in comment:
            vocab.append(word)
    vocab = [word for word in set(vocab) if word != '']
    return vocab


def embed_vocab(corpus, embedding, dim, verbose=None):
    vocab = get_vocab(corpus)
    # remove non-embeddable words
    vocab = [word for word in vocab if word in embedding.keys()]
    # apply embedding to vocab
    embedded_vocab = {word: embedding[word] for word in vocab}
    # apply pca
    pca = PCA(n_components=dim)
    reduced_vocab = pca.fit_transform([values for values in embedded_vocab.values()])
    # define the reduction dictionary
    reduction = dict(zip(vocab, reduced_vocab))
    if verbose:
        return pca, reduction
    else:
        return reduction


def embed_corpus(corpus, embedding, dim):
    # reduce each embedded word:
    reduction = embed_vocab(corpus, embedding, dim)
    # rebuild the dataset
    reduced_corpus = []
    for document in tqdm.tqdm(corpus):
        reduced_document = [reduction.get(word) for word in document]
        reduced_corpus.append(reduced_document)
    return reduced_corpus

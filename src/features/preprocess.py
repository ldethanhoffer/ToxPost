"""
prepocess.py  (author: Louis de thanhoffer de Volcsey / git: ldethanhoffer)

A module that preprocesses a tokenized document. It contains the functions:



"""
import csv
from src.load import load_data
from src.features.clean_corpus import clean_corpus
from src.features.shrink_corpus import shrink_corpus
from src.features.embed_corpus import embed_corpus
from resources.glove.load_embedding import load_embedding


def preprocess():
    raw_path = "./data/raw/data.csv"
    cleaned_path = "./data/cleaned/data.csv"
    shrunken_path = "./data/shrunken/data.csv"
    embedded_path = "./data/embedded/data.csv"

    embedding_file = "./resources/glove/glove.twitter.27B.25d.txt"
    length = 100
    dim = 20

    # load the data:
    print("..loading the data..")
    data = load_data(raw_path, header=True, id=True)
    corpus = [datapoint[0] for datapoint in data]
    labels = [datapoint[1] for datapoint in data]

    # clean the documents in the corpus:
    print("..using NLP to clean each comment in the corpus..")
    cleaned_corpus = clean_corpus(corpus)
    cleaned_data = [list(datapoint) for datapoint in zip(cleaned_corpus, labels)]
    # export the results:
    print("..writing the results to the cleaned data file")
    with open(cleaned_path, "w") as f:
        wr = csv.writer(f, delimiter="\n")
        wr.writerow(cleaned_data)

    # shrink the documents in the corpus:
    print("..using TfIdf to shrink each comment in the corpus to size {}..\n".format(length))
    shrunken_corpus = shrink_corpus(cleaned_corpus, length)
    shrunken_data = [list(datapoint) for datapoint in zip(shrunken_corpus, labels)]
    # export the results:
    print("..writing the results to the shrunken data file..")
    with open(shrunken_path, "w") as f:
        wr = csv.writer(f, delimiter="\n")
        wr.writerow(shrunken_data)

    # obtain the Glove embedding:
    print("\n..loading the Glove embedding..")
    embedding = load_embedding(embedding_file)

    # embed the documents in the corpus:
    print("..using PCA to reduce the embedding space of each word to size {}..".format(dim))
    embedded_corpus = embed_corpus(shrunken_corpus, embedding, dim)
    embedded_data = [list(datapoint) for datapoint in zip(embedded_corpus, labels)]
    print("..writing the results to the embedded data file. (this takes a while)..")
    # export the results:
    with open(embedded_path, "w") as f:
        wr = csv.writer(f, delimiter="\n")
        wr.writerow(embedded_data)
    print("\n..data preprocesssed..")

    # balance the dataset


# Driver
if __name__ == "src.features.preprocess":
    preprocess()

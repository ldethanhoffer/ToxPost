"""
prepocess.py  (author: Louis de thanhoffer de Volcsey / git: ldethanhoffer)

A module that preprocesses a tokenized document. It contains the functions:

clean_document

input:
    a tokenized document
    (optional) a list of admissible words
returns:
    a cleaned document (optionally only containing admissible words)

reduce_length

"""

import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

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


def shrink_document(document, length):

    def TfIdf(corpus):
        # transform each tokenized comment into one single string in order to use TfIdf
        data_as_str = data_as_strings(data)
        # compute Tfidf scores of the corpus:
        tfidf = TfidfVectorizer()
        scores = tfidf.fit_transform(data_as_str)
        # transform sparse matrix:
        scores = scores.todense()
        # TfIdf also assigns an index to each individual word. The following retrieves the word assigned to the index
        column_to_vocab = tfidf.get_feature_names()
        TfIdf_info = {'scores': scores, 'column_to_vocab': column_to_vocab}

        return TfIdf_info

    return shrunken_document

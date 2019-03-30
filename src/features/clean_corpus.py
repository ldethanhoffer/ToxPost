# -*- coding: utf-8 -*-

import re
import tqdm
from nltk.corpus import stopwords
import string


"""""
clean_corpus
input:
    a tokenized document
output:
    a cleaned document
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


# The functions below cleans up each document in the corpus.
def clean_corpus(corpus):

    # The functions below replaces a word according to the custom replacement list defined above:
    def replace(word, replacement_list):
        for a, b in replacement_list:
            word = word.replace(a, b)
        return word

    # We begin by defining a function that cleans up each indiviual word:
    def clean_word(word):
        cleaned_word = word
        # strip extra whitespaces:
        cleaned_word = cleaned_word.lstrip().rstrip()
        # Remove if hyperlink, still work on it
        # Remove hashtags:
        cleaned_word = re.sub(r"#", "", cleaned_word)
        # remove numbers:
        cleaned_word = re.sub(r"$\d+\W+|\b\d+\b|\W+\d+$", "", cleaned_word)
        # Remove apostrophes:
        cleaned_word = re.sub(r"’", "", cleaned_word)
        # remove Punctuation and split 's, 't, 've with a space for filter:
        cleaned_word = cleaned_word.translate(str.maketrans('', '', string.punctuation))
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
        return cleaned_word

    # next we clean each document in the corpus:
    def clean_document(document):
        # clean each word in the document:
        cleaned_document = [clean_word(word) for word in document]
        # remove empty words:
        cleaned_document = [word for word in cleaned_document if word != ""]
        return cleaned_document

    # finally, we clean the whole corpus:
    return [clean_document(document) for document in tqdm.tqdm(corpus)]

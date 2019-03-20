"""
prepocess.py  (author: Louis de thanhoffer de Volcsey / git: ldethanhoffer)

"""

import re
import nltk

nltk.download("stopwords")
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
def clean_comment(comment, admissible_words=None):

    # The functions below replaces a word according to the custom replacement list defined above:
    def replace(word, replacement_list):
        for a, b in replacement_list:
            word = word.replace(a, b)
        return word

    # We begin by defining a function that cleans up each indiviual word:
    def clean_word(word, admissible_words=None):

        clean_word = word

        # Remove hyperlinks
        clean_word = re.sub(r"https?:\/\/.*\/\w*", '', clean_word)

        # Remove hashtags:
        clean_word = re.sub(r"#", '', clean_word)

        # remove numbers:
        clean_word = re.sub(r'$\d+\W+|\b\d+\b|\W+\d+$', '', clean_word)

        # Remove apostrophes:
        clean_word = re.sub(r"’", "", clean_word)

        # remove Punctuation and split 's, 't, 've with a space for filter:
        clean_word = re.sub(r'['+string.punctuation+']+', "", clean_word)

        # Remove if 2 or fewer letters:
        clean_word = re.sub(r'^\w\w?$', '', clean_word)

        # remove if an article:
        clean_word = re.sub('(\s+)(a|an|and|the)(\s+)', ' ', clean_word)

        # replace words according to the custom list of replacements defined above:
        clean_word = replace(clean_word, replacement_list)

        # lowercase the word:
        clean_word = clean_word.lower()

        # remove if stopword:
        if clean_word in stopwords:
            clean_word = ''

        # remove any words that are not admissible
        if admissible_words != None and clean_word not in admissible_words:
            clean_word = ""

        return clean_word

    # clean each word:
    cleaned_comment = [clean_word(word, admissible_words) for word in comment]

    # remove empty strings
    cleaned_comment = [word for word in cleaned_comment if word != '']

    # transform into a tuple:
    cleaned_comment = tuple(cleaned_comment)

    return cleaned_comment

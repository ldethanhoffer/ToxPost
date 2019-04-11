# Detection of Toxic Comments

### Overview
------------

This project aims to describe the toxicity of a comment posted on an online forum.
To follow along, check out this [Jupyter notebook](https://github.com/ldethanhoffer/ToxPost/blob/master/notebooks/previous_version.ipynb)

To achieve this, we'll design and train a neural net on a corpus of 150.000 comments obtained from [Youtube](https://www.youtube.com).

For each comment in the corpus, we assign a __toxicity__ label consisting of 6 different binary  __categories__:

<center>
toxicity, severe toxicity, obscanity, threat, insult and identity hate
</center>




Next, we will preprocess the data

After that, we will fit both a feedforward neural net as well as an LSTM

In order to fit a neural net on the corpus, we will embed each word into a vector space using the 


### directory structure
------------


```
├── data
│    ├── raw                <- the raw data
│    ├── cleaned            <- the cleaned data 
│    ├── shrunken           <- the shrunken data
│    └── embedded           <- the embedded data
│
├── models                  <- the Trained models
│
│
├── notebooks
│     ├── explore            <- guides you through the data exploration steps 
│     └── preprocess         <- guides you through the preprocessing steps   
│
│
├── src 
│     ├── __init__.py         <- turns the file into a py module
│     │
│     ├── data                <- scipt to import the raw data
│     │   └── make_dataset.py
│     │
│     ├── features            <- preprocess raw features from the dataset
│     │   ├── clean_corpus.py
│     │   ├── shrink_corpus.py
│     │   ├── embed_corpus.py
│     │   └── preprocess.py
│     │
│     ├── models         <- Scripts to train models and then use trained models to make predictions                 
│     │   ├── predict_model.py
│     │   └── train_model.py
│     │
│     └── visualization  <- Scripts to create exploratory and results oriented visualizations
│      └── visualize.py
│
├── requirements.txt
│   
└── README.md    <- the file you're looking at!          
```


### preprocessing pipeline
------------
To see how the was preprocessed, follow along the [preprocessing notebook](https://github.com/ldethanhoffer/ToxPost/blob/master/notebooks/preprocessing.ipynb). 
We can summarize the pipeline as follows:
```
1. cleaning each comment: removing numbers, links, stopwords, hyphenation, non ascii-characters etc.  
2. using TfIdf to shrink each comment to a length of at most 60
3. embedding each word in a comment in 100d space using GloVe
4. applying pca to in turn embed each word into 25d space
5. apply the necessary padding
```



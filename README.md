# Detection of Toxic Comments

This project aims to describe the toxicity of a comment posted on an online forum.

We will break down the concept of toxicity into 6 different __categories__:

<center>
toxicity, severe toxicity, obscanity, threat, insult and identity hate
</center>


To achieve this, we'll design and train a neural net on a dataset of YouTube comments retreived from a [Kaggle competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data).  

This dataset will consist of a __corpus__ of  __comments__ together with their associated __labels__ which in turn consist of 6 binary  __categories__

We'll begin by _cleaning up_ the comments:  
Since a lot of the words in each comments are irrelevant for the purposes of detecting toxicity: stopwords, hyperlinks, spelling mistakes, characters, etc.. It makes sense to remove those from the comments

Next, we'll reduce the data

First, we will analyze the distribution of comment lengths 

We will analyze the distribution of labels. it will be clear that the data is skewed in the sense that the overwhelming majority of comments exhibit no toxicity, to remedy this we will shrink the dataset


Next, we will preprocess the data

After that, we will fit both a feedforward neural net as well as an LSTM

In order to fit a neural net on the corpus, we will embed each word into a vector space using the 


### directory structure
------------


```
│
├── data
│   ├── raw             <- the original, immutable data
│   ├── interim         <- cleaned, embedded data, not yet balanced
│   └── processed       <- the final dataset, ready to be trained
│
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks
│   ├── preprocess     <- guides you through the preprocessing steps   
│
│
├── src 
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make predictions                 
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
│
├── requirements.txt
│   
└── README.md          
```

# Detection of Toxic Comments

This project aims to describe the toxicity of a comment posted on an online forum.

We will break down the concept of toxicity into 6 different __categories__:
<br><br>
<center>
toxicity, severe toxicity, obscanity, threat, insult and identity hate
</center>
<br><br>

To achieve this, we'll design and train a neural net on a dataset of YouTube comments retreived from a [Kaggle competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data).  
This dataset will consist of a __corpus__ of  __comments__ together with their associated __labels__ which in turn consist of 6 binary  __categories__

We'll begin by _cleaning up_ the comments:  
Since a lot of the words in each comments are irrelevant for the purposes of detecting toxicity: stopwords, hyperlinks, spelling mistakes, characters, etc.. It makes sense to remove those from the comments

Next, we'll reduce the data

First, we will analyze the distribution of comment lengths 

We will analyze the distribution of labels. it will be clear that the data is skewed in the sense that the overwhelming majority of comments exhibit no toxicity, to remedy this we will shrink the dataset


Next, we will preprocess the data

After that, we will fit both a feedforward neural net as well as an LSTM

In order to fit a neural net on the corpus, we will embed each word into a vector space using the [Glove embedding](https://nlp.stanford.edu/projects/glove/) 

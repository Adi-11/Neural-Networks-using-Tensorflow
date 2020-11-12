import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.python.keras import activations
from tensorflow.python.keras.engine.training import Model


# required data-set
movie_data = keras.datasets.imdb


# dividing data-set into test and tranning set
(train_movie_data, train_label), (test_movie_data,
                                  test_lebels) = movie_data.load_data(num_words=10000)
# '''num_words=1000''' means only take those words that are 10000 more frequent and leave those words that are occuring only 1s or twice

# it will print the list of integers which are encoded form of words for making computer to read data Easily
# print(train_movie_data[0])

# Retrieves a dict mapping words to their index in the IMDB dataset.
word_index = movie_data.get_word_index()

# Dict having word as key and integer as value
word_index = {key: (value+3) for key, value in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

# reversing the initial dict such that integer will be key and words will be value
revers_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Now getting trimming reviews of maximum of of length 250 and also called as pre-processing for making our data consistance for Nural Networks 
train_movie_data = keras.preprocessing.sequence.pad_sequences(
    train_movie_data, value=word_index["<PAD>"], padding="post", maxlen=250)

test_movie_data = keras.preprocessing.sequence.pad_sequences(
    test_movie_data, value=word_index["<PAD>"], padding="post", maxlen=250)


# decode all of the data into human readable words/data
def decode_review(text):
    return " ".join([revers_word_index.get(i, "?") for i in text])


# printing of human readable reviews
# print(decode_review(test_movie_data[0]))

# print(len(test_movie_data[0]), len(test_movie_data[1]))



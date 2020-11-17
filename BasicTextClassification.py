import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.python.keras import activations
from tensorflow.python.keras.engine.training import Model
np.set_printoptions(suppress=True)


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


# model
'''
Setting up the layers
Sequential groups a linear stack of layers into a tf.keras.Model 
Since, the final output is just binary value type either positive or negative so the last layer will have only single node
'''
model = keras.Sequential()
# this layer will group words of similar kind using word vectors
model.add(keras.layers.Embedding(10000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
# This layer will give output between 0 and 1 thats what sigmoid function do....
model.add(keras.layers.Dense(1, activation="sigmoid"))


# Prints a string summary of the network.
print(model.summary())

'''
Compile the model
Before the model is ready for training, it needs a few more settings. These are added during the model's compile step:

Loss function —This measures how accurate the model is during training. You want to minimize this function to "steer" the model in 
the right direction.

Optimizer —This is how the model is updated based on the data it sees and its loss function.

Metrics —Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.
'''

model.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=["accuracy"])


# getting the validation data to each the performance of the models based on the tunnings.
x_val = train_movie_data[:10000]
x_train = train_movie_data[10000:]

y_val = train_label[:10000]
y_train = train_label[10000:]


fit_model = model.fit(x_train, y_train, epochs=40,
                      batch_size=512, validation_data=(x_val, y_val), verbose=1)

results = model.evaluate(test_movie_data, test_lebels)
print(results)

# prediction
'''
test_review = test_movie_data[0]
predict = model.predict([[test_review]])
print("Review: ")
print(decode_review(test_review))
print("Prediction: " + str(predict[0]))
print("Actual: " + str(test_lebels[0]))
print(results)
'''

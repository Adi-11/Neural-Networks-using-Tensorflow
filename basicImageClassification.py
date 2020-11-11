# importing reqiired libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.engine import input_spec

# required data-set
img_data_set = keras.datasets.fashion_mnist

# dividing data-set into test and tranning set
(train_images, train_labels), (test_images, test_lebels) = img_data_set.load_data()

# Naming of lebels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Making the pixels value small for easy working
train_images = train_images / 255.0
test_images = test_images / 255.0

print(train_images[0])

# images representation
# plt.imshow(train_images[0], cmap=plt.cm.binary)
# plt.show()


'''
Setting up the layers
Sequential groups a linear stack of layers into a tf.keras.Model 
'''
model = keras.Sequential([
    #Initial Layer of size 784
    keras.layers.Flatten(input_shape=(28, 28)),
    #Hidden Layers of size 128 and each node is connected to each of the Initial layer's Node
    keras.layers.Dense(128, activation="relu"),
    #Output layer
    keras.layers.Dense(10, activation="softmax")
])

'''
Compile the model
Before the model is ready for training, it needs a few more settings. These are added during the model's compile step:

Loss function —This measures how accurate the model is during training. You want to minimize this function to "steer" the model in 
the right direction.

Optimizer —This is how the model is updated based on the data it sees and its loss function.

Metrics —Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.
'''
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_accuracy = model.evaluate(test_images, test_lebels)

print("Tested acc: ", test_accuracy)
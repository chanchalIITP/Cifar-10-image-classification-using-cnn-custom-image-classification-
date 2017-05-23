import deepnet as net
import tflearn.datasets.mnist as mnist
from sklearn.preprocessing import LabelEncoder

import numpy
from keras.utils import np_utils


# Get the model
model = net.model

from keras.datasets import cifar10

(X, Y), (testX, testY) = cifar10.load_data()

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_Y = np_utils.to_categorical(encoded_Y)

encoder.fit(testY)
encoded_U = encoder.transform(testY)
dummy_U = np_utils.to_categorical(encoded_U)

# Load data
#X, Y, testX, testY = mnist.load_data(one_hot=True)
X = X.reshape([-3, 32, 32, 3])
testX = testX.reshape([-3, 32, 32, 3])

model.fit(X, dummy_Y, n_epoch=100, validation_set=(testX, dummy_U), show_metric=True, run_id="deep_nn")

model.save('final-model10.tflearn')
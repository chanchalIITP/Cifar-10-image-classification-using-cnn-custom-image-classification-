import deepnet as net
import random 
import tflearn.datasets.mnist as mnist
from skimage import io
import deepnet as net
import tflearn.datasets.mnist as mnist
from sklearn.preprocessing import LabelEncoder

import numpy
from keras.utils import np_utils

from keras.datasets import cifar10

(X, Y), (testX, testY) = cifar10.load_data()


model = net.model
path_to_model = 'final-model10.tflearn'

(_, _), (testX, _ )= cifar10.load_data()
model.load(path_to_model)



# Randomly take an image from the test set
rand_index = random.randint(0, len(testX) - 1)
x = testX[rand_index].reshape((32, 32, 3))

result = model.predict([x])[0] # Predict
prediction = result.index(max(result)) # The index represents the number predicted in this case
print("Prediction", prediction)

io.imsave('cifar17.jpg', x.reshape(32, 32,3)) # This shows the image in the computer for you to see
from __future__ import print_function

import densenet_fast as densenet
import numpy as np
import sklearn.metrics as metrics

from keras.datasets import cifar10
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

batch_size = 64
nb_classes = 10
nb_epoch = 250

img_rows, img_cols = 32, 32
img_channels = 3

img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (img_rows, img_cols, img_channels)
depth = 40
nb_dense_block = 3
growth_rate = 12
nb_filter = 16
dropout_rate = 0.0 # 0.0 for data augmentation

model = densenet.create_dense_net(nb_classes, img_dim, depth, nb_dense_block, growth_rate, nb_filter,
                                  dropout_rate=dropout_rate)
print("Model created")

model.summary()
optimizer = Adam(lr=1e-4) # Using Adam instead of SGD to speed up training
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
print("Finished compiling")
print("Building model...")

(trainX, trainY), (testX, testY) = cifar10.load_data()

trainX = trainX.astype('float32')
testX = testX.astype('float32')

Y_train = np_utils.to_categorical(trainY, nb_classes)
Y_test = np_utils.to_categorical(testY, nb_classes)

generator = ImageDataGenerator(rotation_range=15,
                               width_shift_range=5./32,
                               height_shift_range=5./32)

generator.fit(trainX, seed=0)

# Load model
model.load_weights("weights/DenseNet-Fast-40-12-CIFAR10.h5")
print("Model loaded.")

# model.fit_generator(generator.flow(trainX, Y_train, batch_size=batch_size), samples_per_epoch=len(trainX), nb_epoch=nb_epoch,
#                    callbacks=[ModelCheckpoint("weights/DenseNet-Fast-40-12-CIFAR10.h5", monitor="val_acc", save_best_only=True,
#                                               save_weights_only=True)],
#                    validation_data=(testX, Y_test),
#                    nb_val_samples=testX.shape[0], verbose=2)

yPreds = model.predict(testX)
yPred = np.argmax(yPreds, axis=1)
yTrue = testY

accuracy = metrics.accuracy_score(yTrue, yPred) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)


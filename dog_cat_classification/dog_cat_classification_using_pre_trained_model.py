import os
import gc
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
import datetime

from keras import models
from keras import layers
from keras import optimizers
from keras.applications import InceptionResNetV2
from keras.preprocessing.image import ImageDataGenerator

def read_process_image(list_of_images, nrows, ncloumns):

    X = []  # images
    y = []  # labels

    for image in list_of_images:
        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows, ncloumns), interpolation=cv2.INTER_CUBIC))

        if 'dog' in image:
            y.append(1)
        else:
            y.append(0)
    return X, y



train_dir = "train"

train_dogs = ["train/{}".format(i) for i in os.listdir(train_dir) if 'dog' in i]
train_cats = ["train/{}".format(i) for i in os.listdir(train_dir) if 'cat' in i]

train_imgs = train_dogs[:2000] + train_cats[:2000]
random.shuffle(train_imgs)

del train_cats
del train_dogs

gc.collect()

nrows = 150
ncloumns = 150
channels = 3

X, y = read_process_image(train_imgs, nrows, ncloumns)

# Show sample images plotted
#show_sample_images(X, y)

# plot number of images in each class
#show_no_img_in_each_class(y)

# Convert input image and label to numpy array
X = np.array(X)
y = np.array(y)

# print shapes of train images and labels

print("shape of the input images: ", X.shape)
print("shape of the input labels: ", y.shape)

# Divide train data into train and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)


# Creating pre trained model
conv_base = InceptionResNetV2(weights="imagenet", include_top=False, input_shape=(150,150,3))
model = models.Sequential()

model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.summary()

print("Number of trainable weights before freezing cov_base: ", model.trainable_weights)
conv_base.trainable = False
print("Number of trainable weights before freezing cov_base: ", model.trainable_weights)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])



# As pixcels values are ranging from 0 to 255, we need normalise it to 0 to 1 by dividing  with 255
# There is an inbuilt function in keras which do more than this. Which creates more images by rotating, scaling etc.
# This process is called image augmentation
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2,
                                   height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# We are not doing image augmentation as it is not required for the test data.
test_datagen = ImageDataGenerator(rescale=1./255)

# define following parameters which will be useful in following steps.
batch_size = 32
ntrain = X_train.shape[0]

# Now create image generators to train the data
# we need to pass batch size which tells the image generator to take only those many images at a time.
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
test_generator = test_datagen.flow(X_test, y_test, batch_size=batch_size)

# Now train the model
# Steps per epoch tells ourt model how many images we want to process before making a gradient update
# Epochs tells about how many times to go over training data
train_result = model.fit_generator(train_generator,
                                   steps_per_epoch=ntrain // batch_size,
                                   epochs=20,
                                   validation_data=test_generator,
                                   validation_steps=ntrain // batch_size)

print(datetime.datetime.now())

train_accuracy = train_result.history["acc"]
val_accuracy = train_result.history["val_acc"]
train_loss = train_result.history["loss"]
val_loss = train_result.history['val_loss']

epochs = range(1, len(train_accuracy) + 1)

# Train and validation accuracy plot
plt.plot(epochs, train_accuracy, 'b', label="training accuracy")
plt.plot(epochs, val_accuracy, 'r', label="validation accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.figure()

# Train and validation loss plot
plt.plot(epochs, train_loss, 'b', label="training loss")
plt.plot(epochs, val_loss,'r', label="validation loss")
plt.title("Training and Validation Loss")
plt.legend()

plt.show(block=True)

# Save the model
model.save_weights('pretrained_model_weights.h5')
model.save('pretrained_model_keras.h5')

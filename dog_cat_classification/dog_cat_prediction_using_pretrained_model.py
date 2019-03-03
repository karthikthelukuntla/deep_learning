from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2


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


test_dir = "test"

nrows = 150
ncloumns = 150
channels = 3

test_imgs = ["test/{}".format(i) for i in os.listdir(test_dir)]

X, y = read_process_image(test_imgs, nrows, ncloumns)

X = np.array(X)

print(os.getcwd())

model = load_model("pretrained_model_keras.h5")
model.load_weights("pretrained_model_weights.h5")

test_datagen = ImageDataGenerator(rescale=1./255)

text_labels = []
i = 0
fig = plt.figure(figsize=(12, 10))
fig.tight_layout()
fig.subplots_adjust(top=0.8, bottom=0.01, hspace=1.5, wspace=0.4)
for batch in test_datagen.flow(X, batch_size=1):

    pred = model.predict(batch)

    if pred > 0.5:
        text_labels.append("dog")
    else:
        text_labels.append("cat")

    ax = plt.subplot(5, 2, i + 1)
    ax.title.set_text("This is " + text_labels[i])
    ax.imshow(batch[0])

    i += 1

    if i % 10 == 0:
        break;

plt.show(block=True)
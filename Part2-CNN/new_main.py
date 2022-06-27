# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 15:34:26 2022

@author: idankas
"""


# %% Setup
import numpy as np
from preprocessing import preprocessing_step
import imageio as iio
import os
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from augmentation import images_augmentation
import pandas as pd
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from flowers_data_generator import DataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

height = 200
width = 200
channels = 3
n_classes = 102
input_shape = (height, width, channels)
epochs = 200
batch_size = 256
# %% -------------------------- EX. 2 - CNN -----------------------------------

# Parameters
new_img_size = 200

# Data Loading
# Read Images & Preprocessing Step
file_names = os.listdir(os.path.join(os.getcwd(),'102flowers'))
images = []
for idx in range(len(file_names)):
    current_img = os.path.join(os.getcwd(),'102flowers',file_names[idx])
    img = iio.imread(current_img)
    pp = preprocessing_step(img, new_img_size)
    processed_img = pp.preprocess()
    images.append(processed_img)
images = np.array(images)


# Read Labels
mat = loadmat('imagelabels.mat')
labels = mat['labels']
labels = labels.reshape([labels.shape[1],])

# Suffle
images, labels = shuffle(images, labels, random_state=0)

labels = pd.Categorical(labels)
labels = to_categorical(labels)
labels = labels[:,1:]

# Data Spliting - Train, Validation and Test - repeat twice!!!
X_train, X_rest, y_train, y_rest = train_test_split(images, labels, test_size=0.5, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=42)

# Augmentation for train data
aug = images_augmentation(X_train, y_train)
X_train_aug , y_train_aug = aug.make_augmentations()


# %% Create data generators

train_data_generator = DataGenerator(X_train_aug, y_train_aug, augment=False)
valid_data_generator = DataGenerator(X_val, y_val, augment=False)


# %% First Network - Mobilenet
# Model Creation
base_model = MobileNet(weights='imagenet', include_top=False)
model = Sequential()
model.add(InputLayer(input_shape=input_shape))
model.add(base_model)
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))

model.summary()

# Model Compiling and Training
optimizer = Adam(lr=0.0001)
#early stopping to monitor the validation loss and avoid overfitting
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True)
#reducing learning rate on plateau
rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience= 5, factor= 0.5, min_lr= 1e-6, verbose=1)

#model compiling
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
# Training
model_history = model.fit_generator(train_data_generator,
                                    validation_data=valid_data_generator,
                                    callbacks=[early_stop, rlrop],
                                    verbose=1,
                                    epochs=epochs)

#saving the trained model weights as data file in .h5 format
model.save_weights("flowers_Mobilenet2_weights.h5")
model.save('flowers_Mobilenet2_model.h5')

# Training Visualization
plt.suptitle('Loss and Accuracy Plots', fontsize=18)

plt.subplot(1,2,1)
plt.plot(model_history.history['loss'], label='Training Loss')
plt.plot(model_history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Number of epochs', fontsize=15)
plt.ylabel('Loss', fontsize=15)

plt.subplot(1,2,2)
plt.plot(model_history.history['accuracy'], label='Train Accuracy')
plt.plot(model_history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.xlabel('Number of epochs', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.show()

# Prediction and Eva;uation
y_pred = model.predict_generator(DataGenerator(X_test, mode='predict', augment=False, shuffle=False), verbose=1)
y_pred = np.argmax(y_pred, axis=1)
test_accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred)

print('Test Accuracy: ', round((test_accuracy * 100), 2), "%")

target = ["Category {}".format(i) for i in range(n_classes)]
print(classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target))

# %% Second Network - InceptionV3
# Model Creation
base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False)
model = Sequential()
model.add(InputLayer(input_shape=input_shape))
model.add(base_model)
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))

model.summary()

# Model Compiling and Training
optimizer = Adam(lr=0.0001)
#early stopping to monitor the validation loss and avoid overfitting
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True)
#reducing learning rate on plateau
rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience= 5, factor= 0.5, min_lr= 1e-6, verbose=1)

#model compiling
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
# Training
model_history = model.fit_generator(train_data_generator,
                                    validation_data=valid_data_generator,
                                    callbacks=[early_stop, rlrop],
                                    verbose=1,
                                    epochs=epochs)

#saving the trained model weights as data file in .h5 format
model.save_weights("flowers_Inception2_weights.h5")
model.save('flowers_Inception2_model.h5')

# Training Visualization
plt.suptitle('Loss and Accuracy Plots', fontsize=18)

plt.subplot(1,2,1)
plt.plot(model_history.history['loss'], label='Training Loss')
plt.plot(model_history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Number of epochs', fontsize=15)
plt.ylabel('Loss', fontsize=15)

plt.subplot(1,2,2)
plt.plot(model_history.history['accuracy'], label='Train Accuracy')
plt.plot(model_history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.xlabel('Number of epochs', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.show()

# Prediction and Eva;uation
y_pred = model.predict_generator(DataGenerator(X_test, mode='predict', augment=False, shuffle=False), verbose=1)
y_pred = np.argmax(y_pred, axis=1)
test_accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred)

print('Test Accuracy: ', round((test_accuracy * 100), 2), "%")

target = ["Category {}".format(i) for i in range(n_classes)]
print(classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target))



#%% For report
from pylab import rcParams
from numpy import expand_dims
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Data Visualization
rcParams['figure.figsize'] = 8,8

num_row = 4
num_col = 4

#to get 4 * 4 = 16 images together
imageId = np.random.randint(0, len(X_train), num_row * num_col)
#imageId

fig, axes = plt.subplots(num_row, num_col)

for i in range(0, num_row):
    for j in range(0, num_col):
        k = (i*num_col)+j
        axes[i,j].imshow(X_train[imageId[k]])
        axes[i,j].axis('off')

# Augmentations visualization

# load the image
img = load_img(os.path.join(os.getcwd(),'102flowers','image_00579.jpg'))
# convert to numpy array
data = img_to_array(img)
# expand dimension to one sample
samples = expand_dims(data, 0)
# create image data augmentation generator
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
# prepare iterator
it = datagen.flow(samples, batch_size=1)
fig, axes = plt.subplots(3,3)
# generate samples and plot
for i in range(0, 3):
    for j in range(0, 3):
        batch = it.next()
        image = batch[0].astype('uint8')
        axes[i,j].imshow(image)
        axes[i,j].axis('off')


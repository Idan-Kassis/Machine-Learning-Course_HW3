# -*- coding: utf-8 -*-
"""
Created on Wed May 18 13:20:40 2022

@author: Dror
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.utils import shuffle


class images_augmentation():
    
    def __init__(self, X_train, y_train, num_augmentations = 10):
        self.X_train = X_train
        self.y_train = y_train
        self.num_augmentations = 8
    
    def make_augmentations(self):
        datagen = ImageDataGenerator(
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest')
        datagen.fit(self.X_train, augment=True)
        
        # concatenate 
        counter = 0
        
        for x_batch, y_batch in datagen.flow(self.X_train, self.y_train, batch_size = self.X_train.shape[0]):
            self.X_train = np.concatenate((self.X_train,x_batch), axis = 0)
            self.y_train = np.concatenate((self.y_train,y_batch), axis = 0)
            counter += 1
            if (counter == self.num_augmentations): # end after num_augmentations variations
                counter = 0
                break
        
        # Suffle
        self.X_train, self.y_train = shuffle(self.X_train, self.y_train, random_state=0)
        return self.X_train, self.y_train

    
        
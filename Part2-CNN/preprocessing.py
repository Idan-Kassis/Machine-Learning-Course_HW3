# -*- coding: utf-8 -*-
"""
Created on Mon May  9 20:42:13 2022

@author: Dror
"""
import cv2


class preprocessing_step():
    
    def __init__(self,image, new_size):
        self.image = image
        self.min = 0
        self.max = 255
        self.new_size = new_size
        
    def normalize(self,img):
        norm_img = (img - self.min)/(self.max - self.min)
        return norm_img
        
    def resize(self,img):
        resized_image = cv2.resize(img, (self.new_size, self.new_size)) 
        return resized_image
        
    def preprocess(self):
        img = self.image
        # Normalize
        image = self.normalize(img)
        # Resize
        image = self.resize(image)
        return image
    
        
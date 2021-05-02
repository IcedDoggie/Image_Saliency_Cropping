#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import scipy.io
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os

# keras library
import keras
import tensorflow as tf
import keras.preprocessing.image as image
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from keras.layers.core import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50
import pydot, graphviz
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications.resnet50 import preprocess_input
print(keras.__version__)

# own libraries
from networks import unet
from utilities import binarization, path_reader, read_images_from_path
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
# ### Data Understanding
# images    -- train, val, test. Normal looking images
# maps      -- train, val. grayscale points of attention 
# fixations -- train, val, test.

# All pathing goes here

root_dir = '/home/babeen/Documents/Image Saliency Cropping/'
images_path = root_dir + 'images/'
maps_path = root_dir + 'maps/'
fixation_path = root_dir + 'fixations/'
binarized_maps_path = root_dir + 'binarized_maps/'

images_train_path = images_path + 'train/'
images_val_path = images_path + 'val/'
images_test_path = images_path + 'test/'

maps_train_path = maps_path + 'train/'
maps_val_path = maps_path + 'val/'
maps_test_path = maps_path + 'test/'

fixations_train_path = fixation_path + 'train/'
fixations_val_path = fixation_path + 'val/'
fixations_test_path = fixation_path + 'test/'

binarized_maps_train_path = binarized_maps_path + 'train/'
binarized_maps_val_path = binarized_maps_path + 'val/'
binarized_maps_test_path = binarized_maps_path + 'test/'



# initialized optimizers and losses
learning_rate = 0.0001
adam = optimizers.Adam(lr=learning_rate, decay=1e-7)
epochs = 10
batch_size = 2
spatial_size = 256


# image data generator
images_train_datagen = image.ImageDataGenerator(rescale=1. / 255, preprocessing_function=preprocess_input, data_format='channels_last')
images_val_datagen = image.ImageDataGenerator(rescale=1. / 255, preprocessing_function=preprocess_input, data_format='channels_last')
images_test_datagen = image.ImageDataGenerator(rescale=1. / 255, preprocessing_function=preprocess_input, data_format='channels_last')

maps_train_datagen = image.ImageDataGenerator(preprocessing_function=None, data_format='channels_last')
maps_val_datagen = image.ImageDataGenerator(preprocessing_function=None, data_format='channels_last')
maps_test_datagen = image.ImageDataGenerator(preprocessing_function=None, data_format='channels_last')

# flow
images_train_generator = images_train_datagen.flow_from_directory(directory=images_train_path, classes=None, class_mode=None, batch_size=batch_size, target_size=(spatial_size, spatial_size), shuffle=False)
images_val_generator = images_val_datagen.flow_from_directory(directory=images_val_path, classes=None, class_mode=None, batch_size=batch_size, target_size=(spatial_size, spatial_size), shuffle=False)
images_test_generator = images_test_datagen.flow_from_directory(directory=images_test_path, classes=None, class_mode=None, batch_size=batch_size, target_size=(spatial_size, spatial_size), shuffle=False)

maps_train_generator = maps_train_datagen.flow_from_directory(directory=maps_train_path, classes=None, class_mode=None, batch_size=batch_size, color_mode='grayscale', target_size=(spatial_size, spatial_size), shuffle=False)
maps_val_generator = maps_val_datagen.flow_from_directory(directory=maps_val_path, classes=None, class_mode=None, batch_size=batch_size, color_mode='grayscale', target_size=(spatial_size, spatial_size), shuffle=False)
maps_test_generator = maps_test_datagen.flow_from_directory(directory=maps_test_path, classes=None, class_mode=None, batch_size=batch_size, color_mode='grayscale', target_size=(spatial_size, spatial_size), shuffle=False)



# compile model
model = unet()
model.compile(loss='mean_squared_error', optimizer=adam, metrics=[metrics.categorical_accuracy])

# # get items out of generators

for epoch in range(epochs):
	print("Epoch %i / %i" % (epoch, epochs))
	batches = 0
	for imgs, maps in zip(images_train_generator, maps_train_generator):
		
		model.fit(imgs, maps, batch_size = batch_size, epochs = 1, shuffle=False)
		batches += 1
		print("Samples Trained %i / %i" % (int(batches * batch_size), len(images_train_generator)))
		

# model.fit(x=images_train_generator, y=None, epochs=epochs)








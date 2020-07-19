# -*- coding: utf-8 -*-
"""
Created on Thu Jul 2 16:26:18 2020

@author: Xiao
"""
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import models

# Set path
path = os.path.join('D:\\kaggle13')
train_type = 'train_org'
val_type = 'test_org'

# Train dir
train_dir = os.path.join(path,train_type)
train_cat = os.path.join(train_dir, 'cat')
train_dog = os.path.join(train_dir, 'dog')

# Validate dir
val_dir = os.path.join(path, val_type) 
val_cat = os.path.join(val_dir, 'cat') 
val_dog = os.path.join(val_dir, 'dog') 

# Saved model
saved_model = 'model-' + str(datetime.date.today()) + '-' + train_type + '.h5' 


# Number of data
num_train_cat = len(os.listdir(train_cat))
num_train_dog = len(os.listdir(train_dog))
num_train = num_train_cat + num_train_dog

num_val_cat = len(os.listdir(val_cat))
num_val_dog = len(os.listdir(val_dog))
num_val = num_val_cat + num_val_dog


# Model setting
batch_size = 64
epochs = 100
img_height = 256
img_width = 256


# Data flow
train_generator = ImageDataGenerator(rescale=1./255)
validation_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_generator.flow_from_directory(
    batch_size=batch_size, 
    directory=train_dir, 
    shuffle=True,
    target_size=(img_height, img_width), 
    class_mode='binary')
val_data_gen = validation_generator.flow_from_directory(
    batch_size=batch_size, 
    directory=val_dir,
    target_size=(img_height, img_width), 
    class_mode='binary')


# Model
model = models.model_2(img_height, img_width)
model.summary()

# Train
history = model.fit(
    train_data_gen,
    steps_per_epoch = num_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps = num_val // batch_size,
)

# Save the model
model.save(saved_model)

# output results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
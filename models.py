# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 20:16:34 2020

@author: Xiao
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, SpatialDropout2D

def model_0(img_height, img_width):
    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=(img_height, img_width, 3)),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1)
        ])
    model.compile(optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy'])
    return model

def model_1(img_height, img_width):
    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=(img_height, img_width, 3)),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.4),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return model

def model_2(img_height, img_width):
    model = Sequential([
        Conv2D(6, 5, activation='relu', input_shape=(img_height, img_width, 3)),
        MaxPooling2D(),
        Conv2D(16, 5, activation='relu'),
        MaxPooling2D(),
        Conv2D(16, 5, activation='relu'),
        SpatialDropout2D(0.5),
        Flatten(),
        Dense(5000),
        Dense(200),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return model
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 2 17:56:51 2020

@author: Xiao
"""

import numpy as np
import tensorflow as tf

reload_model = tf.keras.models.load_model('./model-2020-07-16-train_org.h5')
reload_model.summary()
path = 'D:\\kaggle13\\test_org\\cat\\cat.1.jpg'
path2 = 'D:\\kaggle13\\test_org\\dog\\dog.3.jpg'

image = tf.keras.preprocessing.image.load_img(path2, target_size=(200,200))

input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr]) / 255.0  # Convert single image to a batch.
predictions = reload_model.predict(input_arr)
print(predictions)
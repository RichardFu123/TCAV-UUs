# -*- coding: utf-8 -*-
"""
Created on Thu Jul 2 17:56:51 2020

@author: Xiao
"""

import os
import numpy as np
import tensorflow as tf
import get_black_cat

reload_model = tf.keras.models.load_model('./model-2020-07-19-train_org.h5')
reload_model.summary()
path = 'D:\\kaggle13\\test_org\\cat\\cat.1.jpg'
path_black_cat = get_black_cat.path_black_cat

def one_image_array(path):
	image = tf.keras.preprocessing.image.load_img(path, target_size=(256,256))
	input_arr = tf.keras.preprocessing.image.img_to_array(image)
	return input_arr

#input_arr = np.array([one_image_array(path)]) / 255.0  # Convert single image to a batch.
#predictions = reload_model.predict(input_arr)
#print(predictions)

dirs_black_cat = os.listdir(path_black_cat)
input_arr = []
for cats in dirs_black_cat:
	input_arr.append(one_image_array(os.path.join(path_black_cat, cats)))

input_arr = np.array(input_arr)/255.0
predictions = reload_model.predict(input_arr)
num_of_dogs = 0
confidence_ave = 0.
result_cat = open('org_cat.txt', 'w')
result_dog = open('org_dog.txt', 'w')
for prediction in predictions:
	num_of_dogs += np.argmax(prediction)
	confidence_ave += prediction[0]
	result_cat.write(str(prediction[0])+'\n')
	result_dog.write(str(prediction[1])+'\n')

result_cat.close()
result_dog.close()
	
acc = (len(predictions) - num_of_dogs)*1./len(predictions)
confidence_ave = confidence_ave/len(predictions)
print(acc)
print(confidence_ave)

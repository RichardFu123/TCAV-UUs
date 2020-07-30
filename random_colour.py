# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 23:02:11 2020

@author: Xiao
"""
import cv2
import numpy as np
import random 
import os

def random_256():
    return random.randint(0,255)

def random_rgb():
    return [random_256(),random_256(), random_256()]

def one_layer(height, width, value):
    return np.ones((height, width))*value

def one_image_rgb(height, width,random_rgb):
    img = np.ones((height,width,3))
    for i in range(height):
        for j in range(width):
            img[i][j] = random_rgb
    return img
    
def save_one_image(img, path):
    cv2.imwrite(path,img)
    
def make_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def images_in_one_file(height, width, rgb, path, numbers):
    make_dir_if_not_exists(path)
    img = one_image_rgb(height, width,rgb)
    rgb_name = "_".join([str(i) for i in rgb])
    for i in range(numbers):
        img_path = os.path.join(path,rgb_name+"_"+str(i)+".png")
        save_one_image(img,img_path)
if __name__ == "__main__":
    for i in range(100):
        images_in_one_file(256,256,random_rgb(),"c"+str(i),40)
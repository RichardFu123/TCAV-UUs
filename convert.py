# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 21:10:33 2020

@author: Xiao
"""

import cv2
import numpy as np

base = cv2.imread('./cat.1927.jpg')
base = cv2.resize(base,(256,256), interpolation=cv2.INTER_CUBIC)
filter = cv2.imread('./striped_0023.jpg')
filter = cv2.resize(filter,(256,256), interpolation=cv2.INTER_CUBIC)
trans = base//2+filter//2
cv2.imwrite("trans7.png",trans)
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 21:18:08 2020

@author: Xiao
"""

import matplotlib.pyplot as plt
import numpy as np

n = 144
X = np.arange(n)
file_art_cat = open('art_cat.txt', 'r')
file_art_dog = open('art_dog.txt', 'r')
file_org_cat = open('org_cat.txt', 'r')
file_org_dog = open('org_dog.txt', 'r')

art_cat = np.zeros(144)
art_dog = np.zeros(144)
org_cat = np.zeros(144)
org_dog = np.zeros(144)
for i in range(n):
    art_cat[i] = float(file_art_cat.readline())
    art_dog[i] = float(file_art_dog.readline())
    org_cat[i] = float(file_org_cat.readline())
    org_dog[i] = float(file_org_dog.readline())
    
file_art_cat.close()
file_art_dog.close()
file_org_cat.close()
file_org_dog.close()

print(np.mean(art_cat))
print(np.median(art_cat))
print(np.mean(art_dog))
print(np.median(art_dog))
print(np.mean(org_cat))
print(np.median(org_cat))
print(np.mean(org_dog))
print(np.median(org_dog))

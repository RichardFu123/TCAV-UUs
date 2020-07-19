# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 10:48:11 2020

@author: Xiao
"""

import os
import shutil

path_org = 'D:/kaggle13/test_org/cat/'
path_art = 'D:/kaggle13/train/cat/'
path_black_cat = 'D:/kaggle13/test_black_cat/cat/'

dirs_org = os.listdir(path_org)
dirs_art = os.listdir(path_art)
for cat in dirs_org:
    if cat not in dirs_art:
        shutil.copy(os.path.join(path_org,cat),path_black_cat)
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 20:06:42 2021

@author: breog
"""
#IMPORTS

#Standard
import pandas as pd



input_path='./../input/'
train_name="train.csv"
test_name="test.csv"

train_df=pd.read_csv(input_path+train_name)
test_df=pd.read_csv(input_path+test_name)
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 20:06:42 2021

@author: breog
"""
#https://www.kaggle.com/mnassrib/titanic-logistic-regression-with-python

#IMPORTS

#Standard
import pandas as pd



input_path='./../input/'
train_name="train.csv"
test_name="test.csv"

train_df=pd.read_csv(input_path+train_name)
test_df=pd.read_csv(input_path+test_name)


#Transform categorical into dummies
pd.get_dummies(train_df,columns=['Sex','Embarked','Pclass'])
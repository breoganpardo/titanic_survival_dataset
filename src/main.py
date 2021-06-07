# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 20:06:42 2021

@author: breog
"""
#https://www.kaggle.com/mnassrib/titanic-logistic-regression-with-python

#IMPORTS
import pandas as pd
import numpy as np

from ml_utils import split_X_y

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.metrics import plot_confusion_matrix

#VARIABLES
#Set current working directory to this file location
#abspath = os.path.abspath(__file__)
#dname = os.path.dirname(abspath)
#os.chdir(dname)

input_path='./../input/'
train_name="train.csv"
test_name="test.csv"

train_df=pd.read_csv(input_path+train_name)
test_df=pd.read_csv(input_path+test_name)

X_train_df,y_train_df=split_X_y(train_df,'Survived')

####################################################
############### X_TRAIN processing ##################
####################################################
#1. Simple Preprocessing

#Drop values that require further analysis
feature_drop_list=['Name','Ticket','Cabin']

axis,inplace=1,True
X_train_df.drop(labels=feature_drop_list, axis=axis,inplace=inplace)
test_df.drop(labels=feature_drop_list, axis=axis,inplace=inplace)
##############################################
####X_TRAIN: Base line train (drop all nan)###
##############################################

#X_train_base=X_train_df.dropna(axis=0, how='any',subset=None, inplace=False)
simple_imputer_list=['Age']
one_hot_list=['Sex','Pclass','Embarked']
trans_ct=ColumnTransformer([('SimpleImputer', SimpleImputer(strategy="median"), simple_imputer_list),
                           ('encoder', OneHotEncoder(), one_hot_list)],
                           remainder='passthrough')

X_train_base=trans_ct.fit_transform(X_train_df)
X_test_base=trans_ct.transform(test_df)
X_test_base=np.nan_to_num(X_test_base)
                               
##############################################
###### X_TRAIN V01: (age filled in) ##########
##############################################
#
#Deal with empty values in Age

#Option 1: Use a regressor

#Prepare data set
labels,inplace=['Age','PassengerId']
train_age_df=X_train_df.dropna(axis=0,inplace=False)
X_age=train_age_df.drop(labels=labels, axis=axis,inplace=False)
y_age=train_age_df.loc[:,'Age']

one_hot_list=['Sex','Pclass','Embarked']
trans_ct=ColumnTransformer(
        [('encoder', OneHotEncoder(), one_hot_list)],
        remainder='passthrough')
X_age_ct=trans_ct.fit_transform(X_age)

#Define regressors and competition
rg_gb_age=GradientBoostingRegressor()

#CV_scores=cross_validate(rg_gb,X_age_ct,y_age,cv=3,scoring='r2')
#Mean R2 is 0.237, which is low. With data in X we can only explain 23% of the variability of Age...
#But still is better than that inputing the mean
rg_gb_age.fit(X_age_ct,y_age)

#Option 2:_____________________________________
#
#
#
#
#


#Finally we input empty age values using the best option

#empty_age_records=X_train_df.loc[rule1,:]
#X_empty_age=empty_age_records.drop(labels=labels, axis=axis,inplace=inplace)
#X_empty_age_transf=trans_ct.transform(X_empty_age)

#Imputing empty values using only training data info to avoid data leakage
#For X,1.Select empty values,2. drop columns not needed,3. columns transformer
X_train_v01=X_train_df.copy(deep=True)
X_test_v01=test_df.copy(deep=True)

rule1=X_train_v01['Age'].isnull()
rule2=X_test_v01['Age'].isnull()

X_train_v01.loc[rule1,'Age']=rg_gb_age.predict(trans_ct.transform(X_train_v01.loc[rule1,:].drop(labels=labels, axis=axis,inplace=False)))
X_test_v01.loc[rule2,'Age']=rg_gb_age.predict(trans_ct.transform(X_test_v01.loc[rule2,:].drop(labels=labels, axis=axis,inplace=False)))


one_hot_list=['Sex','Pclass','Embarked']
trans_ct=ColumnTransformer(
        [('encoder', OneHotEncoder(), one_hot_list)], remainder='passthrough')

X_train_v01=trans_ct.fit_transform(X_train_v01)
X_test_v01=trans_ct.transform(X_test_v01)
X_test_v01=np.nan_to_num(X_test_v01)


###########################################
####### PREPARE X_TRAIN Datasets###########
###########################################


#2.Prepare data sets for ML 

#    #Separate training set into train, validations
#train_size=0.85
#random_state=20210329
#
#X_train,X_val,y_train,y_val=train_test_split(X_train_df,y_train_df,
#                                             train_size=train_size,
#                                             random_state=random_state)

#3.Train classifiers
random_state=20210607
log_clf=LogisticRegression(random_state=random_state,max_iter=10000)
rf_clf=RandomForestClassifier(random_state=random_state)

log_clf2=LogisticRegression(random_state=random_state,max_iter=10000)
rf_clf2=RandomForestClassifier(random_state=random_state)


##################################
##### Base line results ##########
##################################
cross_validate(log_clf, X=X_train_base, y=y_train_df,cv=3,scoring=('accuracy','recall'))
plot_confusion_matrix(log_clf,X=X_train_base, y_true=y_train_df)


log_clf.fit(X_train_base,y_train_df)
rf_clf.fit(X_train_base,y_train_df)

log_clf2.fit(X_train_v01,y_train_df)
rf_clf2.fit(X_train_v01,y_train_df)

plot_confusion_matrix(log_clf,X=X_train_base, y_true=y_train_df)
#Conclusions: no string variables, no empty values

#4. Evaluate on validation set
#confusion_matrix(y_val,log_clf.predict(X_val))
#confusion_matrix(y_val,rf_clf.predict(X_val))
log_clf.get_params()
log_clf.score(X_val,y_val)
rf_clf.score(X_val,y_val)

confusion_matrix(y_val,log_clf.predict(X_val))

#5. Output prediction on test set
#PassengerId,Survived
data={'PassengerId':test_df['PassengerId'],'Survived':rf_clf.predict(X_test)}
output_df=pd.DataFrame(data=data)

version='01'
output_path=f'./../output/prediction_{version}.csv'
output_df.to_csv(output_path,index=False)



# -*- coding: utf-8 -*-
"""
Created on Sat May 20 18:11:55 2023

@author: Aparna
"""
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 11:51:53 2023

@author: Aparna
"""

import pytest
from regression_class import LogisticRegression2
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.ensemble import ExtraTreesClassifier
###-------------------------------------------------------------------------###
###                      Feature Classification                        ###
###-------------------------------------------------------------------------###
# load the data as data frame
df = pd.read_csv("ProcessedClevelandData.csv")
# set up response variables
outcome="num"

# Use feature selection to assess which variables have higher impact
# set up response and explainatory variables
explainatory_variable = df.iloc[:,0:13] #independent variables
response_variable = df.iloc[:,-1] #outcome 

model = ExtraTreesClassifier()
model.fit(explainatory_variable,response_variable)
print(model.feature_importances_)
#use inbuilt class feature_importances of tree based classifiers
feat_importances = pd.Series(model.feature_importances_, index=explainatory_variable.columns)
feat_importances.nlargest(13).plot(kind='barh')
plt.show()




###-------------------------------------------------------------------------###
###                      logistic regression for classification                         ###
###-------------------------------------------------------------------------###


# run logistic regression analysis on the data
model = LogisticRegression2()
model.fit(df,outcome)
#model.fit(df,outcome,'age','trestbps','chol','thalach','oldpeak')
print(model.Summary(df))
#print(model.predict(df))
print(model.accuracy(df))
model.plot_CM(df)

###-------------------------------------------------------------------------###
###      Comparison with logistic regression from standard python module    ###
###-------------------------------------------------------------------------###
# logistic regression with standard python module for comparison
def binary_data(df,outcome):
    X=df.loc[:, df.columns != outcome]
    X.to_numpy()
    y=df.loc[:, df.columns == outcome] 
    y.to_numpy()
    return X.to_numpy(), y.to_numpy()

X,y =binary_data(df,outcome)

#binonmial_model = sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial())
logit_model = sm.Logit(y, sm.add_constant(X))
#res = binonmial_model.fit()
res = logit_model.fit()
res.summary()

###-------------------------------------------------------------------------###
###                      Naive Bayes Classification                         ###
###-------------------------------------------------------------------------###

# 20% of the total data are selected for training
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) 

# Instantiate the Gaussian Naive Bayes classifier 
gnb = GaussianNB() 
# Train clasifier
y_train = y_train.flatten()
gnb.fit(x_train,y_train)
# Predict the outcome in test set
y_pred = classifier.predict(x_test)

# Precision, recall, F1-score, and support for each class
print(classification_report(y_test,y_pred)) 
ax = plt.subplot()
# Display the performance of the classifier by comparing the actual and predicted classes
sns.heatmap(cm_test, annot=True, fmt='g', ax = ax); #annot=True to annotate cells


















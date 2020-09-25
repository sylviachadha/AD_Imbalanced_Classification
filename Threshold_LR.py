#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 16:59:50 2020

@author: sylviachadha
"""

# Threhold Decision using roc curve in Logistic regression

# roc curve for logistic regression model
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from matplotlib import pyplot

# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)

# split into train/test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)


# fit a model
model = LogisticRegression(solver='lbfgs')
model.fit(trainX, trainy)

# predict probabilities
yhat = model.predict_proba(testX)
#yhat2 = model.predict(testX) - Normally u use predict


len(yhat)
# 5000 test samples values u get in terms of probabilities


# keep probabilities for the positive outcome only
yhat1 = yhat[:, 1]

# 5000

# calculate roc curves
fpr, tpr, thresholds = roc_curve(testy, yhat1)

# plot the roc curve for the model
pyplot.plot([0,1], [0,1], linestyle='--', label='No Skill') 
# Above line will plot 0 and 1 values on x-axis and corresponding 0 and 1
# values on y axis so a line will be created by coordinates (0,0) and (1,1)

pyplot.plot(fpr, tpr, marker='.', label='Logistic')

# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate') 
pyplot.legend()
# show the plot
pyplot.show()


from numpy import sqrt
from numpy import argmax
gmeans = sqrt(tpr * (1-fpr))

# locate the index of the largest g-mean
ix = argmax(gmeans)
print('Best Threshold=%f, G-mean=%.3f' % (thresholds[ix], gmeans[ix]))

# plot the roc curve for the model
pyplot.plot([0,1], [0,1], linestyle='--', label='No Skill') 
pyplot.plot(fpr, tpr, marker='.', label='Logistic') 
pyplot.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best') # axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
# show the plot
pyplot.show()











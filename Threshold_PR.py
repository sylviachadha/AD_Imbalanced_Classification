#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 18:00:02 2020

@author: sylviachadha
"""


# Determining threhold using precision recall curve

# Evaluating the predicted probabilities for a logistic regression model
# using a precision-recall curve

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot
from numpy import argmax

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

# keep probabilities for the positive outcome only
yhat1 = yhat[:, 1]

# calculate pr-curve
precision, recall, thresholds = precision_recall_curve(testy, yhat1)

# plot the roc curve for the model
no_skill = len(testy[testy==1]) / len(testy)
pyplot.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill') 
pyplot.plot(recall, precision, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
pyplot.legend()
# show the plot
pyplot.show()


# Which threshold would give us the best result?

# If we are interested in a threshold that results in the best balance of 
# precision and recall, then this is the same as optimizing the F-measure 
# that summarizes the harmonic mean of both measures.

# convert to f-measure
fscore = (2 * precision * recall) / (precision + recall)

# locate the index of the largest f-measure
ix = argmax(fscore)
print('Best Threshold=%f, F-measure=%.3f' % (thresholds[ix], fscore[ix]))

# plot the roc curve for the model
no_skill = len(testy[testy==1]) / len(testy)
pyplot.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill') 
pyplot.plot(recall, precision, marker='.', label='Logistic') 
pyplot.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best') # axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
pyplot.legend()
# show the plot
pyplot.show()















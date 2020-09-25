#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 18:38:45 2020

@author: sylviachadha
"""


# Optimal Threshold Tuning


# logistic regression for imbalanced classification

# evaluating the F-measure of a logistic regression using the default 
# threshold.

from sklearn.datasets import make_classification 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score
from numpy import arange
from numpy import argmax

# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, 
                           n_clusters_per_class=1, weights=[0.99], flip_y=0, 
                           random_state=4)

# split into train/test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)


# fit a model
model = LogisticRegression(solver='lbfgs') 
model.fit(trainX, trainy)

# predict labels
yhat = model.predict(testX)
# In this one we are directly using the function of predict labels which
# is model.predict instead of model.predictproba so default threshold of
# 0.5 is used in the model with model.predict

# evaluate the model
score = f1_score(testy, yhat) 
print('F-measure: %.5f' % score)

# F-measure: 0.70130 using default threshold 0.5 [model.predict function]


# Now we can use the same model on the same dataset and instead of predicting 
# class labels directly, we can predict probabilities

# predict probabilities
yhat1 = model.predict_proba(testX)

# keep probabilities for the positive outcome only
probs = yhat1[:, 1]

# Next, we can then define a set of thresholds to evaluate the probabilities.
# we will test all thresholds between 0.0 and 1.0 with a step size of 0.001, 
# that is, we will test 0.0, 0.001, 0.002, 0.003, and so on to 0.999.

# define thresholds
thresholds = arange(0, 1, 0.001)

# evaluate each threshold
# We will define a to labels() function to do this that will take the 
# probabilities and threshold as an argument and return an array of integers
# in {0, 1}.
# Mapping all values equal to or greater than the threshold to 1 and all 
# values less than the threshold to 0.


# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')

# We can then call this function for each threshold and evaluate the 
# resulting labels using the f1 score()
    
# evaluate each threshold
scores = [f1_score(testy, to_labels(probs, t)) for t in thresholds]


# locate the array index that has the largest score (best F-measure) and we
# will have the optimal threshold

# get best threshold
ix = argmax(scores)
print('Threshold=%.3f, F-measure=%.5f' % (thresholds[ix], scores[ix]))


#Running the example reports the optimal threshold as 0.251 (compared to 
# the default of 0.5) that achieves an F-measure of about 0.75 (compared to
# 0.70).
















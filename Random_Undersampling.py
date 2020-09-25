#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 20:49:29 2020

@author: sylviachadha
"""


# Data Sampling
# Random Undersample

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import RandomUnderSampler

# define dataset
X, y = make_classification(n_samples=10000, weights=[0.99], flip_y=0)

# summarize class distribution
print(Counter(y))

# define undersample strategy
undersample = RandomUnderSampler(sampling_strategy='majority')

# fit and apply the transform
X_under, y_under = undersample.fit_resample(X, y)

# summarize class distribution
print(Counter(y_under))


# We can define an example of fitting a decision tree on an imbalanced 
# classification dataset with the undersampling transform applied to the 
# training dataset on each split of a repeated 10-fold cross-validation

# example of evaluating a decision tree with random undersampling
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler


# define dataset
X, y = make_classification(n_samples=10000, weights=[0.99], flip_y=0)

# define pipeline
steps = [('under', RandomUnderSampler()), ('model', DecisionTreeClassifier())] 
pipeline = Pipeline(steps=steps)

# evaluate pipeline
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(pipeline, X, y, scoring='f1_micro', cv=cv, n_jobs=-1) 
score = mean(scores)
print('F-measure: %.3f' % score)










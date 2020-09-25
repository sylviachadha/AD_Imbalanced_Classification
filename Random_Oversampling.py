#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 19:14:27 2020

@author: sylviachadha
"""

# Data Sampling to handle class imbalance (by modifying training datset to
# balance the class distribution)

# Random Oversampling

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler

# define dataset
X, y = make_classification(n_samples=10000, weights=[0.99], flip_y=0)

# summarize class distribution
print(Counter(y))

# define oversampling strategy
oversample = RandomOverSampler(sampling_strategy='minority')

# fit and apply the transform
X_over, y_over = oversample.fit_resample(X, y)

# summarize class distribution
print(Counter(y_over))

#The model is evaluated using repeated 10-fold cross-validation with three
#repeats, and the oversampling is performed on the training dataset within 
# each fold separately,

# Evaluating a decision tree on an imbalanced dataset with a 1:100 class 
# distribution
# Template to test oversampling with ur dataset and learning algorithm

from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler


# define dataset
X, y = make_classification(n_samples=10000, weights=[0.99], flip_y=0)


# define pipeline
steps = [('over', RandomOverSampler()), ('model', DecisionTreeClassifier())] 
pipeline = Pipeline(steps=steps)


# evaluate pipeline
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(pipeline, X, y, scoring='f1_micro', cv=cv, n_jobs=-1) 
score = mean(scores)
print('F-measure: %.3f' % score)

















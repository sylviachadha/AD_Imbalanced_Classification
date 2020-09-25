#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 18:37:58 2020

@author: sylviachadha
"""


# Grid Search Probability Calibration With KNN

# evaluate knn with uncalibrated probabilities for imbalanced classification
# Default neighbourhood size of 5

from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)

# define model
model = KNeighborsClassifier()

# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)

 # summarize performance
print('Mean ROC AUC: %.3f' % mean(scores))


# Approach 2 Grid Searching Probability Caliberation with KNN Model

# grid search probability calibration with knn for imbalance classification
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV


# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)


# define model
model = KNeighborsClassifier()

# wrap the model
calibrated = CalibratedClassifierCV(model)

# define grid
param_grid = dict(cv=[2,3,4], method=['sigmoid','isotonic'])

# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# define grid search
grid = GridSearchCV(estimator=calibrated, param_grid=param_grid, n_jobs=-1, cv=cv,
scoring='roc_auc')

# execute the grid search
grid_result = grid.fit(X, y)


# report the best configuration
print('Best: %f using %s' % (grid_result.best_score_, grid_result.best_params_)) 

# report all configurations
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('%f (%f) with: %r' % (mean, stdev, param))













#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 20:03:21 2020

@author: sylviachadha
"""

# Compounding effect of dataset size

# CREATING DATASETS BY SPECIFYING PERCENTAGE OF EACH CLASS

# Synthetic imbalanced binary classification dataset.

# vary the dataset size for a 1:100 imbalanced dataset i.e. the ratio
# of majority : minority class is 99%:1% but we want to see how although
# same ratio but dataset size will affect imbalanced classification
# more dataset size more helpful as minority samples increase

# Plotting 1:100 class distribution using four different sizes


from collections import Counter
from sklearn.datasets import make_classification
from matplotlib import pyplot
from numpy import where

# dataset sizes
sizes = [100, 1000, 10000, 100000]

# create and plot a dataset with each size
for i in range(len(sizes)):
    # determine the dataset size
    n = sizes[i]
    # create the dataset
    X, y = make_classification(n_samples=n, n_features=2, n_redundant=0,
      n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
    # summarize class distribution
    counter = Counter(y)
    print('Size=%d, Ratio=%s' % (n, counter)) 
    # define subplot
    pyplot.subplot(2, 2, 1+i) 
    pyplot.title('n=%d' % n) 
    pyplot.xticks([])
    pyplot.yticks([])
    # scatter plot of examples by class label 
    for label, _ in counter.items():
        row_ix = where(y == label)[0]
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
    pyplot.legend()
    
# show the figure
pyplot.show()
    
    
    
    
    
    
    
    
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 19:48:45 2017

@author: cxwang
"""

import numpy as np
from imblearn.metrics import classification_report_imbalanced
from imblearn import over_sampling as os
from imblearn import pipeline as pl
from imblearn.ensemble import EasyEnsemble # easy ensemnble
from imblearn.under_sampling import ClusterCentroids,EditedNearestNeighbours,NearMiss # down sample
from imblearn.over_sampling import ADASYN,SMOTE # up sample
from imblearn.combine import SMOTEENN # combine up and down sample
from collections import Counter
import config.config as config
easy_ensemble_num = 5

def easy_ensemble(train_set,train_label):
    ee = EasyEnsemble(ratio='auto',
                 return_indices=True,
                 random_state=None,
                 replacement=False,
                 n_subsets=easy_ensemble_num)
    X_resampled, y_resampled,idx_resampled = ee.fit_sample(train_set,train_label)
    return X_resampled, y_resampled
    
def cluster_centroids(X,y):# cc
    print('Original dataset shape {}'.format(Counter(y)))
    cc = ClusterCentroids(random_state=42)
    X_res, y_res = cc.fit_sample(X, y)
    print('Resampled dataset shape {}'.format(Counter(y_res)))
    return X_res,y_res
    
def edit_nearest_neribours(X,y):#enn
    print('Original dataset shape {}'.format(Counter(y)))
    enn = EditedNearestNeighbours(random_state=42)
    X_res, y_res = enn.fit_sample(X, y) 
    print('Resampled dataset shape {}'.format(Counter(y_res)))
    return X_res,y_res
    
def near_miss(X,y):#nm
    print('Original dataset shape {}'.format(Counter(y)))
    nm =NearMiss(random_state=42)
    X_res, y_res = nm.fit_sample(X, y) 
    print('Resampled dataset shape {}'.format(Counter(y_res)))
    return X_res,y_res
    
def adasyn(X,y):
    print('Original dataset shape {}'.format(Counter(y)))
    ada = ADASYN(random_state=42)
    X_res, y_res = ada.fit_sample(X, y)
    print('Resampled dataset shape {}'.format(Counter(y_res)))
    return X_res,y_res
    
def smote(X,y):
    print('Original dataset shape {}'.format(Counter(y)))
    smt = SMOTE(random_state=42)
    X_res, y_res = smt.fit_sample(X, y)
    print('Resampled dataset shape {}'.format(Counter(y_res)))
    return X_res,y_res
    
def smoteenn(X,y):
    print('Original dataset shape {}'.format(Counter(y)))
    sme = SMOTEENN(random_state=42)
    X_res, y_res = sme.fit_sample(X, y)
    print('Resampled dataset shape {}'.format(Counter(y_res)))
    return X_res,y_res

    

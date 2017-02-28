# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 14:44:28 2017

@author: cxwang
"""
import numpy as np
from collections import Counter
from imblearn.metrics import classification_report_imbalanced
from sklearn.datasets import make_multilabel_classification as make_ml_clf
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC
from imblearn.ensemble import EasyEnsemble
from sklearn.cross_validation import train_test_split

RANDOM_STATE = 42
RANDOM_SEED = np.random.randint(2 ** 10)
# Generate a dataset
X, y = make_ml_clf(n_classes=5,n_samples=2000, n_features=5,\
                                     n_labels=1 ,allow_unlabeled=True,\
                                     return_distributions=False,\
                                    random_state=RANDOM_SEED)
"""                                    
X, y = make_classification(n_classes=5, class_sep=5,\
    weights=[0.1,0.3,0.1,0.05,0.45], n_informative=3, n_redundant=1, flip_y=0,\
    n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
"""
print('Original dataset shape {}'.format(Counter([np.argmax(l) for l in y])))

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2)

y_train = [np.argmax(yt) for yt in y_train]
y_test = [np.argmax(yt) for yt in y_test]

ee = EasyEnsemble(ratio='auto',
                 return_indices=True,
                 random_state=None,
                 replacement=False,
                 n_subsets=6)
                
X_resampled, y_resampled,idx_resampled = ee.fit_sample(X_train,y_train)
p_list = []
for idx,x in enumerate(X_resampled):
    print('Resampled dataset shape {}'.format(Counter(y_resampled[idx])))
    svc = LinearSVC(random_state=RANDOM_STATE).fit(X_resampled[idx],y_resampled[idx])
    p = svc.predict(X_test)
    print p.shape
    p_list.append(p)
p_list = np.array(p_list)
from scipy.stats import mode
p_final = mode(p_list)[0][0]
accu = np.sum(p_final==y_test)/float(X_test.shape[0])
print(classification_report_imbalanced(y_test, p_final))



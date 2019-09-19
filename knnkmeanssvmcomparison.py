import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

data = load_digits()
xtr,xts,ytr,yts = train_test_split(
    data['data'],
    data['target'],
    test_size = .1
)

modelKMeans = KMeans()
modelKNN = KNeighborsClassifier()
modelSVM = SVC(gamma = 'auto')

modelKNN.fit(xtr,ytr)
modelSVM.fit(xtr,ytr)

# 1. manual scoring
# print(modelKNN.score(xts,yts))
# print(modelSVM.score(xts,yts))
# 4. k-fold cross validation
print(cross_val_score(
    KNeighborsClassifier(n_neighbors=43), xtr, ytr,
    cv=10
))
print(cross_val_score(
    SVC(gamma='auto'), xtr, ytr,
    cv=10
))
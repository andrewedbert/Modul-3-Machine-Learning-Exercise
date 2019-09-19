import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn import tree

dataOliv = fetch_olivetti_faces()
# print(dir(dataOliv))
# print(dataOliv.data[0].shape)
# print(dataOliv.images[0].shape)
# print(len(dataOliv.data))

# plt.subplot(121)
# plt.imshow(dataOliv.images[0], cmap='gray')
# plt.subplot(122)
# plt.imshow(dataOliv.data[0].reshape(64,64), cmap='gray')
# plt.show()

xtrain,xtest,ytrain,ytest=train_test_split(
    dataOliv['data'],
    dataOliv['target'],
    test_size=.05
)

# print(len(xtrain))
# print(len(xtest))

model=tree.DecisionTreeClassifier()
model.fit(xtrain,ytrain)

# print(model.score(xtrain,ytrain))
# print(model.predict([xtest[0]])[0])
# print(ytest[0])

plt.imshow(xtest[0].reshape(64,64),cmap='gray')
plt.show()
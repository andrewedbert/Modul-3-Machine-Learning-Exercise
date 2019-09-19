import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

x = np.arange(10)
y = np.array([0,0,0,0,0,1,1,1,1,1])

# Logistic Regression => y = 1/(1 + e^(-mx+c))
model = LogisticRegression(solver='lbfgs')
model.fit(x.reshape(-1,1),y)
yBest = model.predict(x.reshape(-1,1))
# print(model.coef_)
# print(model.intercept_)
# print(model.score(x.reshape(-1,1),y))
# y = 1/(1 + e^(-mx+c))
# y = 1/(1 + e^(-z))

def plot(z):
    return 1/(1+np.exp(-z))
grs = plot((model.coef_ * x.reshape(-1,1)) + model.intercept_)
plt.scatter(x,y)
plt.plot(x,grs)
plt.show()
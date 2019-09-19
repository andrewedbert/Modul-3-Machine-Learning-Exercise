import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

data = np.random.RandomState(1)
# print(data.rand(10))

x = 5 * data.rand(50)
y = 2 * x + data.randn(50)
# print(x)
# print(y)
# plt.scatter(x,y)
# plt.show()

# Simple Linear Regression
model = LinearRegression()
model.fit(x.reshape(-1,1),y)
# print(model.score(x.reshape(-1,1),y))
yBest = model.predict(x.reshape(-1,1))
# print(yBest)

# Polynomial Linear Regression
model2 = make_pipeline(
    PolynomialFeatures(7),
    LinearRegression()
)
model2.fit(x.reshape(-1,1),y)
yBest2 = model2.predict(x.reshape(-1,1))

# Lasso Regression (L1 regularization) + absolute value B
modelL = make_pipeline(
    PolynomialFeatures(7),
    Lasso(alpha=1e-15, normalize=True, max_iter=100000)
)
modelL.fit(x.reshape(-1,1),y)
yBestL = modelL.predict(x.reshape(-1,1))

# Ridge Regression L2 + k.lambda
modelR = make_pipeline(
    PolynomialFeatures(7),
    Ridge(alpha=1e-15, normalize=True)
)
modelR.fit(x.reshape(-1,1),y)
yBestR = modelR.predict(x.reshape(-1,1))

# Plotting
plt.figure('Regression',figsize=(10,10))

plt.subplot(221)
plt.title('Simple Linear Regression')
plt.scatter(x,y, color='y')
plt.plot(np.sort(x), np.sort(yBest), color='b')

plt.subplot(222)
plt.title('Polynomial Linear Regression')
plt.scatter(x,y, color='y')
plt.plot(np.sort(x), np.sort(yBest2), color='r')

plt.subplot(223)
plt.title('Lasso Regression')
plt.scatter(x,y, color='y')
plt.plot(np.sort(x), np.sort(yBestL), color='purple')

plt.subplot(224)
plt.title('Ridge Regression')
plt.scatter(x,y, color='y')
plt.plot(np.sort(x), np.sort(yBestR), color='g')

plt.show()
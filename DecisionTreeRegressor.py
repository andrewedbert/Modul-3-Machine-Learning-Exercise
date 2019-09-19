import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

x = np.sort(np.random.randn(1000))
y = np.sin(x)

modelLR = LinearRegression()
modelLR.fit(x.reshape(-1,1),y)

modelDTR = DecisionTreeRegressor()
modelDTR.fit(x.reshape(-1,1),y)

modelDTR5 = DecisionTreeRegressor(max_depth=5)
modelDTR5.fit(x.reshape(-1,1),y)

yLR = modelLR.predict(x.reshape(-1,1))
yDTR = modelDTR.predict(x.reshape(-1,1))
yDTR5 = modelDTR5.predict(x.reshape(-1,1))

plt.plot(
    x,y,'ro',
    x,yLR,'g-',
    x,yDTR,'b-',
    x,yDTR5,'y-'
)
plt.show()
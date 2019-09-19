import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

data = {
    'usia':np.array([20,21,22,23,24,25,26,27,28,29,30]),
    'beliAsuransi':np.array([0,0,0,0,0,1,1,1,1,1,1])
}

df = pd.DataFrame(data)
# print(df.head())
# plt.scatter(df['usia'],df['beliAsuransi'])
# plt.show()

model = LogisticRegression(solver='lbfgs')
model.fit(df[['usia']],df['beliAsuransi'])

print(model.predict([[35]]))


# def plot(z):
#     return 1/(1+np.exp(-z))
# grs = plot((model.coef_ * df[['usia']]) + model.intercept_)
# plt.scatter(df['usia'],df['beliAsuransi'])
# plt.plot(df['usia'],grs)
# plt.show()
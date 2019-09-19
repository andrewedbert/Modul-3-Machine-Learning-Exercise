import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

data = {
    'luas':[2500,3000,3200,3600,4000],
    'kamar':[2,3,3,2,4],
    'usia':[10,15,20,18,8],
    'harga':[500,550,620,600,720]
}
df = pd.DataFrame(data)
print(df['luas'])

# multivariate linear regression
# y = m1x1 + m2x2 + m3x3 + c

# model = linear_model.LinearRegression()
# model.fit(df[['luas','kamar','usia']],df['harga'])

# print(model.coef_)
# print(model.intercept_)
# luas, kamar, usia
# print(model.predict([[ 1000, 5, 1]]))

# correlation
# corr = df.corr()
# # print(corr)

# import seaborn as sb
# sb.heatmap(corr)
# plt.show()
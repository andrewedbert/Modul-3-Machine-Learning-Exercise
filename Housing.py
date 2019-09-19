import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_csv('housing.csv')
df = df.fillna(method='ffill')
# print(df)
# corr = df.corr()
# print(corr)
# sb.heatmap(corr)
# plt.xticks(rotation=45)
# plt.show()

# y = m1x1 + m2x2 + m3x3 + m4x4 + m5x5 + m6x6 + m7x7 + m8x8 + m9x9 + c

model = linear_model.LinearRegression()
model.fit(df[['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income']],df['median_house_value'])

# print(model.coef_)
# print(model.intercept_)

plt.plot(
    df['total_rooms'],df['total_bedrooms'],
    df['longitude'],df['latitude']
)
plt.grid(True)
plt.show()
import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_excel('indo_12_1.xls',skiprows=3, skipfooter=2, na_values=['-'])
df.rename(columns={'Unnamed: 0':'Provinsi'},inplace=True)
# print(df[df['Provinsi']=='INDONESIA'])
df = df.set_index('Provinsi')
df = df.interpolate()
df = df.transpose()
# print(df['Kepulauan Riau'])

x = list(df.index)
# print(x)

df1 = pd.DataFrame(x)
# df1.rename(columns={0:'Tahun'})
# print(df1)

model = linear_model.LinearRegression()
model.fit(df1,df['Kepulauan Riau'])
df2 = df1.append({0:2050},ignore_index=True)
# print(df2)
# print(model.coef_)
# print(model.intercept_)
# print(model.predict([[2050]]))
plt.plot(
    df1, df['Kepulauan Riau'], 'r-',
    df2, model.predict(df2), 'g--'
)
plt.grid(True)
plt.show()
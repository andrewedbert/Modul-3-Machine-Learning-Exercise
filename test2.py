import pandas as pd
import numpy as np

df = pd.read_excel('indo_12_1.xls', skiprows=3, skipfooter=3, na_values=['-'])
df.rename(columns={'Unnamed: 0':'Provinsi'},inplace=True)
df = df.replace('-', np.NaN)
# df.isnull().sum()
# df = df.fillna({
#     1971:'x',1980:'y',1990:'z',1995:'b',2000:'c'
# })
# df = df.fillna(method='ffill',axis='index')
# 0:'index', 1:'columns'
# df = df.interpolate()
# df = df.dropna() 
# how='all'=> buang baris yang tidak ada value sama sekali, thresh=int => buang sesuai jumlah yang kosong, subset=[nama kolom] => buang baris yang kosong berdasarkan kolom
# df[2010].max()
print(df[['Provinsi',2010]][df[2010]==df[2010].max()])

# df[df['Provinsi']=='Jawa Barat']

# jabar = df[df['Provinsi']=='Jawa Barat']
# thjabar = list(jabar.columns)[1:]
# jmljabar = list(jabar.values[0])[1:]

# jatim = df[df['Provinsi']=='Jawa Timur']
# thjatim = list(jatim.columns)[1:]
# jmljatim = list(jatim.values[0])[1:]

# # %matplotlib inline
# import matplotlib.pyplot as plt
# plt.style.use('seaborn')
# plt.plot(
#     thjabar, jmljabar, 'r-', thjatim, jmljatim, 'b-'
# )
# plt.grid(True)
# plt.legend(['Jawa Barat','Jawa Timur'])
# plt.show()

# df1 = pd.read_excel('indo_12_1.xls', skiprows=3, skipfooter=3, na_values=['-'])
# df2 = pd.read_excel('indo_12_1.xls', skiprows=3, skipfooter=2, na_values=['-'])

# x = df1[[1971]][df1[1971]==df1[1971].min()]
# y = df1[[2000]][df1[2000]==df1[2000].max()]
# z = df2[[2010]][df2[2010]==df2[2010].max()]

# thx = list(df1[df1[1971]==df1[1971].min()].columns)[1:]
# jmlx = list(df1[df1[1971]==df1[1971].min()].values[0])[1:]

# thy = list(df1[df1[2000]==df1[2000].max()].columns)[1:]
# jmly = list(df1[df1[2000]==df1[2000].max()].values[0])[1:]

# thz = list(df2[df2[2010]==df2[2010].max()].columns)[1:]
# jmlz = list(df2[df2[2010]==df2[2010].max()].values[0])[1:]


# plt.style.use('seaborn')
# plt.plot(thx,jmlx,'r-',thy,jmly,'b-',thz,jmlz,'g-')
# plt.grid(True)
# plt.legend([list(df1[df1[1971]==df1[1971].min()].values[0])[0],list(df1[df1[2000]==df1[2000].max()].values[0])[0],list(df2[df2[2010]==df2[2010].max()].values[0])[0]])
# plt.show()
import requests
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

host = 'http://api.openweathermap.org/data/2.5/forecast?q='
kota = 'Bogor'
appid = '&APPID=a3515f2287ffb8f930d39233c31465fa'
url = host + kota + appid

data = requests.get(url)
data = data.json()
listCuaca = data['list']
# print(listCuaca)

list2 = []
for i in listCuaca:
    waktu = i['dt_txt']
    suhu = i['main']['temp']-273
    tekanan = i['main']['pressure']
    lembab = i['main']['humidity']
    angin = i['wind']['speed']
    data = {
        'waktu':waktu,
        'suhu':suhu,
        'tekanan':tekanan,
        'kelembapan':lembab,
        'angin':angin
    }
    list2.append(data)

df = pd.DataFrame(list2)
df['time'] = pd.to_datetime(df['waktu'])
df = df.drop('waktu',axis=1)
# print(type(df['time'].iloc[0]))
# df = df.set_index('time')
print(df)

# model_suhu = DecisionTreeRegressor()
# model_suhu.fit(np.array(df.index).reshape(-1,1),df['suhu'])
# ySP = model_suhu.predict([np.arange(70)])
# dates = np.array(df['time'])
# dates = dates.astype(np.float64)
# modelLR = LinearRegression()
# modelLR.fit(df.index.reshape(-1,1),df['suhu'])
# yLR = modelLR.predict(df.index.reshape(-1,1))
# plt.plot(
#     df.index,df['suhu'],'ro',
#     np.arange(70),ySP,'b-'
#     # df.index.reshape(-1,1),yLR,'g-',
# )
# plt.show()
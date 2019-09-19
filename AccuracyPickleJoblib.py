import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import joblib

dataCH = fetch_california_housing()
# print(dir(dataCH))
# print(dataCH['data']['shape'])
# print(dataCH['feature_names'])
# print(dataCH.target[0])

dfCH = pd.DataFrame(
    dataCH['data'],
    columns = dataCH['feature_names']
)
dfCH['Price'] = dataCH['target']
# dfCH

x = dfCH[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']]
y = dfCH['Price']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.1)

# dfCH.columns
# print(x_test.iloc[0])

# Linear Regression Model
model = LinearRegression()
model.fit(x_train, y_train)

# prediksi
# print(x_test.iloc[0])
# print(model.predict([x_test.iloc[0]]))
# print(model.predict([[
#     2.890600, 16.000000, 5.413043, 1.034161, 652.000000, 2.024845, 32.850000, -116.890000
# ]]))

# score accuracy
# print(model.score(x_train,y_train))
# print(model.score(x_test,y_test))

# save model ML: pickle
# with open('modelPickle.pkl','wb') as modelku:
#     pickle.dump(model, modelku)

# save model ML: joblib
# joblib.dump(model,'modelJoblib.joblib')

# load model ML: pickle
# with open('modelPickle.pkl', 'rb') as modelku:
#     modelLoad = pickle.load(modelku)

# print(model.predict([[
#     2.890600, 16.000000, 5.413043, 1.034161, 652.000000, 2.024845, 32.850000, -116.890000
# ]]))

# load model ML: joblib
modelLoad = joblib.load('modelJoblib.joblib')

print(model.predict([[
    2.890600, 16.000000, 5.413043, 1.034161, 652.000000, 2.024845, 32.850000, -116.890000
]]))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

data = {
    'luas':np.arange(100,300,20),
    'harga': [500,665,720,795,885,1200,1500,1600,1775,2000]
}

df = pd.DataFrame(data)
# print(df)

# plotting
# plt.scatter(df['luas'],df['harga'])
# plt.show()

# linear regression method / Model ML
model = linear_model.LinearRegression()

# training model dengan data yang kita punya
# model.fit(dataindependent, datadependent)
model.fit(df[['luas']],df['harga'])

m = model.coef_
c = model.intercept_
# print(m[0])
# print(c)

# prediksi
# print(model.predict([[ 100 ]]))
# print(model.predict([[ 3000 ]]))
# print(model.predict(df[['luas']]))


# plot data asli + best fit line
plt.plot(
    df['luas'], df['harga'], 'ro',
    df['luas'], model.predict(df[['luas']]), 'g-'
)
plt.grid(True)
plt.xlabel('Luas (m2)')
plt.ylabel('Harga (Rp)')
plt.legend(['Data','Best Fit Line'])
plt.show()


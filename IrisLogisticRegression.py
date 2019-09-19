import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageOps
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

dataIris = load_iris()
# print(dir(dataIris))

dfIris = pd.DataFrame(
    dataIris['data'],
    columns = ['SL','SW','PL','PW']
)
# print(dfIris.head())

dfIris['target']=dataIris['target']
dfIris['jenis']=dfIris['target'].apply(
    lambda l: dataIris['target_names'][l]
)
# print(dfIris[dfIris['target']==2].head())
print(dfIris)

a, b, c, d = train_test_split(
    dfIris[['SL','SW','PL','PW']],
    dfIris['target'],
    test_size = .05
)
# print('training', len(a))
# print('training', len(b))

model = LogisticRegression(
    solver='lbfgs',
    multi_class='auto',
    max_iter=1000
)
model.fit(a, c)
# print(model.coef_)
# print(model.intercept_)

prediksi = model.predict(b)
# print(prediksi)
# print(d.values)

# visualisation
# plot data asli SL vs SW
plt.subplot(221)
plt.plot(
    dfIris[dfIris['target']==0]['SL'],
    dfIris[dfIris['target']==0]['SW'],
    'ro',
    dfIris[dfIris['target']==1]['SL'],
    dfIris[dfIris['target']==1]['SW'],
    'go',
    dfIris[dfIris['target']==2]['SL'],
    dfIris[dfIris['target']==2]['SW'],
    'bo'
)
pred = model.predict(dfIris[['SL','SW','PL','PW']])
# print(pred)
dfIris['prediksi'] = pred
# print(dfIris.head())
# plot data prediksi SL vs SW
plt.subplot(222)
plt.plot(
    dfIris[dfIris['prediksi']==0]['SL'],
    dfIris[dfIris['prediksi']==0]['SW'],
    'ro',
    dfIris[dfIris['prediksi']==1]['SL'],
    dfIris[dfIris['prediksi']==1]['SW'],
    'go',
    dfIris[dfIris['prediksi']==2]['SL'],
    dfIris[dfIris['prediksi']==2]['SW'],
    'bo'
)

# plot data asli PL vs PW
plt.subplot(223)
plt.plot(
    dfIris[dfIris['target']==0]['PL'],
    dfIris[dfIris['target']==0]['PW'],
    'ro',
    dfIris[dfIris['target']==1]['PL'],
    dfIris[dfIris['target']==1]['PW'],
    'go',
    dfIris[dfIris['target']==2]['PL'],
    dfIris[dfIris['target']==2]['PW'],
    'bo'
)

# plot data prediksi PL vs PW
plt.subplot(224)
plt.plot(
    dfIris[dfIris['prediksi']==0]['PL'],
    dfIris[dfIris['prediksi']==0]['PW'],
    'ro',
    dfIris[dfIris['prediksi']==1]['PL'],
    dfIris[dfIris['prediksi']==1]['PW'],
    'go',
    dfIris[dfIris['prediksi']==2]['PL'],
    dfIris[dfIris['prediksi']==2]['PW'],
    'bo'
)
# plt.show()
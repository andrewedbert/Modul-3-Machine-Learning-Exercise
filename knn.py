import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# data
dataIris = load_iris()
df = pd.DataFrame(
    dataIris['data'],
    columns = ['SL', 'SW', 'PL', 'PW']
)
df['target'] = dataIris['target']
df['spesies'] = df['target'].apply(
    lambda row: dataIris['target_names'][row]
)
# print(df.head())

# split
xtr,xts,ytr,yts = train_test_split(
    df[['SL', 'SW', 'PL', 'PW']],
    df['target'],
    test_size = .05
)

# print(len(xtr))
# print(len(xts))


# k di KNN ? 
# => sqrt(total data point)
# => ganjil
def nilai_k():
    k = round((len(xts)+len(xtr)) ** .5)
    if (k % 2 == 0):
        return k + 1
    else:
        return k
# print(nilai_k())

# KNN
model = KNeighborsClassifier(
    n_neighbors = nilai_k()
)
model.fit(xtr,ytr)

# print(xts.iloc[0])
# print(model.predict([xts.iloc[0]]))
# print(yts.iloc[0])
# print(model.predict_proba([xts.iloc[0]]))

df['pred'] = model.predict(df[['SL', 'SW', 'PL', 'PW']])
# print(df.head())

plt.subplot(221)
plt.plot(
    df[df['target'] == 0]['PL'],
    df[df['target'] == 0]['PW'],
    'ro'
)
plt.plot(
    df[df['target'] == 1]['PL'],
    df[df['target'] == 1]['PW'],
    'go'
)
plt.plot(
    df[df['target'] == 2]['PL'],
    df[df['target'] == 2]['PW'],
    'yo'
)
plt.subplot(222)
plt.plot(
    df[df['pred'] == 0]['PL'],
    df[df['pred'] == 0]['PW'],
    'ro'
)
plt.plot(
    df[df['pred'] == 1]['PL'],
    df[df['pred'] == 1]['PW'],
    'go'
)
plt.plot(
    df[df['pred'] == 2]['PL'],
    df[df['pred'] == 2]['PW'],
    'yo'
)
plt.subplot(223)
plt.plot(
    df[df['target'] == 0]['SL'],
    df[df['target'] == 0]['SW'],
    'ro'
)
plt.plot(
    df[df['target'] == 1]['SL'],
    df[df['target'] == 1]['SW'],
    'go'
)
plt.plot(
    df[df['target'] == 2]['SL'],
    df[df['target'] == 2]['SW'],
    'yo'
)
plt.subplot(224)
plt.plot(
    df[df['pred'] == 0]['SL'],
    df[df['pred'] == 0]['SW'],
    'ro'
)
plt.plot(
    df[df['pred'] == 1]['SL'],
    df[df['pred'] == 1]['SW'],
    'go'
)
plt.plot(
    df[df['pred'] == 2]['SL'],
    df[df['pred'] == 2]['SW'],
    'yo'
)
plt.show()
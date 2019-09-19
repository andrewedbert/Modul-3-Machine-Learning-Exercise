import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

dataIris = load_iris()
df = pd.DataFrame(
    dataIris['data'],
    columns = ['SL', 'SW', 'PL', 'PW']
)

df['target'] = dataIris['target']
df['spesies'] = df['target'].apply(
    lambda row: dataIris['target_names'][row]
)

model = KMeans(
    n_clusters = len(dataIris['target_names'])
)
model.fit(df[['PL','PW']], df['target'])
df['predP'] = model.predict(df[['PL','PW']])
# print(df.head())

centroidP = model.cluster_centers_
# print(centroidP)

# plt.subplot(221)
# plt.plot(
#     df[df['target'] == 0]['PL'],
#     df[df['target'] == 0]['PW'],
#     'ro'
# )
# plt.plot(
#     df[df['target'] == 1]['PL'],
#     df[df['target'] == 1]['PW'],
#     'go'
# )
# plt.plot(
#     df[df['target'] == 2]['PL'],
#     df[df['target'] == 2]['PW'],
#     'yo'
# )
# plt.scatter(
#     centroidP[:,0],
#     centroidP[:,1],
#     marker = '*',
#     color = 'pink',
#     s = 250
# )
plt.grid(True)
# plt.show()

# k means => sse + elbow methods

sse = []
for i in range(1,11):
    model = KMeans(n_clusters = i)
    model.fit(df[['PL','PW']])
    sse.append(model.inertia_)
# print(sse)

plt.plot(range(1,11), sse)
plt.show()
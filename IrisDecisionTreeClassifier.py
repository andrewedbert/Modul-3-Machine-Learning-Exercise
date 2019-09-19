import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.datasets import load_iris

dataIris = load_iris()
# print(dataIris['data'])
dfIris = pd.DataFrame(
    dataIris['data'],
    columns = ['SL','SW','PL','PW']
)
dfIris['target']=dataIris['target']
dfIris['jenis']=dfIris['target'].apply(
    lambda l: dataIris['target_names'][l]
)

letarget = LabelEncoder()
dfIris['letarget'] = letarget.fit_transform(dfIris['target'])
lejenis = LabelEncoder()
dfIris['lejenis'] = lejenis.fit_transform(dfIris['jenis'])
# print(dfIris)

model = tree.DecisionTreeClassifier()
model.fit(dfIris[['SL','SW','PL','PW']], dfIris['lejenis'])

# print(model.predict([[9,3,7,3]]))
# print(dfIris['jenis'][dfIris['lejenis']==0])

# tree.export_graphviz(
#     model.fit(dfIris[['SL','SW','PL','PW']], dfIris['lejenis']),
#     out_file='iris.dot',
#     feature_names=['SL','SW','PL','PW'],
#     class_names=['setosa','versicolor','virginica']
# )
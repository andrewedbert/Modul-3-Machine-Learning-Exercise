import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.datasets import load_digits

dataDg = load_digits()
# print(dir(dataDg))
# print(len(dataDg['target']))

dfDg = pd.DataFrame(
    dataDg['data'],
    columns=np.arange(len(dataDg['data'][0]))
)
dfDg['target']=dataDg['target']
# print(dfDg)

letarget = LabelEncoder()
dfDg['letarget'] = letarget.fit_transform(dfDg['target'])

x = dfDg.drop(['target','letarget'],axis=1)
y = dfDg['letarget']
model = tree.DecisionTreeClassifier()
model.fit(x,y)

# print(x.iloc[0])
# print(y.iloc[0])
# tree.export_graphviz(
#     model.fit(x,y),
#     out_file='digits.dot',
#     feature_names=x.iloc[0],
#     class_names=y.iloc[0]
# )
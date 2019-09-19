import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

data = load_iris()
# print(dir(data))
df = pd.DataFrame(
    data['data'],
    columns = ['SL','SW','PL','PW']
)
df['target'] = data['target']
df['species'] = df['target'].apply(
    lambda x: data['target_names'][x]
)
# print(df.head())

# splitting = 5% test
xtr,xts,ytr,yts = train_test_split(
    df[['SL','SW','PL','PW']],
    df['target'],
    test_size=.05
)
# print(len(xtr))
# print(len(ytr))

# logistic reg
model = LogisticRegression(solver='lbfgs')
model.fit(xtr,ytr)
# print(xts)
# print(yts)
# print(model.predict(xts))

print(model.predict([[9,9,9,9]]))
print(model.predict_proba([[9,9,9,9]]))
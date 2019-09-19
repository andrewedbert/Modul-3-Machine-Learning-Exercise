import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = load_digits()
# print(dir(data))
# print(data['data'])
df = pd.DataFrame(
    data['data'],
    columns=np.arange(len(data['data'][0]))
)
df['target']=data['target']
xtr,xts,ytr,yts = train_test_split(
    df[np.arange(len(data['data'][0]))],
    df['target'],
    test_size=.05
)
model = LogisticRegression(solver='lbfgs')
model.fit(xtr,ytr)
# print(xts.iloc[0])
# print(model.predict(xts)[0])
print(model.predict_proba(xts))
# plt.imshow(data['images'][model.predict(xts)[0]])
# plt.show()
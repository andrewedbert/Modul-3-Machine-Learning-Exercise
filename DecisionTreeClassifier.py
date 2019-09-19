import numpy as np
import pandas as pd

df = pd.DataFrame([
    {'kantor':'Google','job':'GM','gaji>20':1},
    {'kantor':'Google','job':'Sales','gaji>20':0},
    {'kantor':'Google','job':'Staf IT','gaji>20':1},
    {'kantor':'Google','job':'Staf DS','gaji>20':0},
    {'kantor':'FB','job':'GM','gaji>20':0},
    {'kantor':'FB','job':'Sales','gaji>20':0},
    {'kantor':'FB','job':'Staf IT','gaji>20':0},
    {'kantor':'FB','job':'Staf DS','gaji>20':0},
    {'kantor':'Grab','job':'GM','gaji>20':1},
    {'kantor':'Grab','job':'Sales','gaji>20':1},
    {'kantor':'Grab','job':'Staf IT','gaji>20':1},
    {'kantor':'Grab','job':'Staf DS','gaji>20':1}
])
# print(df)
y = df['gaji>20']
x = df.drop('gaji>20',axis=1)
# print(x)

from sklearn.preprocessing import LabelEncoder
leKantor = LabelEncoder()
x['leKantor'] = leKantor.fit_transform(x['kantor'])
leJob = LabelEncoder()
x['leJob'] = leKantor.fit_transform(x['job'])
x = x.drop(['kantor','job'], axis=1)
# print(x)

from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(x, y)

# FB=0 Google=1 Grab=2
# GM=0 Sales=1 Staf DS=2 Staf IT=3
print(model.predict([[0,0]]))
print(model.predict([[1,1]]))
print(model.predict([[2,1]]))

# decision tree image
tree.export_graphviz(
    model.fit(x,y),
    out_file='x.dot',
    feature_names=['kantor','job'],
    class_names=['kurang20','lebih20']
)
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston

dataBoston = load_boston()
# print(dir(dataBoston))
# print(dataBoston['data'][0])
# print(dataBoston['feature_names'])
# print(dataBoston['target'])

df = pd.DataFrame(
    dataBoston['data'],
    columns=dataBoston['feature_names']
)
print(df)
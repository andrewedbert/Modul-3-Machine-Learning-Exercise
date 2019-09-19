import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

dataDg = load_digits()

y = str(input('Tanggal: '))

for i in range(len(y)):
    plt.subplot(1, len(y), i+1)
    plt.imshow(dataDg['images'][int(y[i])])
    plt.title('{}'.format(dataDg['target'][int(y[i])]))
plt.show()
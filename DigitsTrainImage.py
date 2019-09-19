import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageOps
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

dataDg = load_digits()
# print(dir(dataDg))
# print(dataDg['data'][0])
# print(dataDg['images'][0])
# print(dataDg['target'][0])

# splitting datasets
xtr, xts, ytr, yts = train_test_split(
    dataDg['data'],
    dataDg['target'],
    test_size = .1
)
# print(len(xtr))
# print(len(ytr))

# logistic regression
model = LogisticRegression(
    solver='lbfgs', 
    multi_class='auto',
    max_iter=10000
)
model.fit(xtr,ytr)

# visualization + prediction xts[0]
# print(xts[0])

# for i in range(10):
#     akurasi = round(model.score(xts, yts) * 100,2)
#     prediksi = model.predict(xts[i].reshape(1,-1))[0]
#     plt.subplot(2,5,i+1)
#     plt.imshow(xts[i].reshape(8,8))
#     plt.title(
#         f'P = {prediksi} | D = {yts[i]} | A = {akurasi} %'
#     )
# plt.show()

gbr = Image.open('2.jpg').convert('L')
gbr = gbr.resize((8,8))
gbr = PIL.ImageOps.invert(gbr)
gbrArr = np.array(gbr)
gbrArr2 = gbrArr.reshape(1,64)
# print(gbrArr)

# out = Image.fromarray(gbrArr,'L')
# out.show()
plt.imshow(gbrArr, cmap='gray')
prediksi = model.predict(gbrArr2.reshape(1,-1))
print(prediksi)
plt.show()
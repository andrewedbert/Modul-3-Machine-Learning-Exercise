from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

gbr = Image.open('messi.jpg').convert('L') # RGBA CMYK
# print(gbr)
arrGbr = np.array(gbr)
# print(arrGbr.shape)

# plt.imshow(arrGbr, cmap='gray')
# plt.show()

out = Image.fromarray(arrGbr, 'L')
out.save('messi2.jpg')
out.show()
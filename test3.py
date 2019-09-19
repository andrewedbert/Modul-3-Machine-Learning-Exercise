import numpy as np
# a, b = (3, 2)
# x = np.linspace(0, 1, a)
# y = np.linspace(0, 1, b)
# k, l = np.meshgrid(x, y)
# print(k)
# print(l)

x = np.array([[1,2,3],[4,5,6]])
y = np.array([[4,4,4],[2,2,2]])
print(x.ravel())
print(np.c_[x.ravel(),y.ravel()])
import numpy as np


arr1 = np.array([1, 2, 1, 2, 2, 3])
arr0 = np.array([1, 1, 2, 2, 3])
index = np.array(np.where(arr0 == 1))
print(arr1[index][0])
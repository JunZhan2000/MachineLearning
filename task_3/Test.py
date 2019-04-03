import numpy as np
import pandas as pd

data = np.array(range(10))
n = 10
# test_data = np.random.choice(data, n, replace=False)
test_data = np.array([1, 2, 3, 6, 7, 8])
print(data)
print(test_data)
print(np.setdiff1d(data, test_data))

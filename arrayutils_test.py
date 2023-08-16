import numpy as np
import pandas as pd
import time
import arrayutils as au


num_rows = 10000
num_cols = 10000


arr = np.arange(num_rows * num_cols, dtype="float64").reshape((num_rows, num_cols), order="F")
df = pd.DataFrame(arr)


t1 = time.time()
arr2 = au.rolling_mean(arr, 20, min_periods=1)

t2 = time.time()
df_mean = df.rolling(20, min_periods=1).mean()
t3 = time.time()

arr3 = df_mean.to_numpy()
print(t2 - t1)
print(t3 - t2)
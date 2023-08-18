import numpy as np
import pandas as pd
import time
import arrayutils as au
from numba_impl import rolling_mean_numba, rolling_mean_numba_parallel, rank_numba, rank_numba_parallel


def compare_rolling_mean_impl(array: np.ndarray, num_threads: int):
    df = pd.DataFrame(array)

    t1 = time.time()
    df_mean = df.rolling(20, min_periods=1).mean()

    t2 = time.time()
    arr2 = rolling_mean_numba(array, 20, 1)

    t3 = time.time()
    arr3 = rolling_mean_numba_parallel(array, 20, 1)

    t4 = time.time()
    arr4 = au.rolling_mean(array, 20, min_periods=1, num_threads=1)

    t5 = time.time()
    arr5 = au.rolling_mean(array, 20, min_periods=1, num_threads=num_threads)  # 0 means using all CPUs

    t6 = time.time()
    arr1 = df_mean.to_numpy()

    print(f"Rolling Mean - Pandas Rolling: {t2 - t1}")
    print(f"Rolling Mean - Numba w/o Parallelization: {t3 - t2}")
    print(f"Rolling Mean - Numba w/ Parallelization: {t4 - t3}")
    print(f"Rolling Mean - C++ w/o Parallelization: {t5 - t4}")
    print(f"Rolling Mean - C++ w/ Parallelization: {t6 - t5}")
    print()


def compare_rank_impl(array: np.ndarray, num_threads: int):
    df = pd.DataFrame(array)
    t1 = time.time()
    df_rank = df.rank(axis=1, pct=True)

    t2 = time.time()
    arr2 = rank_numba(array)

    t3 = time.time()
    arr3 = rank_numba_parallel(array)

    t4 = time.time()
    arr4 = au.rank(array, num_threads=1)

    t5 = time.time()
    arr5 = au.rank(array, num_threads=num_threads)

    t6 = time.time()
    arr1 = df_rank.to_numpy()

    print(f"Cross-Sectional Rank - Pandas Rank: {t2 - t1}")
    print(f"Cross-Sectional Rank - Numba w/o Parallelization: {t3 - t2}")
    print(f"Cross-Sectional Rank - Numba w/ Parallelization: {t4 - t3}")
    print(f"Cross-Sectional Rank - C++ w/o Parallelization: {t5 - t4}")
    print(f"Cross-Sectional Rank - C++ w/ Parallelization: {t6 - t5}")
    print()


if __name__ == "__main__":
    # np.arange(6).reshape((2, 3), order='C')
    #   ->  [[0 1 2]
    #        [3 4 5]]
    # (each row is in a contiguous memory block)

    # np.arange(6).reshape((2, 3), order='F')
    #   ->  [[0 2 4]
    #        [1 3 5]]
    # (each column is in a contiguous memory block)

    num_rows = 5000
    num_cols = 5000
    size = num_rows * num_cols
    arr_1d = np.random.random(size)
    arr_1d[np.random.choice(size, size // 5, False)] = np.nan  # set 1/5 of the elements to nan

    order = 'F'
    print(f"[Order = '{order}']")
    arr = arr_1d.reshape((num_rows, num_cols), order=order)
    compare_rolling_mean_impl(arr, 0)
    compare_rank_impl(arr, 0)

    order = 'C'
    print(f"[Order = '{order}']")
    arr = arr_1d.reshape((num_rows, num_cols), order=order)
    compare_rolling_mean_impl(arr, 0)
    compare_rank_impl(arr, 0)

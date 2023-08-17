import numpy as np
import pandas as pd
import numba as nb
import time
import arrayutils as au


def rolling_mean_python(array: np.ndarray, window: int, min_periods: int):
    # Used Kahan summation algorithm
    #   https://en.wikipedia.org/wiki/Kahan_summation_algorithm
    # See also pandas source code:
    #   https://github.com/pandas-dev/pandas/blob/main/pandas/_libs/window/aggregations.pyx
    assert array.ndim == 2
    out_array = np.empty_like(array)
    for i in nb.prange(array.shape[1]):
        s = 0.0
        c = 0.0
        cnt = 0

        for j in range(array.shape[0]):
            if j >= window:
                old_val = array[j - window, i]
                if np.isfinite(old_val):
                    y = old_val + c
                    t = s - y
                    z = t - s
                    c = z + y
                    s = t
                    cnt -= 1

            val = array[j, i]
            if np.isfinite(val):
                y = val - c
                t = s + y
                z = t - s
                c = z - y
                s = t
                cnt += 1

            if cnt >= min_periods:
                out_array[j, i] = s / cnt
            else:
                out_array[j, i] = np.nan

    return out_array


rolling_mean_numba_c = nb.njit(nb.float64[:, :](nb.float64[:, :], nb.int64, nb.int64),
                               parallel=False)(rolling_mean_python)
rolling_mean_numba_f = nb.njit(nb.float64[::1, :](nb.float64[::1, :], nb.int64, nb.int64),
                               parallel=False)(rolling_mean_python)


def rolling_mean_numba(array: np.ndarray, window: int, min_periods: int):
    if array.flags.fortran:
        return rolling_mean_numba_f(array, window, min_periods)
    return rolling_mean_numba_c(array, window, min_periods)


rolling_mean_numba_parallel_c = nb.njit(nb.float64[:, :](nb.float64[:, :], nb.int64, nb.int64),
                                        parallel=True)(rolling_mean_python)
rolling_mean_numba_parallel_f = nb.njit(nb.float64[::1, :](nb.float64[::1, :], nb.int64, nb.int64),
                                        parallel=True)(rolling_mean_python)


def rolling_mean_numba_parallel(array: np.ndarray, window: int, min_periods: int):
    if array.flags.fortran:
        return rolling_mean_numba_parallel_f(array, window, min_periods)
    return rolling_mean_numba_parallel_c(array, window, min_periods)


def rank_python(array: np.ndarray):
    assert array.ndim == 2
    out_array = np.empty_like(array)

    for i in nb.prange(array.shape[0]):
        row = array[i]
        args = np.argsort(row)
        valid_mask = np.isfinite(row)
        out_array[i, ~valid_mask] = np.nan
        valid_cnt = np.sum(valid_mask)
        j = 0
        while j < valid_cnt:
            j1 = j
            val = row[args[j]]
            while True:
                j += 1
                if j >= valid_cnt or row[args[j]] != val:
                    break
            out_val = (j + j1 + 1) / (2 * valid_cnt)
            for k in range(j1, j):
                out_array[i, args[k]] = out_val
    return out_array


rank_numba_c = nb.njit(nb.float64[:, :](nb.float64[:, :]), parallel=False)(rank_python)
rank_numba_f = nb.njit(nb.float64[::1, :](nb.float64[::1, :]), parallel=False)(rank_python)


def rank_numba(array: np.ndarray):
    if array.flags.fortran:
        return rank_numba_f(array)
    return rank_numba_c(array)


rank_numba_parallel_c = nb.njit(nb.float64[:, :](nb.float64[:, :]), parallel=True)(rank_python)
rank_numba_parallel_f = nb.njit(nb.float64[::1, :](nb.float64[::1, :]), parallel=True)(rank_python)


def rank_numba_parallel(array: np.ndarray):
    if array.flags.fortran:
        return rank_numba_parallel_f(array)
    return rank_numba_parallel_c(array)


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
    arr_1d[np.random.choice(size, size // 5, False)] = np.nan

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

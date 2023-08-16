import numpy as np
import pandas as pd
import numba as nb
import time
import arrayutils as au


def rolling_mean_numpy(array: np.ndarray, window: int):
    assert array.ndim == 2
    out_array = np.empty_like(array)
    out_array[:window - 1, :] = np.nan
    out_array[window - 1:, :] = np.nanmean(np.lib.stride_tricks.sliding_window_view(array, window, 0), axis=-1)
    return out_array


def rolling_mean_python(array: np.ndarray, window: int, min_periods: int):
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


rolling_mean_c = nb.njit(nb.float64[::, :](nb.float64[::, :], nb.int64, nb.int64),
                         parallel=False)(rolling_mean_python)
rolling_mean_f = nb.njit(nb.float64[::1, :](nb.float64[::1, :], nb.int64, nb.int64),
                         parallel=False)(rolling_mean_python)


def rolling_mean(array: np.ndarray, window: int, min_periods: int):
    if array.flags.fortran:
        return rolling_mean_f(array, window, min_periods)
    return rolling_mean_c(array, window, min_periods)


rolling_mean_c_parallel = nb.njit(nb.float64[::, :](nb.float64[::, :], nb.int64, nb.int64),
                                  parallel=True)(rolling_mean_python)
rolling_mean_f_parallel = nb.njit(nb.float64[::1, :](nb.float64[::1, :], nb.int64, nb.int64),
                                  parallel=True)(rolling_mean_python)


def rolling_mean_parallel(array: np.ndarray, window: int, min_periods: int):
    if array.flags.fortran:
        return rolling_mean_f_parallel(array, window, min_periods)
    return rolling_mean_c_parallel(array, window, min_periods)


def run(order):
    num_rows = 2000
    num_cols = 2000

    print(f"[Order = '{order}']")

    arr = np.arange(num_rows * num_cols, dtype="float64").reshape((num_rows, num_cols), order=order)
    df = pd.DataFrame(arr)

    t0 = time.time()
    _ = rolling_mean_python(arr, 20, 20)

    t1 = time.time()
    _ = rolling_mean_numpy(arr, 20)

    t2 = time.time()
    df_mean = df.rolling(20, min_periods=20).mean()

    t3 = time.time()
    _ = rolling_mean(arr, 20, 20)

    t4 = time.time()
    _ = rolling_mean_parallel(arr, 20, 20)

    t5 = time.time()
    _ = au.rolling_mean(arr, 20, min_periods=20, num_threads=0)  # 0 means using all CPUs

    t6 = time.time()

    print(f"Pure Python: {t1 - t0}")
    print(f"Numpy Stride Tricks: {t2 - t1}")
    print(f"Pandas Rolling: {t3 - t2}")
    print(f"Numba w/o Parallelization: {t4 - t3}")
    print(f"Numba w/ Parallelization: {t5 - t4}")
    print(f"C++ w/ Multithreading: {t6 - t5}")
    print()


if __name__ == "__main__":
    run('C')
    run('F')

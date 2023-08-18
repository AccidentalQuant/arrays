import numpy as np
import numba as nb


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

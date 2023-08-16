#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <cmath>
#include <vector>
#include <thread>
#include <algorithm>

#include <Python.h>
#include <numpy/arrayobject.h>


static void calc_rolling_mean_1d
(const npy_float64* in_start, npy_float64* out_start, long window, long min_periods,
long size, long in_stride, long out_stride) {
  const npy_float64* in_curr = in_start, *in_old = in_start;
  npy_float64* out_curr = out_start;

  npy_float64 sum = 0.0, c = 0.0, y, t, z;
  long cnt = 0;

  for (long i = 0; i < size; ++i) {
    if (i >= window) {
      if (std::isfinite(*in_old)) {
        y = *in_old + c;
        t = sum - y;
        z = t - sum;
        c = z + y;
        sum = t;
        --cnt;
      }

      in_old += in_stride;
    }

    if (std::isfinite(*in_curr)) {
      y = *in_curr - c;
      t = sum + y;
      z = t - sum;
      c = z - y;
      sum = t;
      ++cnt;
    }

    if (cnt >= min_periods) {
      *out_curr = sum / static_cast<npy_float64>(cnt);
    } else {
      *out_curr = NAN;
    }

    in_curr += in_stride;
    out_curr += out_stride;
  }
}

static void calc_rolling_mean
(const npy_float64* in, npy_float64* out, long window, long min_periods, long num_rows, long num_cols,
long in_row_stride, long in_col_stride, long out_row_stride, long out_col_stride, long col_start, long col_end) {
  for (long i = col_start; i < col_end; ++i) {
    calc_rolling_mean_1d(in + in_col_stride * i, out + out_col_stride * i, window, min_periods,
                         num_rows, in_row_stride, out_row_stride);
  }
}

static void calc_rolling_mean_parallel
(const npy_float64* in, npy_float64* out, long window, long min_periods, long num_rows, long num_cols,
long in_row_stride, long in_col_stride, long out_row_stride, long out_col_stride, long num_threads) {
  long num_threads_actual = std::min(num_cols, num_threads);
  long cols_per_thread = num_cols / num_threads_actual;
  long rem = num_cols % num_threads_actual;

  std::vector<std::thread> threads;

  for (long i = 0; i < num_threads_actual; ++i) {
    long col_start = (i < rem) ? (cols_per_thread + 1) * i : cols_per_thread * i + rem;
    long col_end = col_start + cols_per_thread + ((i < rem) ? 1 : 0);
    threads.emplace_back(calc_rolling_mean, in, out, window, min_periods, num_rows, num_cols, in_row_stride,
                         in_col_stride, out_row_stride, out_col_stride, col_start, col_end);
  }

  for (auto& thread : threads) {
    thread.join();
  }
}


static PyObject* rolling_mean(PyObject* self, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"array", "window", "min_periods", "num_threads", nullptr};
  PyObject* array_arg;
  long window;
  long min_periods = 0;
  long num_threads = (std::thread::hardware_concurrency() + 1) / 2;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Ol|ll", const_cast<char**>(keywords),
                                   &array_arg, &window, &min_periods, &num_threads)) {
    return nullptr;
  }

  if (window <= 0) {
    PyErr_SetString(PyExc_ValueError, "`window` must be a positive integer");
    return nullptr;
  }

  if (min_periods <= 0 || min_periods > window) {
    min_periods = window;
  }

  if (num_threads <= 0 || num_threads > std::thread::hardware_concurrency()) {
    num_threads = std::thread::hardware_concurrency();
  }

  PyObject* array_py = PyArray_FROM_OTF(array_arg, NPY_FLOAT64, NPY_ARRAY_ALIGNED);
  auto* array = reinterpret_cast<PyArrayObject*>(array_py);

  if (PyArray_NDIM(array) != 2) {
    PyErr_SetString(PyExc_ValueError, "`array` must be a 2D array");
    Py_DECREF(array_py);
    return nullptr;
  }

  npy_intp* dims = PyArray_DIMS(array);
  npy_intp* strides = PyArray_STRIDES(array);

  long num_rows = dims[0];
  long num_cols = dims[1];
  long row_stride = strides[0] / sizeof(npy_float64);
  long col_stride = strides[1] / sizeof(npy_float64);

  const auto* data_in = reinterpret_cast<const npy_float64*>(PyArray_DATA(array));

  PyObject* array_out_py = PyArray_SimpleNew(2, dims, NPY_FLOAT64);
  auto* data_out = reinterpret_cast<npy_float64*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(array_out_py)));

  if (num_threads == 1) {
    calc_rolling_mean(data_in, data_out, window, min_periods, num_rows, num_cols, row_stride,
                      col_stride, num_cols, 1, 0, num_cols);
  }
  else {
    calc_rolling_mean_parallel(data_in, data_out, window, min_periods, num_rows, num_cols, row_stride,
                               col_stride, num_cols, 1, num_threads);
  }

  Py_DECREF(array_py);
  return array_out_py;
}

static PyMethodDef arrayutils_methods[] = {
  {"rolling_mean",
   reinterpret_cast<PyCFunction>(rolling_mean),
   METH_VARARGS | METH_KEYWORDS, "rolling mean"},
  {nullptr, nullptr, 0, nullptr}        /* Sentinel */
};

static struct PyModuleDef module_def = {
  PyModuleDef_HEAD_INIT,
  "arrayutils",
  nullptr,
  -1,
  arrayutils_methods,
  nullptr,
  nullptr,
  nullptr,
  nullptr
};

PyMODINIT_FUNC PyInit_arrayutils(void) {
  PyObject *m;
  m = PyModule_Create(&module_def);
  if (!m) {
      return nullptr;
  }

  import_array();
  return m;
}

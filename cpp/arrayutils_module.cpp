#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <cmath>
#include <vector>
#include <thread>
#include <algorithm>

#include <Python.h>
#include <numpy/arrayobject.h>


template<typename T, int TYPE_NUM>
static bool parse_2d_array_arg
(PyObject* array_arg, PyObject*& in_array_py, PyObject*& out_array_py, long& num_rows, long& num_cols,
long& in_row_stride, long& in_col_stride, long& out_row_stride, long& out_col_stride, const T*& in_ptr, T*& out_ptr)
{
  in_array_py = PyArray_FROM_OTF(array_arg, TYPE_NUM, NPY_ARRAY_ALIGNED);
  auto* in_array = reinterpret_cast<PyArrayObject*>(in_array_py);
  if (PyArray_NDIM(in_array) != 2) {
    PyErr_SetString(PyExc_ValueError, "`array` must be a 2D array");
    Py_DECREF(in_array_py);
    return false;
  }

  npy_intp* dims = PyArray_DIMS(in_array);
  npy_intp* strides = PyArray_STRIDES(in_array);

  num_rows = dims[0];
  num_cols = dims[1];
  in_row_stride = strides[0] / sizeof(T);
  in_col_stride = strides[1] / sizeof(T);

  int is_fortran = in_col_stride > in_row_stride;
  out_row_stride = is_fortran ? 1 : num_cols;
  out_col_stride = is_fortran ? num_rows : 1;

  PyArray_Descr* descr = PyArray_DescrFromType(TYPE_NUM);

  in_ptr = reinterpret_cast<const T*>(PyArray_DATA(in_array));

  out_array_py = PyArray_NewFromDescr(&PyArray_Type, descr, 2, dims, nullptr, nullptr, is_fortran, nullptr);
  out_ptr = reinterpret_cast<T*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(out_array_py)));
  return true;
}

static void calc_column_rolling_mean
(const npy_float64* in_start, npy_float64* out_start, long window, long min_periods,
long size, long in_stride, long out_stride)
{
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
    calc_column_rolling_mean(in + in_col_stride * i, out + out_col_stride * i, window, min_periods,
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


static PyObject* rolling_mean(PyObject* self, PyObject* args, PyObject* kwargs)
{
  static const char* keywords[] = {"array", "window", "min_periods", "num_threads", nullptr};
  PyObject* array_arg;
  long window;
  long min_periods = 0;
  long num_threads = std::thread::hardware_concurrency();

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

  PyObject* in_array_py, *out_array_py;
  long num_rows, num_cols, in_row_stride, in_col_stride, out_row_stride, out_col_stride;
  const npy_float64* in_ptr;
  npy_float64* out_ptr;

  if (!parse_2d_array_arg<npy_float64, NPY_FLOAT64>(
        array_arg, in_array_py, out_array_py, num_rows, num_cols, in_row_stride, in_col_stride, out_row_stride,
        out_col_stride, in_ptr, out_ptr
      )) {
    return nullptr;
  }

  if (num_threads == 1) {
    calc_rolling_mean(in_ptr, out_ptr, window, min_periods, num_rows, num_cols, in_row_stride,
                      in_col_stride, out_row_stride, out_col_stride, 0, num_cols);
  }
  else {
    calc_rolling_mean_parallel(in_ptr, out_ptr, window, min_periods, num_rows, num_cols, in_row_stride,
                               in_col_stride, out_row_stride, out_col_stride, num_threads);
  }

  Py_DECREF(in_array_py);
  return out_array_py;
}

static void calc_row_rank
(const npy_float64* in_start, npy_float64* out_start, long size, long in_stride, long out_stride, long* idx_buffer)
{
  const npy_float64* in_curr;;
  npy_float64* out_curr;
  long valid_cnt = 0;

  in_curr = in_start;
  out_curr = out_start;

  for (long i = 0; i < size; ++i) {
    if (std::isfinite(*in_curr)) {
      idx_buffer[valid_cnt] = i;
      ++valid_cnt;
    }
    else {
      *out_curr = NAN;
    }
    in_curr += in_stride;
    out_curr += out_stride;
  }

  std::sort(idx_buffer, idx_buffer + valid_cnt,
            [in_start, in_stride](long idx1, long idx2)
            {return in_start[idx1 * in_stride] < in_start[idx2 * in_stride];}
           );
  
  long i = 0, i1, j;
  npy_float64 val, out_val;

  while (i < valid_cnt) {
    i1 = i;
    val = in_start[idx_buffer[i] * in_stride];
    do {
      ++i;
    } while (i < valid_cnt && in_start[idx_buffer[i] * in_stride] == val);

    out_val = static_cast<npy_float64>(i1 + i + 1) / static_cast<npy_float64>(2 * valid_cnt);
    for (j = i1; j < i; ++j) {
      out_start[idx_buffer[j] * out_stride] = out_val;
    }
  }
}

static void calc_rank
(const npy_float64* in, npy_float64* out, long num_rows, long num_cols, long in_row_stride, long in_col_stride,
long out_row_stride, long out_col_stride, long row_start, long row_end) {
  long* idx_buffer = new long[num_cols];
  for (long i = row_start; i < row_end; ++i) {
    calc_row_rank(in + in_row_stride * i, out + out_row_stride * i, num_cols, in_col_stride, out_col_stride, idx_buffer);
  }
  delete[] idx_buffer;
}

static void calc_rank_parallel
(const npy_float64* in, npy_float64* out, long num_rows, long num_cols, long in_row_stride, long in_col_stride,
long out_row_stride, long out_col_stride, long num_threads) {
  long num_threads_actual = std::min(num_rows, num_threads);
  long rows_per_thread = num_rows / num_threads_actual;
  long rem = num_rows % num_threads_actual;

  std::vector<std::thread> threads;

  for (long i = 0; i < num_threads_actual; ++i) {
    long row_start = (i < rem) ? (rows_per_thread + 1) * i : rows_per_thread * i + rem;
    long row_end = row_start + rows_per_thread + ((i < rem) ? 1 : 0);
    threads.emplace_back(calc_rank, in, out, num_rows, num_cols, in_row_stride, in_col_stride, out_row_stride,
                         out_col_stride, row_start, row_end);
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

static PyObject* rank(PyObject* self, PyObject* args, PyObject* kwargs)
{
  static const char* keywords[] = {"array", "num_threads", nullptr};
  PyObject* array_arg;
  long num_threads = std::thread::hardware_concurrency();

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|l", const_cast<char**>(keywords),
                                   &array_arg, &num_threads)) {
    return nullptr;
  }

  if (num_threads <= 0 || num_threads > std::thread::hardware_concurrency()) {
    num_threads = std::thread::hardware_concurrency();
  }

  PyObject* in_array_py, *out_array_py;
  long num_rows, num_cols, in_row_stride, in_col_stride, out_row_stride, out_col_stride;
  const npy_float64* in_ptr;
  npy_float64* out_ptr;

  if (!parse_2d_array_arg<npy_float64, NPY_FLOAT64>(
        array_arg, in_array_py, out_array_py, num_rows, num_cols, in_row_stride, in_col_stride, out_row_stride,
        out_col_stride, in_ptr, out_ptr
      )) {
    return nullptr;
  }

  if (num_threads == 1) {
    calc_rank(in_ptr, out_ptr, num_rows, num_cols, in_row_stride, in_col_stride,
              out_row_stride, out_col_stride, 0, num_cols);
  }
  else {
    calc_rank_parallel(in_ptr, out_ptr, num_rows, num_cols, in_row_stride, in_col_stride,
                       out_row_stride, out_col_stride, num_threads);
  }

  Py_DECREF(in_array_py);
  return out_array_py;
}

static PyMethodDef arrayutils_methods[] =
{
  {"rolling_mean",
   reinterpret_cast<PyCFunction>(rolling_mean),
   METH_VARARGS | METH_KEYWORDS, "rolling mean"},
   {"rank",
   reinterpret_cast<PyCFunction>(rank),
   METH_VARARGS | METH_KEYWORDS, "cross-sectional rank"},
  {nullptr, nullptr, 0, nullptr}        /* Sentinel */
};

static struct PyModuleDef module_def =
{
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

PyMODINIT_FUNC PyInit_arrayutils(void)
{
  PyObject *m;
  m = PyModule_Create(&module_def);
  if (!m) {
      return nullptr;
  }

  import_array();
  return m;
}

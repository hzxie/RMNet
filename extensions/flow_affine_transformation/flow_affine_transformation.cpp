/**
 * @Author: Haozhe Xie
 * @Date:   2020-08-07 10:37:26
 * @Last Modified by:   Haozhe Xie
 * @Last Modified time: 2020-08-25 15:59:34
 * @Email:  cshzxie@gmail.com
 *
 * References:
 * - https://stackoverflow.com/questions/34571945/migrating-to-numpy-api-1-7
 * - https://stackoverflow.com/a/47027598/1841143
 * -
 * http://pageperso.lif.univ-mrs.fr/~francois.denis/IAAM1/numpy-html-1.14.0/reference/c-api.array.html
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <cmath>
#include <numpy/arrayobject.h>

// Initialize Numpy and make sure that be called only once.
// Without initialize, both PyArray_SimpleNew and PyArray_Return cause
// segmentation fault error.
int initNumpy() {
  import_array();
  return 1;
}
const static int NUMPY_INITIALIZED = initNumpy();

/**
 * Update the optical flow after the affine transformation applied to two
 * frames.
 *
 * @param  self unused dummy var
 * @param  args array[3]: optical flow, transformation matrix 1, transformation
 * matrix 2
 * @return array: the new optical flow after affine transformation
 */
static PyObject *updateOpticalFlow(PyObject *self, PyObject *args) {
  PyObject *arrays[3];
  if (!PyArg_ParseTuple(args, "OOO", &arrays[0], &arrays[1], &arrays[2])) {
    return NULL;
  }

  PyArrayObject *opticalFlowArrObj =
      reinterpret_cast<PyArrayObject *>(arrays[0]);
  PyArrayObject *trMatrix1ArrObj = reinterpret_cast<PyArrayObject *>(arrays[1]);
  PyArrayObject *trMatrix2ArrObj = reinterpret_cast<PyArrayObject *>(arrays[2]);

  float *opticalFlow = static_cast<float *>(PyArray_DATA(opticalFlowArrObj));
  float *trMatrix1 = static_cast<float *>(PyArray_DATA(trMatrix1ArrObj));
  float *trMatrix2 = static_cast<float *>(PyArray_DATA(trMatrix2ArrObj));

  npy_intp *opticalFlowShape = PyArray_SHAPE(opticalFlowArrObj);
  size_t height = opticalFlowShape[0], width = opticalFlowShape[1];

  PyArrayObject *newOpticalFlowArrObj =
      reinterpret_cast<PyArrayObject *>(PyArray_SimpleNew(
          PyArray_NDIM(opticalFlowArrObj), opticalFlowShape, NPY_FLOAT32));
  float *newOpticalFlow =
      static_cast<float *>(PyArray_DATA(newOpticalFlowArrObj));

  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      // NOTE: i -> Y; j -> X
      size_t ofArrayIndex = (i * width + j) * 2;
      float x2 = std::round(trMatrix2[0] * j + trMatrix2[1] * i + trMatrix2[2]);
      float y2 = std::round(trMatrix2[3] * j + trMatrix2[4] * i + trMatrix2[5]);

      float x1 = j + opticalFlow[ofArrayIndex];
      float y1 = i + opticalFlow[ofArrayIndex + 1];
      x1 = std::round(trMatrix1[0] * x1 + trMatrix1[1] * y1 + trMatrix1[2]);
      y1 = std::round(trMatrix1[3] * x1 + trMatrix1[4] * y1 + trMatrix1[5]);

      x1 = x1 < 0 ? 0 : (x1 >= width ? width - 1 : x1);
      y1 = y1 < 0 ? 0 : (y1 >= height ? height - 1 : y1);
      x2 = x2 < 0 ? 0 : (x2 >= width ? width - 1 : x2);
      y2 = y2 < 0 ? 0 : (y2 >= height ? height - 1 : y2);

      newOpticalFlow[ofArrayIndex] = x1 - x2;
      newOpticalFlow[ofArrayIndex + 1] = y1 - y2;
    }
  }
  return PyArray_Return(newOpticalFlowArrObj);
}

static PyMethodDef flowAffineTransformationMethods[] = {
    {"update_optical_flow", updateOpticalFlow, METH_VARARGS,
     "Update the value of optical flow for affine transformation"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef cModPyDem = {
    PyModuleDef_HEAD_INIT, "flow_affine_transformation",
    "Update the value of optical flow for affine transformation", -1,
    flowAffineTransformationMethods};

PyMODINIT_FUNC PyInit_flow_affine_transformation(void) {
  return PyModule_Create(&cModPyDem);
}

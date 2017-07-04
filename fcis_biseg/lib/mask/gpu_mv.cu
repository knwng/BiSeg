// ------------------------------------------------------------------
// Fully Convolutional Instance-aware Semantic Segmentation
// Copyright (c) 2017 Microsoft
// Licensed under The Apache-2.0 License [see LICENSE for details]
// Written by Haozhi Qi
// ------------------------------------------------------------------

#include "gpu_mv.hpp"
#include <iostream>

const int CAFFE_CUDA_NUM_THREADS = 512;

inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}


#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

__device__ float bilinear_interpolate(const float* bottom_data,
                                      const int input_height, const int input_width,
                                      float inverse_y, float inverse_x) {

  // deal with cases that inverse elements are out of feature map boundary
  if (inverse_y <= 0) inverse_y = 0;
  if (inverse_x <= 0) inverse_x = 0;
  
  int h_low = (int) inverse_y;
  int w_low = (int) inverse_x;
  int h_high;
  int w_high;

  // handle boundary case
  if (h_low >= input_height - 1) {
    h_high = h_low = input_height - 1;
    inverse_y = (float) h_low;
  } else {
    h_high = h_low + 1;
  }

  if (w_low >= input_width - 1) {
    w_high = w_low = input_width - 1;
    inverse_x = (float) w_low;
  } else {
    w_high = w_low + 1;
  }

  float lh = inverse_y - h_low;
  float lw = inverse_x - w_low;
  float hh = 1 - lh, hw = 1 - lw;
  // corner point of interpolation
  float v1 = bottom_data[h_low * input_width + w_low];
  float v2 = bottom_data[h_low * input_width + w_high];
  float v3 = bottom_data[h_high * input_width + w_low];
  float v4 = bottom_data[h_high * input_width + w_high];
  // weight for each corner
  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  // do bilinear interpolation
  float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

__global__ void mask_render(const int nthreads, const float* input_box, const float* input_mask, const int box_dim, const int mask_size,
                 const int image_height, const int image_width, float* target_buffer) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // target buffer's size if (n * h * w)
    int w = index % image_width;
    int h = (index / image_width) % image_height;
    int n = index / image_width / image_height;
    // get the n-th boxes
    const float* offset_box = input_box + n * box_dim;
    const float* offset_mask = input_mask + n * mask_size * mask_size;
    const float box_x1 = offset_box[0];
    const float box_y1 = offset_box[1];
    const float box_x2 = offset_box[2];
    const float box_y2 = offset_box[3];
    // check whether pixel is out of box bound
    if (w < box_x1 || w > box_x2 || h < box_y1 || h > box_y2) {
      target_buffer[index] = 0.0;
      continue;
    }
    const float box_width = box_x2 - box_x1 + 1.0;
    const float box_height = box_y2 - box_y1 + 1.0;
    const float ratio_w = (float) mask_size / box_width;
    const float ratio_h = (float) mask_size / box_height;
    const float inverse_x = ((float) w - box_x1 + 0.5) * ratio_w - 0.5;
    const float inverse_y = ((float) h - box_y1 + 0.5) * ratio_h - 0.5;

    target_buffer[index] = bilinear_interpolate(offset_mask, mask_size, mask_size, inverse_y, inverse_x);
  }
}

__global__ void mask_aggregate(const int nthreads, const float* render_mask, float* aggregate_mask, const int* candidate_inds, const int* candidate_starts, const float* candidate_weights, const int image_height, const int image_width, const float binary_thresh) {
  // render_mask: num_boxes * image_height * image_width
  // aggregate_mask: output_num * image_height * image_width
  CUDA_KERNEL_LOOP(index, nthreads) {
    int w = index % image_width;
    int h = (index / image_width) % image_height;
    int n = index / image_width / image_height;
    // get candidate_inds, candidate_start
    int candidate_start = (n == 0) ? 0 : candidate_starts[n-1];
    int candidate_end = candidate_starts[n];
    // output value will be summation of (mask * mask_weight)
    float val = 0.0;
    for (int i = candidate_start; i < candidate_end; ++i) {
      int input_mask_ind = candidate_inds[i];
      int offset_render_mask = (input_mask_ind * image_height + h) * image_width + w;
      const float mask_val = render_mask[offset_render_mask] >= binary_thresh ? 1.f : 0.f;
      val += (mask_val * candidate_weights[i]); 
    }
    aggregate_mask[index] = val;
  }
}

__global__ void reduce_mask_col(const int nthreads, const float* masks, int image_height, int image_width, const float binary_thresh, bool* output_buffer) {
  // nthreads will be output_num * image_width
  CUDA_KERNEL_LOOP(index, nthreads) {
    int w = index % image_width;
    int n = index / image_width;
    output_buffer[index] = false;
    for (int i = 0; i < image_height; ++i) {
      if (masks[(n * image_height + i) * image_width + w] >= binary_thresh) {
        output_buffer[index] = true;
        break;
      }
    }
  }
}

__global__ void reduce_mask_row(const int nthreads, const float* masks, int image_height, int image_width, const float binary_thresh, bool* output_buffer) {
  // nthreads will be output_num * image_width
  CUDA_KERNEL_LOOP(index, nthreads) {
    int h = index % image_height;
    int n = index / image_height;
    output_buffer[index] = false;
    for (int i = 0; i < image_width; ++i) {
      if (masks[(n * image_height + h) * image_width + i] >= binary_thresh) {
        output_buffer[index] = true;
        break;
      }
    }
  }
}

__global__ void reduce_bounding_x(const int nthreads, const bool* reduced_col, int* output_buffer, const int image_width) {
  // nthreads will be output_num * 2
  CUDA_KERNEL_LOOP(index, nthreads) {
    int x = index % 2;
    int n = index / 2;
    output_buffer[index] = image_width / 2;
    if (x == 0) {
      for (int i = 0; i < image_width; ++i) {
        if (reduced_col[n * image_width + i]) {
          output_buffer[index] = i;
          break;
        }
      }   
    } else {
      for (int i = image_width - 1; i >= 0; --i) {
        if (reduced_col[n * image_width + i]) {
          output_buffer[index] = i;
          break;
        }
      }
    }
  }
}

__global__ void reduce_bounding_y(const int nthreads, const bool* reduced_row, int* output_buffer, const int image_height) {
  // nthreads will be output_num * 2
  CUDA_KERNEL_LOOP(index, nthreads) {
    int x = index % 2;
    int n = index / 2;
    output_buffer[index] = image_height / 2;
    if (x == 0) {
      for (int i = 0; i < image_height; ++i) {
        if (reduced_row[n * image_height + i]) {
          output_buffer[index] = i;
          break;
        }
      }   
    } else {
      for (int i = image_height - 1; i >= 0; --i) {
        if (reduced_row[n * image_height + i]) {
          output_buffer[index] = i;
          break;
        }
      }
    }
  }
}


__global__ void mask_resize(const int nthreads, const float* original_mask, const int* bounding_x, const int* bounding_y, float* resized_mask, const int mask_size, const int image_height, const int image_width) {
  // output size should be result_num * mask_size * mask_size
  // original_mask should be result_num * image_height * image_width
  // bounding_x should be result_num * 2
  // bounding_y should be result_num * 2
  CUDA_KERNEL_LOOP(index, nthreads) {
    int w = index % mask_size;
    int h = (index / mask_size) % mask_size;
    int n = index / mask_size / mask_size;
    int bbox_x1 = bounding_x[n * 2];
    int bbox_x2 = bounding_x[n * 2 + 1];
    int bbox_y1 = bounding_y[n * 2];
    int bbox_y2 = bounding_y[n * 2 + 1];
    float bbox_width = bbox_x2 - bbox_x1 + 1.0;
    float bbox_height = bbox_y2 - bbox_y1 + 1.0;
    float ratio_w = bbox_width / static_cast<float>(mask_size);
    float ratio_h = bbox_height / static_cast<float>(mask_size);
    float inverse_x = bbox_x1 + static_cast<float>(w + 0.5) * ratio_w - 0.5;
    float inverse_y = bbox_y1 + static_cast<float>(h + 0.5) * ratio_h - 0.5;
    const float* offset_mask = original_mask + n * image_height * image_width;
    resized_mask[index] = bilinear_interpolate(offset_mask, image_height, image_width, inverse_y, inverse_x);
  }
}

void _mv(const float* all_boxes, const float* all_masks, const int all_boxes_num, const int* candidate_inds, const int* candidate_start, const float* candidate_weights, const int candidate_num, const float binary_thresh, const int image_height, const int image_width, const int box_dim, const int mask_size, const int result_num, float* finalize_output_mask, int* finalize_output_box, const int device_id) {

  // allocate device memory
  float* dev_boxes = NULL;
  float* dev_masks = NULL;
  int* dev_candidate_inds = NULL;
  float* dev_candidate_weights = NULL;
  int* dev_candidate_start = NULL;
  
  CUDA_CHECK(cudaMalloc(&dev_boxes, all_boxes_num * box_dim * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dev_boxes, all_boxes, all_boxes_num * box_dim * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&dev_masks, all_boxes_num * mask_size * mask_size * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dev_masks, all_masks, all_boxes_num * mask_size * mask_size * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&dev_candidate_inds, candidate_num * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dev_candidate_inds, candidate_inds, candidate_num * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&dev_candidate_weights, candidate_num * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dev_candidate_weights, candidate_weights, candidate_num * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&dev_candidate_start, result_num * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dev_candidate_start, candidate_start, result_num * sizeof(int),
                        cudaMemcpyHostToDevice));

  // 1. Masks are of size mask_size x mask_size, to do aggregation
  //    first resize them to image scale (image_height x image_width)
  //    result n x image_height x image_width buffer
  const int render_mask_num = all_boxes_num * image_height * image_width;
  float* dev_render_mask = NULL;
  CUDA_CHECK(cudaMalloc(&dev_render_mask, render_mask_num * sizeof(float)));
  
  mask_render<<<CAFFE_GET_BLOCKS(render_mask_num), CAFFE_CUDA_NUM_THREADS>>> (render_mask_num, dev_boxes, dev_masks, box_dim, mask_size, image_height, image_width, dev_render_mask);
  CUDA_POST_KERNEL_CHECK;

  // 2. After we get above buffer, we need to merge certain masks
  //    to get new masks according to candidate_weights and candidate_inds
  //    new_mask = \sum (old_mask * old_mask_weight)
  const int output_mask_num = result_num * image_height * image_width;
  float* dev_output_mask = NULL;
  CUDA_CHECK(cudaMalloc(&dev_output_mask, output_mask_num * sizeof(float)));
  mask_aggregate<<<CAFFE_GET_BLOCKS(output_mask_num), CAFFE_CUDA_NUM_THREADS>>> (output_mask_num, dev_render_mask, dev_output_mask, dev_candidate_inds, dev_candidate_start, dev_candidate_weights, image_height, image_width, binary_thresh);

  CUDA_POST_KERNEL_CHECK;

  // 3. After we get new masks buffer (result_num * image_height * image_width)
  //    we then find the mask boundary, this is achieved by two reduction operation
  //    then the tight mask boundary can be obtained
  int reduced_col_num = result_num * image_width;
  bool* reduced_col_buffer = NULL;
  CUDA_CHECK(cudaMalloc(&reduced_col_buffer, reduced_col_num * sizeof(bool)));
  reduce_mask_col<<<CAFFE_GET_BLOCKS(reduced_col_num), CAFFE_CUDA_NUM_THREADS>>> (reduced_col_num, dev_output_mask, image_height, image_width, binary_thresh, reduced_col_buffer);
  
  int reduced_bound_x_num = result_num * 2;
  int* reduced_bound_x = NULL;
  CUDA_CHECK(cudaMalloc(&reduced_bound_x, reduced_bound_x_num * sizeof(int)));
  reduce_bounding_x<<<CAFFE_GET_BLOCKS(reduced_bound_x_num), CAFFE_CUDA_NUM_THREADS>>> (reduced_bound_x_num, reduced_col_buffer, reduced_bound_x, image_width);
  
  // find vertical boundary
  int reduced_row_num = result_num * image_height;
  bool* reduced_row_buffer = NULL;
  CUDA_CHECK(cudaMalloc(&reduced_row_buffer, reduced_row_num * sizeof(bool)));
  reduce_mask_row<<<CAFFE_GET_BLOCKS(reduced_row_num), CAFFE_CUDA_NUM_THREADS>>> (reduced_row_num, dev_output_mask, image_height, image_width, binary_thresh, reduced_row_buffer);
  
  int reduced_bound_y_num = result_num * 2;
  int* reduced_bound_y = NULL;
  CUDA_CHECK(cudaMalloc(&reduced_bound_y, reduced_bound_y_num * sizeof(int)));
  reduce_bounding_y<<<CAFFE_GET_BLOCKS(reduced_bound_y_num), CAFFE_CUDA_NUM_THREADS>>> (reduced_bound_y_num, reduced_row_buffer, reduced_bound_y, image_height);

  // 4. Once we get tight mask boundary, we could use it to resize masks back
  //    to mask_size x mask_size
  const int resized_mask_num = result_num * mask_size * mask_size;
  float* resized_mask = NULL;
  CUDA_CHECK(cudaMalloc(&resized_mask, resized_mask_num * sizeof(float)));
  mask_resize<<<CAFFE_GET_BLOCKS(resized_mask_num), CAFFE_CUDA_NUM_THREADS>>> (resized_mask_num, dev_output_mask, reduced_bound_x, reduced_bound_y, resized_mask, mask_size, image_height, image_width);

  // copy back boxes to cpu
  int* cpu_bound_x = (int*) malloc(reduced_bound_x_num * sizeof(int));
  cudaMemcpy(cpu_bound_x, reduced_bound_x, reduced_bound_x_num * sizeof(int), cudaMemcpyDeviceToHost);
  int* cpu_bound_y = (int*) malloc(reduced_bound_y_num * sizeof(int));
  cudaMemcpy(cpu_bound_y, reduced_bound_y, reduced_bound_y_num * sizeof(int), cudaMemcpyDeviceToHost);
  int cnt = 0;
  for (int i = 0; i < result_num; i ++) {
    finalize_output_box[i*4] = cpu_bound_x[cnt];
    finalize_output_box[i*4+1] = cpu_bound_y[cnt];
    finalize_output_box[i*4+2] = cpu_bound_x[cnt+1];
    finalize_output_box[i*4+3] = cpu_bound_y[cnt+1];
    cnt += 2;
  }
  // copy back masks to cpu
  CUDA_CHECK(cudaMemcpy(finalize_output_mask, resized_mask, resized_mask_num * sizeof(float), 
                        cudaMemcpyDeviceToHost));
  
  // free gpu memories     
  CUDA_CHECK(cudaFree(dev_boxes));
  CUDA_CHECK(cudaFree(dev_masks));
  CUDA_CHECK(cudaFree(dev_candidate_inds));
  CUDA_CHECK(cudaFree(dev_candidate_start));
  CUDA_CHECK(cudaFree(dev_candidate_weights));
  CUDA_CHECK(cudaFree(dev_render_mask));
  CUDA_CHECK(cudaFree(resized_mask));
  CUDA_CHECK(cudaFree(dev_output_mask));
  CUDA_CHECK(cudaFree(reduced_col_buffer));
  CUDA_CHECK(cudaFree(reduced_bound_x));
  CUDA_CHECK(cudaFree(reduced_row_buffer));
  CUDA_CHECK(cudaFree(reduced_bound_y));
}


/* Generated by Cython 0.24 */

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#ifndef Py_PYTHON_H
    #error Python headers needed to compile C extensions, please install development version of Python.
#elif PY_VERSION_HEX < 0x02060000 || (0x03000000 <= PY_VERSION_HEX && PY_VERSION_HEX < 0x03020000)
    #error Cython requires Python 2.6+ or Python 3.2+.
#else
#define CYTHON_ABI "0_24"
#include <stddef.h>
#ifndef offsetof
  #define offsetof(type, member) ( (size_t) & ((type*)0) -> member )
#endif
#if !defined(WIN32) && !defined(MS_WINDOWS)
  #ifndef __stdcall
    #define __stdcall
  #endif
  #ifndef __cdecl
    #define __cdecl
  #endif
  #ifndef __fastcall
    #define __fastcall
  #endif
#endif
#ifndef DL_IMPORT
  #define DL_IMPORT(t) t
#endif
#ifndef DL_EXPORT
  #define DL_EXPORT(t) t
#endif
#ifndef PY_LONG_LONG
  #define PY_LONG_LONG LONG_LONG
#endif
#ifndef Py_HUGE_VAL
  #define Py_HUGE_VAL HUGE_VAL
#endif
#ifdef PYPY_VERSION
  #define CYTHON_COMPILING_IN_PYPY 1
  #define CYTHON_COMPILING_IN_CPYTHON 0
#else
  #define CYTHON_COMPILING_IN_PYPY 0
  #define CYTHON_COMPILING_IN_CPYTHON 1
#endif
#if !defined(CYTHON_USE_PYLONG_INTERNALS) && CYTHON_COMPILING_IN_CPYTHON && PY_VERSION_HEX >= 0x02070000
  #define CYTHON_USE_PYLONG_INTERNALS 1
#endif
#if CYTHON_USE_PYLONG_INTERNALS
  #include "longintrepr.h"
  #undef SHIFT
  #undef BASE
  #undef MASK
#endif
#if CYTHON_COMPILING_IN_PYPY && PY_VERSION_HEX < 0x02070600 && !defined(Py_OptimizeFlag)
  #define Py_OptimizeFlag 0
#endif
#define __PYX_BUILD_PY_SSIZE_T "n"
#define CYTHON_FORMAT_SSIZE_T "z"
#if PY_MAJOR_VERSION < 3
  #define __Pyx_BUILTIN_MODULE_NAME "__builtin__"
  #define __Pyx_PyCode_New(a, k, l, s, f, code, c, n, v, fv, cell, fn, name, fline, lnos)\
          PyCode_New(a+k, l, s, f, code, c, n, v, fv, cell, fn, name, fline, lnos)
  #define __Pyx_DefaultClassType PyClass_Type
#else
  #define __Pyx_BUILTIN_MODULE_NAME "builtins"
  #define __Pyx_PyCode_New(a, k, l, s, f, code, c, n, v, fv, cell, fn, name, fline, lnos)\
          PyCode_New(a, k, l, s, f, code, c, n, v, fv, cell, fn, name, fline, lnos)
  #define __Pyx_DefaultClassType PyType_Type
#endif
#ifndef Py_TPFLAGS_CHECKTYPES
  #define Py_TPFLAGS_CHECKTYPES 0
#endif
#ifndef Py_TPFLAGS_HAVE_INDEX
  #define Py_TPFLAGS_HAVE_INDEX 0
#endif
#ifndef Py_TPFLAGS_HAVE_NEWBUFFER
  #define Py_TPFLAGS_HAVE_NEWBUFFER 0
#endif
#ifndef Py_TPFLAGS_HAVE_FINALIZE
  #define Py_TPFLAGS_HAVE_FINALIZE 0
#endif
#if PY_VERSION_HEX > 0x03030000 && defined(PyUnicode_KIND)
  #define CYTHON_PEP393_ENABLED 1
  #define __Pyx_PyUnicode_READY(op)       (likely(PyUnicode_IS_READY(op)) ?\
                                              0 : _PyUnicode_Ready((PyObject *)(op)))
  #define __Pyx_PyUnicode_GET_LENGTH(u)   PyUnicode_GET_LENGTH(u)
  #define __Pyx_PyUnicode_READ_CHAR(u, i) PyUnicode_READ_CHAR(u, i)
  #define __Pyx_PyUnicode_KIND(u)         PyUnicode_KIND(u)
  #define __Pyx_PyUnicode_DATA(u)         PyUnicode_DATA(u)
  #define __Pyx_PyUnicode_READ(k, d, i)   PyUnicode_READ(k, d, i)
  #define __Pyx_PyUnicode_IS_TRUE(u)      (0 != (likely(PyUnicode_IS_READY(u)) ? PyUnicode_GET_LENGTH(u) : PyUnicode_GET_SIZE(u)))
#else
  #define CYTHON_PEP393_ENABLED 0
  #define __Pyx_PyUnicode_READY(op)       (0)
  #define __Pyx_PyUnicode_GET_LENGTH(u)   PyUnicode_GET_SIZE(u)
  #define __Pyx_PyUnicode_READ_CHAR(u, i) ((Py_UCS4)(PyUnicode_AS_UNICODE(u)[i]))
  #define __Pyx_PyUnicode_KIND(u)         (sizeof(Py_UNICODE))
  #define __Pyx_PyUnicode_DATA(u)         ((void*)PyUnicode_AS_UNICODE(u))
  #define __Pyx_PyUnicode_READ(k, d, i)   ((void)(k), (Py_UCS4)(((Py_UNICODE*)d)[i]))
  #define __Pyx_PyUnicode_IS_TRUE(u)      (0 != PyUnicode_GET_SIZE(u))
#endif
#if CYTHON_COMPILING_IN_PYPY
  #define __Pyx_PyUnicode_Concat(a, b)      PyNumber_Add(a, b)
  #define __Pyx_PyUnicode_ConcatSafe(a, b)  PyNumber_Add(a, b)
#else
  #define __Pyx_PyUnicode_Concat(a, b)      PyUnicode_Concat(a, b)
  #define __Pyx_PyUnicode_ConcatSafe(a, b)  ((unlikely((a) == Py_None) || unlikely((b) == Py_None)) ?\
      PyNumber_Add(a, b) : __Pyx_PyUnicode_Concat(a, b))
#endif
#if CYTHON_COMPILING_IN_PYPY && !defined(PyUnicode_Contains)
  #define PyUnicode_Contains(u, s)  PySequence_Contains(u, s)
#endif
#if CYTHON_COMPILING_IN_PYPY && !defined(PyObject_Format)
  #define PyObject_Format(obj, fmt)  PyObject_CallMethod(obj, "__format__", "O", fmt)
#endif
#if CYTHON_COMPILING_IN_PYPY && !defined(PyObject_Malloc)
  #define PyObject_Malloc(s)   PyMem_Malloc(s)
  #define PyObject_Free(p)     PyMem_Free(p)
  #define PyObject_Realloc(p)  PyMem_Realloc(p)
#endif
#define __Pyx_PyString_FormatSafe(a, b)   ((unlikely((a) == Py_None)) ? PyNumber_Remainder(a, b) : __Pyx_PyString_Format(a, b))
#define __Pyx_PyUnicode_FormatSafe(a, b)  ((unlikely((a) == Py_None)) ? PyNumber_Remainder(a, b) : PyUnicode_Format(a, b))
#if PY_MAJOR_VERSION >= 3
  #define __Pyx_PyString_Format(a, b)  PyUnicode_Format(a, b)
#else
  #define __Pyx_PyString_Format(a, b)  PyString_Format(a, b)
#endif
#if PY_MAJOR_VERSION < 3 && !defined(PyObject_ASCII)
  #define PyObject_ASCII(o)            PyObject_Repr(o)
#endif
#if PY_MAJOR_VERSION >= 3
  #define PyBaseString_Type            PyUnicode_Type
  #define PyStringObject               PyUnicodeObject
  #define PyString_Type                PyUnicode_Type
  #define PyString_Check               PyUnicode_Check
  #define PyString_CheckExact          PyUnicode_CheckExact
#endif
#if PY_MAJOR_VERSION >= 3
  #define __Pyx_PyBaseString_Check(obj) PyUnicode_Check(obj)
  #define __Pyx_PyBaseString_CheckExact(obj) PyUnicode_CheckExact(obj)
#else
  #define __Pyx_PyBaseString_Check(obj) (PyString_Check(obj) || PyUnicode_Check(obj))
  #define __Pyx_PyBaseString_CheckExact(obj) (PyString_CheckExact(obj) || PyUnicode_CheckExact(obj))
#endif
#ifndef PySet_CheckExact
  #define PySet_CheckExact(obj)        (Py_TYPE(obj) == &PySet_Type)
#endif
#define __Pyx_TypeCheck(obj, type) PyObject_TypeCheck(obj, (PyTypeObject *)type)
#if PY_MAJOR_VERSION >= 3
  #define PyIntObject                  PyLongObject
  #define PyInt_Type                   PyLong_Type
  #define PyInt_Check(op)              PyLong_Check(op)
  #define PyInt_CheckExact(op)         PyLong_CheckExact(op)
  #define PyInt_FromString             PyLong_FromString
  #define PyInt_FromUnicode            PyLong_FromUnicode
  #define PyInt_FromLong               PyLong_FromLong
  #define PyInt_FromSize_t             PyLong_FromSize_t
  #define PyInt_FromSsize_t            PyLong_FromSsize_t
  #define PyInt_AsLong                 PyLong_AsLong
  #define PyInt_AS_LONG                PyLong_AS_LONG
  #define PyInt_AsSsize_t              PyLong_AsSsize_t
  #define PyInt_AsUnsignedLongMask     PyLong_AsUnsignedLongMask
  #define PyInt_AsUnsignedLongLongMask PyLong_AsUnsignedLongLongMask
  #define PyNumber_Int                 PyNumber_Long
#endif
#if PY_MAJOR_VERSION >= 3
  #define PyBoolObject                 PyLongObject
#endif
#if PY_MAJOR_VERSION >= 3 && CYTHON_COMPILING_IN_PYPY
  #ifndef PyUnicode_InternFromString
    #define PyUnicode_InternFromString(s) PyUnicode_FromString(s)
  #endif
#endif
#if PY_VERSION_HEX < 0x030200A4
  typedef long Py_hash_t;
  #define __Pyx_PyInt_FromHash_t PyInt_FromLong
  #define __Pyx_PyInt_AsHash_t   PyInt_AsLong
#else
  #define __Pyx_PyInt_FromHash_t PyInt_FromSsize_t
  #define __Pyx_PyInt_AsHash_t   PyInt_AsSsize_t
#endif
#if PY_MAJOR_VERSION >= 3
  #define __Pyx_PyMethod_New(func, self, klass) ((self) ? PyMethod_New(func, self) : PyInstanceMethod_New(func))
#else
  #define __Pyx_PyMethod_New(func, self, klass) PyMethod_New(func, self, klass)
#endif
#if PY_VERSION_HEX >= 0x030500B1
#define __Pyx_PyAsyncMethodsStruct PyAsyncMethods
#define __Pyx_PyType_AsAsync(obj) (Py_TYPE(obj)->tp_as_async)
#elif CYTHON_COMPILING_IN_CPYTHON && PY_MAJOR_VERSION >= 3
typedef struct {
    unaryfunc am_await;
    unaryfunc am_aiter;
    unaryfunc am_anext;
} __Pyx_PyAsyncMethodsStruct;
#define __Pyx_PyType_AsAsync(obj) ((__Pyx_PyAsyncMethodsStruct*) (Py_TYPE(obj)->tp_reserved))
#else
#define __Pyx_PyType_AsAsync(obj) NULL
#endif
#ifndef CYTHON_RESTRICT
  #if defined(__GNUC__)
    #define CYTHON_RESTRICT __restrict__
  #elif defined(_MSC_VER) && _MSC_VER >= 1400
    #define CYTHON_RESTRICT __restrict
  #elif defined (__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
    #define CYTHON_RESTRICT restrict
  #else
    #define CYTHON_RESTRICT
  #endif
#endif
#define __Pyx_void_to_None(void_result) ((void)(void_result), Py_INCREF(Py_None), Py_None)

#ifndef CYTHON_INLINE
  #if defined(__GNUC__)
    #define CYTHON_INLINE __inline__
  #elif defined(_MSC_VER)
    #define CYTHON_INLINE __inline
  #elif defined (__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
    #define CYTHON_INLINE inline
  #else
    #define CYTHON_INLINE
  #endif
#endif

#if defined(WIN32) || defined(MS_WINDOWS)
  #define _USE_MATH_DEFINES
#endif
#include <math.h>
#ifdef NAN
#define __PYX_NAN() ((float) NAN)
#else
static CYTHON_INLINE float __PYX_NAN() {
  float value;
  memset(&value, 0xFF, sizeof(value));
  return value;
}
#endif


#define __PYX_ERR(f_index, lineno, Ln_error) \
{ \
  __pyx_filename = __pyx_f[f_index]; __pyx_lineno = lineno; __pyx_clineno = __LINE__; goto Ln_error; \
}

#if PY_MAJOR_VERSION >= 3
  #define __Pyx_PyNumber_Divide(x,y)         PyNumber_TrueDivide(x,y)
  #define __Pyx_PyNumber_InPlaceDivide(x,y)  PyNumber_InPlaceTrueDivide(x,y)
#else
  #define __Pyx_PyNumber_Divide(x,y)         PyNumber_Divide(x,y)
  #define __Pyx_PyNumber_InPlaceDivide(x,y)  PyNumber_InPlaceDivide(x,y)
#endif

#ifndef __PYX_EXTERN_C
  #ifdef __cplusplus
    #define __PYX_EXTERN_C extern "C"
  #else
    #define __PYX_EXTERN_C extern
  #endif
#endif

#define __PYX_HAVE__lib__mask__gpu_mv
#define __PYX_HAVE_API__lib__mask__gpu_mv
#include "string.h"
#include "stdio.h"
#include "stdlib.h"
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "gpu_mv.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif /* _OPENMP */

#ifdef PYREX_WITHOUT_ASSERTIONS
#define CYTHON_WITHOUT_ASSERTIONS
#endif

#ifndef CYTHON_UNUSED
# if defined(__GNUC__)
#   if !(defined(__cplusplus)) || (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4))
#     define CYTHON_UNUSED __attribute__ ((__unused__))
#   else
#     define CYTHON_UNUSED
#   endif
# elif defined(__ICC) || (defined(__INTEL_COMPILER) && !defined(_MSC_VER))
#   define CYTHON_UNUSED __attribute__ ((__unused__))
# else
#   define CYTHON_UNUSED
# endif
#endif
#ifndef CYTHON_NCP_UNUSED
# if CYTHON_COMPILING_IN_CPYTHON
#  define CYTHON_NCP_UNUSED
# else
#  define CYTHON_NCP_UNUSED CYTHON_UNUSED
# endif
#endif
typedef struct {PyObject **p; const char *s; const Py_ssize_t n; const char* encoding;
                const char is_unicode; const char is_str; const char intern; } __Pyx_StringTabEntry;

#define __PYX_DEFAULT_STRING_ENCODING_IS_ASCII 0
#define __PYX_DEFAULT_STRING_ENCODING_IS_DEFAULT 0
#define __PYX_DEFAULT_STRING_ENCODING ""
#define __Pyx_PyObject_FromString __Pyx_PyBytes_FromString
#define __Pyx_PyObject_FromStringAndSize __Pyx_PyBytes_FromStringAndSize
#define __Pyx_uchar_cast(c) ((unsigned char)c)
#define __Pyx_long_cast(x) ((long)x)
#define __Pyx_fits_Py_ssize_t(v, type, is_signed)  (\
    (sizeof(type) < sizeof(Py_ssize_t))  ||\
    (sizeof(type) > sizeof(Py_ssize_t) &&\
          likely(v < (type)PY_SSIZE_T_MAX ||\
                 v == (type)PY_SSIZE_T_MAX)  &&\
          (!is_signed || likely(v > (type)PY_SSIZE_T_MIN ||\
                                v == (type)PY_SSIZE_T_MIN)))  ||\
    (sizeof(type) == sizeof(Py_ssize_t) &&\
          (is_signed || likely(v < (type)PY_SSIZE_T_MAX ||\
                               v == (type)PY_SSIZE_T_MAX)))  )
#if defined (__cplusplus) && __cplusplus >= 201103L
    #include <cstdlib>
    #define __Pyx_sst_abs(value) std::abs(value)
#elif SIZEOF_INT >= SIZEOF_SIZE_T
    #define __Pyx_sst_abs(value) abs(value)
#elif SIZEOF_LONG >= SIZEOF_SIZE_T
    #define __Pyx_sst_abs(value) labs(value)
#elif defined (_MSC_VER) && defined (_M_X64)
    #define __Pyx_sst_abs(value) _abs64(value)
#elif defined (__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
    #define __Pyx_sst_abs(value) llabs(value)
#elif defined (__GNUC__)
    #define __Pyx_sst_abs(value) __builtin_llabs(value)
#else
    #define __Pyx_sst_abs(value) ((value<0) ? -value : value)
#endif
static CYTHON_INLINE char* __Pyx_PyObject_AsString(PyObject*);
static CYTHON_INLINE char* __Pyx_PyObject_AsStringAndSize(PyObject*, Py_ssize_t* length);
#define __Pyx_PyByteArray_FromString(s) PyByteArray_FromStringAndSize((const char*)s, strlen((const char*)s))
#define __Pyx_PyByteArray_FromStringAndSize(s, l) PyByteArray_FromStringAndSize((const char*)s, l)
#define __Pyx_PyBytes_FromString        PyBytes_FromString
#define __Pyx_PyBytes_FromStringAndSize PyBytes_FromStringAndSize
static CYTHON_INLINE PyObject* __Pyx_PyUnicode_FromString(const char*);
#if PY_MAJOR_VERSION < 3
    #define __Pyx_PyStr_FromString        __Pyx_PyBytes_FromString
    #define __Pyx_PyStr_FromStringAndSize __Pyx_PyBytes_FromStringAndSize
#else
    #define __Pyx_PyStr_FromString        __Pyx_PyUnicode_FromString
    #define __Pyx_PyStr_FromStringAndSize __Pyx_PyUnicode_FromStringAndSize
#endif
#define __Pyx_PyObject_AsSString(s)    ((signed char*) __Pyx_PyObject_AsString(s))
#define __Pyx_PyObject_AsUString(s)    ((unsigned char*) __Pyx_PyObject_AsString(s))
#define __Pyx_PyObject_FromCString(s)  __Pyx_PyObject_FromString((const char*)s)
#define __Pyx_PyBytes_FromCString(s)   __Pyx_PyBytes_FromString((const char*)s)
#define __Pyx_PyByteArray_FromCString(s)   __Pyx_PyByteArray_FromString((const char*)s)
#define __Pyx_PyStr_FromCString(s)     __Pyx_PyStr_FromString((const char*)s)
#define __Pyx_PyUnicode_FromCString(s) __Pyx_PyUnicode_FromString((const char*)s)
#if PY_MAJOR_VERSION < 3
static CYTHON_INLINE size_t __Pyx_Py_UNICODE_strlen(const Py_UNICODE *u)
{
    const Py_UNICODE *u_end = u;
    while (*u_end++) ;
    return (size_t)(u_end - u - 1);
}
#else
#define __Pyx_Py_UNICODE_strlen Py_UNICODE_strlen
#endif
#define __Pyx_PyUnicode_FromUnicode(u)       PyUnicode_FromUnicode(u, __Pyx_Py_UNICODE_strlen(u))
#define __Pyx_PyUnicode_FromUnicodeAndLength PyUnicode_FromUnicode
#define __Pyx_PyUnicode_AsUnicode            PyUnicode_AsUnicode
#define __Pyx_NewRef(obj) (Py_INCREF(obj), obj)
#define __Pyx_Owned_Py_None(b) __Pyx_NewRef(Py_None)
#define __Pyx_PyBool_FromLong(b) ((b) ? __Pyx_NewRef(Py_True) : __Pyx_NewRef(Py_False))
static CYTHON_INLINE int __Pyx_PyObject_IsTrue(PyObject*);
static CYTHON_INLINE PyObject* __Pyx_PyNumber_IntOrLong(PyObject* x);
static CYTHON_INLINE Py_ssize_t __Pyx_PyIndex_AsSsize_t(PyObject*);
static CYTHON_INLINE PyObject * __Pyx_PyInt_FromSize_t(size_t);
#if CYTHON_COMPILING_IN_CPYTHON
#define __pyx_PyFloat_AsDouble(x) (PyFloat_CheckExact(x) ? PyFloat_AS_DOUBLE(x) : PyFloat_AsDouble(x))
#else
#define __pyx_PyFloat_AsDouble(x) PyFloat_AsDouble(x)
#endif
#define __pyx_PyFloat_AsFloat(x) ((float) __pyx_PyFloat_AsDouble(x))
#if PY_MAJOR_VERSION >= 3
#define __Pyx_PyNumber_Int(x) (PyLong_CheckExact(x) ? __Pyx_NewRef(x) : PyNumber_Long(x))
#else
#define __Pyx_PyNumber_Int(x) (PyInt_CheckExact(x) ? __Pyx_NewRef(x) : PyNumber_Int(x))
#endif
#define __Pyx_PyNumber_Float(x) (PyFloat_CheckExact(x) ? __Pyx_NewRef(x) : PyNumber_Float(x))
#if PY_MAJOR_VERSION < 3 && __PYX_DEFAULT_STRING_ENCODING_IS_ASCII
static int __Pyx_sys_getdefaultencoding_not_ascii;
static int __Pyx_init_sys_getdefaultencoding_params(void) {
    PyObject* sys;
    PyObject* default_encoding = NULL;
    PyObject* ascii_chars_u = NULL;
    PyObject* ascii_chars_b = NULL;
    const char* default_encoding_c;
    sys = PyImport_ImportModule("sys");
    if (!sys) goto bad;
    default_encoding = PyObject_CallMethod(sys, (char*) "getdefaultencoding", NULL);
    Py_DECREF(sys);
    if (!default_encoding) goto bad;
    default_encoding_c = PyBytes_AsString(default_encoding);
    if (!default_encoding_c) goto bad;
    if (strcmp(default_encoding_c, "ascii") == 0) {
        __Pyx_sys_getdefaultencoding_not_ascii = 0;
    } else {
        char ascii_chars[128];
        int c;
        for (c = 0; c < 128; c++) {
            ascii_chars[c] = c;
        }
        __Pyx_sys_getdefaultencoding_not_ascii = 1;
        ascii_chars_u = PyUnicode_DecodeASCII(ascii_chars, 128, NULL);
        if (!ascii_chars_u) goto bad;
        ascii_chars_b = PyUnicode_AsEncodedString(ascii_chars_u, default_encoding_c, NULL);
        if (!ascii_chars_b || !PyBytes_Check(ascii_chars_b) || memcmp(ascii_chars, PyBytes_AS_STRING(ascii_chars_b), 128) != 0) {
            PyErr_Format(
                PyExc_ValueError,
                "This module compiled with c_string_encoding=ascii, but default encoding '%.200s' is not a superset of ascii.",
                default_encoding_c);
            goto bad;
        }
        Py_DECREF(ascii_chars_u);
        Py_DECREF(ascii_chars_b);
    }
    Py_DECREF(default_encoding);
    return 0;
bad:
    Py_XDECREF(default_encoding);
    Py_XDECREF(ascii_chars_u);
    Py_XDECREF(ascii_chars_b);
    return -1;
}
#endif
#if __PYX_DEFAULT_STRING_ENCODING_IS_DEFAULT && PY_MAJOR_VERSION >= 3
#define __Pyx_PyUnicode_FromStringAndSize(c_str, size) PyUnicode_DecodeUTF8(c_str, size, NULL)
#else
#define __Pyx_PyUnicode_FromStringAndSize(c_str, size) PyUnicode_Decode(c_str, size, __PYX_DEFAULT_STRING_ENCODING, NULL)
#if __PYX_DEFAULT_STRING_ENCODING_IS_DEFAULT
static char* __PYX_DEFAULT_STRING_ENCODING;
static int __Pyx_init_sys_getdefaultencoding_params(void) {
    PyObject* sys;
    PyObject* default_encoding = NULL;
    char* default_encoding_c;
    sys = PyImport_ImportModule("sys");
    if (!sys) goto bad;
    default_encoding = PyObject_CallMethod(sys, (char*) (const char*) "getdefaultencoding", NULL);
    Py_DECREF(sys);
    if (!default_encoding) goto bad;
    default_encoding_c = PyBytes_AsString(default_encoding);
    if (!default_encoding_c) goto bad;
    __PYX_DEFAULT_STRING_ENCODING = (char*) malloc(strlen(default_encoding_c));
    if (!__PYX_DEFAULT_STRING_ENCODING) goto bad;
    strcpy(__PYX_DEFAULT_STRING_ENCODING, default_encoding_c);
    Py_DECREF(default_encoding);
    return 0;
bad:
    Py_XDECREF(default_encoding);
    return -1;
}
#endif
#endif


/* Test for GCC > 2.95 */
#if defined(__GNUC__)     && (__GNUC__ > 2 || (__GNUC__ == 2 && (__GNUC_MINOR__ > 95)))
  #define likely(x)   __builtin_expect(!!(x), 1)
  #define unlikely(x) __builtin_expect(!!(x), 0)
#else /* !__GNUC__ or GCC < 2.95 */
  #define likely(x)   (x)
  #define unlikely(x) (x)
#endif /* __GNUC__ */

static PyObject *__pyx_m;
static PyObject *__pyx_d;
static PyObject *__pyx_b;
static PyObject *__pyx_empty_tuple;
static PyObject *__pyx_empty_bytes;
static PyObject *__pyx_empty_unicode;
static int __pyx_lineno;
static int __pyx_clineno = 0;
static const char * __pyx_cfilenm= __FILE__;
static const char *__pyx_filename;

/* None.proto */
#if !defined(CYTHON_CCOMPLEX)
  #if defined(__cplusplus)
    #define CYTHON_CCOMPLEX 1
  #elif defined(_Complex_I)
    #define CYTHON_CCOMPLEX 1
  #else
    #define CYTHON_CCOMPLEX 0
  #endif
#endif
#if CYTHON_CCOMPLEX
  #ifdef __cplusplus
    #include <complex>
  #else
    #include <complex.h>
  #endif
#endif
#if CYTHON_CCOMPLEX && !defined(__cplusplus) && defined(__sun__) && defined(__GNUC__)
  #undef _Complex_I
  #define _Complex_I 1.0fj
#endif


static const char *__pyx_f[] = {
  "lib\\mask\\gpu_mv.pyx",
  "__init__.pxd",
  "type.pxd",
};
/* BufferFormatStructs.proto */
#define IS_UNSIGNED(type) (((type) -1) > 0)
struct __Pyx_StructField_;
#define __PYX_BUF_FLAGS_PACKED_STRUCT (1 << 0)
typedef struct {
  const char* name;
  struct __Pyx_StructField_* fields;
  size_t size;
  size_t arraysize[8];
  int ndim;
  char typegroup;
  char is_unsigned;
  int flags;
} __Pyx_TypeInfo;
typedef struct __Pyx_StructField_ {
  __Pyx_TypeInfo* type;
  const char* name;
  size_t offset;
} __Pyx_StructField;
typedef struct {
  __Pyx_StructField* field;
  size_t parent_offset;
} __Pyx_BufFmt_StackElem;
typedef struct {
  __Pyx_StructField root;
  __Pyx_BufFmt_StackElem* head;
  size_t fmt_offset;
  size_t new_count, enc_count;
  size_t struct_alignment;
  int is_complex;
  char enc_type;
  char new_packmode;
  char enc_packmode;
  char is_valid_array;
} __Pyx_BufFmt_Context;


/* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":725
 * # in Cython to enable them only on the right systems.
 * 
 * ctypedef npy_int8       int8_t             # <<<<<<<<<<<<<<
 * ctypedef npy_int16      int16_t
 * ctypedef npy_int32      int32_t
 */
typedef npy_int8 __pyx_t_5numpy_int8_t;

/* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":726
 * 
 * ctypedef npy_int8       int8_t
 * ctypedef npy_int16      int16_t             # <<<<<<<<<<<<<<
 * ctypedef npy_int32      int32_t
 * ctypedef npy_int64      int64_t
 */
typedef npy_int16 __pyx_t_5numpy_int16_t;

/* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":727
 * ctypedef npy_int8       int8_t
 * ctypedef npy_int16      int16_t
 * ctypedef npy_int32      int32_t             # <<<<<<<<<<<<<<
 * ctypedef npy_int64      int64_t
 * #ctypedef npy_int96      int96_t
 */
typedef npy_int32 __pyx_t_5numpy_int32_t;

/* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":728
 * ctypedef npy_int16      int16_t
 * ctypedef npy_int32      int32_t
 * ctypedef npy_int64      int64_t             # <<<<<<<<<<<<<<
 * #ctypedef npy_int96      int96_t
 * #ctypedef npy_int128     int128_t
 */
typedef npy_int64 __pyx_t_5numpy_int64_t;

/* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":732
 * #ctypedef npy_int128     int128_t
 * 
 * ctypedef npy_uint8      uint8_t             # <<<<<<<<<<<<<<
 * ctypedef npy_uint16     uint16_t
 * ctypedef npy_uint32     uint32_t
 */
typedef npy_uint8 __pyx_t_5numpy_uint8_t;

/* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":733
 * 
 * ctypedef npy_uint8      uint8_t
 * ctypedef npy_uint16     uint16_t             # <<<<<<<<<<<<<<
 * ctypedef npy_uint32     uint32_t
 * ctypedef npy_uint64     uint64_t
 */
typedef npy_uint16 __pyx_t_5numpy_uint16_t;

/* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":734
 * ctypedef npy_uint8      uint8_t
 * ctypedef npy_uint16     uint16_t
 * ctypedef npy_uint32     uint32_t             # <<<<<<<<<<<<<<
 * ctypedef npy_uint64     uint64_t
 * #ctypedef npy_uint96     uint96_t
 */
typedef npy_uint32 __pyx_t_5numpy_uint32_t;

/* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":735
 * ctypedef npy_uint16     uint16_t
 * ctypedef npy_uint32     uint32_t
 * ctypedef npy_uint64     uint64_t             # <<<<<<<<<<<<<<
 * #ctypedef npy_uint96     uint96_t
 * #ctypedef npy_uint128    uint128_t
 */
typedef npy_uint64 __pyx_t_5numpy_uint64_t;

/* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":739
 * #ctypedef npy_uint128    uint128_t
 * 
 * ctypedef npy_float32    float32_t             # <<<<<<<<<<<<<<
 * ctypedef npy_float64    float64_t
 * #ctypedef npy_float80    float80_t
 */
typedef npy_float32 __pyx_t_5numpy_float32_t;

/* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":740
 * 
 * ctypedef npy_float32    float32_t
 * ctypedef npy_float64    float64_t             # <<<<<<<<<<<<<<
 * #ctypedef npy_float80    float80_t
 * #ctypedef npy_float128   float128_t
 */
typedef npy_float64 __pyx_t_5numpy_float64_t;

/* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":749
 * # The int types are mapped a bit surprising --
 * # numpy.int corresponds to 'l' and numpy.long to 'q'
 * ctypedef npy_long       int_t             # <<<<<<<<<<<<<<
 * ctypedef npy_longlong   long_t
 * ctypedef npy_longlong   longlong_t
 */
typedef npy_long __pyx_t_5numpy_int_t;

/* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":750
 * # numpy.int corresponds to 'l' and numpy.long to 'q'
 * ctypedef npy_long       int_t
 * ctypedef npy_longlong   long_t             # <<<<<<<<<<<<<<
 * ctypedef npy_longlong   longlong_t
 * 
 */
typedef npy_longlong __pyx_t_5numpy_long_t;

/* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":751
 * ctypedef npy_long       int_t
 * ctypedef npy_longlong   long_t
 * ctypedef npy_longlong   longlong_t             # <<<<<<<<<<<<<<
 * 
 * ctypedef npy_ulong      uint_t
 */
typedef npy_longlong __pyx_t_5numpy_longlong_t;

/* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":753
 * ctypedef npy_longlong   longlong_t
 * 
 * ctypedef npy_ulong      uint_t             # <<<<<<<<<<<<<<
 * ctypedef npy_ulonglong  ulong_t
 * ctypedef npy_ulonglong  ulonglong_t
 */
typedef npy_ulong __pyx_t_5numpy_uint_t;

/* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":754
 * 
 * ctypedef npy_ulong      uint_t
 * ctypedef npy_ulonglong  ulong_t             # <<<<<<<<<<<<<<
 * ctypedef npy_ulonglong  ulonglong_t
 * 
 */
typedef npy_ulonglong __pyx_t_5numpy_ulong_t;

/* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":755
 * ctypedef npy_ulong      uint_t
 * ctypedef npy_ulonglong  ulong_t
 * ctypedef npy_ulonglong  ulonglong_t             # <<<<<<<<<<<<<<
 * 
 * ctypedef npy_intp       intp_t
 */
typedef npy_ulonglong __pyx_t_5numpy_ulonglong_t;

/* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":757
 * ctypedef npy_ulonglong  ulonglong_t
 * 
 * ctypedef npy_intp       intp_t             # <<<<<<<<<<<<<<
 * ctypedef npy_uintp      uintp_t
 * 
 */
typedef npy_intp __pyx_t_5numpy_intp_t;

/* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":758
 * 
 * ctypedef npy_intp       intp_t
 * ctypedef npy_uintp      uintp_t             # <<<<<<<<<<<<<<
 * 
 * ctypedef npy_double     float_t
 */
typedef npy_uintp __pyx_t_5numpy_uintp_t;

/* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":760
 * ctypedef npy_uintp      uintp_t
 * 
 * ctypedef npy_double     float_t             # <<<<<<<<<<<<<<
 * ctypedef npy_double     double_t
 * ctypedef npy_longdouble longdouble_t
 */
typedef npy_double __pyx_t_5numpy_float_t;

/* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":761
 * 
 * ctypedef npy_double     float_t
 * ctypedef npy_double     double_t             # <<<<<<<<<<<<<<
 * ctypedef npy_longdouble longdouble_t
 * 
 */
typedef npy_double __pyx_t_5numpy_double_t;

/* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":762
 * ctypedef npy_double     float_t
 * ctypedef npy_double     double_t
 * ctypedef npy_longdouble longdouble_t             # <<<<<<<<<<<<<<
 * 
 * ctypedef npy_cfloat      cfloat_t
 */
typedef npy_longdouble __pyx_t_5numpy_longdouble_t;
/* None.proto */
#if CYTHON_CCOMPLEX
  #ifdef __cplusplus
    typedef ::std::complex< float > __pyx_t_float_complex;
  #else
    typedef float _Complex __pyx_t_float_complex;
  #endif
#else
    typedef struct { float real, imag; } __pyx_t_float_complex;
#endif

/* None.proto */
#if CYTHON_CCOMPLEX
  #ifdef __cplusplus
    typedef ::std::complex< double > __pyx_t_double_complex;
  #else
    typedef double _Complex __pyx_t_double_complex;
  #endif
#else
    typedef struct { double real, imag; } __pyx_t_double_complex;
#endif


/*--- Type declarations ---*/

/* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":764
 * ctypedef npy_longdouble longdouble_t
 * 
 * ctypedef npy_cfloat      cfloat_t             # <<<<<<<<<<<<<<
 * ctypedef npy_cdouble     cdouble_t
 * ctypedef npy_clongdouble clongdouble_t
 */
typedef npy_cfloat __pyx_t_5numpy_cfloat_t;

/* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":765
 * 
 * ctypedef npy_cfloat      cfloat_t
 * ctypedef npy_cdouble     cdouble_t             # <<<<<<<<<<<<<<
 * ctypedef npy_clongdouble clongdouble_t
 * 
 */
typedef npy_cdouble __pyx_t_5numpy_cdouble_t;

/* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":766
 * ctypedef npy_cfloat      cfloat_t
 * ctypedef npy_cdouble     cdouble_t
 * ctypedef npy_clongdouble clongdouble_t             # <<<<<<<<<<<<<<
 * 
 * ctypedef npy_cdouble     complex_t
 */
typedef npy_clongdouble __pyx_t_5numpy_clongdouble_t;

/* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":768
 * ctypedef npy_clongdouble clongdouble_t
 * 
 * ctypedef npy_cdouble     complex_t             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_MultiIterNew1(a):
 */
typedef npy_cdouble __pyx_t_5numpy_complex_t;

/* --- Runtime support code (head) --- */
/* Refnanny.proto */
#ifndef CYTHON_REFNANNY
  #define CYTHON_REFNANNY 0
#endif
#if CYTHON_REFNANNY
  typedef struct {
    void (*INCREF)(void*, PyObject*, int);
    void (*DECREF)(void*, PyObject*, int);
    void (*GOTREF)(void*, PyObject*, int);
    void (*GIVEREF)(void*, PyObject*, int);
    void* (*SetupContext)(const char*, int, const char*);
    void (*FinishContext)(void**);
  } __Pyx_RefNannyAPIStruct;
  static __Pyx_RefNannyAPIStruct *__Pyx_RefNanny = NULL;
  static __Pyx_RefNannyAPIStruct *__Pyx_RefNannyImportAPI(const char *modname);
  #define __Pyx_RefNannyDeclarations void *__pyx_refnanny = NULL;
#ifdef WITH_THREAD
  #define __Pyx_RefNannySetupContext(name, acquire_gil)\
          if (acquire_gil) {\
              PyGILState_STATE __pyx_gilstate_save = PyGILState_Ensure();\
              __pyx_refnanny = __Pyx_RefNanny->SetupContext((name), __LINE__, __FILE__);\
              PyGILState_Release(__pyx_gilstate_save);\
          } else {\
              __pyx_refnanny = __Pyx_RefNanny->SetupContext((name), __LINE__, __FILE__);\
          }
#else
  #define __Pyx_RefNannySetupContext(name, acquire_gil)\
          __pyx_refnanny = __Pyx_RefNanny->SetupContext((name), __LINE__, __FILE__)
#endif
  #define __Pyx_RefNannyFinishContext()\
          __Pyx_RefNanny->FinishContext(&__pyx_refnanny)
  #define __Pyx_INCREF(r)  __Pyx_RefNanny->INCREF(__pyx_refnanny, (PyObject *)(r), __LINE__)
  #define __Pyx_DECREF(r)  __Pyx_RefNanny->DECREF(__pyx_refnanny, (PyObject *)(r), __LINE__)
  #define __Pyx_GOTREF(r)  __Pyx_RefNanny->GOTREF(__pyx_refnanny, (PyObject *)(r), __LINE__)
  #define __Pyx_GIVEREF(r) __Pyx_RefNanny->GIVEREF(__pyx_refnanny, (PyObject *)(r), __LINE__)
  #define __Pyx_XINCREF(r)  do { if((r) != NULL) {__Pyx_INCREF(r); }} while(0)
  #define __Pyx_XDECREF(r)  do { if((r) != NULL) {__Pyx_DECREF(r); }} while(0)
  #define __Pyx_XGOTREF(r)  do { if((r) != NULL) {__Pyx_GOTREF(r); }} while(0)
  #define __Pyx_XGIVEREF(r) do { if((r) != NULL) {__Pyx_GIVEREF(r);}} while(0)
#else
  #define __Pyx_RefNannyDeclarations
  #define __Pyx_RefNannySetupContext(name, acquire_gil)
  #define __Pyx_RefNannyFinishContext()
  #define __Pyx_INCREF(r) Py_INCREF(r)
  #define __Pyx_DECREF(r) Py_DECREF(r)
  #define __Pyx_GOTREF(r)
  #define __Pyx_GIVEREF(r)
  #define __Pyx_XINCREF(r) Py_XINCREF(r)
  #define __Pyx_XDECREF(r) Py_XDECREF(r)
  #define __Pyx_XGOTREF(r)
  #define __Pyx_XGIVEREF(r)
#endif
#define __Pyx_XDECREF_SET(r, v) do {\
        PyObject *tmp = (PyObject *) r;\
        r = v; __Pyx_XDECREF(tmp);\
    } while (0)
#define __Pyx_DECREF_SET(r, v) do {\
        PyObject *tmp = (PyObject *) r;\
        r = v; __Pyx_DECREF(tmp);\
    } while (0)
#define __Pyx_CLEAR(r)    do { PyObject* tmp = ((PyObject*)(r)); r = NULL; __Pyx_DECREF(tmp);} while(0)
#define __Pyx_XCLEAR(r)   do { if((r) != NULL) {PyObject* tmp = ((PyObject*)(r)); r = NULL; __Pyx_DECREF(tmp);}} while(0)

/* RaiseArgTupleInvalid.proto */
static void __Pyx_RaiseArgtupleInvalid(const char* func_name, int exact,
    Py_ssize_t num_min, Py_ssize_t num_max, Py_ssize_t num_found);

/* RaiseDoubleKeywords.proto */
static void __Pyx_RaiseDoubleKeywordsError(const char* func_name, PyObject* kw_name);

/* ParseKeywords.proto */
static int __Pyx_ParseOptionalKeywords(PyObject *kwds, PyObject **argnames[],\
    PyObject *kwds2, PyObject *values[], Py_ssize_t num_pos_args,\
    const char* function_name);

/* ArgTypeTest.proto */
static CYTHON_INLINE int __Pyx_ArgTypeTest(PyObject *obj, PyTypeObject *type, int none_allowed,
    const char *name, int exact);

/* BufferFormatCheck.proto */
static CYTHON_INLINE int  __Pyx_GetBufferAndValidate(Py_buffer* buf, PyObject* obj,
    __Pyx_TypeInfo* dtype, int flags, int nd, int cast, __Pyx_BufFmt_StackElem* stack);
static CYTHON_INLINE void __Pyx_SafeReleaseBuffer(Py_buffer* info);
static const char* __Pyx_BufFmt_CheckString(__Pyx_BufFmt_Context* ctx, const char* ts);
static void __Pyx_BufFmt_Init(__Pyx_BufFmt_Context* ctx,
                              __Pyx_BufFmt_StackElem* stack,
                              __Pyx_TypeInfo* type); // PROTO

/* PyObjectGetAttrStr.proto */
#if CYTHON_COMPILING_IN_CPYTHON
static CYTHON_INLINE PyObject* __Pyx_PyObject_GetAttrStr(PyObject* obj, PyObject* attr_name) {
    PyTypeObject* tp = Py_TYPE(obj);
    if (likely(tp->tp_getattro))
        return tp->tp_getattro(obj, attr_name);
#if PY_MAJOR_VERSION < 3
    if (likely(tp->tp_getattr))
        return tp->tp_getattr(obj, PyString_AS_STRING(attr_name));
#endif
    return PyObject_GetAttr(obj, attr_name);
}
#else
#define __Pyx_PyObject_GetAttrStr(o,n) PyObject_GetAttr(o,n)
#endif

/* GetBuiltinName.proto */
static PyObject *__Pyx_GetBuiltinName(PyObject *name);

/* GetModuleGlobalName.proto */
static CYTHON_INLINE PyObject *__Pyx_GetModuleGlobalName(PyObject *name);

/* PyObjectCall.proto */
#if CYTHON_COMPILING_IN_CPYTHON
static CYTHON_INLINE PyObject* __Pyx_PyObject_Call(PyObject *func, PyObject *arg, PyObject *kw);
#else
#define __Pyx_PyObject_Call(func, arg, kw) PyObject_Call(func, arg, kw)
#endif

/* ExtTypeTest.proto */
static CYTHON_INLINE int __Pyx_TypeTest(PyObject *obj, PyTypeObject *type);

/* BufferIndexError.proto */
static void __Pyx_RaiseBufferIndexError(int axis);

#define __Pyx_BufPtrStrided2d(type, buf, i0, s0, i1, s1) (type)((char*)buf + i0 * s0 + i1 * s1)
#define __Pyx_BufPtrStrided4d(type, buf, i0, s0, i1, s1, i2, s2, i3, s3) (type)((char*)buf + i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3)
#define __Pyx_BufPtrStrided1d(type, buf, i0, s0) (type)((char*)buf + i0 * s0)
/* PyThreadStateGet.proto */
#if CYTHON_COMPILING_IN_CPYTHON
#define __Pyx_PyThreadState_declare  PyThreadState *__pyx_tstate;
#define __Pyx_PyThreadState_assign  __pyx_tstate = PyThreadState_GET();
#else
#define __Pyx_PyThreadState_declare
#define __Pyx_PyThreadState_assign
#endif

/* PyErrFetchRestore.proto */
#if CYTHON_COMPILING_IN_CPYTHON
#define __Pyx_ErrRestoreWithState(type, value, tb)  __Pyx_ErrRestoreInState(PyThreadState_GET(), type, value, tb)
#define __Pyx_ErrFetchWithState(type, value, tb)    __Pyx_ErrFetchInState(PyThreadState_GET(), type, value, tb)
#define __Pyx_ErrRestore(type, value, tb)  __Pyx_ErrRestoreInState(__pyx_tstate, type, value, tb)
#define __Pyx_ErrFetch(type, value, tb)    __Pyx_ErrFetchInState(__pyx_tstate, type, value, tb)
static CYTHON_INLINE void __Pyx_ErrRestoreInState(PyThreadState *tstate, PyObject *type, PyObject *value, PyObject *tb);
static CYTHON_INLINE void __Pyx_ErrFetchInState(PyThreadState *tstate, PyObject **type, PyObject **value, PyObject **tb);
#else
#define __Pyx_ErrRestoreWithState(type, value, tb)  PyErr_Restore(type, value, tb)
#define __Pyx_ErrFetchWithState(type, value, tb)  PyErr_Fetch(type, value, tb)
#define __Pyx_ErrRestore(type, value, tb)  PyErr_Restore(type, value, tb)
#define __Pyx_ErrFetch(type, value, tb)  PyErr_Fetch(type, value, tb)
#endif

/* RaiseException.proto */
static void __Pyx_Raise(PyObject *type, PyObject *value, PyObject *tb, PyObject *cause);

/* DictGetItem.proto */
#if PY_MAJOR_VERSION >= 3 && !CYTHON_COMPILING_IN_PYPY
static PyObject *__Pyx_PyDict_GetItem(PyObject *d, PyObject* key) {
    PyObject *value;
    value = PyDict_GetItemWithError(d, key);
    if (unlikely(!value)) {
        if (!PyErr_Occurred()) {
            PyObject* args = PyTuple_Pack(1, key);
            if (likely(args))
                PyErr_SetObject(PyExc_KeyError, args);
            Py_XDECREF(args);
        }
        return NULL;
    }
    Py_INCREF(value);
    return value;
}
#else
    #define __Pyx_PyDict_GetItem(d, key) PyObject_GetItem(d, key)
#endif

/* RaiseTooManyValuesToUnpack.proto */
static CYTHON_INLINE void __Pyx_RaiseTooManyValuesError(Py_ssize_t expected);

/* RaiseNeedMoreValuesToUnpack.proto */
static CYTHON_INLINE void __Pyx_RaiseNeedMoreValuesError(Py_ssize_t index);

/* RaiseNoneIterError.proto */
static CYTHON_INLINE void __Pyx_RaiseNoneNotIterableError(void);

/* Import.proto */
static PyObject *__Pyx_Import(PyObject *name, PyObject *from_list, int level);

/* CodeObjectCache.proto */
typedef struct {
    PyCodeObject* code_object;
    int code_line;
} __Pyx_CodeObjectCacheEntry;
struct __Pyx_CodeObjectCache {
    int count;
    int max_count;
    __Pyx_CodeObjectCacheEntry* entries;
};
static struct __Pyx_CodeObjectCache __pyx_code_cache = {0,0,NULL};
static int __pyx_bisect_code_objects(__Pyx_CodeObjectCacheEntry* entries, int count, int code_line);
static PyCodeObject *__pyx_find_code_object(int code_line);
static void __pyx_insert_code_object(int code_line, PyCodeObject* code_object);

/* AddTraceback.proto */
static void __Pyx_AddTraceback(const char *funcname, int c_line,
                               int py_line, const char *filename);

/* BufferStructDeclare.proto */
typedef struct {
  Py_ssize_t shape, strides, suboffsets;
} __Pyx_Buf_DimInfo;
typedef struct {
  size_t refcount;
  Py_buffer pybuffer;
} __Pyx_Buffer;
typedef struct {
  __Pyx_Buffer *rcbuffer;
  char *data;
  __Pyx_Buf_DimInfo diminfo[8];
} __Pyx_LocalBuf_ND;

#if PY_MAJOR_VERSION < 3
    static int __Pyx_GetBuffer(PyObject *obj, Py_buffer *view, int flags);
    static void __Pyx_ReleaseBuffer(Py_buffer *view);
#else
    #define __Pyx_GetBuffer PyObject_GetBuffer
    #define __Pyx_ReleaseBuffer PyBuffer_Release
#endif


/* None.proto */
static Py_ssize_t __Pyx_zeros[] = {0, 0, 0, 0, 0, 0, 0, 0};
static Py_ssize_t __Pyx_minusones[] = {-1, -1, -1, -1, -1, -1, -1, -1};

/* CIntToPy.proto */
static CYTHON_INLINE PyObject* __Pyx_PyInt_From_int(int value);

/* CIntToPy.proto */
static CYTHON_INLINE PyObject* __Pyx_PyInt_From_Py_intptr_t(Py_intptr_t value);

/* None.proto */
#if CYTHON_CCOMPLEX
  #ifdef __cplusplus
    #define __Pyx_CREAL(z) ((z).real())
    #define __Pyx_CIMAG(z) ((z).imag())
  #else
    #define __Pyx_CREAL(z) (__real__(z))
    #define __Pyx_CIMAG(z) (__imag__(z))
  #endif
#else
    #define __Pyx_CREAL(z) ((z).real)
    #define __Pyx_CIMAG(z) ((z).imag)
#endif
#if defined(__cplusplus) && CYTHON_CCOMPLEX         && (defined(_WIN32) || defined(__clang__) || (defined(__GNUC__) && (__GNUC__ >= 5 || __GNUC__ == 4 && __GNUC_MINOR__ >= 4 )) || __cplusplus >= 201103)
    #define __Pyx_SET_CREAL(z,x) ((z).real(x))
    #define __Pyx_SET_CIMAG(z,y) ((z).imag(y))
#else
    #define __Pyx_SET_CREAL(z,x) __Pyx_CREAL(z) = (x)
    #define __Pyx_SET_CIMAG(z,y) __Pyx_CIMAG(z) = (y)
#endif

/* None.proto */
static CYTHON_INLINE __pyx_t_float_complex __pyx_t_float_complex_from_parts(float, float);

/* None.proto */
#if CYTHON_CCOMPLEX
    #define __Pyx_c_eqf(a, b)   ((a)==(b))
    #define __Pyx_c_sumf(a, b)  ((a)+(b))
    #define __Pyx_c_difff(a, b) ((a)-(b))
    #define __Pyx_c_prodf(a, b) ((a)*(b))
    #define __Pyx_c_quotf(a, b) ((a)/(b))
    #define __Pyx_c_negf(a)     (-(a))
  #ifdef __cplusplus
    #define __Pyx_c_is_zerof(z) ((z)==(float)0)
    #define __Pyx_c_conjf(z)    (::std::conj(z))
    #if 1
        #define __Pyx_c_absf(z)     (::std::abs(z))
        #define __Pyx_c_powf(a, b)  (::std::pow(a, b))
    #endif
  #else
    #define __Pyx_c_is_zerof(z) ((z)==0)
    #define __Pyx_c_conjf(z)    (conjf(z))
    #if 1
        #define __Pyx_c_absf(z)     (cabsf(z))
        #define __Pyx_c_powf(a, b)  (cpowf(a, b))
    #endif
 #endif
#else
    static CYTHON_INLINE int __Pyx_c_eqf(__pyx_t_float_complex, __pyx_t_float_complex);
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_sumf(__pyx_t_float_complex, __pyx_t_float_complex);
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_difff(__pyx_t_float_complex, __pyx_t_float_complex);
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_prodf(__pyx_t_float_complex, __pyx_t_float_complex);
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_quotf(__pyx_t_float_complex, __pyx_t_float_complex);
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_negf(__pyx_t_float_complex);
    static CYTHON_INLINE int __Pyx_c_is_zerof(__pyx_t_float_complex);
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_conjf(__pyx_t_float_complex);
    #if 1
        static CYTHON_INLINE float __Pyx_c_absf(__pyx_t_float_complex);
        static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_powf(__pyx_t_float_complex, __pyx_t_float_complex);
    #endif
#endif

/* None.proto */
static CYTHON_INLINE __pyx_t_double_complex __pyx_t_double_complex_from_parts(double, double);

/* None.proto */
#if CYTHON_CCOMPLEX
    #define __Pyx_c_eq(a, b)   ((a)==(b))
    #define __Pyx_c_sum(a, b)  ((a)+(b))
    #define __Pyx_c_diff(a, b) ((a)-(b))
    #define __Pyx_c_prod(a, b) ((a)*(b))
    #define __Pyx_c_quot(a, b) ((a)/(b))
    #define __Pyx_c_neg(a)     (-(a))
  #ifdef __cplusplus
    #define __Pyx_c_is_zero(z) ((z)==(double)0)
    #define __Pyx_c_conj(z)    (::std::conj(z))
    #if 1
        #define __Pyx_c_abs(z)     (::std::abs(z))
        #define __Pyx_c_pow(a, b)  (::std::pow(a, b))
    #endif
  #else
    #define __Pyx_c_is_zero(z) ((z)==0)
    #define __Pyx_c_conj(z)    (conj(z))
    #if 1
        #define __Pyx_c_abs(z)     (cabs(z))
        #define __Pyx_c_pow(a, b)  (cpow(a, b))
    #endif
 #endif
#else
    static CYTHON_INLINE int __Pyx_c_eq(__pyx_t_double_complex, __pyx_t_double_complex);
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_sum(__pyx_t_double_complex, __pyx_t_double_complex);
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_diff(__pyx_t_double_complex, __pyx_t_double_complex);
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_prod(__pyx_t_double_complex, __pyx_t_double_complex);
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_quot(__pyx_t_double_complex, __pyx_t_double_complex);
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_neg(__pyx_t_double_complex);
    static CYTHON_INLINE int __Pyx_c_is_zero(__pyx_t_double_complex);
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_conj(__pyx_t_double_complex);
    #if 1
        static CYTHON_INLINE double __Pyx_c_abs(__pyx_t_double_complex);
        static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_pow(__pyx_t_double_complex, __pyx_t_double_complex);
    #endif
#endif

/* CIntToPy.proto */
static CYTHON_INLINE PyObject* __Pyx_PyInt_From_enum__NPY_TYPES(enum NPY_TYPES value);

/* CIntFromPy.proto */
static CYTHON_INLINE npy_int32 __Pyx_PyInt_As_npy_int32(PyObject *);

/* CIntFromPy.proto */
static CYTHON_INLINE int __Pyx_PyInt_As_int(PyObject *);

/* CIntToPy.proto */
static CYTHON_INLINE PyObject* __Pyx_PyInt_From_long(long value);

/* CIntFromPy.proto */
static CYTHON_INLINE long __Pyx_PyInt_As_long(PyObject *);

/* CheckBinaryVersion.proto */
static int __Pyx_check_binary_version(void);

/* PyIdentifierFromString.proto */
#if !defined(__Pyx_PyIdentifier_FromString)
#if PY_MAJOR_VERSION < 3
  #define __Pyx_PyIdentifier_FromString(s) PyString_FromString(s)
#else
  #define __Pyx_PyIdentifier_FromString(s) PyUnicode_FromString(s)
#endif
#endif

/* ModuleImport.proto */
static PyObject *__Pyx_ImportModule(const char *name);

/* TypeImport.proto */
static PyTypeObject *__Pyx_ImportType(const char *module_name, const char *class_name, size_t size, int strict);

/* InitStrings.proto */
static int __Pyx_InitStrings(__Pyx_StringTabEntry *t);


/* Module declarations from 'cpython.buffer' */

/* Module declarations from 'libc.string' */

/* Module declarations from 'libc.stdio' */

/* Module declarations from '__builtin__' */

/* Module declarations from 'cpython.type' */
static PyTypeObject *__pyx_ptype_7cpython_4type_type = 0;

/* Module declarations from 'cpython' */

/* Module declarations from 'cpython.object' */

/* Module declarations from 'cpython.ref' */

/* Module declarations from 'libc.stdlib' */

/* Module declarations from 'numpy' */

/* Module declarations from 'numpy' */
static PyTypeObject *__pyx_ptype_5numpy_dtype = 0;
static PyTypeObject *__pyx_ptype_5numpy_flatiter = 0;
static PyTypeObject *__pyx_ptype_5numpy_broadcast = 0;
static PyTypeObject *__pyx_ptype_5numpy_ndarray = 0;
static PyTypeObject *__pyx_ptype_5numpy_ufunc = 0;
static CYTHON_INLINE char *__pyx_f_5numpy__util_dtypestring(PyArray_Descr *, char *, char *, int *); /*proto*/

/* Module declarations from 'lib.mask.gpu_mv' */
static __Pyx_TypeInfo __Pyx_TypeInfo_nn___pyx_t_5numpy_float32_t = { "float32_t", NULL, sizeof(__pyx_t_5numpy_float32_t), { 0 }, 0, 'R', 0, 0 };
static __Pyx_TypeInfo __Pyx_TypeInfo_nn___pyx_t_5numpy_int32_t = { "int32_t", NULL, sizeof(__pyx_t_5numpy_int32_t), { 0 }, 0, IS_UNSIGNED(__pyx_t_5numpy_int32_t) ? 'U' : 'I', IS_UNSIGNED(__pyx_t_5numpy_int32_t), 0 };
#define __Pyx_MODULE_NAME "lib.mask.gpu_mv"
int __pyx_module_is_main_lib__mask__gpu_mv = 0;

/* Implementation of 'lib.mask.gpu_mv' */
static PyObject *__pyx_builtin_ValueError;
static PyObject *__pyx_builtin_range;
static PyObject *__pyx_builtin_RuntimeError;
static const char __pyx_k_mv[] = "mv";
static const char __pyx_k_np[] = "np";
static const char __pyx_k_main[] = "__main__";
static const char __pyx_k_test[] = "__test__";
static const char __pyx_k_dtype[] = "dtype";
static const char __pyx_k_int32[] = "int32";
static const char __pyx_k_numpy[] = "numpy";
static const char __pyx_k_range[] = "range";
static const char __pyx_k_zeros[] = "zeros";
static const char __pyx_k_import[] = "__import__";
static const char __pyx_k_float32[] = "float32";
static const char __pyx_k_all_boxes[] = "all_boxes";
static const char __pyx_k_all_masks[] = "all_masks";
static const char __pyx_k_boxes_dim[] = "boxes_dim";
static const char __pyx_k_device_id[] = "device_id";
static const char __pyx_k_mask_size[] = "mask_size";
static const char __pyx_k_ValueError[] = "ValueError";
static const char __pyx_k_result_box[] = "result_box";
static const char __pyx_k_result_num[] = "result_num";
static const char __pyx_k_all_box_num[] = "all_box_num";
static const char __pyx_k_image_width[] = "image_width";
static const char __pyx_k_result_mask[] = "result_mask";
static const char __pyx_k_RuntimeError[] = "RuntimeError";
static const char __pyx_k_image_height[] = "image_height";
static const char __pyx_k_binary_thresh[] = "binary_thresh";
static const char __pyx_k_candidate_num[] = "candidate_num";
static const char __pyx_k_candidate_inds[] = "candidate_inds";
static const char __pyx_k_candidate_start[] = "candidate_start";
static const char __pyx_k_lib_mask_gpu_mv[] = "lib.mask.gpu_mv";
static const char __pyx_k_candidate_weights[] = "candidate_weights";
static const char __pyx_k_ndarray_is_not_C_contiguous[] = "ndarray is not C contiguous";
static const char __pyx_k_D_v_haoq_github_fcis_prerelease[] = "D:\\v-haoq\\github\\fcis_prerelease\\lib\\mask\\gpu_mv.pyx";
static const char __pyx_k_unknown_dtype_code_in_numpy_pxd[] = "unknown dtype code in numpy.pxd (%d)";
static const char __pyx_k_Format_string_allocated_too_shor[] = "Format string allocated too short, see comment in numpy.pxd";
static const char __pyx_k_Non_native_byte_order_not_suppor[] = "Non-native byte order not supported";
static const char __pyx_k_ndarray_is_not_Fortran_contiguou[] = "ndarray is not Fortran contiguous";
static const char __pyx_k_Format_string_allocated_too_shor_2[] = "Format string allocated too short.";
static PyObject *__pyx_kp_s_D_v_haoq_github_fcis_prerelease;
static PyObject *__pyx_kp_u_Format_string_allocated_too_shor;
static PyObject *__pyx_kp_u_Format_string_allocated_too_shor_2;
static PyObject *__pyx_kp_u_Non_native_byte_order_not_suppor;
static PyObject *__pyx_n_s_RuntimeError;
static PyObject *__pyx_n_s_ValueError;
static PyObject *__pyx_n_s_all_box_num;
static PyObject *__pyx_n_s_all_boxes;
static PyObject *__pyx_n_s_all_masks;
static PyObject *__pyx_n_s_binary_thresh;
static PyObject *__pyx_n_s_boxes_dim;
static PyObject *__pyx_n_s_candidate_inds;
static PyObject *__pyx_n_s_candidate_num;
static PyObject *__pyx_n_s_candidate_start;
static PyObject *__pyx_n_s_candidate_weights;
static PyObject *__pyx_n_s_device_id;
static PyObject *__pyx_n_s_dtype;
static PyObject *__pyx_n_s_float32;
static PyObject *__pyx_n_s_image_height;
static PyObject *__pyx_n_s_image_width;
static PyObject *__pyx_n_s_import;
static PyObject *__pyx_n_s_int32;
static PyObject *__pyx_n_s_lib_mask_gpu_mv;
static PyObject *__pyx_n_s_main;
static PyObject *__pyx_n_s_mask_size;
static PyObject *__pyx_n_s_mv;
static PyObject *__pyx_kp_u_ndarray_is_not_C_contiguous;
static PyObject *__pyx_kp_u_ndarray_is_not_Fortran_contiguou;
static PyObject *__pyx_n_s_np;
static PyObject *__pyx_n_s_numpy;
static PyObject *__pyx_n_s_range;
static PyObject *__pyx_n_s_result_box;
static PyObject *__pyx_n_s_result_mask;
static PyObject *__pyx_n_s_result_num;
static PyObject *__pyx_n_s_test;
static PyObject *__pyx_kp_u_unknown_dtype_code_in_numpy_pxd;
static PyObject *__pyx_n_s_zeros;
static PyObject *__pyx_pf_3lib_4mask_6gpu_mv_mv(CYTHON_UNUSED PyObject *__pyx_self, PyArrayObject *__pyx_v_all_boxes, PyArrayObject *__pyx_v_all_masks, PyArrayObject *__pyx_v_candidate_inds, PyArrayObject *__pyx_v_candidate_start, PyArrayObject *__pyx_v_candidate_weights, __pyx_t_5numpy_float32_t __pyx_v_binary_thresh, __pyx_t_5numpy_int32_t __pyx_v_image_height, __pyx_t_5numpy_int32_t __pyx_v_image_width, __pyx_t_5numpy_int32_t __pyx_v_device_id); /* proto */
static int __pyx_pf_5numpy_7ndarray___getbuffer__(PyArrayObject *__pyx_v_self, Py_buffer *__pyx_v_info, int __pyx_v_flags); /* proto */
static void __pyx_pf_5numpy_7ndarray_2__releasebuffer__(PyArrayObject *__pyx_v_self, Py_buffer *__pyx_v_info); /* proto */
static PyObject *__pyx_int_1;
static PyObject *__pyx_tuple_;
static PyObject *__pyx_tuple__2;
static PyObject *__pyx_tuple__3;
static PyObject *__pyx_tuple__4;
static PyObject *__pyx_tuple__5;
static PyObject *__pyx_tuple__6;
static PyObject *__pyx_tuple__7;
static PyObject *__pyx_codeobj__8;

/* "lib/mask/gpu_mv.pyx":19
 * # masks: n * 1 * 21 * 21
 * # scores: n * 21
 * def mv(np.ndarray[np.float32_t, ndim=2] all_boxes,             # <<<<<<<<<<<<<<
 *                 np.ndarray[np.float32_t, ndim=4] all_masks,
 *                 np.ndarray[np.int32_t, ndim=1] candidate_inds,
 */

/* Python wrapper */
static PyObject *__pyx_pw_3lib_4mask_6gpu_mv_1mv(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyMethodDef __pyx_mdef_3lib_4mask_6gpu_mv_1mv = {"mv", (PyCFunction)__pyx_pw_3lib_4mask_6gpu_mv_1mv, METH_VARARGS|METH_KEYWORDS, 0};
static PyObject *__pyx_pw_3lib_4mask_6gpu_mv_1mv(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyArrayObject *__pyx_v_all_boxes = 0;
  PyArrayObject *__pyx_v_all_masks = 0;
  PyArrayObject *__pyx_v_candidate_inds = 0;
  PyArrayObject *__pyx_v_candidate_start = 0;
  PyArrayObject *__pyx_v_candidate_weights = 0;
  __pyx_t_5numpy_float32_t __pyx_v_binary_thresh;
  __pyx_t_5numpy_int32_t __pyx_v_image_height;
  __pyx_t_5numpy_int32_t __pyx_v_image_width;
  __pyx_t_5numpy_int32_t __pyx_v_device_id;
  PyObject *__pyx_r = 0;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("mv (wrapper)", 0);
  {
    static PyObject **__pyx_pyargnames[] = {&__pyx_n_s_all_boxes,&__pyx_n_s_all_masks,&__pyx_n_s_candidate_inds,&__pyx_n_s_candidate_start,&__pyx_n_s_candidate_weights,&__pyx_n_s_binary_thresh,&__pyx_n_s_image_height,&__pyx_n_s_image_width,&__pyx_n_s_device_id,0};
    PyObject* values[9] = {0,0,0,0,0,0,0,0,0};
    if (unlikely(__pyx_kwds)) {
      Py_ssize_t kw_args;
      const Py_ssize_t pos_args = PyTuple_GET_SIZE(__pyx_args);
      switch (pos_args) {
        case  9: values[8] = PyTuple_GET_ITEM(__pyx_args, 8);
        case  8: values[7] = PyTuple_GET_ITEM(__pyx_args, 7);
        case  7: values[6] = PyTuple_GET_ITEM(__pyx_args, 6);
        case  6: values[5] = PyTuple_GET_ITEM(__pyx_args, 5);
        case  5: values[4] = PyTuple_GET_ITEM(__pyx_args, 4);
        case  4: values[3] = PyTuple_GET_ITEM(__pyx_args, 3);
        case  3: values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
        case  2: values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
        case  1: values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
        case  0: break;
        default: goto __pyx_L5_argtuple_error;
      }
      kw_args = PyDict_Size(__pyx_kwds);
      switch (pos_args) {
        case  0:
        if (likely((values[0] = PyDict_GetItem(__pyx_kwds, __pyx_n_s_all_boxes)) != 0)) kw_args--;
        else goto __pyx_L5_argtuple_error;
        case  1:
        if (likely((values[1] = PyDict_GetItem(__pyx_kwds, __pyx_n_s_all_masks)) != 0)) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("mv", 0, 8, 9, 1); __PYX_ERR(0, 19, __pyx_L3_error)
        }
        case  2:
        if (likely((values[2] = PyDict_GetItem(__pyx_kwds, __pyx_n_s_candidate_inds)) != 0)) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("mv", 0, 8, 9, 2); __PYX_ERR(0, 19, __pyx_L3_error)
        }
        case  3:
        if (likely((values[3] = PyDict_GetItem(__pyx_kwds, __pyx_n_s_candidate_start)) != 0)) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("mv", 0, 8, 9, 3); __PYX_ERR(0, 19, __pyx_L3_error)
        }
        case  4:
        if (likely((values[4] = PyDict_GetItem(__pyx_kwds, __pyx_n_s_candidate_weights)) != 0)) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("mv", 0, 8, 9, 4); __PYX_ERR(0, 19, __pyx_L3_error)
        }
        case  5:
        if (likely((values[5] = PyDict_GetItem(__pyx_kwds, __pyx_n_s_binary_thresh)) != 0)) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("mv", 0, 8, 9, 5); __PYX_ERR(0, 19, __pyx_L3_error)
        }
        case  6:
        if (likely((values[6] = PyDict_GetItem(__pyx_kwds, __pyx_n_s_image_height)) != 0)) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("mv", 0, 8, 9, 6); __PYX_ERR(0, 19, __pyx_L3_error)
        }
        case  7:
        if (likely((values[7] = PyDict_GetItem(__pyx_kwds, __pyx_n_s_image_width)) != 0)) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("mv", 0, 8, 9, 7); __PYX_ERR(0, 19, __pyx_L3_error)
        }
        case  8:
        if (kw_args > 0) {
          PyObject* value = PyDict_GetItem(__pyx_kwds, __pyx_n_s_device_id);
          if (value) { values[8] = value; kw_args--; }
        }
      }
      if (unlikely(kw_args > 0)) {
        if (unlikely(__Pyx_ParseOptionalKeywords(__pyx_kwds, __pyx_pyargnames, 0, values, pos_args, "mv") < 0)) __PYX_ERR(0, 19, __pyx_L3_error)
      }
    } else {
      switch (PyTuple_GET_SIZE(__pyx_args)) {
        case  9: values[8] = PyTuple_GET_ITEM(__pyx_args, 8);
        case  8: values[7] = PyTuple_GET_ITEM(__pyx_args, 7);
        values[6] = PyTuple_GET_ITEM(__pyx_args, 6);
        values[5] = PyTuple_GET_ITEM(__pyx_args, 5);
        values[4] = PyTuple_GET_ITEM(__pyx_args, 4);
        values[3] = PyTuple_GET_ITEM(__pyx_args, 3);
        values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
        values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
        values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
        break;
        default: goto __pyx_L5_argtuple_error;
      }
    }
    __pyx_v_all_boxes = ((PyArrayObject *)values[0]);
    __pyx_v_all_masks = ((PyArrayObject *)values[1]);
    __pyx_v_candidate_inds = ((PyArrayObject *)values[2]);
    __pyx_v_candidate_start = ((PyArrayObject *)values[3]);
    __pyx_v_candidate_weights = ((PyArrayObject *)values[4]);
    __pyx_v_binary_thresh = __pyx_PyFloat_AsFloat(values[5]); if (unlikely((__pyx_v_binary_thresh == (npy_float32)-1) && PyErr_Occurred())) __PYX_ERR(0, 24, __pyx_L3_error)
    __pyx_v_image_height = __Pyx_PyInt_As_npy_int32(values[6]); if (unlikely((__pyx_v_image_height == (npy_int32)-1) && PyErr_Occurred())) __PYX_ERR(0, 25, __pyx_L3_error)
    __pyx_v_image_width = __Pyx_PyInt_As_npy_int32(values[7]); if (unlikely((__pyx_v_image_width == (npy_int32)-1) && PyErr_Occurred())) __PYX_ERR(0, 26, __pyx_L3_error)
    if (values[8]) {
      __pyx_v_device_id = __Pyx_PyInt_As_npy_int32(values[8]); if (unlikely((__pyx_v_device_id == (npy_int32)-1) && PyErr_Occurred())) __PYX_ERR(0, 27, __pyx_L3_error)
    } else {
      __pyx_v_device_id = ((__pyx_t_5numpy_int32_t)0);
    }
  }
  goto __pyx_L4_argument_unpacking_done;
  __pyx_L5_argtuple_error:;
  __Pyx_RaiseArgtupleInvalid("mv", 0, 8, 9, PyTuple_GET_SIZE(__pyx_args)); __PYX_ERR(0, 19, __pyx_L3_error)
  __pyx_L3_error:;
  __Pyx_AddTraceback("lib.mask.gpu_mv.mv", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __Pyx_RefNannyFinishContext();
  return NULL;
  __pyx_L4_argument_unpacking_done:;
  if (unlikely(!__Pyx_ArgTypeTest(((PyObject *)__pyx_v_all_boxes), __pyx_ptype_5numpy_ndarray, 1, "all_boxes", 0))) __PYX_ERR(0, 19, __pyx_L1_error)
  if (unlikely(!__Pyx_ArgTypeTest(((PyObject *)__pyx_v_all_masks), __pyx_ptype_5numpy_ndarray, 1, "all_masks", 0))) __PYX_ERR(0, 20, __pyx_L1_error)
  if (unlikely(!__Pyx_ArgTypeTest(((PyObject *)__pyx_v_candidate_inds), __pyx_ptype_5numpy_ndarray, 1, "candidate_inds", 0))) __PYX_ERR(0, 21, __pyx_L1_error)
  if (unlikely(!__Pyx_ArgTypeTest(((PyObject *)__pyx_v_candidate_start), __pyx_ptype_5numpy_ndarray, 1, "candidate_start", 0))) __PYX_ERR(0, 22, __pyx_L1_error)
  if (unlikely(!__Pyx_ArgTypeTest(((PyObject *)__pyx_v_candidate_weights), __pyx_ptype_5numpy_ndarray, 1, "candidate_weights", 0))) __PYX_ERR(0, 23, __pyx_L1_error)
  __pyx_r = __pyx_pf_3lib_4mask_6gpu_mv_mv(__pyx_self, __pyx_v_all_boxes, __pyx_v_all_masks, __pyx_v_candidate_inds, __pyx_v_candidate_start, __pyx_v_candidate_weights, __pyx_v_binary_thresh, __pyx_v_image_height, __pyx_v_image_width, __pyx_v_device_id);

  /* function exit code */
  goto __pyx_L0;
  __pyx_L1_error:;
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

static PyObject *__pyx_pf_3lib_4mask_6gpu_mv_mv(CYTHON_UNUSED PyObject *__pyx_self, PyArrayObject *__pyx_v_all_boxes, PyArrayObject *__pyx_v_all_masks, PyArrayObject *__pyx_v_candidate_inds, PyArrayObject *__pyx_v_candidate_start, PyArrayObject *__pyx_v_candidate_weights, __pyx_t_5numpy_float32_t __pyx_v_binary_thresh, __pyx_t_5numpy_int32_t __pyx_v_image_height, __pyx_t_5numpy_int32_t __pyx_v_image_width, __pyx_t_5numpy_int32_t __pyx_v_device_id) {
  int __pyx_v_all_box_num;
  int __pyx_v_boxes_dim;
  int __pyx_v_mask_size;
  int __pyx_v_candidate_num;
  int __pyx_v_result_num;
  PyArrayObject *__pyx_v_result_mask = 0;
  PyArrayObject *__pyx_v_result_box = 0;
  __Pyx_LocalBuf_ND __pyx_pybuffernd_all_boxes;
  __Pyx_Buffer __pyx_pybuffer_all_boxes;
  __Pyx_LocalBuf_ND __pyx_pybuffernd_all_masks;
  __Pyx_Buffer __pyx_pybuffer_all_masks;
  __Pyx_LocalBuf_ND __pyx_pybuffernd_candidate_inds;
  __Pyx_Buffer __pyx_pybuffer_candidate_inds;
  __Pyx_LocalBuf_ND __pyx_pybuffernd_candidate_start;
  __Pyx_Buffer __pyx_pybuffer_candidate_start;
  __Pyx_LocalBuf_ND __pyx_pybuffernd_candidate_weights;
  __Pyx_Buffer __pyx_pybuffer_candidate_weights;
  __Pyx_LocalBuf_ND __pyx_pybuffernd_result_box;
  __Pyx_Buffer __pyx_pybuffer_result_box;
  __Pyx_LocalBuf_ND __pyx_pybuffernd_result_mask;
  __Pyx_Buffer __pyx_pybuffer_result_mask;
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  PyObject *__pyx_t_2 = NULL;
  PyObject *__pyx_t_3 = NULL;
  PyObject *__pyx_t_4 = NULL;
  PyObject *__pyx_t_5 = NULL;
  PyArrayObject *__pyx_t_6 = NULL;
  PyArrayObject *__pyx_t_7 = NULL;
  int __pyx_t_8;
  Py_ssize_t __pyx_t_9;
  Py_ssize_t __pyx_t_10;
  int __pyx_t_11;
  Py_ssize_t __pyx_t_12;
  Py_ssize_t __pyx_t_13;
  Py_ssize_t __pyx_t_14;
  Py_ssize_t __pyx_t_15;
  Py_ssize_t __pyx_t_16;
  Py_ssize_t __pyx_t_17;
  Py_ssize_t __pyx_t_18;
  Py_ssize_t __pyx_t_19;
  Py_ssize_t __pyx_t_20;
  Py_ssize_t __pyx_t_21;
  Py_ssize_t __pyx_t_22;
  Py_ssize_t __pyx_t_23;
  Py_ssize_t __pyx_t_24;
  __Pyx_RefNannySetupContext("mv", 0);
  __pyx_pybuffer_result_mask.pybuffer.buf = NULL;
  __pyx_pybuffer_result_mask.refcount = 0;
  __pyx_pybuffernd_result_mask.data = NULL;
  __pyx_pybuffernd_result_mask.rcbuffer = &__pyx_pybuffer_result_mask;
  __pyx_pybuffer_result_box.pybuffer.buf = NULL;
  __pyx_pybuffer_result_box.refcount = 0;
  __pyx_pybuffernd_result_box.data = NULL;
  __pyx_pybuffernd_result_box.rcbuffer = &__pyx_pybuffer_result_box;
  __pyx_pybuffer_all_boxes.pybuffer.buf = NULL;
  __pyx_pybuffer_all_boxes.refcount = 0;
  __pyx_pybuffernd_all_boxes.data = NULL;
  __pyx_pybuffernd_all_boxes.rcbuffer = &__pyx_pybuffer_all_boxes;
  __pyx_pybuffer_all_masks.pybuffer.buf = NULL;
  __pyx_pybuffer_all_masks.refcount = 0;
  __pyx_pybuffernd_all_masks.data = NULL;
  __pyx_pybuffernd_all_masks.rcbuffer = &__pyx_pybuffer_all_masks;
  __pyx_pybuffer_candidate_inds.pybuffer.buf = NULL;
  __pyx_pybuffer_candidate_inds.refcount = 0;
  __pyx_pybuffernd_candidate_inds.data = NULL;
  __pyx_pybuffernd_candidate_inds.rcbuffer = &__pyx_pybuffer_candidate_inds;
  __pyx_pybuffer_candidate_start.pybuffer.buf = NULL;
  __pyx_pybuffer_candidate_start.refcount = 0;
  __pyx_pybuffernd_candidate_start.data = NULL;
  __pyx_pybuffernd_candidate_start.rcbuffer = &__pyx_pybuffer_candidate_start;
  __pyx_pybuffer_candidate_weights.pybuffer.buf = NULL;
  __pyx_pybuffer_candidate_weights.refcount = 0;
  __pyx_pybuffernd_candidate_weights.data = NULL;
  __pyx_pybuffernd_candidate_weights.rcbuffer = &__pyx_pybuffer_candidate_weights;
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_pybuffernd_all_boxes.rcbuffer->pybuffer, (PyObject*)__pyx_v_all_boxes, &__Pyx_TypeInfo_nn___pyx_t_5numpy_float32_t, PyBUF_FORMAT| PyBUF_STRIDES, 2, 0, __pyx_stack) == -1)) __PYX_ERR(0, 19, __pyx_L1_error)
  }
  __pyx_pybuffernd_all_boxes.diminfo[0].strides = __pyx_pybuffernd_all_boxes.rcbuffer->pybuffer.strides[0]; __pyx_pybuffernd_all_boxes.diminfo[0].shape = __pyx_pybuffernd_all_boxes.rcbuffer->pybuffer.shape[0]; __pyx_pybuffernd_all_boxes.diminfo[1].strides = __pyx_pybuffernd_all_boxes.rcbuffer->pybuffer.strides[1]; __pyx_pybuffernd_all_boxes.diminfo[1].shape = __pyx_pybuffernd_all_boxes.rcbuffer->pybuffer.shape[1];
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_pybuffernd_all_masks.rcbuffer->pybuffer, (PyObject*)__pyx_v_all_masks, &__Pyx_TypeInfo_nn___pyx_t_5numpy_float32_t, PyBUF_FORMAT| PyBUF_STRIDES, 4, 0, __pyx_stack) == -1)) __PYX_ERR(0, 19, __pyx_L1_error)
  }
  __pyx_pybuffernd_all_masks.diminfo[0].strides = __pyx_pybuffernd_all_masks.rcbuffer->pybuffer.strides[0]; __pyx_pybuffernd_all_masks.diminfo[0].shape = __pyx_pybuffernd_all_masks.rcbuffer->pybuffer.shape[0]; __pyx_pybuffernd_all_masks.diminfo[1].strides = __pyx_pybuffernd_all_masks.rcbuffer->pybuffer.strides[1]; __pyx_pybuffernd_all_masks.diminfo[1].shape = __pyx_pybuffernd_all_masks.rcbuffer->pybuffer.shape[1]; __pyx_pybuffernd_all_masks.diminfo[2].strides = __pyx_pybuffernd_all_masks.rcbuffer->pybuffer.strides[2]; __pyx_pybuffernd_all_masks.diminfo[2].shape = __pyx_pybuffernd_all_masks.rcbuffer->pybuffer.shape[2]; __pyx_pybuffernd_all_masks.diminfo[3].strides = __pyx_pybuffernd_all_masks.rcbuffer->pybuffer.strides[3]; __pyx_pybuffernd_all_masks.diminfo[3].shape = __pyx_pybuffernd_all_masks.rcbuffer->pybuffer.shape[3];
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_pybuffernd_candidate_inds.rcbuffer->pybuffer, (PyObject*)__pyx_v_candidate_inds, &__Pyx_TypeInfo_nn___pyx_t_5numpy_int32_t, PyBUF_FORMAT| PyBUF_STRIDES, 1, 0, __pyx_stack) == -1)) __PYX_ERR(0, 19, __pyx_L1_error)
  }
  __pyx_pybuffernd_candidate_inds.diminfo[0].strides = __pyx_pybuffernd_candidate_inds.rcbuffer->pybuffer.strides[0]; __pyx_pybuffernd_candidate_inds.diminfo[0].shape = __pyx_pybuffernd_candidate_inds.rcbuffer->pybuffer.shape[0];
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_pybuffernd_candidate_start.rcbuffer->pybuffer, (PyObject*)__pyx_v_candidate_start, &__Pyx_TypeInfo_nn___pyx_t_5numpy_int32_t, PyBUF_FORMAT| PyBUF_STRIDES, 1, 0, __pyx_stack) == -1)) __PYX_ERR(0, 19, __pyx_L1_error)
  }
  __pyx_pybuffernd_candidate_start.diminfo[0].strides = __pyx_pybuffernd_candidate_start.rcbuffer->pybuffer.strides[0]; __pyx_pybuffernd_candidate_start.diminfo[0].shape = __pyx_pybuffernd_candidate_start.rcbuffer->pybuffer.shape[0];
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_pybuffernd_candidate_weights.rcbuffer->pybuffer, (PyObject*)__pyx_v_candidate_weights, &__Pyx_TypeInfo_nn___pyx_t_5numpy_float32_t, PyBUF_FORMAT| PyBUF_STRIDES, 1, 0, __pyx_stack) == -1)) __PYX_ERR(0, 19, __pyx_L1_error)
  }
  __pyx_pybuffernd_candidate_weights.diminfo[0].strides = __pyx_pybuffernd_candidate_weights.rcbuffer->pybuffer.strides[0]; __pyx_pybuffernd_candidate_weights.diminfo[0].shape = __pyx_pybuffernd_candidate_weights.rcbuffer->pybuffer.shape[0];

  /* "lib/mask/gpu_mv.pyx":28
 *                 np.int32_t image_width,
 *                 np.int32_t device_id = 0):
 *     cdef int all_box_num = all_boxes.shape[0]             # <<<<<<<<<<<<<<
 *     cdef int boxes_dim = all_boxes.shape[1]
 *     cdef int mask_size = all_masks.shape[3]
 */
  __pyx_v_all_box_num = (__pyx_v_all_boxes->dimensions[0]);

  /* "lib/mask/gpu_mv.pyx":29
 *                 np.int32_t device_id = 0):
 *     cdef int all_box_num = all_boxes.shape[0]
 *     cdef int boxes_dim = all_boxes.shape[1]             # <<<<<<<<<<<<<<
 *     cdef int mask_size = all_masks.shape[3]
 *     cdef int candidate_num = candidate_inds.shape[0]
 */
  __pyx_v_boxes_dim = (__pyx_v_all_boxes->dimensions[1]);

  /* "lib/mask/gpu_mv.pyx":30
 *     cdef int all_box_num = all_boxes.shape[0]
 *     cdef int boxes_dim = all_boxes.shape[1]
 *     cdef int mask_size = all_masks.shape[3]             # <<<<<<<<<<<<<<
 *     cdef int candidate_num = candidate_inds.shape[0]
 *     cdef int result_num = candidate_start.shape[0]
 */
  __pyx_v_mask_size = (__pyx_v_all_masks->dimensions[3]);

  /* "lib/mask/gpu_mv.pyx":31
 *     cdef int boxes_dim = all_boxes.shape[1]
 *     cdef int mask_size = all_masks.shape[3]
 *     cdef int candidate_num = candidate_inds.shape[0]             # <<<<<<<<<<<<<<
 *     cdef int result_num = candidate_start.shape[0]
 *     cdef np.ndarray[np.float32_t, ndim=4] \
 */
  __pyx_v_candidate_num = (__pyx_v_candidate_inds->dimensions[0]);

  /* "lib/mask/gpu_mv.pyx":32
 *     cdef int mask_size = all_masks.shape[3]
 *     cdef int candidate_num = candidate_inds.shape[0]
 *     cdef int result_num = candidate_start.shape[0]             # <<<<<<<<<<<<<<
 *     cdef np.ndarray[np.float32_t, ndim=4] \
 *         result_mask = np.zeros((result_num, 1, all_masks.shape[2], all_masks.shape[3]), dtype=np.float32)
 */
  __pyx_v_result_num = (__pyx_v_candidate_start->dimensions[0]);

  /* "lib/mask/gpu_mv.pyx":34
 *     cdef int result_num = candidate_start.shape[0]
 *     cdef np.ndarray[np.float32_t, ndim=4] \
 *         result_mask = np.zeros((result_num, 1, all_masks.shape[2], all_masks.shape[3]), dtype=np.float32)             # <<<<<<<<<<<<<<
 *     cdef np.ndarray[np.int32_t, ndim=2] \
 *         result_box = np.zeros((result_num, boxes_dim), dtype=np.int32)
 */
  __pyx_t_1 = __Pyx_GetModuleGlobalName(__pyx_n_s_np); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 34, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_2 = __Pyx_PyObject_GetAttrStr(__pyx_t_1, __pyx_n_s_zeros); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 34, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_2);
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_t_1 = __Pyx_PyInt_From_int(__pyx_v_result_num); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 34, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_3 = __Pyx_PyInt_From_Py_intptr_t((__pyx_v_all_masks->dimensions[2])); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 34, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_3);
  __pyx_t_4 = __Pyx_PyInt_From_Py_intptr_t((__pyx_v_all_masks->dimensions[3])); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 34, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_4);
  __pyx_t_5 = PyTuple_New(4); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 34, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_5);
  __Pyx_GIVEREF(__pyx_t_1);
  PyTuple_SET_ITEM(__pyx_t_5, 0, __pyx_t_1);
  __Pyx_INCREF(__pyx_int_1);
  __Pyx_GIVEREF(__pyx_int_1);
  PyTuple_SET_ITEM(__pyx_t_5, 1, __pyx_int_1);
  __Pyx_GIVEREF(__pyx_t_3);
  PyTuple_SET_ITEM(__pyx_t_5, 2, __pyx_t_3);
  __Pyx_GIVEREF(__pyx_t_4);
  PyTuple_SET_ITEM(__pyx_t_5, 3, __pyx_t_4);
  __pyx_t_1 = 0;
  __pyx_t_3 = 0;
  __pyx_t_4 = 0;
  __pyx_t_4 = PyTuple_New(1); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 34, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_4);
  __Pyx_GIVEREF(__pyx_t_5);
  PyTuple_SET_ITEM(__pyx_t_4, 0, __pyx_t_5);
  __pyx_t_5 = 0;
  __pyx_t_5 = PyDict_New(); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 34, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_5);
  __pyx_t_3 = __Pyx_GetModuleGlobalName(__pyx_n_s_np); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 34, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_3);
  __pyx_t_1 = __Pyx_PyObject_GetAttrStr(__pyx_t_3, __pyx_n_s_float32); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 34, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
  if (PyDict_SetItem(__pyx_t_5, __pyx_n_s_dtype, __pyx_t_1) < 0) __PYX_ERR(0, 34, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_t_1 = __Pyx_PyObject_Call(__pyx_t_2, __pyx_t_4, __pyx_t_5); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 34, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
  __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
  if (!(likely(((__pyx_t_1) == Py_None) || likely(__Pyx_TypeTest(__pyx_t_1, __pyx_ptype_5numpy_ndarray))))) __PYX_ERR(0, 34, __pyx_L1_error)
  __pyx_t_6 = ((PyArrayObject *)__pyx_t_1);
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_pybuffernd_result_mask.rcbuffer->pybuffer, (PyObject*)__pyx_t_6, &__Pyx_TypeInfo_nn___pyx_t_5numpy_float32_t, PyBUF_FORMAT| PyBUF_STRIDES, 4, 0, __pyx_stack) == -1)) {
      __pyx_v_result_mask = ((PyArrayObject *)Py_None); __Pyx_INCREF(Py_None); __pyx_pybuffernd_result_mask.rcbuffer->pybuffer.buf = NULL;
      __PYX_ERR(0, 33, __pyx_L1_error)
    } else {__pyx_pybuffernd_result_mask.diminfo[0].strides = __pyx_pybuffernd_result_mask.rcbuffer->pybuffer.strides[0]; __pyx_pybuffernd_result_mask.diminfo[0].shape = __pyx_pybuffernd_result_mask.rcbuffer->pybuffer.shape[0]; __pyx_pybuffernd_result_mask.diminfo[1].strides = __pyx_pybuffernd_result_mask.rcbuffer->pybuffer.strides[1]; __pyx_pybuffernd_result_mask.diminfo[1].shape = __pyx_pybuffernd_result_mask.rcbuffer->pybuffer.shape[1]; __pyx_pybuffernd_result_mask.diminfo[2].strides = __pyx_pybuffernd_result_mask.rcbuffer->pybuffer.strides[2]; __pyx_pybuffernd_result_mask.diminfo[2].shape = __pyx_pybuffernd_result_mask.rcbuffer->pybuffer.shape[2]; __pyx_pybuffernd_result_mask.diminfo[3].strides = __pyx_pybuffernd_result_mask.rcbuffer->pybuffer.strides[3]; __pyx_pybuffernd_result_mask.diminfo[3].shape = __pyx_pybuffernd_result_mask.rcbuffer->pybuffer.shape[3];
    }
  }
  __pyx_t_6 = 0;
  __pyx_v_result_mask = ((PyArrayObject *)__pyx_t_1);
  __pyx_t_1 = 0;

  /* "lib/mask/gpu_mv.pyx":36
 *         result_mask = np.zeros((result_num, 1, all_masks.shape[2], all_masks.shape[3]), dtype=np.float32)
 *     cdef np.ndarray[np.int32_t, ndim=2] \
 *         result_box = np.zeros((result_num, boxes_dim), dtype=np.int32)             # <<<<<<<<<<<<<<
 *     if all_boxes.shape[0] > 0:
 *         _mv(&all_boxes[0, 0], &all_masks[0, 0, 0, 0], all_box_num, &candidate_inds[0], &candidate_start[0], &candidate_weights[0], candidate_num, binary_thresh, image_height, image_width, boxes_dim, mask_size, candidate_start.shape[0], &result_mask[0,0,0,0], &result_box[0,0], device_id)
 */
  __pyx_t_1 = __Pyx_GetModuleGlobalName(__pyx_n_s_np); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 36, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_5 = __Pyx_PyObject_GetAttrStr(__pyx_t_1, __pyx_n_s_zeros); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 36, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_5);
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_t_1 = __Pyx_PyInt_From_int(__pyx_v_result_num); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 36, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_4 = __Pyx_PyInt_From_int(__pyx_v_boxes_dim); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 36, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_4);
  __pyx_t_2 = PyTuple_New(2); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 36, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_2);
  __Pyx_GIVEREF(__pyx_t_1);
  PyTuple_SET_ITEM(__pyx_t_2, 0, __pyx_t_1);
  __Pyx_GIVEREF(__pyx_t_4);
  PyTuple_SET_ITEM(__pyx_t_2, 1, __pyx_t_4);
  __pyx_t_1 = 0;
  __pyx_t_4 = 0;
  __pyx_t_4 = PyTuple_New(1); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 36, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_4);
  __Pyx_GIVEREF(__pyx_t_2);
  PyTuple_SET_ITEM(__pyx_t_4, 0, __pyx_t_2);
  __pyx_t_2 = 0;
  __pyx_t_2 = PyDict_New(); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 36, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_2);
  __pyx_t_1 = __Pyx_GetModuleGlobalName(__pyx_n_s_np); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 36, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_3 = __Pyx_PyObject_GetAttrStr(__pyx_t_1, __pyx_n_s_int32); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 36, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_3);
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  if (PyDict_SetItem(__pyx_t_2, __pyx_n_s_dtype, __pyx_t_3) < 0) __PYX_ERR(0, 36, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
  __pyx_t_3 = __Pyx_PyObject_Call(__pyx_t_5, __pyx_t_4, __pyx_t_2); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 36, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_3);
  __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
  __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  if (!(likely(((__pyx_t_3) == Py_None) || likely(__Pyx_TypeTest(__pyx_t_3, __pyx_ptype_5numpy_ndarray))))) __PYX_ERR(0, 36, __pyx_L1_error)
  __pyx_t_7 = ((PyArrayObject *)__pyx_t_3);
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_pybuffernd_result_box.rcbuffer->pybuffer, (PyObject*)__pyx_t_7, &__Pyx_TypeInfo_nn___pyx_t_5numpy_int32_t, PyBUF_FORMAT| PyBUF_STRIDES, 2, 0, __pyx_stack) == -1)) {
      __pyx_v_result_box = ((PyArrayObject *)Py_None); __Pyx_INCREF(Py_None); __pyx_pybuffernd_result_box.rcbuffer->pybuffer.buf = NULL;
      __PYX_ERR(0, 35, __pyx_L1_error)
    } else {__pyx_pybuffernd_result_box.diminfo[0].strides = __pyx_pybuffernd_result_box.rcbuffer->pybuffer.strides[0]; __pyx_pybuffernd_result_box.diminfo[0].shape = __pyx_pybuffernd_result_box.rcbuffer->pybuffer.shape[0]; __pyx_pybuffernd_result_box.diminfo[1].strides = __pyx_pybuffernd_result_box.rcbuffer->pybuffer.strides[1]; __pyx_pybuffernd_result_box.diminfo[1].shape = __pyx_pybuffernd_result_box.rcbuffer->pybuffer.shape[1];
    }
  }
  __pyx_t_7 = 0;
  __pyx_v_result_box = ((PyArrayObject *)__pyx_t_3);
  __pyx_t_3 = 0;

  /* "lib/mask/gpu_mv.pyx":37
 *     cdef np.ndarray[np.int32_t, ndim=2] \
 *         result_box = np.zeros((result_num, boxes_dim), dtype=np.int32)
 *     if all_boxes.shape[0] > 0:             # <<<<<<<<<<<<<<
 *         _mv(&all_boxes[0, 0], &all_masks[0, 0, 0, 0], all_box_num, &candidate_inds[0], &candidate_start[0], &candidate_weights[0], candidate_num, binary_thresh, image_height, image_width, boxes_dim, mask_size, candidate_start.shape[0], &result_mask[0,0,0,0], &result_box[0,0], device_id)
 *     return result_mask, result_box
 */
  __pyx_t_8 = (((__pyx_v_all_boxes->dimensions[0]) > 0) != 0);
  if (__pyx_t_8) {

    /* "lib/mask/gpu_mv.pyx":38
 *         result_box = np.zeros((result_num, boxes_dim), dtype=np.int32)
 *     if all_boxes.shape[0] > 0:
 *         _mv(&all_boxes[0, 0], &all_masks[0, 0, 0, 0], all_box_num, &candidate_inds[0], &candidate_start[0], &candidate_weights[0], candidate_num, binary_thresh, image_height, image_width, boxes_dim, mask_size, candidate_start.shape[0], &result_mask[0,0,0,0], &result_box[0,0], device_id)             # <<<<<<<<<<<<<<
 *     return result_mask, result_box
 */
    __pyx_t_9 = 0;
    __pyx_t_10 = 0;
    __pyx_t_11 = -1;
    if (__pyx_t_9 < 0) {
      __pyx_t_9 += __pyx_pybuffernd_all_boxes.diminfo[0].shape;
      if (unlikely(__pyx_t_9 < 0)) __pyx_t_11 = 0;
    } else if (unlikely(__pyx_t_9 >= __pyx_pybuffernd_all_boxes.diminfo[0].shape)) __pyx_t_11 = 0;
    if (__pyx_t_10 < 0) {
      __pyx_t_10 += __pyx_pybuffernd_all_boxes.diminfo[1].shape;
      if (unlikely(__pyx_t_10 < 0)) __pyx_t_11 = 1;
    } else if (unlikely(__pyx_t_10 >= __pyx_pybuffernd_all_boxes.diminfo[1].shape)) __pyx_t_11 = 1;
    if (unlikely(__pyx_t_11 != -1)) {
      __Pyx_RaiseBufferIndexError(__pyx_t_11);
      __PYX_ERR(0, 38, __pyx_L1_error)
    }
    __pyx_t_12 = 0;
    __pyx_t_13 = 0;
    __pyx_t_14 = 0;
    __pyx_t_15 = 0;
    __pyx_t_11 = -1;
    if (__pyx_t_12 < 0) {
      __pyx_t_12 += __pyx_pybuffernd_all_masks.diminfo[0].shape;
      if (unlikely(__pyx_t_12 < 0)) __pyx_t_11 = 0;
    } else if (unlikely(__pyx_t_12 >= __pyx_pybuffernd_all_masks.diminfo[0].shape)) __pyx_t_11 = 0;
    if (__pyx_t_13 < 0) {
      __pyx_t_13 += __pyx_pybuffernd_all_masks.diminfo[1].shape;
      if (unlikely(__pyx_t_13 < 0)) __pyx_t_11 = 1;
    } else if (unlikely(__pyx_t_13 >= __pyx_pybuffernd_all_masks.diminfo[1].shape)) __pyx_t_11 = 1;
    if (__pyx_t_14 < 0) {
      __pyx_t_14 += __pyx_pybuffernd_all_masks.diminfo[2].shape;
      if (unlikely(__pyx_t_14 < 0)) __pyx_t_11 = 2;
    } else if (unlikely(__pyx_t_14 >= __pyx_pybuffernd_all_masks.diminfo[2].shape)) __pyx_t_11 = 2;
    if (__pyx_t_15 < 0) {
      __pyx_t_15 += __pyx_pybuffernd_all_masks.diminfo[3].shape;
      if (unlikely(__pyx_t_15 < 0)) __pyx_t_11 = 3;
    } else if (unlikely(__pyx_t_15 >= __pyx_pybuffernd_all_masks.diminfo[3].shape)) __pyx_t_11 = 3;
    if (unlikely(__pyx_t_11 != -1)) {
      __Pyx_RaiseBufferIndexError(__pyx_t_11);
      __PYX_ERR(0, 38, __pyx_L1_error)
    }
    __pyx_t_16 = 0;
    __pyx_t_11 = -1;
    if (__pyx_t_16 < 0) {
      __pyx_t_16 += __pyx_pybuffernd_candidate_inds.diminfo[0].shape;
      if (unlikely(__pyx_t_16 < 0)) __pyx_t_11 = 0;
    } else if (unlikely(__pyx_t_16 >= __pyx_pybuffernd_candidate_inds.diminfo[0].shape)) __pyx_t_11 = 0;
    if (unlikely(__pyx_t_11 != -1)) {
      __Pyx_RaiseBufferIndexError(__pyx_t_11);
      __PYX_ERR(0, 38, __pyx_L1_error)
    }
    __pyx_t_17 = 0;
    __pyx_t_11 = -1;
    if (__pyx_t_17 < 0) {
      __pyx_t_17 += __pyx_pybuffernd_candidate_start.diminfo[0].shape;
      if (unlikely(__pyx_t_17 < 0)) __pyx_t_11 = 0;
    } else if (unlikely(__pyx_t_17 >= __pyx_pybuffernd_candidate_start.diminfo[0].shape)) __pyx_t_11 = 0;
    if (unlikely(__pyx_t_11 != -1)) {
      __Pyx_RaiseBufferIndexError(__pyx_t_11);
      __PYX_ERR(0, 38, __pyx_L1_error)
    }
    __pyx_t_18 = 0;
    __pyx_t_11 = -1;
    if (__pyx_t_18 < 0) {
      __pyx_t_18 += __pyx_pybuffernd_candidate_weights.diminfo[0].shape;
      if (unlikely(__pyx_t_18 < 0)) __pyx_t_11 = 0;
    } else if (unlikely(__pyx_t_18 >= __pyx_pybuffernd_candidate_weights.diminfo[0].shape)) __pyx_t_11 = 0;
    if (unlikely(__pyx_t_11 != -1)) {
      __Pyx_RaiseBufferIndexError(__pyx_t_11);
      __PYX_ERR(0, 38, __pyx_L1_error)
    }
    __pyx_t_19 = 0;
    __pyx_t_20 = 0;
    __pyx_t_21 = 0;
    __pyx_t_22 = 0;
    __pyx_t_11 = -1;
    if (__pyx_t_19 < 0) {
      __pyx_t_19 += __pyx_pybuffernd_result_mask.diminfo[0].shape;
      if (unlikely(__pyx_t_19 < 0)) __pyx_t_11 = 0;
    } else if (unlikely(__pyx_t_19 >= __pyx_pybuffernd_result_mask.diminfo[0].shape)) __pyx_t_11 = 0;
    if (__pyx_t_20 < 0) {
      __pyx_t_20 += __pyx_pybuffernd_result_mask.diminfo[1].shape;
      if (unlikely(__pyx_t_20 < 0)) __pyx_t_11 = 1;
    } else if (unlikely(__pyx_t_20 >= __pyx_pybuffernd_result_mask.diminfo[1].shape)) __pyx_t_11 = 1;
    if (__pyx_t_21 < 0) {
      __pyx_t_21 += __pyx_pybuffernd_result_mask.diminfo[2].shape;
      if (unlikely(__pyx_t_21 < 0)) __pyx_t_11 = 2;
    } else if (unlikely(__pyx_t_21 >= __pyx_pybuffernd_result_mask.diminfo[2].shape)) __pyx_t_11 = 2;
    if (__pyx_t_22 < 0) {
      __pyx_t_22 += __pyx_pybuffernd_result_mask.diminfo[3].shape;
      if (unlikely(__pyx_t_22 < 0)) __pyx_t_11 = 3;
    } else if (unlikely(__pyx_t_22 >= __pyx_pybuffernd_result_mask.diminfo[3].shape)) __pyx_t_11 = 3;
    if (unlikely(__pyx_t_11 != -1)) {
      __Pyx_RaiseBufferIndexError(__pyx_t_11);
      __PYX_ERR(0, 38, __pyx_L1_error)
    }
    __pyx_t_23 = 0;
    __pyx_t_24 = 0;
    __pyx_t_11 = -1;
    if (__pyx_t_23 < 0) {
      __pyx_t_23 += __pyx_pybuffernd_result_box.diminfo[0].shape;
      if (unlikely(__pyx_t_23 < 0)) __pyx_t_11 = 0;
    } else if (unlikely(__pyx_t_23 >= __pyx_pybuffernd_result_box.diminfo[0].shape)) __pyx_t_11 = 0;
    if (__pyx_t_24 < 0) {
      __pyx_t_24 += __pyx_pybuffernd_result_box.diminfo[1].shape;
      if (unlikely(__pyx_t_24 < 0)) __pyx_t_11 = 1;
    } else if (unlikely(__pyx_t_24 >= __pyx_pybuffernd_result_box.diminfo[1].shape)) __pyx_t_11 = 1;
    if (unlikely(__pyx_t_11 != -1)) {
      __Pyx_RaiseBufferIndexError(__pyx_t_11);
      __PYX_ERR(0, 38, __pyx_L1_error)
    }
    _mv((&(*__Pyx_BufPtrStrided2d(__pyx_t_5numpy_float32_t *, __pyx_pybuffernd_all_boxes.rcbuffer->pybuffer.buf, __pyx_t_9, __pyx_pybuffernd_all_boxes.diminfo[0].strides, __pyx_t_10, __pyx_pybuffernd_all_boxes.diminfo[1].strides))), (&(*__Pyx_BufPtrStrided4d(__pyx_t_5numpy_float32_t *, __pyx_pybuffernd_all_masks.rcbuffer->pybuffer.buf, __pyx_t_12, __pyx_pybuffernd_all_masks.diminfo[0].strides, __pyx_t_13, __pyx_pybuffernd_all_masks.diminfo[1].strides, __pyx_t_14, __pyx_pybuffernd_all_masks.diminfo[2].strides, __pyx_t_15, __pyx_pybuffernd_all_masks.diminfo[3].strides))), __pyx_v_all_box_num, (&(*__Pyx_BufPtrStrided1d(const int *, __pyx_pybuffernd_candidate_inds.rcbuffer->pybuffer.buf, __pyx_t_16, __pyx_pybuffernd_candidate_inds.diminfo[0].strides))), (&(*__Pyx_BufPtrStrided1d(const int *, __pyx_pybuffernd_candidate_start.rcbuffer->pybuffer.buf, __pyx_t_17, __pyx_pybuffernd_candidate_start.diminfo[0].strides))), (&(*__Pyx_BufPtrStrided1d(__pyx_t_5numpy_float32_t *, __pyx_pybuffernd_candidate_weights.rcbuffer->pybuffer.buf, __pyx_t_18, __pyx_pybuffernd_candidate_weights.diminfo[0].strides))), __pyx_v_candidate_num, __pyx_v_binary_thresh, __pyx_v_image_height, __pyx_v_image_width, __pyx_v_boxes_dim, __pyx_v_mask_size, (__pyx_v_candidate_start->dimensions[0]), (&(*__Pyx_BufPtrStrided4d(__pyx_t_5numpy_float32_t *, __pyx_pybuffernd_result_mask.rcbuffer->pybuffer.buf, __pyx_t_19, __pyx_pybuffernd_result_mask.diminfo[0].strides, __pyx_t_20, __pyx_pybuffernd_result_mask.diminfo[1].strides, __pyx_t_21, __pyx_pybuffernd_result_mask.diminfo[2].strides, __pyx_t_22, __pyx_pybuffernd_result_mask.diminfo[3].strides))), (&(*__Pyx_BufPtrStrided2d(int *, __pyx_pybuffernd_result_box.rcbuffer->pybuffer.buf, __pyx_t_23, __pyx_pybuffernd_result_box.diminfo[0].strides, __pyx_t_24, __pyx_pybuffernd_result_box.diminfo[1].strides))), __pyx_v_device_id);

    /* "lib/mask/gpu_mv.pyx":37
 *     cdef np.ndarray[np.int32_t, ndim=2] \
 *         result_box = np.zeros((result_num, boxes_dim), dtype=np.int32)
 *     if all_boxes.shape[0] > 0:             # <<<<<<<<<<<<<<
 *         _mv(&all_boxes[0, 0], &all_masks[0, 0, 0, 0], all_box_num, &candidate_inds[0], &candidate_start[0], &candidate_weights[0], candidate_num, binary_thresh, image_height, image_width, boxes_dim, mask_size, candidate_start.shape[0], &result_mask[0,0,0,0], &result_box[0,0], device_id)
 *     return result_mask, result_box
 */
  }

  /* "lib/mask/gpu_mv.pyx":39
 *     if all_boxes.shape[0] > 0:
 *         _mv(&all_boxes[0, 0], &all_masks[0, 0, 0, 0], all_box_num, &candidate_inds[0], &candidate_start[0], &candidate_weights[0], candidate_num, binary_thresh, image_height, image_width, boxes_dim, mask_size, candidate_start.shape[0], &result_mask[0,0,0,0], &result_box[0,0], device_id)
 *     return result_mask, result_box             # <<<<<<<<<<<<<<
 */
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_3 = PyTuple_New(2); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 39, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_3);
  __Pyx_INCREF(((PyObject *)__pyx_v_result_mask));
  __Pyx_GIVEREF(((PyObject *)__pyx_v_result_mask));
  PyTuple_SET_ITEM(__pyx_t_3, 0, ((PyObject *)__pyx_v_result_mask));
  __Pyx_INCREF(((PyObject *)__pyx_v_result_box));
  __Pyx_GIVEREF(((PyObject *)__pyx_v_result_box));
  PyTuple_SET_ITEM(__pyx_t_3, 1, ((PyObject *)__pyx_v_result_box));
  __pyx_r = __pyx_t_3;
  __pyx_t_3 = 0;
  goto __pyx_L0;

  /* "lib/mask/gpu_mv.pyx":19
 * # masks: n * 1 * 21 * 21
 * # scores: n * 21
 * def mv(np.ndarray[np.float32_t, ndim=2] all_boxes,             # <<<<<<<<<<<<<<
 *                 np.ndarray[np.float32_t, ndim=4] all_masks,
 *                 np.ndarray[np.int32_t, ndim=1] candidate_inds,
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_XDECREF(__pyx_t_2);
  __Pyx_XDECREF(__pyx_t_3);
  __Pyx_XDECREF(__pyx_t_4);
  __Pyx_XDECREF(__pyx_t_5);
  { PyObject *__pyx_type, *__pyx_value, *__pyx_tb;
    __Pyx_PyThreadState_declare
    __Pyx_PyThreadState_assign
    __Pyx_ErrFetch(&__pyx_type, &__pyx_value, &__pyx_tb);
    __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_all_boxes.rcbuffer->pybuffer);
    __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_all_masks.rcbuffer->pybuffer);
    __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_candidate_inds.rcbuffer->pybuffer);
    __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_candidate_start.rcbuffer->pybuffer);
    __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_candidate_weights.rcbuffer->pybuffer);
    __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_result_box.rcbuffer->pybuffer);
    __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_result_mask.rcbuffer->pybuffer);
  __Pyx_ErrRestore(__pyx_type, __pyx_value, __pyx_tb);}
  __Pyx_AddTraceback("lib.mask.gpu_mv.mv", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  goto __pyx_L2;
  __pyx_L0:;
  __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_all_boxes.rcbuffer->pybuffer);
  __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_all_masks.rcbuffer->pybuffer);
  __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_candidate_inds.rcbuffer->pybuffer);
  __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_candidate_start.rcbuffer->pybuffer);
  __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_candidate_weights.rcbuffer->pybuffer);
  __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_result_box.rcbuffer->pybuffer);
  __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_result_mask.rcbuffer->pybuffer);
  __pyx_L2:;
  __Pyx_XDECREF((PyObject *)__pyx_v_result_mask);
  __Pyx_XDECREF((PyObject *)__pyx_v_result_box);
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":197
 *         # experimental exception made for __getbuffer__ and __releasebuffer__
 *         # -- the details of this may change.
 *         def __getbuffer__(ndarray self, Py_buffer* info, int flags):             # <<<<<<<<<<<<<<
 *             # This implementation of getbuffer is geared towards Cython
 *             # requirements, and does not yet fullfill the PEP.
 */

/* Python wrapper */
static CYTHON_UNUSED int __pyx_pw_5numpy_7ndarray_1__getbuffer__(PyObject *__pyx_v_self, Py_buffer *__pyx_v_info, int __pyx_v_flags); /*proto*/
static CYTHON_UNUSED int __pyx_pw_5numpy_7ndarray_1__getbuffer__(PyObject *__pyx_v_self, Py_buffer *__pyx_v_info, int __pyx_v_flags) {
  int __pyx_r;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("__getbuffer__ (wrapper)", 0);
  __pyx_r = __pyx_pf_5numpy_7ndarray___getbuffer__(((PyArrayObject *)__pyx_v_self), ((Py_buffer *)__pyx_v_info), ((int)__pyx_v_flags));

  /* function exit code */
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

static int __pyx_pf_5numpy_7ndarray___getbuffer__(PyArrayObject *__pyx_v_self, Py_buffer *__pyx_v_info, int __pyx_v_flags) {
  int __pyx_v_copy_shape;
  int __pyx_v_i;
  int __pyx_v_ndim;
  int __pyx_v_endian_detector;
  int __pyx_v_little_endian;
  int __pyx_v_t;
  char *__pyx_v_f;
  PyArray_Descr *__pyx_v_descr = 0;
  int __pyx_v_offset;
  int __pyx_v_hasfields;
  int __pyx_r;
  __Pyx_RefNannyDeclarations
  int __pyx_t_1;
  int __pyx_t_2;
  PyObject *__pyx_t_3 = NULL;
  int __pyx_t_4;
  int __pyx_t_5;
  PyObject *__pyx_t_6 = NULL;
  char *__pyx_t_7;
  __Pyx_RefNannySetupContext("__getbuffer__", 0);
  if (__pyx_v_info != NULL) {
    __pyx_v_info->obj = Py_None; __Pyx_INCREF(Py_None);
    __Pyx_GIVEREF(__pyx_v_info->obj);
  }

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":203
 *             # of flags
 * 
 *             if info == NULL: return             # <<<<<<<<<<<<<<
 * 
 *             cdef int copy_shape, i, ndim
 */
  __pyx_t_1 = ((__pyx_v_info == NULL) != 0);
  if (__pyx_t_1) {
    __pyx_r = 0;
    goto __pyx_L0;
  }

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":206
 * 
 *             cdef int copy_shape, i, ndim
 *             cdef int endian_detector = 1             # <<<<<<<<<<<<<<
 *             cdef bint little_endian = ((<char*>&endian_detector)[0] != 0)
 * 
 */
  __pyx_v_endian_detector = 1;

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":207
 *             cdef int copy_shape, i, ndim
 *             cdef int endian_detector = 1
 *             cdef bint little_endian = ((<char*>&endian_detector)[0] != 0)             # <<<<<<<<<<<<<<
 * 
 *             ndim = PyArray_NDIM(self)
 */
  __pyx_v_little_endian = ((((char *)(&__pyx_v_endian_detector))[0]) != 0);

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":209
 *             cdef bint little_endian = ((<char*>&endian_detector)[0] != 0)
 * 
 *             ndim = PyArray_NDIM(self)             # <<<<<<<<<<<<<<
 * 
 *             if sizeof(npy_intp) != sizeof(Py_ssize_t):
 */
  __pyx_v_ndim = PyArray_NDIM(__pyx_v_self);

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":211
 *             ndim = PyArray_NDIM(self)
 * 
 *             if sizeof(npy_intp) != sizeof(Py_ssize_t):             # <<<<<<<<<<<<<<
 *                 copy_shape = 1
 *             else:
 */
  __pyx_t_1 = (((sizeof(npy_intp)) != (sizeof(Py_ssize_t))) != 0);
  if (__pyx_t_1) {

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":212
 * 
 *             if sizeof(npy_intp) != sizeof(Py_ssize_t):
 *                 copy_shape = 1             # <<<<<<<<<<<<<<
 *             else:
 *                 copy_shape = 0
 */
    __pyx_v_copy_shape = 1;

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":211
 *             ndim = PyArray_NDIM(self)
 * 
 *             if sizeof(npy_intp) != sizeof(Py_ssize_t):             # <<<<<<<<<<<<<<
 *                 copy_shape = 1
 *             else:
 */
    goto __pyx_L4;
  }

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":214
 *                 copy_shape = 1
 *             else:
 *                 copy_shape = 0             # <<<<<<<<<<<<<<
 * 
 *             if ((flags & pybuf.PyBUF_C_CONTIGUOUS == pybuf.PyBUF_C_CONTIGUOUS)
 */
  /*else*/ {
    __pyx_v_copy_shape = 0;
  }
  __pyx_L4:;

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":216
 *                 copy_shape = 0
 * 
 *             if ((flags & pybuf.PyBUF_C_CONTIGUOUS == pybuf.PyBUF_C_CONTIGUOUS)             # <<<<<<<<<<<<<<
 *                 and not PyArray_CHKFLAGS(self, NPY_C_CONTIGUOUS)):
 *                 raise ValueError(u"ndarray is not C contiguous")
 */
  __pyx_t_2 = (((__pyx_v_flags & PyBUF_C_CONTIGUOUS) == PyBUF_C_CONTIGUOUS) != 0);
  if (__pyx_t_2) {
  } else {
    __pyx_t_1 = __pyx_t_2;
    goto __pyx_L6_bool_binop_done;
  }

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":217
 * 
 *             if ((flags & pybuf.PyBUF_C_CONTIGUOUS == pybuf.PyBUF_C_CONTIGUOUS)
 *                 and not PyArray_CHKFLAGS(self, NPY_C_CONTIGUOUS)):             # <<<<<<<<<<<<<<
 *                 raise ValueError(u"ndarray is not C contiguous")
 * 
 */
  __pyx_t_2 = ((!(PyArray_CHKFLAGS(__pyx_v_self, NPY_C_CONTIGUOUS) != 0)) != 0);
  __pyx_t_1 = __pyx_t_2;
  __pyx_L6_bool_binop_done:;

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":216
 *                 copy_shape = 0
 * 
 *             if ((flags & pybuf.PyBUF_C_CONTIGUOUS == pybuf.PyBUF_C_CONTIGUOUS)             # <<<<<<<<<<<<<<
 *                 and not PyArray_CHKFLAGS(self, NPY_C_CONTIGUOUS)):
 *                 raise ValueError(u"ndarray is not C contiguous")
 */
  if (__pyx_t_1) {

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":218
 *             if ((flags & pybuf.PyBUF_C_CONTIGUOUS == pybuf.PyBUF_C_CONTIGUOUS)
 *                 and not PyArray_CHKFLAGS(self, NPY_C_CONTIGUOUS)):
 *                 raise ValueError(u"ndarray is not C contiguous")             # <<<<<<<<<<<<<<
 * 
 *             if ((flags & pybuf.PyBUF_F_CONTIGUOUS == pybuf.PyBUF_F_CONTIGUOUS)
 */
    __pyx_t_3 = __Pyx_PyObject_Call(__pyx_builtin_ValueError, __pyx_tuple_, NULL); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 218, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_3);
    __Pyx_Raise(__pyx_t_3, 0, 0, 0);
    __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
    __PYX_ERR(1, 218, __pyx_L1_error)

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":216
 *                 copy_shape = 0
 * 
 *             if ((flags & pybuf.PyBUF_C_CONTIGUOUS == pybuf.PyBUF_C_CONTIGUOUS)             # <<<<<<<<<<<<<<
 *                 and not PyArray_CHKFLAGS(self, NPY_C_CONTIGUOUS)):
 *                 raise ValueError(u"ndarray is not C contiguous")
 */
  }

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":220
 *                 raise ValueError(u"ndarray is not C contiguous")
 * 
 *             if ((flags & pybuf.PyBUF_F_CONTIGUOUS == pybuf.PyBUF_F_CONTIGUOUS)             # <<<<<<<<<<<<<<
 *                 and not PyArray_CHKFLAGS(self, NPY_F_CONTIGUOUS)):
 *                 raise ValueError(u"ndarray is not Fortran contiguous")
 */
  __pyx_t_2 = (((__pyx_v_flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS) != 0);
  if (__pyx_t_2) {
  } else {
    __pyx_t_1 = __pyx_t_2;
    goto __pyx_L9_bool_binop_done;
  }

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":221
 * 
 *             if ((flags & pybuf.PyBUF_F_CONTIGUOUS == pybuf.PyBUF_F_CONTIGUOUS)
 *                 and not PyArray_CHKFLAGS(self, NPY_F_CONTIGUOUS)):             # <<<<<<<<<<<<<<
 *                 raise ValueError(u"ndarray is not Fortran contiguous")
 * 
 */
  __pyx_t_2 = ((!(PyArray_CHKFLAGS(__pyx_v_self, NPY_F_CONTIGUOUS) != 0)) != 0);
  __pyx_t_1 = __pyx_t_2;
  __pyx_L9_bool_binop_done:;

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":220
 *                 raise ValueError(u"ndarray is not C contiguous")
 * 
 *             if ((flags & pybuf.PyBUF_F_CONTIGUOUS == pybuf.PyBUF_F_CONTIGUOUS)             # <<<<<<<<<<<<<<
 *                 and not PyArray_CHKFLAGS(self, NPY_F_CONTIGUOUS)):
 *                 raise ValueError(u"ndarray is not Fortran contiguous")
 */
  if (__pyx_t_1) {

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":222
 *             if ((flags & pybuf.PyBUF_F_CONTIGUOUS == pybuf.PyBUF_F_CONTIGUOUS)
 *                 and not PyArray_CHKFLAGS(self, NPY_F_CONTIGUOUS)):
 *                 raise ValueError(u"ndarray is not Fortran contiguous")             # <<<<<<<<<<<<<<
 * 
 *             info.buf = PyArray_DATA(self)
 */
    __pyx_t_3 = __Pyx_PyObject_Call(__pyx_builtin_ValueError, __pyx_tuple__2, NULL); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 222, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_3);
    __Pyx_Raise(__pyx_t_3, 0, 0, 0);
    __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
    __PYX_ERR(1, 222, __pyx_L1_error)

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":220
 *                 raise ValueError(u"ndarray is not C contiguous")
 * 
 *             if ((flags & pybuf.PyBUF_F_CONTIGUOUS == pybuf.PyBUF_F_CONTIGUOUS)             # <<<<<<<<<<<<<<
 *                 and not PyArray_CHKFLAGS(self, NPY_F_CONTIGUOUS)):
 *                 raise ValueError(u"ndarray is not Fortran contiguous")
 */
  }

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":224
 *                 raise ValueError(u"ndarray is not Fortran contiguous")
 * 
 *             info.buf = PyArray_DATA(self)             # <<<<<<<<<<<<<<
 *             info.ndim = ndim
 *             if copy_shape:
 */
  __pyx_v_info->buf = PyArray_DATA(__pyx_v_self);

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":225
 * 
 *             info.buf = PyArray_DATA(self)
 *             info.ndim = ndim             # <<<<<<<<<<<<<<
 *             if copy_shape:
 *                 # Allocate new buffer for strides and shape info.
 */
  __pyx_v_info->ndim = __pyx_v_ndim;

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":226
 *             info.buf = PyArray_DATA(self)
 *             info.ndim = ndim
 *             if copy_shape:             # <<<<<<<<<<<<<<
 *                 # Allocate new buffer for strides and shape info.
 *                 # This is allocated as one block, strides first.
 */
  __pyx_t_1 = (__pyx_v_copy_shape != 0);
  if (__pyx_t_1) {

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":229
 *                 # Allocate new buffer for strides and shape info.
 *                 # This is allocated as one block, strides first.
 *                 info.strides = <Py_ssize_t*>stdlib.malloc(sizeof(Py_ssize_t) * <size_t>ndim * 2)             # <<<<<<<<<<<<<<
 *                 info.shape = info.strides + ndim
 *                 for i in range(ndim):
 */
    __pyx_v_info->strides = ((Py_ssize_t *)malloc((((sizeof(Py_ssize_t)) * ((size_t)__pyx_v_ndim)) * 2)));

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":230
 *                 # This is allocated as one block, strides first.
 *                 info.strides = <Py_ssize_t*>stdlib.malloc(sizeof(Py_ssize_t) * <size_t>ndim * 2)
 *                 info.shape = info.strides + ndim             # <<<<<<<<<<<<<<
 *                 for i in range(ndim):
 *                     info.strides[i] = PyArray_STRIDES(self)[i]
 */
    __pyx_v_info->shape = (__pyx_v_info->strides + __pyx_v_ndim);

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":231
 *                 info.strides = <Py_ssize_t*>stdlib.malloc(sizeof(Py_ssize_t) * <size_t>ndim * 2)
 *                 info.shape = info.strides + ndim
 *                 for i in range(ndim):             # <<<<<<<<<<<<<<
 *                     info.strides[i] = PyArray_STRIDES(self)[i]
 *                     info.shape[i] = PyArray_DIMS(self)[i]
 */
    __pyx_t_4 = __pyx_v_ndim;
    for (__pyx_t_5 = 0; __pyx_t_5 < __pyx_t_4; __pyx_t_5+=1) {
      __pyx_v_i = __pyx_t_5;

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":232
 *                 info.shape = info.strides + ndim
 *                 for i in range(ndim):
 *                     info.strides[i] = PyArray_STRIDES(self)[i]             # <<<<<<<<<<<<<<
 *                     info.shape[i] = PyArray_DIMS(self)[i]
 *             else:
 */
      (__pyx_v_info->strides[__pyx_v_i]) = (PyArray_STRIDES(__pyx_v_self)[__pyx_v_i]);

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":233
 *                 for i in range(ndim):
 *                     info.strides[i] = PyArray_STRIDES(self)[i]
 *                     info.shape[i] = PyArray_DIMS(self)[i]             # <<<<<<<<<<<<<<
 *             else:
 *                 info.strides = <Py_ssize_t*>PyArray_STRIDES(self)
 */
      (__pyx_v_info->shape[__pyx_v_i]) = (PyArray_DIMS(__pyx_v_self)[__pyx_v_i]);
    }

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":226
 *             info.buf = PyArray_DATA(self)
 *             info.ndim = ndim
 *             if copy_shape:             # <<<<<<<<<<<<<<
 *                 # Allocate new buffer for strides and shape info.
 *                 # This is allocated as one block, strides first.
 */
    goto __pyx_L11;
  }

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":235
 *                     info.shape[i] = PyArray_DIMS(self)[i]
 *             else:
 *                 info.strides = <Py_ssize_t*>PyArray_STRIDES(self)             # <<<<<<<<<<<<<<
 *                 info.shape = <Py_ssize_t*>PyArray_DIMS(self)
 *             info.suboffsets = NULL
 */
  /*else*/ {
    __pyx_v_info->strides = ((Py_ssize_t *)PyArray_STRIDES(__pyx_v_self));

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":236
 *             else:
 *                 info.strides = <Py_ssize_t*>PyArray_STRIDES(self)
 *                 info.shape = <Py_ssize_t*>PyArray_DIMS(self)             # <<<<<<<<<<<<<<
 *             info.suboffsets = NULL
 *             info.itemsize = PyArray_ITEMSIZE(self)
 */
    __pyx_v_info->shape = ((Py_ssize_t *)PyArray_DIMS(__pyx_v_self));
  }
  __pyx_L11:;

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":237
 *                 info.strides = <Py_ssize_t*>PyArray_STRIDES(self)
 *                 info.shape = <Py_ssize_t*>PyArray_DIMS(self)
 *             info.suboffsets = NULL             # <<<<<<<<<<<<<<
 *             info.itemsize = PyArray_ITEMSIZE(self)
 *             info.readonly = not PyArray_ISWRITEABLE(self)
 */
  __pyx_v_info->suboffsets = NULL;

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":238
 *                 info.shape = <Py_ssize_t*>PyArray_DIMS(self)
 *             info.suboffsets = NULL
 *             info.itemsize = PyArray_ITEMSIZE(self)             # <<<<<<<<<<<<<<
 *             info.readonly = not PyArray_ISWRITEABLE(self)
 * 
 */
  __pyx_v_info->itemsize = PyArray_ITEMSIZE(__pyx_v_self);

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":239
 *             info.suboffsets = NULL
 *             info.itemsize = PyArray_ITEMSIZE(self)
 *             info.readonly = not PyArray_ISWRITEABLE(self)             # <<<<<<<<<<<<<<
 * 
 *             cdef int t
 */
  __pyx_v_info->readonly = (!(PyArray_ISWRITEABLE(__pyx_v_self) != 0));

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":242
 * 
 *             cdef int t
 *             cdef char* f = NULL             # <<<<<<<<<<<<<<
 *             cdef dtype descr = self.descr
 *             cdef int offset
 */
  __pyx_v_f = NULL;

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":243
 *             cdef int t
 *             cdef char* f = NULL
 *             cdef dtype descr = self.descr             # <<<<<<<<<<<<<<
 *             cdef int offset
 * 
 */
  __pyx_t_3 = ((PyObject *)__pyx_v_self->descr);
  __Pyx_INCREF(__pyx_t_3);
  __pyx_v_descr = ((PyArray_Descr *)__pyx_t_3);
  __pyx_t_3 = 0;

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":246
 *             cdef int offset
 * 
 *             cdef bint hasfields = PyDataType_HASFIELDS(descr)             # <<<<<<<<<<<<<<
 * 
 *             if not hasfields and not copy_shape:
 */
  __pyx_v_hasfields = PyDataType_HASFIELDS(__pyx_v_descr);

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":248
 *             cdef bint hasfields = PyDataType_HASFIELDS(descr)
 * 
 *             if not hasfields and not copy_shape:             # <<<<<<<<<<<<<<
 *                 # do not call releasebuffer
 *                 info.obj = None
 */
  __pyx_t_2 = ((!(__pyx_v_hasfields != 0)) != 0);
  if (__pyx_t_2) {
  } else {
    __pyx_t_1 = __pyx_t_2;
    goto __pyx_L15_bool_binop_done;
  }
  __pyx_t_2 = ((!(__pyx_v_copy_shape != 0)) != 0);
  __pyx_t_1 = __pyx_t_2;
  __pyx_L15_bool_binop_done:;
  if (__pyx_t_1) {

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":250
 *             if not hasfields and not copy_shape:
 *                 # do not call releasebuffer
 *                 info.obj = None             # <<<<<<<<<<<<<<
 *             else:
 *                 # need to call releasebuffer
 */
    __Pyx_INCREF(Py_None);
    __Pyx_GIVEREF(Py_None);
    __Pyx_GOTREF(__pyx_v_info->obj);
    __Pyx_DECREF(__pyx_v_info->obj);
    __pyx_v_info->obj = Py_None;

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":248
 *             cdef bint hasfields = PyDataType_HASFIELDS(descr)
 * 
 *             if not hasfields and not copy_shape:             # <<<<<<<<<<<<<<
 *                 # do not call releasebuffer
 *                 info.obj = None
 */
    goto __pyx_L14;
  }

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":253
 *             else:
 *                 # need to call releasebuffer
 *                 info.obj = self             # <<<<<<<<<<<<<<
 * 
 *             if not hasfields:
 */
  /*else*/ {
    __Pyx_INCREF(((PyObject *)__pyx_v_self));
    __Pyx_GIVEREF(((PyObject *)__pyx_v_self));
    __Pyx_GOTREF(__pyx_v_info->obj);
    __Pyx_DECREF(__pyx_v_info->obj);
    __pyx_v_info->obj = ((PyObject *)__pyx_v_self);
  }
  __pyx_L14:;

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":255
 *                 info.obj = self
 * 
 *             if not hasfields:             # <<<<<<<<<<<<<<
 *                 t = descr.type_num
 *                 if ((descr.byteorder == c'>' and little_endian) or
 */
  __pyx_t_1 = ((!(__pyx_v_hasfields != 0)) != 0);
  if (__pyx_t_1) {

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":256
 * 
 *             if not hasfields:
 *                 t = descr.type_num             # <<<<<<<<<<<<<<
 *                 if ((descr.byteorder == c'>' and little_endian) or
 *                     (descr.byteorder == c'<' and not little_endian)):
 */
    __pyx_t_4 = __pyx_v_descr->type_num;
    __pyx_v_t = __pyx_t_4;

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":257
 *             if not hasfields:
 *                 t = descr.type_num
 *                 if ((descr.byteorder == c'>' and little_endian) or             # <<<<<<<<<<<<<<
 *                     (descr.byteorder == c'<' and not little_endian)):
 *                     raise ValueError(u"Non-native byte order not supported")
 */
    __pyx_t_2 = ((__pyx_v_descr->byteorder == '>') != 0);
    if (!__pyx_t_2) {
      goto __pyx_L20_next_or;
    } else {
    }
    __pyx_t_2 = (__pyx_v_little_endian != 0);
    if (!__pyx_t_2) {
    } else {
      __pyx_t_1 = __pyx_t_2;
      goto __pyx_L19_bool_binop_done;
    }
    __pyx_L20_next_or:;

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":258
 *                 t = descr.type_num
 *                 if ((descr.byteorder == c'>' and little_endian) or
 *                     (descr.byteorder == c'<' and not little_endian)):             # <<<<<<<<<<<<<<
 *                     raise ValueError(u"Non-native byte order not supported")
 *                 if   t == NPY_BYTE:        f = "b"
 */
    __pyx_t_2 = ((__pyx_v_descr->byteorder == '<') != 0);
    if (__pyx_t_2) {
    } else {
      __pyx_t_1 = __pyx_t_2;
      goto __pyx_L19_bool_binop_done;
    }
    __pyx_t_2 = ((!(__pyx_v_little_endian != 0)) != 0);
    __pyx_t_1 = __pyx_t_2;
    __pyx_L19_bool_binop_done:;

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":257
 *             if not hasfields:
 *                 t = descr.type_num
 *                 if ((descr.byteorder == c'>' and little_endian) or             # <<<<<<<<<<<<<<
 *                     (descr.byteorder == c'<' and not little_endian)):
 *                     raise ValueError(u"Non-native byte order not supported")
 */
    if (__pyx_t_1) {

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":259
 *                 if ((descr.byteorder == c'>' and little_endian) or
 *                     (descr.byteorder == c'<' and not little_endian)):
 *                     raise ValueError(u"Non-native byte order not supported")             # <<<<<<<<<<<<<<
 *                 if   t == NPY_BYTE:        f = "b"
 *                 elif t == NPY_UBYTE:       f = "B"
 */
      __pyx_t_3 = __Pyx_PyObject_Call(__pyx_builtin_ValueError, __pyx_tuple__3, NULL); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 259, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __Pyx_Raise(__pyx_t_3, 0, 0, 0);
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __PYX_ERR(1, 259, __pyx_L1_error)

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":257
 *             if not hasfields:
 *                 t = descr.type_num
 *                 if ((descr.byteorder == c'>' and little_endian) or             # <<<<<<<<<<<<<<
 *                     (descr.byteorder == c'<' and not little_endian)):
 *                     raise ValueError(u"Non-native byte order not supported")
 */
    }

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":260
 *                     (descr.byteorder == c'<' and not little_endian)):
 *                     raise ValueError(u"Non-native byte order not supported")
 *                 if   t == NPY_BYTE:        f = "b"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_UBYTE:       f = "B"
 *                 elif t == NPY_SHORT:       f = "h"
 */
    switch (__pyx_v_t) {
      case NPY_BYTE:
      __pyx_v_f = ((char *)"b");
      break;

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":261
 *                     raise ValueError(u"Non-native byte order not supported")
 *                 if   t == NPY_BYTE:        f = "b"
 *                 elif t == NPY_UBYTE:       f = "B"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_SHORT:       f = "h"
 *                 elif t == NPY_USHORT:      f = "H"
 */
      case NPY_UBYTE:
      __pyx_v_f = ((char *)"B");
      break;

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":262
 *                 if   t == NPY_BYTE:        f = "b"
 *                 elif t == NPY_UBYTE:       f = "B"
 *                 elif t == NPY_SHORT:       f = "h"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_USHORT:      f = "H"
 *                 elif t == NPY_INT:         f = "i"
 */
      case NPY_SHORT:
      __pyx_v_f = ((char *)"h");
      break;

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":263
 *                 elif t == NPY_UBYTE:       f = "B"
 *                 elif t == NPY_SHORT:       f = "h"
 *                 elif t == NPY_USHORT:      f = "H"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_INT:         f = "i"
 *                 elif t == NPY_UINT:        f = "I"
 */
      case NPY_USHORT:
      __pyx_v_f = ((char *)"H");
      break;

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":264
 *                 elif t == NPY_SHORT:       f = "h"
 *                 elif t == NPY_USHORT:      f = "H"
 *                 elif t == NPY_INT:         f = "i"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_UINT:        f = "I"
 *                 elif t == NPY_LONG:        f = "l"
 */
      case NPY_INT:
      __pyx_v_f = ((char *)"i");
      break;

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":265
 *                 elif t == NPY_USHORT:      f = "H"
 *                 elif t == NPY_INT:         f = "i"
 *                 elif t == NPY_UINT:        f = "I"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_LONG:        f = "l"
 *                 elif t == NPY_ULONG:       f = "L"
 */
      case NPY_UINT:
      __pyx_v_f = ((char *)"I");
      break;

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":266
 *                 elif t == NPY_INT:         f = "i"
 *                 elif t == NPY_UINT:        f = "I"
 *                 elif t == NPY_LONG:        f = "l"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_ULONG:       f = "L"
 *                 elif t == NPY_LONGLONG:    f = "q"
 */
      case NPY_LONG:
      __pyx_v_f = ((char *)"l");
      break;

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":267
 *                 elif t == NPY_UINT:        f = "I"
 *                 elif t == NPY_LONG:        f = "l"
 *                 elif t == NPY_ULONG:       f = "L"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_LONGLONG:    f = "q"
 *                 elif t == NPY_ULONGLONG:   f = "Q"
 */
      case NPY_ULONG:
      __pyx_v_f = ((char *)"L");
      break;

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":268
 *                 elif t == NPY_LONG:        f = "l"
 *                 elif t == NPY_ULONG:       f = "L"
 *                 elif t == NPY_LONGLONG:    f = "q"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_ULONGLONG:   f = "Q"
 *                 elif t == NPY_FLOAT:       f = "f"
 */
      case NPY_LONGLONG:
      __pyx_v_f = ((char *)"q");
      break;

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":269
 *                 elif t == NPY_ULONG:       f = "L"
 *                 elif t == NPY_LONGLONG:    f = "q"
 *                 elif t == NPY_ULONGLONG:   f = "Q"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_FLOAT:       f = "f"
 *                 elif t == NPY_DOUBLE:      f = "d"
 */
      case NPY_ULONGLONG:
      __pyx_v_f = ((char *)"Q");
      break;

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":270
 *                 elif t == NPY_LONGLONG:    f = "q"
 *                 elif t == NPY_ULONGLONG:   f = "Q"
 *                 elif t == NPY_FLOAT:       f = "f"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_DOUBLE:      f = "d"
 *                 elif t == NPY_LONGDOUBLE:  f = "g"
 */
      case NPY_FLOAT:
      __pyx_v_f = ((char *)"f");
      break;

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":271
 *                 elif t == NPY_ULONGLONG:   f = "Q"
 *                 elif t == NPY_FLOAT:       f = "f"
 *                 elif t == NPY_DOUBLE:      f = "d"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_LONGDOUBLE:  f = "g"
 *                 elif t == NPY_CFLOAT:      f = "Zf"
 */
      case NPY_DOUBLE:
      __pyx_v_f = ((char *)"d");
      break;

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":272
 *                 elif t == NPY_FLOAT:       f = "f"
 *                 elif t == NPY_DOUBLE:      f = "d"
 *                 elif t == NPY_LONGDOUBLE:  f = "g"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_CFLOAT:      f = "Zf"
 *                 elif t == NPY_CDOUBLE:     f = "Zd"
 */
      case NPY_LONGDOUBLE:
      __pyx_v_f = ((char *)"g");
      break;

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":273
 *                 elif t == NPY_DOUBLE:      f = "d"
 *                 elif t == NPY_LONGDOUBLE:  f = "g"
 *                 elif t == NPY_CFLOAT:      f = "Zf"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_CDOUBLE:     f = "Zd"
 *                 elif t == NPY_CLONGDOUBLE: f = "Zg"
 */
      case NPY_CFLOAT:
      __pyx_v_f = ((char *)"Zf");
      break;

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":274
 *                 elif t == NPY_LONGDOUBLE:  f = "g"
 *                 elif t == NPY_CFLOAT:      f = "Zf"
 *                 elif t == NPY_CDOUBLE:     f = "Zd"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_CLONGDOUBLE: f = "Zg"
 *                 elif t == NPY_OBJECT:      f = "O"
 */
      case NPY_CDOUBLE:
      __pyx_v_f = ((char *)"Zd");
      break;

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":275
 *                 elif t == NPY_CFLOAT:      f = "Zf"
 *                 elif t == NPY_CDOUBLE:     f = "Zd"
 *                 elif t == NPY_CLONGDOUBLE: f = "Zg"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_OBJECT:      f = "O"
 *                 else:
 */
      case NPY_CLONGDOUBLE:
      __pyx_v_f = ((char *)"Zg");
      break;

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":276
 *                 elif t == NPY_CDOUBLE:     f = "Zd"
 *                 elif t == NPY_CLONGDOUBLE: f = "Zg"
 *                 elif t == NPY_OBJECT:      f = "O"             # <<<<<<<<<<<<<<
 *                 else:
 *                     raise ValueError(u"unknown dtype code in numpy.pxd (%d)" % t)
 */
      case NPY_OBJECT:
      __pyx_v_f = ((char *)"O");
      break;
      default:

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":278
 *                 elif t == NPY_OBJECT:      f = "O"
 *                 else:
 *                     raise ValueError(u"unknown dtype code in numpy.pxd (%d)" % t)             # <<<<<<<<<<<<<<
 *                 info.format = f
 *                 return
 */
      __pyx_t_3 = __Pyx_PyInt_From_int(__pyx_v_t); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 278, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_6 = PyUnicode_Format(__pyx_kp_u_unknown_dtype_code_in_numpy_pxd, __pyx_t_3); if (unlikely(!__pyx_t_6)) __PYX_ERR(1, 278, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_6);
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_3 = PyTuple_New(1); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 278, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __Pyx_GIVEREF(__pyx_t_6);
      PyTuple_SET_ITEM(__pyx_t_3, 0, __pyx_t_6);
      __pyx_t_6 = 0;
      __pyx_t_6 = __Pyx_PyObject_Call(__pyx_builtin_ValueError, __pyx_t_3, NULL); if (unlikely(!__pyx_t_6)) __PYX_ERR(1, 278, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_6);
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __Pyx_Raise(__pyx_t_6, 0, 0, 0);
      __Pyx_DECREF(__pyx_t_6); __pyx_t_6 = 0;
      __PYX_ERR(1, 278, __pyx_L1_error)
      break;
    }

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":279
 *                 else:
 *                     raise ValueError(u"unknown dtype code in numpy.pxd (%d)" % t)
 *                 info.format = f             # <<<<<<<<<<<<<<
 *                 return
 *             else:
 */
    __pyx_v_info->format = __pyx_v_f;

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":280
 *                     raise ValueError(u"unknown dtype code in numpy.pxd (%d)" % t)
 *                 info.format = f
 *                 return             # <<<<<<<<<<<<<<
 *             else:
 *                 info.format = <char*>stdlib.malloc(_buffer_format_string_len)
 */
    __pyx_r = 0;
    goto __pyx_L0;

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":255
 *                 info.obj = self
 * 
 *             if not hasfields:             # <<<<<<<<<<<<<<
 *                 t = descr.type_num
 *                 if ((descr.byteorder == c'>' and little_endian) or
 */
  }

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":282
 *                 return
 *             else:
 *                 info.format = <char*>stdlib.malloc(_buffer_format_string_len)             # <<<<<<<<<<<<<<
 *                 info.format[0] = c'^' # Native data types, manual alignment
 *                 offset = 0
 */
  /*else*/ {
    __pyx_v_info->format = ((char *)malloc(0xFF));

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":283
 *             else:
 *                 info.format = <char*>stdlib.malloc(_buffer_format_string_len)
 *                 info.format[0] = c'^' # Native data types, manual alignment             # <<<<<<<<<<<<<<
 *                 offset = 0
 *                 f = _util_dtypestring(descr, info.format + 1,
 */
    (__pyx_v_info->format[0]) = '^';

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":284
 *                 info.format = <char*>stdlib.malloc(_buffer_format_string_len)
 *                 info.format[0] = c'^' # Native data types, manual alignment
 *                 offset = 0             # <<<<<<<<<<<<<<
 *                 f = _util_dtypestring(descr, info.format + 1,
 *                                       info.format + _buffer_format_string_len,
 */
    __pyx_v_offset = 0;

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":285
 *                 info.format[0] = c'^' # Native data types, manual alignment
 *                 offset = 0
 *                 f = _util_dtypestring(descr, info.format + 1,             # <<<<<<<<<<<<<<
 *                                       info.format + _buffer_format_string_len,
 *                                       &offset)
 */
    __pyx_t_7 = __pyx_f_5numpy__util_dtypestring(__pyx_v_descr, (__pyx_v_info->format + 1), (__pyx_v_info->format + 0xFF), (&__pyx_v_offset)); if (unlikely(__pyx_t_7 == NULL)) __PYX_ERR(1, 285, __pyx_L1_error)
    __pyx_v_f = __pyx_t_7;

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":288
 *                                       info.format + _buffer_format_string_len,
 *                                       &offset)
 *                 f[0] = c'\0' # Terminate format string             # <<<<<<<<<<<<<<
 * 
 *         def __releasebuffer__(ndarray self, Py_buffer* info):
 */
    (__pyx_v_f[0]) = '\x00';
  }

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":197
 *         # experimental exception made for __getbuffer__ and __releasebuffer__
 *         # -- the details of this may change.
 *         def __getbuffer__(ndarray self, Py_buffer* info, int flags):             # <<<<<<<<<<<<<<
 *             # This implementation of getbuffer is geared towards Cython
 *             # requirements, and does not yet fullfill the PEP.
 */

  /* function exit code */
  __pyx_r = 0;
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_3);
  __Pyx_XDECREF(__pyx_t_6);
  __Pyx_AddTraceback("numpy.ndarray.__getbuffer__", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = -1;
  if (__pyx_v_info != NULL && __pyx_v_info->obj != NULL) {
    __Pyx_GOTREF(__pyx_v_info->obj);
    __Pyx_DECREF(__pyx_v_info->obj); __pyx_v_info->obj = NULL;
  }
  goto __pyx_L2;
  __pyx_L0:;
  if (__pyx_v_info != NULL && __pyx_v_info->obj == Py_None) {
    __Pyx_GOTREF(Py_None);
    __Pyx_DECREF(Py_None); __pyx_v_info->obj = NULL;
  }
  __pyx_L2:;
  __Pyx_XDECREF((PyObject *)__pyx_v_descr);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":290
 *                 f[0] = c'\0' # Terminate format string
 * 
 *         def __releasebuffer__(ndarray self, Py_buffer* info):             # <<<<<<<<<<<<<<
 *             if PyArray_HASFIELDS(self):
 *                 stdlib.free(info.format)
 */

/* Python wrapper */
static CYTHON_UNUSED void __pyx_pw_5numpy_7ndarray_3__releasebuffer__(PyObject *__pyx_v_self, Py_buffer *__pyx_v_info); /*proto*/
static CYTHON_UNUSED void __pyx_pw_5numpy_7ndarray_3__releasebuffer__(PyObject *__pyx_v_self, Py_buffer *__pyx_v_info) {
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("__releasebuffer__ (wrapper)", 0);
  __pyx_pf_5numpy_7ndarray_2__releasebuffer__(((PyArrayObject *)__pyx_v_self), ((Py_buffer *)__pyx_v_info));

  /* function exit code */
  __Pyx_RefNannyFinishContext();
}

static void __pyx_pf_5numpy_7ndarray_2__releasebuffer__(PyArrayObject *__pyx_v_self, Py_buffer *__pyx_v_info) {
  __Pyx_RefNannyDeclarations
  int __pyx_t_1;
  __Pyx_RefNannySetupContext("__releasebuffer__", 0);

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":291
 * 
 *         def __releasebuffer__(ndarray self, Py_buffer* info):
 *             if PyArray_HASFIELDS(self):             # <<<<<<<<<<<<<<
 *                 stdlib.free(info.format)
 *             if sizeof(npy_intp) != sizeof(Py_ssize_t):
 */
  __pyx_t_1 = (PyArray_HASFIELDS(__pyx_v_self) != 0);
  if (__pyx_t_1) {

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":292
 *         def __releasebuffer__(ndarray self, Py_buffer* info):
 *             if PyArray_HASFIELDS(self):
 *                 stdlib.free(info.format)             # <<<<<<<<<<<<<<
 *             if sizeof(npy_intp) != sizeof(Py_ssize_t):
 *                 stdlib.free(info.strides)
 */
    free(__pyx_v_info->format);

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":291
 * 
 *         def __releasebuffer__(ndarray self, Py_buffer* info):
 *             if PyArray_HASFIELDS(self):             # <<<<<<<<<<<<<<
 *                 stdlib.free(info.format)
 *             if sizeof(npy_intp) != sizeof(Py_ssize_t):
 */
  }

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":293
 *             if PyArray_HASFIELDS(self):
 *                 stdlib.free(info.format)
 *             if sizeof(npy_intp) != sizeof(Py_ssize_t):             # <<<<<<<<<<<<<<
 *                 stdlib.free(info.strides)
 *                 # info.shape was stored after info.strides in the same block
 */
  __pyx_t_1 = (((sizeof(npy_intp)) != (sizeof(Py_ssize_t))) != 0);
  if (__pyx_t_1) {

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":294
 *                 stdlib.free(info.format)
 *             if sizeof(npy_intp) != sizeof(Py_ssize_t):
 *                 stdlib.free(info.strides)             # <<<<<<<<<<<<<<
 *                 # info.shape was stored after info.strides in the same block
 * 
 */
    free(__pyx_v_info->strides);

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":293
 *             if PyArray_HASFIELDS(self):
 *                 stdlib.free(info.format)
 *             if sizeof(npy_intp) != sizeof(Py_ssize_t):             # <<<<<<<<<<<<<<
 *                 stdlib.free(info.strides)
 *                 # info.shape was stored after info.strides in the same block
 */
  }

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":290
 *                 f[0] = c'\0' # Terminate format string
 * 
 *         def __releasebuffer__(ndarray self, Py_buffer* info):             # <<<<<<<<<<<<<<
 *             if PyArray_HASFIELDS(self):
 *                 stdlib.free(info.format)
 */

  /* function exit code */
  __Pyx_RefNannyFinishContext();
}

/* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":770
 * ctypedef npy_cdouble     complex_t
 * 
 * cdef inline object PyArray_MultiIterNew1(a):             # <<<<<<<<<<<<<<
 *     return PyArray_MultiIterNew(1, <void*>a)
 * 
 */

static CYTHON_INLINE PyObject *__pyx_f_5numpy_PyArray_MultiIterNew1(PyObject *__pyx_v_a) {
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  __Pyx_RefNannySetupContext("PyArray_MultiIterNew1", 0);

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":771
 * 
 * cdef inline object PyArray_MultiIterNew1(a):
 *     return PyArray_MultiIterNew(1, <void*>a)             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_MultiIterNew2(a, b):
 */
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_1 = PyArray_MultiIterNew(1, ((void *)__pyx_v_a)); if (unlikely(!__pyx_t_1)) __PYX_ERR(1, 771, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = 0;
  goto __pyx_L0;

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":770
 * ctypedef npy_cdouble     complex_t
 * 
 * cdef inline object PyArray_MultiIterNew1(a):             # <<<<<<<<<<<<<<
 *     return PyArray_MultiIterNew(1, <void*>a)
 * 
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_AddTraceback("numpy.PyArray_MultiIterNew1", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = 0;
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":773
 *     return PyArray_MultiIterNew(1, <void*>a)
 * 
 * cdef inline object PyArray_MultiIterNew2(a, b):             # <<<<<<<<<<<<<<
 *     return PyArray_MultiIterNew(2, <void*>a, <void*>b)
 * 
 */

static CYTHON_INLINE PyObject *__pyx_f_5numpy_PyArray_MultiIterNew2(PyObject *__pyx_v_a, PyObject *__pyx_v_b) {
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  __Pyx_RefNannySetupContext("PyArray_MultiIterNew2", 0);

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":774
 * 
 * cdef inline object PyArray_MultiIterNew2(a, b):
 *     return PyArray_MultiIterNew(2, <void*>a, <void*>b)             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_MultiIterNew3(a, b, c):
 */
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_1 = PyArray_MultiIterNew(2, ((void *)__pyx_v_a), ((void *)__pyx_v_b)); if (unlikely(!__pyx_t_1)) __PYX_ERR(1, 774, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = 0;
  goto __pyx_L0;

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":773
 *     return PyArray_MultiIterNew(1, <void*>a)
 * 
 * cdef inline object PyArray_MultiIterNew2(a, b):             # <<<<<<<<<<<<<<
 *     return PyArray_MultiIterNew(2, <void*>a, <void*>b)
 * 
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_AddTraceback("numpy.PyArray_MultiIterNew2", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = 0;
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":776
 *     return PyArray_MultiIterNew(2, <void*>a, <void*>b)
 * 
 * cdef inline object PyArray_MultiIterNew3(a, b, c):             # <<<<<<<<<<<<<<
 *     return PyArray_MultiIterNew(3, <void*>a, <void*>b, <void*> c)
 * 
 */

static CYTHON_INLINE PyObject *__pyx_f_5numpy_PyArray_MultiIterNew3(PyObject *__pyx_v_a, PyObject *__pyx_v_b, PyObject *__pyx_v_c) {
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  __Pyx_RefNannySetupContext("PyArray_MultiIterNew3", 0);

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":777
 * 
 * cdef inline object PyArray_MultiIterNew3(a, b, c):
 *     return PyArray_MultiIterNew(3, <void*>a, <void*>b, <void*> c)             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_MultiIterNew4(a, b, c, d):
 */
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_1 = PyArray_MultiIterNew(3, ((void *)__pyx_v_a), ((void *)__pyx_v_b), ((void *)__pyx_v_c)); if (unlikely(!__pyx_t_1)) __PYX_ERR(1, 777, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = 0;
  goto __pyx_L0;

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":776
 *     return PyArray_MultiIterNew(2, <void*>a, <void*>b)
 * 
 * cdef inline object PyArray_MultiIterNew3(a, b, c):             # <<<<<<<<<<<<<<
 *     return PyArray_MultiIterNew(3, <void*>a, <void*>b, <void*> c)
 * 
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_AddTraceback("numpy.PyArray_MultiIterNew3", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = 0;
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":779
 *     return PyArray_MultiIterNew(3, <void*>a, <void*>b, <void*> c)
 * 
 * cdef inline object PyArray_MultiIterNew4(a, b, c, d):             # <<<<<<<<<<<<<<
 *     return PyArray_MultiIterNew(4, <void*>a, <void*>b, <void*>c, <void*> d)
 * 
 */

static CYTHON_INLINE PyObject *__pyx_f_5numpy_PyArray_MultiIterNew4(PyObject *__pyx_v_a, PyObject *__pyx_v_b, PyObject *__pyx_v_c, PyObject *__pyx_v_d) {
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  __Pyx_RefNannySetupContext("PyArray_MultiIterNew4", 0);

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":780
 * 
 * cdef inline object PyArray_MultiIterNew4(a, b, c, d):
 *     return PyArray_MultiIterNew(4, <void*>a, <void*>b, <void*>c, <void*> d)             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_MultiIterNew5(a, b, c, d, e):
 */
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_1 = PyArray_MultiIterNew(4, ((void *)__pyx_v_a), ((void *)__pyx_v_b), ((void *)__pyx_v_c), ((void *)__pyx_v_d)); if (unlikely(!__pyx_t_1)) __PYX_ERR(1, 780, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = 0;
  goto __pyx_L0;

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":779
 *     return PyArray_MultiIterNew(3, <void*>a, <void*>b, <void*> c)
 * 
 * cdef inline object PyArray_MultiIterNew4(a, b, c, d):             # <<<<<<<<<<<<<<
 *     return PyArray_MultiIterNew(4, <void*>a, <void*>b, <void*>c, <void*> d)
 * 
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_AddTraceback("numpy.PyArray_MultiIterNew4", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = 0;
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":782
 *     return PyArray_MultiIterNew(4, <void*>a, <void*>b, <void*>c, <void*> d)
 * 
 * cdef inline object PyArray_MultiIterNew5(a, b, c, d, e):             # <<<<<<<<<<<<<<
 *     return PyArray_MultiIterNew(5, <void*>a, <void*>b, <void*>c, <void*> d, <void*> e)
 * 
 */

static CYTHON_INLINE PyObject *__pyx_f_5numpy_PyArray_MultiIterNew5(PyObject *__pyx_v_a, PyObject *__pyx_v_b, PyObject *__pyx_v_c, PyObject *__pyx_v_d, PyObject *__pyx_v_e) {
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  __Pyx_RefNannySetupContext("PyArray_MultiIterNew5", 0);

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":783
 * 
 * cdef inline object PyArray_MultiIterNew5(a, b, c, d, e):
 *     return PyArray_MultiIterNew(5, <void*>a, <void*>b, <void*>c, <void*> d, <void*> e)             # <<<<<<<<<<<<<<
 * 
 * cdef inline char* _util_dtypestring(dtype descr, char* f, char* end, int* offset) except NULL:
 */
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_1 = PyArray_MultiIterNew(5, ((void *)__pyx_v_a), ((void *)__pyx_v_b), ((void *)__pyx_v_c), ((void *)__pyx_v_d), ((void *)__pyx_v_e)); if (unlikely(!__pyx_t_1)) __PYX_ERR(1, 783, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = 0;
  goto __pyx_L0;

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":782
 *     return PyArray_MultiIterNew(4, <void*>a, <void*>b, <void*>c, <void*> d)
 * 
 * cdef inline object PyArray_MultiIterNew5(a, b, c, d, e):             # <<<<<<<<<<<<<<
 *     return PyArray_MultiIterNew(5, <void*>a, <void*>b, <void*>c, <void*> d, <void*> e)
 * 
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_AddTraceback("numpy.PyArray_MultiIterNew5", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = 0;
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":785
 *     return PyArray_MultiIterNew(5, <void*>a, <void*>b, <void*>c, <void*> d, <void*> e)
 * 
 * cdef inline char* _util_dtypestring(dtype descr, char* f, char* end, int* offset) except NULL:             # <<<<<<<<<<<<<<
 *     # Recursive utility function used in __getbuffer__ to get format
 *     # string. The new location in the format string is returned.
 */

static CYTHON_INLINE char *__pyx_f_5numpy__util_dtypestring(PyArray_Descr *__pyx_v_descr, char *__pyx_v_f, char *__pyx_v_end, int *__pyx_v_offset) {
  PyArray_Descr *__pyx_v_child = 0;
  int __pyx_v_endian_detector;
  int __pyx_v_little_endian;
  PyObject *__pyx_v_fields = 0;
  PyObject *__pyx_v_childname = NULL;
  PyObject *__pyx_v_new_offset = NULL;
  PyObject *__pyx_v_t = NULL;
  char *__pyx_r;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  Py_ssize_t __pyx_t_2;
  PyObject *__pyx_t_3 = NULL;
  PyObject *__pyx_t_4 = NULL;
  int __pyx_t_5;
  int __pyx_t_6;
  int __pyx_t_7;
  long __pyx_t_8;
  char *__pyx_t_9;
  __Pyx_RefNannySetupContext("_util_dtypestring", 0);

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":790
 * 
 *     cdef dtype child
 *     cdef int endian_detector = 1             # <<<<<<<<<<<<<<
 *     cdef bint little_endian = ((<char*>&endian_detector)[0] != 0)
 *     cdef tuple fields
 */
  __pyx_v_endian_detector = 1;

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":791
 *     cdef dtype child
 *     cdef int endian_detector = 1
 *     cdef bint little_endian = ((<char*>&endian_detector)[0] != 0)             # <<<<<<<<<<<<<<
 *     cdef tuple fields
 * 
 */
  __pyx_v_little_endian = ((((char *)(&__pyx_v_endian_detector))[0]) != 0);

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":794
 *     cdef tuple fields
 * 
 *     for childname in descr.names:             # <<<<<<<<<<<<<<
 *         fields = descr.fields[childname]
 *         child, new_offset = fields
 */
  if (unlikely(__pyx_v_descr->names == Py_None)) {
    PyErr_SetString(PyExc_TypeError, "'NoneType' object is not iterable");
    __PYX_ERR(1, 794, __pyx_L1_error)
  }
  __pyx_t_1 = __pyx_v_descr->names; __Pyx_INCREF(__pyx_t_1); __pyx_t_2 = 0;
  for (;;) {
    if (__pyx_t_2 >= PyTuple_GET_SIZE(__pyx_t_1)) break;
    #if CYTHON_COMPILING_IN_CPYTHON
    __pyx_t_3 = PyTuple_GET_ITEM(__pyx_t_1, __pyx_t_2); __Pyx_INCREF(__pyx_t_3); __pyx_t_2++; if (unlikely(0 < 0)) __PYX_ERR(1, 794, __pyx_L1_error)
    #else
    __pyx_t_3 = PySequence_ITEM(__pyx_t_1, __pyx_t_2); __pyx_t_2++; if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 794, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_3);
    #endif
    __Pyx_XDECREF_SET(__pyx_v_childname, __pyx_t_3);
    __pyx_t_3 = 0;

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":795
 * 
 *     for childname in descr.names:
 *         fields = descr.fields[childname]             # <<<<<<<<<<<<<<
 *         child, new_offset = fields
 * 
 */
    if (unlikely(__pyx_v_descr->fields == Py_None)) {
      PyErr_SetString(PyExc_TypeError, "'NoneType' object is not subscriptable");
      __PYX_ERR(1, 795, __pyx_L1_error)
    }
    __pyx_t_3 = __Pyx_PyDict_GetItem(__pyx_v_descr->fields, __pyx_v_childname); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 795, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_3);
    if (!(likely(PyTuple_CheckExact(__pyx_t_3))||((__pyx_t_3) == Py_None)||(PyErr_Format(PyExc_TypeError, "Expected %.16s, got %.200s", "tuple", Py_TYPE(__pyx_t_3)->tp_name), 0))) __PYX_ERR(1, 795, __pyx_L1_error)
    __Pyx_XDECREF_SET(__pyx_v_fields, ((PyObject*)__pyx_t_3));
    __pyx_t_3 = 0;

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":796
 *     for childname in descr.names:
 *         fields = descr.fields[childname]
 *         child, new_offset = fields             # <<<<<<<<<<<<<<
 * 
 *         if (end - f) - <int>(new_offset - offset[0]) < 15:
 */
    if (likely(__pyx_v_fields != Py_None)) {
      PyObject* sequence = __pyx_v_fields;
      #if CYTHON_COMPILING_IN_CPYTHON
      Py_ssize_t size = Py_SIZE(sequence);
      #else
      Py_ssize_t size = PySequence_Size(sequence);
      #endif
      if (unlikely(size != 2)) {
        if (size > 2) __Pyx_RaiseTooManyValuesError(2);
        else if (size >= 0) __Pyx_RaiseNeedMoreValuesError(size);
        __PYX_ERR(1, 796, __pyx_L1_error)
      }
      #if CYTHON_COMPILING_IN_CPYTHON
      __pyx_t_3 = PyTuple_GET_ITEM(sequence, 0); 
      __pyx_t_4 = PyTuple_GET_ITEM(sequence, 1); 
      __Pyx_INCREF(__pyx_t_3);
      __Pyx_INCREF(__pyx_t_4);
      #else
      __pyx_t_3 = PySequence_ITEM(sequence, 0); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 796, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_4 = PySequence_ITEM(sequence, 1); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 796, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      #endif
    } else {
      __Pyx_RaiseNoneNotIterableError(); __PYX_ERR(1, 796, __pyx_L1_error)
    }
    if (!(likely(((__pyx_t_3) == Py_None) || likely(__Pyx_TypeTest(__pyx_t_3, __pyx_ptype_5numpy_dtype))))) __PYX_ERR(1, 796, __pyx_L1_error)
    __Pyx_XDECREF_SET(__pyx_v_child, ((PyArray_Descr *)__pyx_t_3));
    __pyx_t_3 = 0;
    __Pyx_XDECREF_SET(__pyx_v_new_offset, __pyx_t_4);
    __pyx_t_4 = 0;

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":798
 *         child, new_offset = fields
 * 
 *         if (end - f) - <int>(new_offset - offset[0]) < 15:             # <<<<<<<<<<<<<<
 *             raise RuntimeError(u"Format string allocated too short, see comment in numpy.pxd")
 * 
 */
    __pyx_t_4 = __Pyx_PyInt_From_int((__pyx_v_offset[0])); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 798, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_4);
    __pyx_t_3 = PyNumber_Subtract(__pyx_v_new_offset, __pyx_t_4); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 798, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_3);
    __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
    __pyx_t_5 = __Pyx_PyInt_As_int(__pyx_t_3); if (unlikely((__pyx_t_5 == (int)-1) && PyErr_Occurred())) __PYX_ERR(1, 798, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
    __pyx_t_6 = ((((__pyx_v_end - __pyx_v_f) - ((int)__pyx_t_5)) < 15) != 0);
    if (__pyx_t_6) {

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":799
 * 
 *         if (end - f) - <int>(new_offset - offset[0]) < 15:
 *             raise RuntimeError(u"Format string allocated too short, see comment in numpy.pxd")             # <<<<<<<<<<<<<<
 * 
 *         if ((child.byteorder == c'>' and little_endian) or
 */
      __pyx_t_3 = __Pyx_PyObject_Call(__pyx_builtin_RuntimeError, __pyx_tuple__4, NULL); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 799, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __Pyx_Raise(__pyx_t_3, 0, 0, 0);
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __PYX_ERR(1, 799, __pyx_L1_error)

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":798
 *         child, new_offset = fields
 * 
 *         if (end - f) - <int>(new_offset - offset[0]) < 15:             # <<<<<<<<<<<<<<
 *             raise RuntimeError(u"Format string allocated too short, see comment in numpy.pxd")
 * 
 */
    }

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":801
 *             raise RuntimeError(u"Format string allocated too short, see comment in numpy.pxd")
 * 
 *         if ((child.byteorder == c'>' and little_endian) or             # <<<<<<<<<<<<<<
 *             (child.byteorder == c'<' and not little_endian)):
 *             raise ValueError(u"Non-native byte order not supported")
 */
    __pyx_t_7 = ((__pyx_v_child->byteorder == '>') != 0);
    if (!__pyx_t_7) {
      goto __pyx_L8_next_or;
    } else {
    }
    __pyx_t_7 = (__pyx_v_little_endian != 0);
    if (!__pyx_t_7) {
    } else {
      __pyx_t_6 = __pyx_t_7;
      goto __pyx_L7_bool_binop_done;
    }
    __pyx_L8_next_or:;

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":802
 * 
 *         if ((child.byteorder == c'>' and little_endian) or
 *             (child.byteorder == c'<' and not little_endian)):             # <<<<<<<<<<<<<<
 *             raise ValueError(u"Non-native byte order not supported")
 *             # One could encode it in the format string and have Cython
 */
    __pyx_t_7 = ((__pyx_v_child->byteorder == '<') != 0);
    if (__pyx_t_7) {
    } else {
      __pyx_t_6 = __pyx_t_7;
      goto __pyx_L7_bool_binop_done;
    }
    __pyx_t_7 = ((!(__pyx_v_little_endian != 0)) != 0);
    __pyx_t_6 = __pyx_t_7;
    __pyx_L7_bool_binop_done:;

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":801
 *             raise RuntimeError(u"Format string allocated too short, see comment in numpy.pxd")
 * 
 *         if ((child.byteorder == c'>' and little_endian) or             # <<<<<<<<<<<<<<
 *             (child.byteorder == c'<' and not little_endian)):
 *             raise ValueError(u"Non-native byte order not supported")
 */
    if (__pyx_t_6) {

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":803
 *         if ((child.byteorder == c'>' and little_endian) or
 *             (child.byteorder == c'<' and not little_endian)):
 *             raise ValueError(u"Non-native byte order not supported")             # <<<<<<<<<<<<<<
 *             # One could encode it in the format string and have Cython
 *             # complain instead, BUT: < and > in format strings also imply
 */
      __pyx_t_3 = __Pyx_PyObject_Call(__pyx_builtin_ValueError, __pyx_tuple__5, NULL); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 803, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __Pyx_Raise(__pyx_t_3, 0, 0, 0);
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __PYX_ERR(1, 803, __pyx_L1_error)

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":801
 *             raise RuntimeError(u"Format string allocated too short, see comment in numpy.pxd")
 * 
 *         if ((child.byteorder == c'>' and little_endian) or             # <<<<<<<<<<<<<<
 *             (child.byteorder == c'<' and not little_endian)):
 *             raise ValueError(u"Non-native byte order not supported")
 */
    }

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":813
 * 
 *         # Output padding bytes
 *         while offset[0] < new_offset:             # <<<<<<<<<<<<<<
 *             f[0] = 120 # "x"; pad byte
 *             f += 1
 */
    while (1) {
      __pyx_t_3 = __Pyx_PyInt_From_int((__pyx_v_offset[0])); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 813, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_4 = PyObject_RichCompare(__pyx_t_3, __pyx_v_new_offset, Py_LT); __Pyx_XGOTREF(__pyx_t_4); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 813, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_4); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(1, 813, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      if (!__pyx_t_6) break;

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":814
 *         # Output padding bytes
 *         while offset[0] < new_offset:
 *             f[0] = 120 # "x"; pad byte             # <<<<<<<<<<<<<<
 *             f += 1
 *             offset[0] += 1
 */
      (__pyx_v_f[0]) = 0x78;

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":815
 *         while offset[0] < new_offset:
 *             f[0] = 120 # "x"; pad byte
 *             f += 1             # <<<<<<<<<<<<<<
 *             offset[0] += 1
 * 
 */
      __pyx_v_f = (__pyx_v_f + 1);

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":816
 *             f[0] = 120 # "x"; pad byte
 *             f += 1
 *             offset[0] += 1             # <<<<<<<<<<<<<<
 * 
 *         offset[0] += child.itemsize
 */
      __pyx_t_8 = 0;
      (__pyx_v_offset[__pyx_t_8]) = ((__pyx_v_offset[__pyx_t_8]) + 1);
    }

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":818
 *             offset[0] += 1
 * 
 *         offset[0] += child.itemsize             # <<<<<<<<<<<<<<
 * 
 *         if not PyDataType_HASFIELDS(child):
 */
    __pyx_t_8 = 0;
    (__pyx_v_offset[__pyx_t_8]) = ((__pyx_v_offset[__pyx_t_8]) + __pyx_v_child->elsize);

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":820
 *         offset[0] += child.itemsize
 * 
 *         if not PyDataType_HASFIELDS(child):             # <<<<<<<<<<<<<<
 *             t = child.type_num
 *             if end - f < 5:
 */
    __pyx_t_6 = ((!(PyDataType_HASFIELDS(__pyx_v_child) != 0)) != 0);
    if (__pyx_t_6) {

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":821
 * 
 *         if not PyDataType_HASFIELDS(child):
 *             t = child.type_num             # <<<<<<<<<<<<<<
 *             if end - f < 5:
 *                 raise RuntimeError(u"Format string allocated too short.")
 */
      __pyx_t_4 = __Pyx_PyInt_From_int(__pyx_v_child->type_num); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 821, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      __Pyx_XDECREF_SET(__pyx_v_t, __pyx_t_4);
      __pyx_t_4 = 0;

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":822
 *         if not PyDataType_HASFIELDS(child):
 *             t = child.type_num
 *             if end - f < 5:             # <<<<<<<<<<<<<<
 *                 raise RuntimeError(u"Format string allocated too short.")
 * 
 */
      __pyx_t_6 = (((__pyx_v_end - __pyx_v_f) < 5) != 0);
      if (__pyx_t_6) {

        /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":823
 *             t = child.type_num
 *             if end - f < 5:
 *                 raise RuntimeError(u"Format string allocated too short.")             # <<<<<<<<<<<<<<
 * 
 *             # Until ticket #99 is fixed, use integers to avoid warnings
 */
        __pyx_t_4 = __Pyx_PyObject_Call(__pyx_builtin_RuntimeError, __pyx_tuple__6, NULL); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 823, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_4);
        __Pyx_Raise(__pyx_t_4, 0, 0, 0);
        __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
        __PYX_ERR(1, 823, __pyx_L1_error)

        /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":822
 *         if not PyDataType_HASFIELDS(child):
 *             t = child.type_num
 *             if end - f < 5:             # <<<<<<<<<<<<<<
 *                 raise RuntimeError(u"Format string allocated too short.")
 * 
 */
      }

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":826
 * 
 *             # Until ticket #99 is fixed, use integers to avoid warnings
 *             if   t == NPY_BYTE:        f[0] =  98 #"b"             # <<<<<<<<<<<<<<
 *             elif t == NPY_UBYTE:       f[0] =  66 #"B"
 *             elif t == NPY_SHORT:       f[0] = 104 #"h"
 */
      __pyx_t_4 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_BYTE); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 826, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      __pyx_t_3 = PyObject_RichCompare(__pyx_v_t, __pyx_t_4, Py_EQ); __Pyx_XGOTREF(__pyx_t_3); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 826, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(1, 826, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 98;
        goto __pyx_L15;
      }

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":827
 *             # Until ticket #99 is fixed, use integers to avoid warnings
 *             if   t == NPY_BYTE:        f[0] =  98 #"b"
 *             elif t == NPY_UBYTE:       f[0] =  66 #"B"             # <<<<<<<<<<<<<<
 *             elif t == NPY_SHORT:       f[0] = 104 #"h"
 *             elif t == NPY_USHORT:      f[0] =  72 #"H"
 */
      __pyx_t_3 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_UBYTE); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 827, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_4 = PyObject_RichCompare(__pyx_v_t, __pyx_t_3, Py_EQ); __Pyx_XGOTREF(__pyx_t_4); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 827, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_4); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(1, 827, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 66;
        goto __pyx_L15;
      }

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":828
 *             if   t == NPY_BYTE:        f[0] =  98 #"b"
 *             elif t == NPY_UBYTE:       f[0] =  66 #"B"
 *             elif t == NPY_SHORT:       f[0] = 104 #"h"             # <<<<<<<<<<<<<<
 *             elif t == NPY_USHORT:      f[0] =  72 #"H"
 *             elif t == NPY_INT:         f[0] = 105 #"i"
 */
      __pyx_t_4 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_SHORT); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 828, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      __pyx_t_3 = PyObject_RichCompare(__pyx_v_t, __pyx_t_4, Py_EQ); __Pyx_XGOTREF(__pyx_t_3); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 828, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(1, 828, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 0x68;
        goto __pyx_L15;
      }

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":829
 *             elif t == NPY_UBYTE:       f[0] =  66 #"B"
 *             elif t == NPY_SHORT:       f[0] = 104 #"h"
 *             elif t == NPY_USHORT:      f[0] =  72 #"H"             # <<<<<<<<<<<<<<
 *             elif t == NPY_INT:         f[0] = 105 #"i"
 *             elif t == NPY_UINT:        f[0] =  73 #"I"
 */
      __pyx_t_3 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_USHORT); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 829, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_4 = PyObject_RichCompare(__pyx_v_t, __pyx_t_3, Py_EQ); __Pyx_XGOTREF(__pyx_t_4); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 829, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_4); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(1, 829, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 72;
        goto __pyx_L15;
      }

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":830
 *             elif t == NPY_SHORT:       f[0] = 104 #"h"
 *             elif t == NPY_USHORT:      f[0] =  72 #"H"
 *             elif t == NPY_INT:         f[0] = 105 #"i"             # <<<<<<<<<<<<<<
 *             elif t == NPY_UINT:        f[0] =  73 #"I"
 *             elif t == NPY_LONG:        f[0] = 108 #"l"
 */
      __pyx_t_4 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_INT); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 830, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      __pyx_t_3 = PyObject_RichCompare(__pyx_v_t, __pyx_t_4, Py_EQ); __Pyx_XGOTREF(__pyx_t_3); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 830, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(1, 830, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 0x69;
        goto __pyx_L15;
      }

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":831
 *             elif t == NPY_USHORT:      f[0] =  72 #"H"
 *             elif t == NPY_INT:         f[0] = 105 #"i"
 *             elif t == NPY_UINT:        f[0] =  73 #"I"             # <<<<<<<<<<<<<<
 *             elif t == NPY_LONG:        f[0] = 108 #"l"
 *             elif t == NPY_ULONG:       f[0] = 76  #"L"
 */
      __pyx_t_3 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_UINT); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 831, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_4 = PyObject_RichCompare(__pyx_v_t, __pyx_t_3, Py_EQ); __Pyx_XGOTREF(__pyx_t_4); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 831, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_4); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(1, 831, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 73;
        goto __pyx_L15;
      }

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":832
 *             elif t == NPY_INT:         f[0] = 105 #"i"
 *             elif t == NPY_UINT:        f[0] =  73 #"I"
 *             elif t == NPY_LONG:        f[0] = 108 #"l"             # <<<<<<<<<<<<<<
 *             elif t == NPY_ULONG:       f[0] = 76  #"L"
 *             elif t == NPY_LONGLONG:    f[0] = 113 #"q"
 */
      __pyx_t_4 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_LONG); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 832, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      __pyx_t_3 = PyObject_RichCompare(__pyx_v_t, __pyx_t_4, Py_EQ); __Pyx_XGOTREF(__pyx_t_3); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 832, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(1, 832, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 0x6C;
        goto __pyx_L15;
      }

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":833
 *             elif t == NPY_UINT:        f[0] =  73 #"I"
 *             elif t == NPY_LONG:        f[0] = 108 #"l"
 *             elif t == NPY_ULONG:       f[0] = 76  #"L"             # <<<<<<<<<<<<<<
 *             elif t == NPY_LONGLONG:    f[0] = 113 #"q"
 *             elif t == NPY_ULONGLONG:   f[0] = 81  #"Q"
 */
      __pyx_t_3 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_ULONG); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 833, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_4 = PyObject_RichCompare(__pyx_v_t, __pyx_t_3, Py_EQ); __Pyx_XGOTREF(__pyx_t_4); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 833, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_4); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(1, 833, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 76;
        goto __pyx_L15;
      }

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":834
 *             elif t == NPY_LONG:        f[0] = 108 #"l"
 *             elif t == NPY_ULONG:       f[0] = 76  #"L"
 *             elif t == NPY_LONGLONG:    f[0] = 113 #"q"             # <<<<<<<<<<<<<<
 *             elif t == NPY_ULONGLONG:   f[0] = 81  #"Q"
 *             elif t == NPY_FLOAT:       f[0] = 102 #"f"
 */
      __pyx_t_4 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_LONGLONG); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 834, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      __pyx_t_3 = PyObject_RichCompare(__pyx_v_t, __pyx_t_4, Py_EQ); __Pyx_XGOTREF(__pyx_t_3); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 834, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(1, 834, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 0x71;
        goto __pyx_L15;
      }

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":835
 *             elif t == NPY_ULONG:       f[0] = 76  #"L"
 *             elif t == NPY_LONGLONG:    f[0] = 113 #"q"
 *             elif t == NPY_ULONGLONG:   f[0] = 81  #"Q"             # <<<<<<<<<<<<<<
 *             elif t == NPY_FLOAT:       f[0] = 102 #"f"
 *             elif t == NPY_DOUBLE:      f[0] = 100 #"d"
 */
      __pyx_t_3 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_ULONGLONG); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 835, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_4 = PyObject_RichCompare(__pyx_v_t, __pyx_t_3, Py_EQ); __Pyx_XGOTREF(__pyx_t_4); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 835, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_4); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(1, 835, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 81;
        goto __pyx_L15;
      }

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":836
 *             elif t == NPY_LONGLONG:    f[0] = 113 #"q"
 *             elif t == NPY_ULONGLONG:   f[0] = 81  #"Q"
 *             elif t == NPY_FLOAT:       f[0] = 102 #"f"             # <<<<<<<<<<<<<<
 *             elif t == NPY_DOUBLE:      f[0] = 100 #"d"
 *             elif t == NPY_LONGDOUBLE:  f[0] = 103 #"g"
 */
      __pyx_t_4 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_FLOAT); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 836, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      __pyx_t_3 = PyObject_RichCompare(__pyx_v_t, __pyx_t_4, Py_EQ); __Pyx_XGOTREF(__pyx_t_3); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 836, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(1, 836, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 0x66;
        goto __pyx_L15;
      }

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":837
 *             elif t == NPY_ULONGLONG:   f[0] = 81  #"Q"
 *             elif t == NPY_FLOAT:       f[0] = 102 #"f"
 *             elif t == NPY_DOUBLE:      f[0] = 100 #"d"             # <<<<<<<<<<<<<<
 *             elif t == NPY_LONGDOUBLE:  f[0] = 103 #"g"
 *             elif t == NPY_CFLOAT:      f[0] = 90; f[1] = 102; f += 1 # Zf
 */
      __pyx_t_3 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_DOUBLE); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 837, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_4 = PyObject_RichCompare(__pyx_v_t, __pyx_t_3, Py_EQ); __Pyx_XGOTREF(__pyx_t_4); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 837, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_4); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(1, 837, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 0x64;
        goto __pyx_L15;
      }

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":838
 *             elif t == NPY_FLOAT:       f[0] = 102 #"f"
 *             elif t == NPY_DOUBLE:      f[0] = 100 #"d"
 *             elif t == NPY_LONGDOUBLE:  f[0] = 103 #"g"             # <<<<<<<<<<<<<<
 *             elif t == NPY_CFLOAT:      f[0] = 90; f[1] = 102; f += 1 # Zf
 *             elif t == NPY_CDOUBLE:     f[0] = 90; f[1] = 100; f += 1 # Zd
 */
      __pyx_t_4 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_LONGDOUBLE); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 838, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      __pyx_t_3 = PyObject_RichCompare(__pyx_v_t, __pyx_t_4, Py_EQ); __Pyx_XGOTREF(__pyx_t_3); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 838, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(1, 838, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 0x67;
        goto __pyx_L15;
      }

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":839
 *             elif t == NPY_DOUBLE:      f[0] = 100 #"d"
 *             elif t == NPY_LONGDOUBLE:  f[0] = 103 #"g"
 *             elif t == NPY_CFLOAT:      f[0] = 90; f[1] = 102; f += 1 # Zf             # <<<<<<<<<<<<<<
 *             elif t == NPY_CDOUBLE:     f[0] = 90; f[1] = 100; f += 1 # Zd
 *             elif t == NPY_CLONGDOUBLE: f[0] = 90; f[1] = 103; f += 1 # Zg
 */
      __pyx_t_3 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_CFLOAT); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 839, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_4 = PyObject_RichCompare(__pyx_v_t, __pyx_t_3, Py_EQ); __Pyx_XGOTREF(__pyx_t_4); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 839, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_4); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(1, 839, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 90;
        (__pyx_v_f[1]) = 0x66;
        __pyx_v_f = (__pyx_v_f + 1);
        goto __pyx_L15;
      }

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":840
 *             elif t == NPY_LONGDOUBLE:  f[0] = 103 #"g"
 *             elif t == NPY_CFLOAT:      f[0] = 90; f[1] = 102; f += 1 # Zf
 *             elif t == NPY_CDOUBLE:     f[0] = 90; f[1] = 100; f += 1 # Zd             # <<<<<<<<<<<<<<
 *             elif t == NPY_CLONGDOUBLE: f[0] = 90; f[1] = 103; f += 1 # Zg
 *             elif t == NPY_OBJECT:      f[0] = 79 #"O"
 */
      __pyx_t_4 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_CDOUBLE); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 840, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      __pyx_t_3 = PyObject_RichCompare(__pyx_v_t, __pyx_t_4, Py_EQ); __Pyx_XGOTREF(__pyx_t_3); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 840, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(1, 840, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 90;
        (__pyx_v_f[1]) = 0x64;
        __pyx_v_f = (__pyx_v_f + 1);
        goto __pyx_L15;
      }

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":841
 *             elif t == NPY_CFLOAT:      f[0] = 90; f[1] = 102; f += 1 # Zf
 *             elif t == NPY_CDOUBLE:     f[0] = 90; f[1] = 100; f += 1 # Zd
 *             elif t == NPY_CLONGDOUBLE: f[0] = 90; f[1] = 103; f += 1 # Zg             # <<<<<<<<<<<<<<
 *             elif t == NPY_OBJECT:      f[0] = 79 #"O"
 *             else:
 */
      __pyx_t_3 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_CLONGDOUBLE); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 841, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_4 = PyObject_RichCompare(__pyx_v_t, __pyx_t_3, Py_EQ); __Pyx_XGOTREF(__pyx_t_4); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 841, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_4); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(1, 841, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 90;
        (__pyx_v_f[1]) = 0x67;
        __pyx_v_f = (__pyx_v_f + 1);
        goto __pyx_L15;
      }

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":842
 *             elif t == NPY_CDOUBLE:     f[0] = 90; f[1] = 100; f += 1 # Zd
 *             elif t == NPY_CLONGDOUBLE: f[0] = 90; f[1] = 103; f += 1 # Zg
 *             elif t == NPY_OBJECT:      f[0] = 79 #"O"             # <<<<<<<<<<<<<<
 *             else:
 *                 raise ValueError(u"unknown dtype code in numpy.pxd (%d)" % t)
 */
      __pyx_t_4 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_OBJECT); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 842, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      __pyx_t_3 = PyObject_RichCompare(__pyx_v_t, __pyx_t_4, Py_EQ); __Pyx_XGOTREF(__pyx_t_3); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 842, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(1, 842, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 79;
        goto __pyx_L15;
      }

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":844
 *             elif t == NPY_OBJECT:      f[0] = 79 #"O"
 *             else:
 *                 raise ValueError(u"unknown dtype code in numpy.pxd (%d)" % t)             # <<<<<<<<<<<<<<
 *             f += 1
 *         else:
 */
      /*else*/ {
        __pyx_t_3 = PyUnicode_Format(__pyx_kp_u_unknown_dtype_code_in_numpy_pxd, __pyx_v_t); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 844, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_3);
        __pyx_t_4 = PyTuple_New(1); if (unlikely(!__pyx_t_4)) __PYX_ERR(1, 844, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_4);
        __Pyx_GIVEREF(__pyx_t_3);
        PyTuple_SET_ITEM(__pyx_t_4, 0, __pyx_t_3);
        __pyx_t_3 = 0;
        __pyx_t_3 = __Pyx_PyObject_Call(__pyx_builtin_ValueError, __pyx_t_4, NULL); if (unlikely(!__pyx_t_3)) __PYX_ERR(1, 844, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_3);
        __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
        __Pyx_Raise(__pyx_t_3, 0, 0, 0);
        __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
        __PYX_ERR(1, 844, __pyx_L1_error)
      }
      __pyx_L15:;

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":845
 *             else:
 *                 raise ValueError(u"unknown dtype code in numpy.pxd (%d)" % t)
 *             f += 1             # <<<<<<<<<<<<<<
 *         else:
 *             # Cython ignores struct boundary information ("T{...}"),
 */
      __pyx_v_f = (__pyx_v_f + 1);

      /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":820
 *         offset[0] += child.itemsize
 * 
 *         if not PyDataType_HASFIELDS(child):             # <<<<<<<<<<<<<<
 *             t = child.type_num
 *             if end - f < 5:
 */
      goto __pyx_L13;
    }

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":849
 *             # Cython ignores struct boundary information ("T{...}"),
 *             # so don't output it
 *             f = _util_dtypestring(child, f, end, offset)             # <<<<<<<<<<<<<<
 *     return f
 * 
 */
    /*else*/ {
      __pyx_t_9 = __pyx_f_5numpy__util_dtypestring(__pyx_v_child, __pyx_v_f, __pyx_v_end, __pyx_v_offset); if (unlikely(__pyx_t_9 == NULL)) __PYX_ERR(1, 849, __pyx_L1_error)
      __pyx_v_f = __pyx_t_9;
    }
    __pyx_L13:;

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":794
 *     cdef tuple fields
 * 
 *     for childname in descr.names:             # <<<<<<<<<<<<<<
 *         fields = descr.fields[childname]
 *         child, new_offset = fields
 */
  }
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":850
 *             # so don't output it
 *             f = _util_dtypestring(child, f, end, offset)
 *     return f             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_r = __pyx_v_f;
  goto __pyx_L0;

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":785
 *     return PyArray_MultiIterNew(5, <void*>a, <void*>b, <void*>c, <void*> d, <void*> e)
 * 
 * cdef inline char* _util_dtypestring(dtype descr, char* f, char* end, int* offset) except NULL:             # <<<<<<<<<<<<<<
 *     # Recursive utility function used in __getbuffer__ to get format
 *     # string. The new location in the format string is returned.
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_XDECREF(__pyx_t_3);
  __Pyx_XDECREF(__pyx_t_4);
  __Pyx_AddTraceback("numpy._util_dtypestring", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_XDECREF((PyObject *)__pyx_v_child);
  __Pyx_XDECREF(__pyx_v_fields);
  __Pyx_XDECREF(__pyx_v_childname);
  __Pyx_XDECREF(__pyx_v_new_offset);
  __Pyx_XDECREF(__pyx_v_t);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":966
 * 
 * 
 * cdef inline void set_array_base(ndarray arr, object base):             # <<<<<<<<<<<<<<
 *      cdef PyObject* baseptr
 *      if base is None:
 */

static CYTHON_INLINE void __pyx_f_5numpy_set_array_base(PyArrayObject *__pyx_v_arr, PyObject *__pyx_v_base) {
  PyObject *__pyx_v_baseptr;
  __Pyx_RefNannyDeclarations
  int __pyx_t_1;
  int __pyx_t_2;
  __Pyx_RefNannySetupContext("set_array_base", 0);

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":968
 * cdef inline void set_array_base(ndarray arr, object base):
 *      cdef PyObject* baseptr
 *      if base is None:             # <<<<<<<<<<<<<<
 *          baseptr = NULL
 *      else:
 */
  __pyx_t_1 = (__pyx_v_base == Py_None);
  __pyx_t_2 = (__pyx_t_1 != 0);
  if (__pyx_t_2) {

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":969
 *      cdef PyObject* baseptr
 *      if base is None:
 *          baseptr = NULL             # <<<<<<<<<<<<<<
 *      else:
 *          Py_INCREF(base) # important to do this before decref below!
 */
    __pyx_v_baseptr = NULL;

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":968
 * cdef inline void set_array_base(ndarray arr, object base):
 *      cdef PyObject* baseptr
 *      if base is None:             # <<<<<<<<<<<<<<
 *          baseptr = NULL
 *      else:
 */
    goto __pyx_L3;
  }

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":971
 *          baseptr = NULL
 *      else:
 *          Py_INCREF(base) # important to do this before decref below!             # <<<<<<<<<<<<<<
 *          baseptr = <PyObject*>base
 *      Py_XDECREF(arr.base)
 */
  /*else*/ {
    Py_INCREF(__pyx_v_base);

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":972
 *      else:
 *          Py_INCREF(base) # important to do this before decref below!
 *          baseptr = <PyObject*>base             # <<<<<<<<<<<<<<
 *      Py_XDECREF(arr.base)
 *      arr.base = baseptr
 */
    __pyx_v_baseptr = ((PyObject *)__pyx_v_base);
  }
  __pyx_L3:;

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":973
 *          Py_INCREF(base) # important to do this before decref below!
 *          baseptr = <PyObject*>base
 *      Py_XDECREF(arr.base)             # <<<<<<<<<<<<<<
 *      arr.base = baseptr
 * 
 */
  Py_XDECREF(__pyx_v_arr->base);

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":974
 *          baseptr = <PyObject*>base
 *      Py_XDECREF(arr.base)
 *      arr.base = baseptr             # <<<<<<<<<<<<<<
 * 
 * cdef inline object get_array_base(ndarray arr):
 */
  __pyx_v_arr->base = __pyx_v_baseptr;

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":966
 * 
 * 
 * cdef inline void set_array_base(ndarray arr, object base):             # <<<<<<<<<<<<<<
 *      cdef PyObject* baseptr
 *      if base is None:
 */

  /* function exit code */
  __Pyx_RefNannyFinishContext();
}

/* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":976
 *      arr.base = baseptr
 * 
 * cdef inline object get_array_base(ndarray arr):             # <<<<<<<<<<<<<<
 *     if arr.base is NULL:
 *         return None
 */

static CYTHON_INLINE PyObject *__pyx_f_5numpy_get_array_base(PyArrayObject *__pyx_v_arr) {
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  int __pyx_t_1;
  __Pyx_RefNannySetupContext("get_array_base", 0);

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":977
 * 
 * cdef inline object get_array_base(ndarray arr):
 *     if arr.base is NULL:             # <<<<<<<<<<<<<<
 *         return None
 *     else:
 */
  __pyx_t_1 = ((__pyx_v_arr->base == NULL) != 0);
  if (__pyx_t_1) {

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":978
 * cdef inline object get_array_base(ndarray arr):
 *     if arr.base is NULL:
 *         return None             # <<<<<<<<<<<<<<
 *     else:
 *         return <object>arr.base
 */
    __Pyx_XDECREF(__pyx_r);
    __Pyx_INCREF(Py_None);
    __pyx_r = Py_None;
    goto __pyx_L0;

    /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":977
 * 
 * cdef inline object get_array_base(ndarray arr):
 *     if arr.base is NULL:             # <<<<<<<<<<<<<<
 *         return None
 *     else:
 */
  }

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":980
 *         return None
 *     else:
 *         return <object>arr.base             # <<<<<<<<<<<<<<
 */
  /*else*/ {
    __Pyx_XDECREF(__pyx_r);
    __Pyx_INCREF(((PyObject *)__pyx_v_arr->base));
    __pyx_r = ((PyObject *)__pyx_v_arr->base);
    goto __pyx_L0;
  }

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":976
 *      arr.base = baseptr
 * 
 * cdef inline object get_array_base(ndarray arr):             # <<<<<<<<<<<<<<
 *     if arr.base is NULL:
 *         return None
 */

  /* function exit code */
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

static PyMethodDef __pyx_methods[] = {
  {0, 0, 0, 0}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef __pyx_moduledef = {
  #if PY_VERSION_HEX < 0x03020000
    { PyObject_HEAD_INIT(NULL) NULL, 0, NULL },
  #else
    PyModuleDef_HEAD_INIT,
  #endif
    "gpu_mv",
    0, /* m_doc */
    -1, /* m_size */
    __pyx_methods /* m_methods */,
    NULL, /* m_reload */
    NULL, /* m_traverse */
    NULL, /* m_clear */
    NULL /* m_free */
};
#endif

static __Pyx_StringTabEntry __pyx_string_tab[] = {
  {&__pyx_kp_s_D_v_haoq_github_fcis_prerelease, __pyx_k_D_v_haoq_github_fcis_prerelease, sizeof(__pyx_k_D_v_haoq_github_fcis_prerelease), 0, 0, 1, 0},
  {&__pyx_kp_u_Format_string_allocated_too_shor, __pyx_k_Format_string_allocated_too_shor, sizeof(__pyx_k_Format_string_allocated_too_shor), 0, 1, 0, 0},
  {&__pyx_kp_u_Format_string_allocated_too_shor_2, __pyx_k_Format_string_allocated_too_shor_2, sizeof(__pyx_k_Format_string_allocated_too_shor_2), 0, 1, 0, 0},
  {&__pyx_kp_u_Non_native_byte_order_not_suppor, __pyx_k_Non_native_byte_order_not_suppor, sizeof(__pyx_k_Non_native_byte_order_not_suppor), 0, 1, 0, 0},
  {&__pyx_n_s_RuntimeError, __pyx_k_RuntimeError, sizeof(__pyx_k_RuntimeError), 0, 0, 1, 1},
  {&__pyx_n_s_ValueError, __pyx_k_ValueError, sizeof(__pyx_k_ValueError), 0, 0, 1, 1},
  {&__pyx_n_s_all_box_num, __pyx_k_all_box_num, sizeof(__pyx_k_all_box_num), 0, 0, 1, 1},
  {&__pyx_n_s_all_boxes, __pyx_k_all_boxes, sizeof(__pyx_k_all_boxes), 0, 0, 1, 1},
  {&__pyx_n_s_all_masks, __pyx_k_all_masks, sizeof(__pyx_k_all_masks), 0, 0, 1, 1},
  {&__pyx_n_s_binary_thresh, __pyx_k_binary_thresh, sizeof(__pyx_k_binary_thresh), 0, 0, 1, 1},
  {&__pyx_n_s_boxes_dim, __pyx_k_boxes_dim, sizeof(__pyx_k_boxes_dim), 0, 0, 1, 1},
  {&__pyx_n_s_candidate_inds, __pyx_k_candidate_inds, sizeof(__pyx_k_candidate_inds), 0, 0, 1, 1},
  {&__pyx_n_s_candidate_num, __pyx_k_candidate_num, sizeof(__pyx_k_candidate_num), 0, 0, 1, 1},
  {&__pyx_n_s_candidate_start, __pyx_k_candidate_start, sizeof(__pyx_k_candidate_start), 0, 0, 1, 1},
  {&__pyx_n_s_candidate_weights, __pyx_k_candidate_weights, sizeof(__pyx_k_candidate_weights), 0, 0, 1, 1},
  {&__pyx_n_s_device_id, __pyx_k_device_id, sizeof(__pyx_k_device_id), 0, 0, 1, 1},
  {&__pyx_n_s_dtype, __pyx_k_dtype, sizeof(__pyx_k_dtype), 0, 0, 1, 1},
  {&__pyx_n_s_float32, __pyx_k_float32, sizeof(__pyx_k_float32), 0, 0, 1, 1},
  {&__pyx_n_s_image_height, __pyx_k_image_height, sizeof(__pyx_k_image_height), 0, 0, 1, 1},
  {&__pyx_n_s_image_width, __pyx_k_image_width, sizeof(__pyx_k_image_width), 0, 0, 1, 1},
  {&__pyx_n_s_import, __pyx_k_import, sizeof(__pyx_k_import), 0, 0, 1, 1},
  {&__pyx_n_s_int32, __pyx_k_int32, sizeof(__pyx_k_int32), 0, 0, 1, 1},
  {&__pyx_n_s_lib_mask_gpu_mv, __pyx_k_lib_mask_gpu_mv, sizeof(__pyx_k_lib_mask_gpu_mv), 0, 0, 1, 1},
  {&__pyx_n_s_main, __pyx_k_main, sizeof(__pyx_k_main), 0, 0, 1, 1},
  {&__pyx_n_s_mask_size, __pyx_k_mask_size, sizeof(__pyx_k_mask_size), 0, 0, 1, 1},
  {&__pyx_n_s_mv, __pyx_k_mv, sizeof(__pyx_k_mv), 0, 0, 1, 1},
  {&__pyx_kp_u_ndarray_is_not_C_contiguous, __pyx_k_ndarray_is_not_C_contiguous, sizeof(__pyx_k_ndarray_is_not_C_contiguous), 0, 1, 0, 0},
  {&__pyx_kp_u_ndarray_is_not_Fortran_contiguou, __pyx_k_ndarray_is_not_Fortran_contiguou, sizeof(__pyx_k_ndarray_is_not_Fortran_contiguou), 0, 1, 0, 0},
  {&__pyx_n_s_np, __pyx_k_np, sizeof(__pyx_k_np), 0, 0, 1, 1},
  {&__pyx_n_s_numpy, __pyx_k_numpy, sizeof(__pyx_k_numpy), 0, 0, 1, 1},
  {&__pyx_n_s_range, __pyx_k_range, sizeof(__pyx_k_range), 0, 0, 1, 1},
  {&__pyx_n_s_result_box, __pyx_k_result_box, sizeof(__pyx_k_result_box), 0, 0, 1, 1},
  {&__pyx_n_s_result_mask, __pyx_k_result_mask, sizeof(__pyx_k_result_mask), 0, 0, 1, 1},
  {&__pyx_n_s_result_num, __pyx_k_result_num, sizeof(__pyx_k_result_num), 0, 0, 1, 1},
  {&__pyx_n_s_test, __pyx_k_test, sizeof(__pyx_k_test), 0, 0, 1, 1},
  {&__pyx_kp_u_unknown_dtype_code_in_numpy_pxd, __pyx_k_unknown_dtype_code_in_numpy_pxd, sizeof(__pyx_k_unknown_dtype_code_in_numpy_pxd), 0, 1, 0, 0},
  {&__pyx_n_s_zeros, __pyx_k_zeros, sizeof(__pyx_k_zeros), 0, 0, 1, 1},
  {0, 0, 0, 0, 0, 0, 0}
};
static int __Pyx_InitCachedBuiltins(void) {
  __pyx_builtin_ValueError = __Pyx_GetBuiltinName(__pyx_n_s_ValueError); if (!__pyx_builtin_ValueError) __PYX_ERR(1, 218, __pyx_L1_error)
  __pyx_builtin_range = __Pyx_GetBuiltinName(__pyx_n_s_range); if (!__pyx_builtin_range) __PYX_ERR(1, 231, __pyx_L1_error)
  __pyx_builtin_RuntimeError = __Pyx_GetBuiltinName(__pyx_n_s_RuntimeError); if (!__pyx_builtin_RuntimeError) __PYX_ERR(1, 799, __pyx_L1_error)
  return 0;
  __pyx_L1_error:;
  return -1;
}

static int __Pyx_InitCachedConstants(void) {
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("__Pyx_InitCachedConstants", 0);

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":218
 *             if ((flags & pybuf.PyBUF_C_CONTIGUOUS == pybuf.PyBUF_C_CONTIGUOUS)
 *                 and not PyArray_CHKFLAGS(self, NPY_C_CONTIGUOUS)):
 *                 raise ValueError(u"ndarray is not C contiguous")             # <<<<<<<<<<<<<<
 * 
 *             if ((flags & pybuf.PyBUF_F_CONTIGUOUS == pybuf.PyBUF_F_CONTIGUOUS)
 */
  __pyx_tuple_ = PyTuple_Pack(1, __pyx_kp_u_ndarray_is_not_C_contiguous); if (unlikely(!__pyx_tuple_)) __PYX_ERR(1, 218, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple_);
  __Pyx_GIVEREF(__pyx_tuple_);

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":222
 *             if ((flags & pybuf.PyBUF_F_CONTIGUOUS == pybuf.PyBUF_F_CONTIGUOUS)
 *                 and not PyArray_CHKFLAGS(self, NPY_F_CONTIGUOUS)):
 *                 raise ValueError(u"ndarray is not Fortran contiguous")             # <<<<<<<<<<<<<<
 * 
 *             info.buf = PyArray_DATA(self)
 */
  __pyx_tuple__2 = PyTuple_Pack(1, __pyx_kp_u_ndarray_is_not_Fortran_contiguou); if (unlikely(!__pyx_tuple__2)) __PYX_ERR(1, 222, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__2);
  __Pyx_GIVEREF(__pyx_tuple__2);

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":259
 *                 if ((descr.byteorder == c'>' and little_endian) or
 *                     (descr.byteorder == c'<' and not little_endian)):
 *                     raise ValueError(u"Non-native byte order not supported")             # <<<<<<<<<<<<<<
 *                 if   t == NPY_BYTE:        f = "b"
 *                 elif t == NPY_UBYTE:       f = "B"
 */
  __pyx_tuple__3 = PyTuple_Pack(1, __pyx_kp_u_Non_native_byte_order_not_suppor); if (unlikely(!__pyx_tuple__3)) __PYX_ERR(1, 259, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__3);
  __Pyx_GIVEREF(__pyx_tuple__3);

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":799
 * 
 *         if (end - f) - <int>(new_offset - offset[0]) < 15:
 *             raise RuntimeError(u"Format string allocated too short, see comment in numpy.pxd")             # <<<<<<<<<<<<<<
 * 
 *         if ((child.byteorder == c'>' and little_endian) or
 */
  __pyx_tuple__4 = PyTuple_Pack(1, __pyx_kp_u_Format_string_allocated_too_shor); if (unlikely(!__pyx_tuple__4)) __PYX_ERR(1, 799, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__4);
  __Pyx_GIVEREF(__pyx_tuple__4);

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":803
 *         if ((child.byteorder == c'>' and little_endian) or
 *             (child.byteorder == c'<' and not little_endian)):
 *             raise ValueError(u"Non-native byte order not supported")             # <<<<<<<<<<<<<<
 *             # One could encode it in the format string and have Cython
 *             # complain instead, BUT: < and > in format strings also imply
 */
  __pyx_tuple__5 = PyTuple_Pack(1, __pyx_kp_u_Non_native_byte_order_not_suppor); if (unlikely(!__pyx_tuple__5)) __PYX_ERR(1, 803, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__5);
  __Pyx_GIVEREF(__pyx_tuple__5);

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":823
 *             t = child.type_num
 *             if end - f < 5:
 *                 raise RuntimeError(u"Format string allocated too short.")             # <<<<<<<<<<<<<<
 * 
 *             # Until ticket #99 is fixed, use integers to avoid warnings
 */
  __pyx_tuple__6 = PyTuple_Pack(1, __pyx_kp_u_Format_string_allocated_too_shor_2); if (unlikely(!__pyx_tuple__6)) __PYX_ERR(1, 823, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__6);
  __Pyx_GIVEREF(__pyx_tuple__6);

  /* "lib/mask/gpu_mv.pyx":19
 * # masks: n * 1 * 21 * 21
 * # scores: n * 21
 * def mv(np.ndarray[np.float32_t, ndim=2] all_boxes,             # <<<<<<<<<<<<<<
 *                 np.ndarray[np.float32_t, ndim=4] all_masks,
 *                 np.ndarray[np.int32_t, ndim=1] candidate_inds,
 */
  __pyx_tuple__7 = PyTuple_Pack(16, __pyx_n_s_all_boxes, __pyx_n_s_all_masks, __pyx_n_s_candidate_inds, __pyx_n_s_candidate_start, __pyx_n_s_candidate_weights, __pyx_n_s_binary_thresh, __pyx_n_s_image_height, __pyx_n_s_image_width, __pyx_n_s_device_id, __pyx_n_s_all_box_num, __pyx_n_s_boxes_dim, __pyx_n_s_mask_size, __pyx_n_s_candidate_num, __pyx_n_s_result_num, __pyx_n_s_result_mask, __pyx_n_s_result_box); if (unlikely(!__pyx_tuple__7)) __PYX_ERR(0, 19, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__7);
  __Pyx_GIVEREF(__pyx_tuple__7);
  __pyx_codeobj__8 = (PyObject*)__Pyx_PyCode_New(9, 0, 16, 0, 0, __pyx_empty_bytes, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_tuple__7, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_kp_s_D_v_haoq_github_fcis_prerelease, __pyx_n_s_mv, 19, __pyx_empty_bytes); if (unlikely(!__pyx_codeobj__8)) __PYX_ERR(0, 19, __pyx_L1_error)
  __Pyx_RefNannyFinishContext();
  return 0;
  __pyx_L1_error:;
  __Pyx_RefNannyFinishContext();
  return -1;
}

static int __Pyx_InitGlobals(void) {
  if (__Pyx_InitStrings(__pyx_string_tab) < 0) __PYX_ERR(0, 1, __pyx_L1_error);
  __pyx_int_1 = PyInt_FromLong(1); if (unlikely(!__pyx_int_1)) __PYX_ERR(0, 1, __pyx_L1_error)
  return 0;
  __pyx_L1_error:;
  return -1;
}

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC initgpu_mv(void); /*proto*/
PyMODINIT_FUNC initgpu_mv(void)
#else
PyMODINIT_FUNC PyInit_gpu_mv(void); /*proto*/
PyMODINIT_FUNC PyInit_gpu_mv(void)
#endif
{
  PyObject *__pyx_t_1 = NULL;
  __Pyx_RefNannyDeclarations
  #if CYTHON_REFNANNY
  __Pyx_RefNanny = __Pyx_RefNannyImportAPI("refnanny");
  if (!__Pyx_RefNanny) {
      PyErr_Clear();
      __Pyx_RefNanny = __Pyx_RefNannyImportAPI("Cython.Runtime.refnanny");
      if (!__Pyx_RefNanny)
          Py_FatalError("failed to import 'refnanny' module");
  }
  #endif
  __Pyx_RefNannySetupContext("PyMODINIT_FUNC PyInit_gpu_mv(void)", 0);
  if (__Pyx_check_binary_version() < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  __pyx_empty_tuple = PyTuple_New(0); if (unlikely(!__pyx_empty_tuple)) __PYX_ERR(0, 1, __pyx_L1_error)
  __pyx_empty_bytes = PyBytes_FromStringAndSize("", 0); if (unlikely(!__pyx_empty_bytes)) __PYX_ERR(0, 1, __pyx_L1_error)
  __pyx_empty_unicode = PyUnicode_FromStringAndSize("", 0); if (unlikely(!__pyx_empty_unicode)) __PYX_ERR(0, 1, __pyx_L1_error)
  #ifdef __Pyx_CyFunction_USED
  if (__pyx_CyFunction_init() < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  #endif
  #ifdef __Pyx_FusedFunction_USED
  if (__pyx_FusedFunction_init() < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  #endif
  #ifdef __Pyx_Coroutine_USED
  if (__pyx_Coroutine_init() < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  #endif
  #ifdef __Pyx_Generator_USED
  if (__pyx_Generator_init() < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  #endif
  #ifdef __Pyx_StopAsyncIteration_USED
  if (__pyx_StopAsyncIteration_init() < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  #endif
  /*--- Library function declarations ---*/
  /*--- Threads initialization code ---*/
  #if defined(__PYX_FORCE_INIT_THREADS) && __PYX_FORCE_INIT_THREADS
  #ifdef WITH_THREAD /* Python build with threading support? */
  PyEval_InitThreads();
  #endif
  #endif
  /*--- Module creation code ---*/
  #if PY_MAJOR_VERSION < 3
  __pyx_m = Py_InitModule4("gpu_mv", __pyx_methods, 0, 0, PYTHON_API_VERSION); Py_XINCREF(__pyx_m);
  #else
  __pyx_m = PyModule_Create(&__pyx_moduledef);
  #endif
  if (unlikely(!__pyx_m)) __PYX_ERR(0, 1, __pyx_L1_error)
  __pyx_d = PyModule_GetDict(__pyx_m); if (unlikely(!__pyx_d)) __PYX_ERR(0, 1, __pyx_L1_error)
  Py_INCREF(__pyx_d);
  __pyx_b = PyImport_AddModule(__Pyx_BUILTIN_MODULE_NAME); if (unlikely(!__pyx_b)) __PYX_ERR(0, 1, __pyx_L1_error)
  #if CYTHON_COMPILING_IN_PYPY
  Py_INCREF(__pyx_b);
  #endif
  if (PyObject_SetAttrString(__pyx_m, "__builtins__", __pyx_b) < 0) __PYX_ERR(0, 1, __pyx_L1_error);
  /*--- Initialize various global constants etc. ---*/
  if (__Pyx_InitGlobals() < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  #if PY_MAJOR_VERSION < 3 && (__PYX_DEFAULT_STRING_ENCODING_IS_ASCII || __PYX_DEFAULT_STRING_ENCODING_IS_DEFAULT)
  if (__Pyx_init_sys_getdefaultencoding_params() < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  #endif
  if (__pyx_module_is_main_lib__mask__gpu_mv) {
    if (PyObject_SetAttrString(__pyx_m, "__name__", __pyx_n_s_main) < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  }
  #if PY_MAJOR_VERSION >= 3
  {
    PyObject *modules = PyImport_GetModuleDict(); if (unlikely(!modules)) __PYX_ERR(0, 1, __pyx_L1_error)
    if (!PyDict_GetItemString(modules, "lib.mask.gpu_mv")) {
      if (unlikely(PyDict_SetItemString(modules, "lib.mask.gpu_mv", __pyx_m) < 0)) __PYX_ERR(0, 1, __pyx_L1_error)
    }
  }
  #endif
  /*--- Builtin init code ---*/
  if (__Pyx_InitCachedBuiltins() < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  /*--- Constants init code ---*/
  if (__Pyx_InitCachedConstants() < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  /*--- Global init code ---*/
  /*--- Variable export code ---*/
  /*--- Function export code ---*/
  /*--- Type init code ---*/
  /*--- Type import code ---*/
  __pyx_ptype_7cpython_4type_type = __Pyx_ImportType(__Pyx_BUILTIN_MODULE_NAME, "type", 
  #if CYTHON_COMPILING_IN_PYPY
  sizeof(PyTypeObject),
  #else
  sizeof(PyHeapTypeObject),
  #endif
  0); if (unlikely(!__pyx_ptype_7cpython_4type_type)) __PYX_ERR(2, 9, __pyx_L1_error)
  __pyx_ptype_5numpy_dtype = __Pyx_ImportType("numpy", "dtype", sizeof(PyArray_Descr), 0); if (unlikely(!__pyx_ptype_5numpy_dtype)) __PYX_ERR(1, 155, __pyx_L1_error)
  __pyx_ptype_5numpy_flatiter = __Pyx_ImportType("numpy", "flatiter", sizeof(PyArrayIterObject), 0); if (unlikely(!__pyx_ptype_5numpy_flatiter)) __PYX_ERR(1, 168, __pyx_L1_error)
  __pyx_ptype_5numpy_broadcast = __Pyx_ImportType("numpy", "broadcast", sizeof(PyArrayMultiIterObject), 0); if (unlikely(!__pyx_ptype_5numpy_broadcast)) __PYX_ERR(1, 172, __pyx_L1_error)
  __pyx_ptype_5numpy_ndarray = __Pyx_ImportType("numpy", "ndarray", sizeof(PyArrayObject), 0); if (unlikely(!__pyx_ptype_5numpy_ndarray)) __PYX_ERR(1, 181, __pyx_L1_error)
  __pyx_ptype_5numpy_ufunc = __Pyx_ImportType("numpy", "ufunc", sizeof(PyUFuncObject), 0); if (unlikely(!__pyx_ptype_5numpy_ufunc)) __PYX_ERR(1, 861, __pyx_L1_error)
  /*--- Variable import code ---*/
  /*--- Function import code ---*/
  /*--- Execution code ---*/
  #if defined(__Pyx_Generator_USED) || defined(__Pyx_Coroutine_USED)
  if (__Pyx_patch_abc() < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  #endif

  /* "lib/mask/gpu_mv.pyx":8
 * # --------------------------------------------------------
 * 
 * import numpy as np             # <<<<<<<<<<<<<<
 * cimport numpy as np
 * 
 */
  __pyx_t_1 = __Pyx_Import(__pyx_n_s_numpy, 0, -1); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 8, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  if (PyDict_SetItem(__pyx_d, __pyx_n_s_np, __pyx_t_1) < 0) __PYX_ERR(0, 8, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;

  /* "lib/mask/gpu_mv.pyx":11
 * cimport numpy as np
 * 
 * assert sizeof(int) == sizeof(np.int32_t)             # <<<<<<<<<<<<<<
 * 
 * cdef extern from "gpu_mv.hpp":
 */
  #ifndef CYTHON_WITHOUT_ASSERTIONS
  if (unlikely(!Py_OptimizeFlag)) {
    if (unlikely(!(((sizeof(int)) == (sizeof(__pyx_t_5numpy_int32_t))) != 0))) {
      PyErr_SetNone(PyExc_AssertionError);
      __PYX_ERR(0, 11, __pyx_L1_error)
    }
  }
  #endif

  /* "lib/mask/gpu_mv.pyx":19
 * # masks: n * 1 * 21 * 21
 * # scores: n * 21
 * def mv(np.ndarray[np.float32_t, ndim=2] all_boxes,             # <<<<<<<<<<<<<<
 *                 np.ndarray[np.float32_t, ndim=4] all_masks,
 *                 np.ndarray[np.int32_t, ndim=1] candidate_inds,
 */
  __pyx_t_1 = PyCFunction_NewEx(&__pyx_mdef_3lib_4mask_6gpu_mv_1mv, NULL, __pyx_n_s_lib_mask_gpu_mv); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 19, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  if (PyDict_SetItem(__pyx_d, __pyx_n_s_mv, __pyx_t_1) < 0) __PYX_ERR(0, 19, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;

  /* "lib/mask/gpu_mv.pyx":1
 * # --------------------------------------------------------             # <<<<<<<<<<<<<<
 * # Fully Convolutional Instance-aware Semantic Segmentation
 * # Copyright (c) 2017 Microsoft
 */
  __pyx_t_1 = PyDict_New(); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 1, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  if (PyDict_SetItem(__pyx_d, __pyx_n_s_test, __pyx_t_1) < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;

  /* "C:/Anaconda2/lib/site-packages/Cython/Includes/numpy/__init__.pxd":976
 *      arr.base = baseptr
 * 
 * cdef inline object get_array_base(ndarray arr):             # <<<<<<<<<<<<<<
 *     if arr.base is NULL:
 *         return None
 */

  /*--- Wrapped vars code ---*/

  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  if (__pyx_m) {
    if (__pyx_d) {
      __Pyx_AddTraceback("init lib.mask.gpu_mv", __pyx_clineno, __pyx_lineno, __pyx_filename);
    }
    Py_DECREF(__pyx_m); __pyx_m = 0;
  } else if (!PyErr_Occurred()) {
    PyErr_SetString(PyExc_ImportError, "init lib.mask.gpu_mv");
  }
  __pyx_L0:;
  __Pyx_RefNannyFinishContext();
  #if PY_MAJOR_VERSION < 3
  return;
  #else
  return __pyx_m;
  #endif
}

/* --- Runtime support code --- */
/* Refnanny */
#if CYTHON_REFNANNY
static __Pyx_RefNannyAPIStruct *__Pyx_RefNannyImportAPI(const char *modname) {
    PyObject *m = NULL, *p = NULL;
    void *r = NULL;
    m = PyImport_ImportModule((char *)modname);
    if (!m) goto end;
    p = PyObject_GetAttrString(m, (char *)"RefNannyAPI");
    if (!p) goto end;
    r = PyLong_AsVoidPtr(p);
end:
    Py_XDECREF(p);
    Py_XDECREF(m);
    return (__Pyx_RefNannyAPIStruct *)r;
}
#endif

/* RaiseArgTupleInvalid */
static void __Pyx_RaiseArgtupleInvalid(
    const char* func_name,
    int exact,
    Py_ssize_t num_min,
    Py_ssize_t num_max,
    Py_ssize_t num_found)
{
    Py_ssize_t num_expected;
    const char *more_or_less;
    if (num_found < num_min) {
        num_expected = num_min;
        more_or_less = "at least";
    } else {
        num_expected = num_max;
        more_or_less = "at most";
    }
    if (exact) {
        more_or_less = "exactly";
    }
    PyErr_Format(PyExc_TypeError,
                 "%.200s() takes %.8s %" CYTHON_FORMAT_SSIZE_T "d positional argument%.1s (%" CYTHON_FORMAT_SSIZE_T "d given)",
                 func_name, more_or_less, num_expected,
                 (num_expected == 1) ? "" : "s", num_found);
}

/* RaiseDoubleKeywords */
static void __Pyx_RaiseDoubleKeywordsError(
    const char* func_name,
    PyObject* kw_name)
{
    PyErr_Format(PyExc_TypeError,
        #if PY_MAJOR_VERSION >= 3
        "%s() got multiple values for keyword argument '%U'", func_name, kw_name);
        #else
        "%s() got multiple values for keyword argument '%s'", func_name,
        PyString_AsString(kw_name));
        #endif
}

/* ParseKeywords */
static int __Pyx_ParseOptionalKeywords(
    PyObject *kwds,
    PyObject **argnames[],
    PyObject *kwds2,
    PyObject *values[],
    Py_ssize_t num_pos_args,
    const char* function_name)
{
    PyObject *key = 0, *value = 0;
    Py_ssize_t pos = 0;
    PyObject*** name;
    PyObject*** first_kw_arg = argnames + num_pos_args;
    while (PyDict_Next(kwds, &pos, &key, &value)) {
        name = first_kw_arg;
        while (*name && (**name != key)) name++;
        if (*name) {
            values[name-argnames] = value;
            continue;
        }
        name = first_kw_arg;
        #if PY_MAJOR_VERSION < 3
        if (likely(PyString_CheckExact(key)) || likely(PyString_Check(key))) {
            while (*name) {
                if ((CYTHON_COMPILING_IN_PYPY || PyString_GET_SIZE(**name) == PyString_GET_SIZE(key))
                        && _PyString_Eq(**name, key)) {
                    values[name-argnames] = value;
                    break;
                }
                name++;
            }
            if (*name) continue;
            else {
                PyObject*** argname = argnames;
                while (argname != first_kw_arg) {
                    if ((**argname == key) || (
                            (CYTHON_COMPILING_IN_PYPY || PyString_GET_SIZE(**argname) == PyString_GET_SIZE(key))
                             && _PyString_Eq(**argname, key))) {
                        goto arg_passed_twice;
                    }
                    argname++;
                }
            }
        } else
        #endif
        if (likely(PyUnicode_Check(key))) {
            while (*name) {
                int cmp = (**name == key) ? 0 :
                #if !CYTHON_COMPILING_IN_PYPY && PY_MAJOR_VERSION >= 3
                    (PyUnicode_GET_SIZE(**name) != PyUnicode_GET_SIZE(key)) ? 1 :
                #endif
                    PyUnicode_Compare(**name, key);
                if (cmp < 0 && unlikely(PyErr_Occurred())) goto bad;
                if (cmp == 0) {
                    values[name-argnames] = value;
                    break;
                }
                name++;
            }
            if (*name) continue;
            else {
                PyObject*** argname = argnames;
                while (argname != first_kw_arg) {
                    int cmp = (**argname == key) ? 0 :
                    #if !CYTHON_COMPILING_IN_PYPY && PY_MAJOR_VERSION >= 3
                        (PyUnicode_GET_SIZE(**argname) != PyUnicode_GET_SIZE(key)) ? 1 :
                    #endif
                        PyUnicode_Compare(**argname, key);
                    if (cmp < 0 && unlikely(PyErr_Occurred())) goto bad;
                    if (cmp == 0) goto arg_passed_twice;
                    argname++;
                }
            }
        } else
            goto invalid_keyword_type;
        if (kwds2) {
            if (unlikely(PyDict_SetItem(kwds2, key, value))) goto bad;
        } else {
            goto invalid_keyword;
        }
    }
    return 0;
arg_passed_twice:
    __Pyx_RaiseDoubleKeywordsError(function_name, key);
    goto bad;
invalid_keyword_type:
    PyErr_Format(PyExc_TypeError,
        "%.200s() keywords must be strings", function_name);
    goto bad;
invalid_keyword:
    PyErr_Format(PyExc_TypeError,
    #if PY_MAJOR_VERSION < 3
        "%.200s() got an unexpected keyword argument '%.200s'",
        function_name, PyString_AsString(key));
    #else
        "%s() got an unexpected keyword argument '%U'",
        function_name, key);
    #endif
bad:
    return -1;
}

/* ArgTypeTest */
static void __Pyx_RaiseArgumentTypeInvalid(const char* name, PyObject *obj, PyTypeObject *type) {
    PyErr_Format(PyExc_TypeError,
        "Argument '%.200s' has incorrect type (expected %.200s, got %.200s)",
        name, type->tp_name, Py_TYPE(obj)->tp_name);
}
static CYTHON_INLINE int __Pyx_ArgTypeTest(PyObject *obj, PyTypeObject *type, int none_allowed,
    const char *name, int exact)
{
    if (unlikely(!type)) {
        PyErr_SetString(PyExc_SystemError, "Missing type object");
        return 0;
    }
    if (none_allowed && obj == Py_None) return 1;
    else if (exact) {
        if (likely(Py_TYPE(obj) == type)) return 1;
        #if PY_MAJOR_VERSION == 2
        else if ((type == &PyBaseString_Type) && likely(__Pyx_PyBaseString_CheckExact(obj))) return 1;
        #endif
    }
    else {
        if (likely(PyObject_TypeCheck(obj, type))) return 1;
    }
    __Pyx_RaiseArgumentTypeInvalid(name, obj, type);
    return 0;
}

/* BufferFormatCheck */
static CYTHON_INLINE int __Pyx_IsLittleEndian(void) {
  unsigned int n = 1;
  return *(unsigned char*)(&n) != 0;
}
static void __Pyx_BufFmt_Init(__Pyx_BufFmt_Context* ctx,
                              __Pyx_BufFmt_StackElem* stack,
                              __Pyx_TypeInfo* type) {
  stack[0].field = &ctx->root;
  stack[0].parent_offset = 0;
  ctx->root.type = type;
  ctx->root.name = "buffer dtype";
  ctx->root.offset = 0;
  ctx->head = stack;
  ctx->head->field = &ctx->root;
  ctx->fmt_offset = 0;
  ctx->head->parent_offset = 0;
  ctx->new_packmode = '@';
  ctx->enc_packmode = '@';
  ctx->new_count = 1;
  ctx->enc_count = 0;
  ctx->enc_type = 0;
  ctx->is_complex = 0;
  ctx->is_valid_array = 0;
  ctx->struct_alignment = 0;
  while (type->typegroup == 'S') {
    ++ctx->head;
    ctx->head->field = type->fields;
    ctx->head->parent_offset = 0;
    type = type->fields->type;
  }
}
static int __Pyx_BufFmt_ParseNumber(const char** ts) {
    int count;
    const char* t = *ts;
    if (*t < '0' || *t > '9') {
      return -1;
    } else {
        count = *t++ - '0';
        while (*t >= '0' && *t < '9') {
            count *= 10;
            count += *t++ - '0';
        }
    }
    *ts = t;
    return count;
}
static int __Pyx_BufFmt_ExpectNumber(const char **ts) {
    int number = __Pyx_BufFmt_ParseNumber(ts);
    if (number == -1)
        PyErr_Format(PyExc_ValueError,\
                     "Does not understand character buffer dtype format string ('%c')", **ts);
    return number;
}
static void __Pyx_BufFmt_RaiseUnexpectedChar(char ch) {
  PyErr_Format(PyExc_ValueError,
               "Unexpected format string character: '%c'", ch);
}
static const char* __Pyx_BufFmt_DescribeTypeChar(char ch, int is_complex) {
  switch (ch) {
    case 'c': return "'char'";
    case 'b': return "'signed char'";
    case 'B': return "'unsigned char'";
    case 'h': return "'short'";
    case 'H': return "'unsigned short'";
    case 'i': return "'int'";
    case 'I': return "'unsigned int'";
    case 'l': return "'long'";
    case 'L': return "'unsigned long'";
    case 'q': return "'long long'";
    case 'Q': return "'unsigned long long'";
    case 'f': return (is_complex ? "'complex float'" : "'float'");
    case 'd': return (is_complex ? "'complex double'" : "'double'");
    case 'g': return (is_complex ? "'complex long double'" : "'long double'");
    case 'T': return "a struct";
    case 'O': return "Python object";
    case 'P': return "a pointer";
    case 's': case 'p': return "a string";
    case 0: return "end";
    default: return "unparseable format string";
  }
}
static size_t __Pyx_BufFmt_TypeCharToStandardSize(char ch, int is_complex) {
  switch (ch) {
    case '?': case 'c': case 'b': case 'B': case 's': case 'p': return 1;
    case 'h': case 'H': return 2;
    case 'i': case 'I': case 'l': case 'L': return 4;
    case 'q': case 'Q': return 8;
    case 'f': return (is_complex ? 8 : 4);
    case 'd': return (is_complex ? 16 : 8);
    case 'g': {
      PyErr_SetString(PyExc_ValueError, "Python does not define a standard format string size for long double ('g')..");
      return 0;
    }
    case 'O': case 'P': return sizeof(void*);
    default:
      __Pyx_BufFmt_RaiseUnexpectedChar(ch);
      return 0;
    }
}
static size_t __Pyx_BufFmt_TypeCharToNativeSize(char ch, int is_complex) {
  switch (ch) {
    case 'c': case 'b': case 'B': case 's': case 'p': return 1;
    case 'h': case 'H': return sizeof(short);
    case 'i': case 'I': return sizeof(int);
    case 'l': case 'L': return sizeof(long);
    #ifdef HAVE_LONG_LONG
    case 'q': case 'Q': return sizeof(PY_LONG_LONG);
    #endif
    case 'f': return sizeof(float) * (is_complex ? 2 : 1);
    case 'd': return sizeof(double) * (is_complex ? 2 : 1);
    case 'g': return sizeof(long double) * (is_complex ? 2 : 1);
    case 'O': case 'P': return sizeof(void*);
    default: {
      __Pyx_BufFmt_RaiseUnexpectedChar(ch);
      return 0;
    }
  }
}
typedef struct { char c; short x; } __Pyx_st_short;
typedef struct { char c; int x; } __Pyx_st_int;
typedef struct { char c; long x; } __Pyx_st_long;
typedef struct { char c; float x; } __Pyx_st_float;
typedef struct { char c; double x; } __Pyx_st_double;
typedef struct { char c; long double x; } __Pyx_st_longdouble;
typedef struct { char c; void *x; } __Pyx_st_void_p;
#ifdef HAVE_LONG_LONG
typedef struct { char c; PY_LONG_LONG x; } __Pyx_st_longlong;
#endif
static size_t __Pyx_BufFmt_TypeCharToAlignment(char ch, CYTHON_UNUSED int is_complex) {
  switch (ch) {
    case '?': case 'c': case 'b': case 'B': case 's': case 'p': return 1;
    case 'h': case 'H': return sizeof(__Pyx_st_short) - sizeof(short);
    case 'i': case 'I': return sizeof(__Pyx_st_int) - sizeof(int);
    case 'l': case 'L': return sizeof(__Pyx_st_long) - sizeof(long);
#ifdef HAVE_LONG_LONG
    case 'q': case 'Q': return sizeof(__Pyx_st_longlong) - sizeof(PY_LONG_LONG);
#endif
    case 'f': return sizeof(__Pyx_st_float) - sizeof(float);
    case 'd': return sizeof(__Pyx_st_double) - sizeof(double);
    case 'g': return sizeof(__Pyx_st_longdouble) - sizeof(long double);
    case 'P': case 'O': return sizeof(__Pyx_st_void_p) - sizeof(void*);
    default:
      __Pyx_BufFmt_RaiseUnexpectedChar(ch);
      return 0;
    }
}
/* These are for computing the padding at the end of the struct to align
   on the first member of the struct. This will probably the same as above,
   but we don't have any guarantees.
 */
typedef struct { short x; char c; } __Pyx_pad_short;
typedef struct { int x; char c; } __Pyx_pad_int;
typedef struct { long x; char c; } __Pyx_pad_long;
typedef struct { float x; char c; } __Pyx_pad_float;
typedef struct { double x; char c; } __Pyx_pad_double;
typedef struct { long double x; char c; } __Pyx_pad_longdouble;
typedef struct { void *x; char c; } __Pyx_pad_void_p;
#ifdef HAVE_LONG_LONG
typedef struct { PY_LONG_LONG x; char c; } __Pyx_pad_longlong;
#endif
static size_t __Pyx_BufFmt_TypeCharToPadding(char ch, CYTHON_UNUSED int is_complex) {
  switch (ch) {
    case '?': case 'c': case 'b': case 'B': case 's': case 'p': return 1;
    case 'h': case 'H': return sizeof(__Pyx_pad_short) - sizeof(short);
    case 'i': case 'I': return sizeof(__Pyx_pad_int) - sizeof(int);
    case 'l': case 'L': return sizeof(__Pyx_pad_long) - sizeof(long);
#ifdef HAVE_LONG_LONG
    case 'q': case 'Q': return sizeof(__Pyx_pad_longlong) - sizeof(PY_LONG_LONG);
#endif
    case 'f': return sizeof(__Pyx_pad_float) - sizeof(float);
    case 'd': return sizeof(__Pyx_pad_double) - sizeof(double);
    case 'g': return sizeof(__Pyx_pad_longdouble) - sizeof(long double);
    case 'P': case 'O': return sizeof(__Pyx_pad_void_p) - sizeof(void*);
    default:
      __Pyx_BufFmt_RaiseUnexpectedChar(ch);
      return 0;
    }
}
static char __Pyx_BufFmt_TypeCharToGroup(char ch, int is_complex) {
  switch (ch) {
    case 'c':
        return 'H';
    case 'b': case 'h': case 'i':
    case 'l': case 'q': case 's': case 'p':
        return 'I';
    case 'B': case 'H': case 'I': case 'L': case 'Q':
        return 'U';
    case 'f': case 'd': case 'g':
        return (is_complex ? 'C' : 'R');
    case 'O':
        return 'O';
    case 'P':
        return 'P';
    default: {
      __Pyx_BufFmt_RaiseUnexpectedChar(ch);
      return 0;
    }
  }
}
static void __Pyx_BufFmt_RaiseExpected(__Pyx_BufFmt_Context* ctx) {
  if (ctx->head == NULL || ctx->head->field == &ctx->root) {
    const char* expected;
    const char* quote;
    if (ctx->head == NULL) {
      expected = "end";
      quote = "";
    } else {
      expected = ctx->head->field->type->name;
      quote = "'";
    }
    PyErr_Format(PyExc_ValueError,
                 "Buffer dtype mismatch, expected %s%s%s but got %s",
                 quote, expected, quote,
                 __Pyx_BufFmt_DescribeTypeChar(ctx->enc_type, ctx->is_complex));
  } else {
    __Pyx_StructField* field = ctx->head->field;
    __Pyx_StructField* parent = (ctx->head - 1)->field;
    PyErr_Format(PyExc_ValueError,
                 "Buffer dtype mismatch, expected '%s' but got %s in '%s.%s'",
                 field->type->name, __Pyx_BufFmt_DescribeTypeChar(ctx->enc_type, ctx->is_complex),
                 parent->type->name, field->name);
  }
}
static int __Pyx_BufFmt_ProcessTypeChunk(__Pyx_BufFmt_Context* ctx) {
  char group;
  size_t size, offset, arraysize = 1;
  if (ctx->enc_type == 0) return 0;
  if (ctx->head->field->type->arraysize[0]) {
    int i, ndim = 0;
    if (ctx->enc_type == 's' || ctx->enc_type == 'p') {
        ctx->is_valid_array = ctx->head->field->type->ndim == 1;
        ndim = 1;
        if (ctx->enc_count != ctx->head->field->type->arraysize[0]) {
            PyErr_Format(PyExc_ValueError,
                         "Expected a dimension of size %zu, got %zu",
                         ctx->head->field->type->arraysize[0], ctx->enc_count);
            return -1;
        }
    }
    if (!ctx->is_valid_array) {
      PyErr_Format(PyExc_ValueError, "Expected %d dimensions, got %d",
                   ctx->head->field->type->ndim, ndim);
      return -1;
    }
    for (i = 0; i < ctx->head->field->type->ndim; i++) {
      arraysize *= ctx->head->field->type->arraysize[i];
    }
    ctx->is_valid_array = 0;
    ctx->enc_count = 1;
  }
  group = __Pyx_BufFmt_TypeCharToGroup(ctx->enc_type, ctx->is_complex);
  do {
    __Pyx_StructField* field = ctx->head->field;
    __Pyx_TypeInfo* type = field->type;
    if (ctx->enc_packmode == '@' || ctx->enc_packmode == '^') {
      size = __Pyx_BufFmt_TypeCharToNativeSize(ctx->enc_type, ctx->is_complex);
    } else {
      size = __Pyx_BufFmt_TypeCharToStandardSize(ctx->enc_type, ctx->is_complex);
    }
    if (ctx->enc_packmode == '@') {
      size_t align_at = __Pyx_BufFmt_TypeCharToAlignment(ctx->enc_type, ctx->is_complex);
      size_t align_mod_offset;
      if (align_at == 0) return -1;
      align_mod_offset = ctx->fmt_offset % align_at;
      if (align_mod_offset > 0) ctx->fmt_offset += align_at - align_mod_offset;
      if (ctx->struct_alignment == 0)
          ctx->struct_alignment = __Pyx_BufFmt_TypeCharToPadding(ctx->enc_type,
                                                                 ctx->is_complex);
    }
    if (type->size != size || type->typegroup != group) {
      if (type->typegroup == 'C' && type->fields != NULL) {
        size_t parent_offset = ctx->head->parent_offset + field->offset;
        ++ctx->head;
        ctx->head->field = type->fields;
        ctx->head->parent_offset = parent_offset;
        continue;
      }
      if ((type->typegroup == 'H' || group == 'H') && type->size == size) {
      } else {
          __Pyx_BufFmt_RaiseExpected(ctx);
          return -1;
      }
    }
    offset = ctx->head->parent_offset + field->offset;
    if (ctx->fmt_offset != offset) {
      PyErr_Format(PyExc_ValueError,
                   "Buffer dtype mismatch; next field is at offset %" CYTHON_FORMAT_SSIZE_T "d but %" CYTHON_FORMAT_SSIZE_T "d expected",
                   (Py_ssize_t)ctx->fmt_offset, (Py_ssize_t)offset);
      return -1;
    }
    ctx->fmt_offset += size;
    if (arraysize)
      ctx->fmt_offset += (arraysize - 1) * size;
    --ctx->enc_count;
    while (1) {
      if (field == &ctx->root) {
        ctx->head = NULL;
        if (ctx->enc_count != 0) {
          __Pyx_BufFmt_RaiseExpected(ctx);
          return -1;
        }
        break;
      }
      ctx->head->field = ++field;
      if (field->type == NULL) {
        --ctx->head;
        field = ctx->head->field;
        continue;
      } else if (field->type->typegroup == 'S') {
        size_t parent_offset = ctx->head->parent_offset + field->offset;
        if (field->type->fields->type == NULL) continue;
        field = field->type->fields;
        ++ctx->head;
        ctx->head->field = field;
        ctx->head->parent_offset = parent_offset;
        break;
      } else {
        break;
      }
    }
  } while (ctx->enc_count);
  ctx->enc_type = 0;
  ctx->is_complex = 0;
  return 0;
}
static CYTHON_INLINE PyObject *
__pyx_buffmt_parse_array(__Pyx_BufFmt_Context* ctx, const char** tsp)
{
    const char *ts = *tsp;
    int i = 0, number;
    int ndim = ctx->head->field->type->ndim;
;
    ++ts;
    if (ctx->new_count != 1) {
        PyErr_SetString(PyExc_ValueError,
                        "Cannot handle repeated arrays in format string");
        return NULL;
    }
    if (__Pyx_BufFmt_ProcessTypeChunk(ctx) == -1) return NULL;
    while (*ts && *ts != ')') {
        switch (*ts) {
            case ' ': case '\f': case '\r': case '\n': case '\t': case '\v':  continue;
            default:  break;
        }
        number = __Pyx_BufFmt_ExpectNumber(&ts);
        if (number == -1) return NULL;
        if (i < ndim && (size_t) number != ctx->head->field->type->arraysize[i])
            return PyErr_Format(PyExc_ValueError,
                        "Expected a dimension of size %zu, got %d",
                        ctx->head->field->type->arraysize[i], number);
        if (*ts != ',' && *ts != ')')
            return PyErr_Format(PyExc_ValueError,
                                "Expected a comma in format string, got '%c'", *ts);
        if (*ts == ',') ts++;
        i++;
    }
    if (i != ndim)
        return PyErr_Format(PyExc_ValueError, "Expected %d dimension(s), got %d",
                            ctx->head->field->type->ndim, i);
    if (!*ts) {
        PyErr_SetString(PyExc_ValueError,
                        "Unexpected end of format string, expected ')'");
        return NULL;
    }
    ctx->is_valid_array = 1;
    ctx->new_count = 1;
    *tsp = ++ts;
    return Py_None;
}
static const char* __Pyx_BufFmt_CheckString(__Pyx_BufFmt_Context* ctx, const char* ts) {
  int got_Z = 0;
  while (1) {
    switch(*ts) {
      case 0:
        if (ctx->enc_type != 0 && ctx->head == NULL) {
          __Pyx_BufFmt_RaiseExpected(ctx);
          return NULL;
        }
        if (__Pyx_BufFmt_ProcessTypeChunk(ctx) == -1) return NULL;
        if (ctx->head != NULL) {
          __Pyx_BufFmt_RaiseExpected(ctx);
          return NULL;
        }
        return ts;
      case ' ':
      case '\r':
      case '\n':
        ++ts;
        break;
      case '<':
        if (!__Pyx_IsLittleEndian()) {
          PyErr_SetString(PyExc_ValueError, "Little-endian buffer not supported on big-endian compiler");
          return NULL;
        }
        ctx->new_packmode = '=';
        ++ts;
        break;
      case '>':
      case '!':
        if (__Pyx_IsLittleEndian()) {
          PyErr_SetString(PyExc_ValueError, "Big-endian buffer not supported on little-endian compiler");
          return NULL;
        }
        ctx->new_packmode = '=';
        ++ts;
        break;
      case '=':
      case '@':
      case '^':
        ctx->new_packmode = *ts++;
        break;
      case 'T':
        {
          const char* ts_after_sub;
          size_t i, struct_count = ctx->new_count;
          size_t struct_alignment = ctx->struct_alignment;
          ctx->new_count = 1;
          ++ts;
          if (*ts != '{') {
            PyErr_SetString(PyExc_ValueError, "Buffer acquisition: Expected '{' after 'T'");
            return NULL;
          }
          if (__Pyx_BufFmt_ProcessTypeChunk(ctx) == -1) return NULL;
          ctx->enc_type = 0;
          ctx->enc_count = 0;
          ctx->struct_alignment = 0;
          ++ts;
          ts_after_sub = ts;
          for (i = 0; i != struct_count; ++i) {
            ts_after_sub = __Pyx_BufFmt_CheckString(ctx, ts);
            if (!ts_after_sub) return NULL;
          }
          ts = ts_after_sub;
          if (struct_alignment) ctx->struct_alignment = struct_alignment;
        }
        break;
      case '}':
        {
          size_t alignment = ctx->struct_alignment;
          ++ts;
          if (__Pyx_BufFmt_ProcessTypeChunk(ctx) == -1) return NULL;
          ctx->enc_type = 0;
          if (alignment && ctx->fmt_offset % alignment) {
            ctx->fmt_offset += alignment - (ctx->fmt_offset % alignment);
          }
        }
        return ts;
      case 'x':
        if (__Pyx_BufFmt_ProcessTypeChunk(ctx) == -1) return NULL;
        ctx->fmt_offset += ctx->new_count;
        ctx->new_count = 1;
        ctx->enc_count = 0;
        ctx->enc_type = 0;
        ctx->enc_packmode = ctx->new_packmode;
        ++ts;
        break;
      case 'Z':
        got_Z = 1;
        ++ts;
        if (*ts != 'f' && *ts != 'd' && *ts != 'g') {
          __Pyx_BufFmt_RaiseUnexpectedChar('Z');
          return NULL;
        }
      case 'c': case 'b': case 'B': case 'h': case 'H': case 'i': case 'I':
      case 'l': case 'L': case 'q': case 'Q':
      case 'f': case 'd': case 'g':
      case 'O': case 'p':
        if (ctx->enc_type == *ts && got_Z == ctx->is_complex &&
            ctx->enc_packmode == ctx->new_packmode) {
          ctx->enc_count += ctx->new_count;
          ctx->new_count = 1;
          got_Z = 0;
          ++ts;
          break;
        }
      case 's':
        if (__Pyx_BufFmt_ProcessTypeChunk(ctx) == -1) return NULL;
        ctx->enc_count = ctx->new_count;
        ctx->enc_packmode = ctx->new_packmode;
        ctx->enc_type = *ts;
        ctx->is_complex = got_Z;
        ++ts;
        ctx->new_count = 1;
        got_Z = 0;
        break;
      case ':':
        ++ts;
        while(*ts != ':') ++ts;
        ++ts;
        break;
      case '(':
        if (!__pyx_buffmt_parse_array(ctx, &ts)) return NULL;
        break;
      default:
        {
          int number = __Pyx_BufFmt_ExpectNumber(&ts);
          if (number == -1) return NULL;
          ctx->new_count = (size_t)number;
        }
    }
  }
}
static CYTHON_INLINE void __Pyx_ZeroBuffer(Py_buffer* buf) {
  buf->buf = NULL;
  buf->obj = NULL;
  buf->strides = __Pyx_zeros;
  buf->shape = __Pyx_zeros;
  buf->suboffsets = __Pyx_minusones;
}
static CYTHON_INLINE int __Pyx_GetBufferAndValidate(
        Py_buffer* buf, PyObject* obj,  __Pyx_TypeInfo* dtype, int flags,
        int nd, int cast, __Pyx_BufFmt_StackElem* stack)
{
  if (obj == Py_None || obj == NULL) {
    __Pyx_ZeroBuffer(buf);
    return 0;
  }
  buf->buf = NULL;
  if (__Pyx_GetBuffer(obj, buf, flags) == -1) goto fail;
  if (buf->ndim != nd) {
    PyErr_Format(PyExc_ValueError,
                 "Buffer has wrong number of dimensions (expected %d, got %d)",
                 nd, buf->ndim);
    goto fail;
  }
  if (!cast) {
    __Pyx_BufFmt_Context ctx;
    __Pyx_BufFmt_Init(&ctx, stack, dtype);
    if (!__Pyx_BufFmt_CheckString(&ctx, buf->format)) goto fail;
  }
  if ((unsigned)buf->itemsize != dtype->size) {
    PyErr_Format(PyExc_ValueError,
      "Item size of buffer (%" CYTHON_FORMAT_SSIZE_T "d byte%s) does not match size of '%s' (%" CYTHON_FORMAT_SSIZE_T "d byte%s)",
      buf->itemsize, (buf->itemsize > 1) ? "s" : "",
      dtype->name, (Py_ssize_t)dtype->size, (dtype->size > 1) ? "s" : "");
    goto fail;
  }
  if (buf->suboffsets == NULL) buf->suboffsets = __Pyx_minusones;
  return 0;
fail:;
  __Pyx_ZeroBuffer(buf);
  return -1;
}
static CYTHON_INLINE void __Pyx_SafeReleaseBuffer(Py_buffer* info) {
  if (info->buf == NULL) return;
  if (info->suboffsets == __Pyx_minusones) info->suboffsets = NULL;
  __Pyx_ReleaseBuffer(info);
}

/* GetBuiltinName */
  static PyObject *__Pyx_GetBuiltinName(PyObject *name) {
    PyObject* result = __Pyx_PyObject_GetAttrStr(__pyx_b, name);
    if (unlikely(!result)) {
        PyErr_Format(PyExc_NameError,
#if PY_MAJOR_VERSION >= 3
            "name '%U' is not defined", name);
#else
            "name '%.200s' is not defined", PyString_AS_STRING(name));
#endif
    }
    return result;
}

/* GetModuleGlobalName */
  static CYTHON_INLINE PyObject *__Pyx_GetModuleGlobalName(PyObject *name) {
    PyObject *result;
#if CYTHON_COMPILING_IN_CPYTHON
    result = PyDict_GetItem(__pyx_d, name);
    if (likely(result)) {
        Py_INCREF(result);
    } else {
#else
    result = PyObject_GetItem(__pyx_d, name);
    if (!result) {
        PyErr_Clear();
#endif
        result = __Pyx_GetBuiltinName(name);
    }
    return result;
}

/* PyObjectCall */
    #if CYTHON_COMPILING_IN_CPYTHON
static CYTHON_INLINE PyObject* __Pyx_PyObject_Call(PyObject *func, PyObject *arg, PyObject *kw) {
    PyObject *result;
    ternaryfunc call = func->ob_type->tp_call;
    if (unlikely(!call))
        return PyObject_Call(func, arg, kw);
    if (unlikely(Py_EnterRecursiveCall((char*)" while calling a Python object")))
        return NULL;
    result = (*call)(func, arg, kw);
    Py_LeaveRecursiveCall();
    if (unlikely(!result) && unlikely(!PyErr_Occurred())) {
        PyErr_SetString(
            PyExc_SystemError,
            "NULL result without error in PyObject_Call");
    }
    return result;
}
#endif

/* ExtTypeTest */
    static CYTHON_INLINE int __Pyx_TypeTest(PyObject *obj, PyTypeObject *type) {
    if (unlikely(!type)) {
        PyErr_SetString(PyExc_SystemError, "Missing type object");
        return 0;
    }
    if (likely(PyObject_TypeCheck(obj, type)))
        return 1;
    PyErr_Format(PyExc_TypeError, "Cannot convert %.200s to %.200s",
                 Py_TYPE(obj)->tp_name, type->tp_name);
    return 0;
}

/* BufferIndexError */
    static void __Pyx_RaiseBufferIndexError(int axis) {
  PyErr_Format(PyExc_IndexError,
     "Out of bounds on buffer access (axis %d)", axis);
}

/* PyErrFetchRestore */
    #if CYTHON_COMPILING_IN_CPYTHON
static CYTHON_INLINE void __Pyx_ErrRestoreInState(PyThreadState *tstate, PyObject *type, PyObject *value, PyObject *tb) {
    PyObject *tmp_type, *tmp_value, *tmp_tb;
    tmp_type = tstate->curexc_type;
    tmp_value = tstate->curexc_value;
    tmp_tb = tstate->curexc_traceback;
    tstate->curexc_type = type;
    tstate->curexc_value = value;
    tstate->curexc_traceback = tb;
    Py_XDECREF(tmp_type);
    Py_XDECREF(tmp_value);
    Py_XDECREF(tmp_tb);
}
static CYTHON_INLINE void __Pyx_ErrFetchInState(PyThreadState *tstate, PyObject **type, PyObject **value, PyObject **tb) {
    *type = tstate->curexc_type;
    *value = tstate->curexc_value;
    *tb = tstate->curexc_traceback;
    tstate->curexc_type = 0;
    tstate->curexc_value = 0;
    tstate->curexc_traceback = 0;
}
#endif

/* RaiseException */
    #if PY_MAJOR_VERSION < 3
static void __Pyx_Raise(PyObject *type, PyObject *value, PyObject *tb,
                        CYTHON_UNUSED PyObject *cause) {
    __Pyx_PyThreadState_declare
    Py_XINCREF(type);
    if (!value || value == Py_None)
        value = NULL;
    else
        Py_INCREF(value);
    if (!tb || tb == Py_None)
        tb = NULL;
    else {
        Py_INCREF(tb);
        if (!PyTraceBack_Check(tb)) {
            PyErr_SetString(PyExc_TypeError,
                "raise: arg 3 must be a traceback or None");
            goto raise_error;
        }
    }
    if (PyType_Check(type)) {
#if CYTHON_COMPILING_IN_PYPY
        if (!value) {
            Py_INCREF(Py_None);
            value = Py_None;
        }
#endif
        PyErr_NormalizeException(&type, &value, &tb);
    } else {
        if (value) {
            PyErr_SetString(PyExc_TypeError,
                "instance exception may not have a separate value");
            goto raise_error;
        }
        value = type;
        type = (PyObject*) Py_TYPE(type);
        Py_INCREF(type);
        if (!PyType_IsSubtype((PyTypeObject *)type, (PyTypeObject *)PyExc_BaseException)) {
            PyErr_SetString(PyExc_TypeError,
                "raise: exception class must be a subclass of BaseException");
            goto raise_error;
        }
    }
    __Pyx_PyThreadState_assign
    __Pyx_ErrRestore(type, value, tb);
    return;
raise_error:
    Py_XDECREF(value);
    Py_XDECREF(type);
    Py_XDECREF(tb);
    return;
}
#else
static void __Pyx_Raise(PyObject *type, PyObject *value, PyObject *tb, PyObject *cause) {
    PyObject* owned_instance = NULL;
    if (tb == Py_None) {
        tb = 0;
    } else if (tb && !PyTraceBack_Check(tb)) {
        PyErr_SetString(PyExc_TypeError,
            "raise: arg 3 must be a traceback or None");
        goto bad;
    }
    if (value == Py_None)
        value = 0;
    if (PyExceptionInstance_Check(type)) {
        if (value) {
            PyErr_SetString(PyExc_TypeError,
                "instance exception may not have a separate value");
            goto bad;
        }
        value = type;
        type = (PyObject*) Py_TYPE(value);
    } else if (PyExceptionClass_Check(type)) {
        PyObject *instance_class = NULL;
        if (value && PyExceptionInstance_Check(value)) {
            instance_class = (PyObject*) Py_TYPE(value);
            if (instance_class != type) {
                int is_subclass = PyObject_IsSubclass(instance_class, type);
                if (!is_subclass) {
                    instance_class = NULL;
                } else if (unlikely(is_subclass == -1)) {
                    goto bad;
                } else {
                    type = instance_class;
                }
            }
        }
        if (!instance_class) {
            PyObject *args;
            if (!value)
                args = PyTuple_New(0);
            else if (PyTuple_Check(value)) {
                Py_INCREF(value);
                args = value;
            } else
                args = PyTuple_Pack(1, value);
            if (!args)
                goto bad;
            owned_instance = PyObject_Call(type, args, NULL);
            Py_DECREF(args);
            if (!owned_instance)
                goto bad;
            value = owned_instance;
            if (!PyExceptionInstance_Check(value)) {
                PyErr_Format(PyExc_TypeError,
                             "calling %R should have returned an instance of "
                             "BaseException, not %R",
                             type, Py_TYPE(value));
                goto bad;
            }
        }
    } else {
        PyErr_SetString(PyExc_TypeError,
            "raise: exception class must be a subclass of BaseException");
        goto bad;
    }
#if PY_VERSION_HEX >= 0x03030000
    if (cause) {
#else
    if (cause && cause != Py_None) {
#endif
        PyObject *fixed_cause;
        if (cause == Py_None) {
            fixed_cause = NULL;
        } else if (PyExceptionClass_Check(cause)) {
            fixed_cause = PyObject_CallObject(cause, NULL);
            if (fixed_cause == NULL)
                goto bad;
        } else if (PyExceptionInstance_Check(cause)) {
            fixed_cause = cause;
            Py_INCREF(fixed_cause);
        } else {
            PyErr_SetString(PyExc_TypeError,
                            "exception causes must derive from "
                            "BaseException");
            goto bad;
        }
        PyException_SetCause(value, fixed_cause);
    }
    PyErr_SetObject(type, value);
    if (tb) {
#if CYTHON_COMPILING_IN_PYPY
        PyObject *tmp_type, *tmp_value, *tmp_tb;
        PyErr_Fetch(&tmp_type, &tmp_value, &tmp_tb);
        Py_INCREF(tb);
        PyErr_Restore(tmp_type, tmp_value, tb);
        Py_XDECREF(tmp_tb);
#else
        PyThreadState *tstate = PyThreadState_GET();
        PyObject* tmp_tb = tstate->curexc_traceback;
        if (tb != tmp_tb) {
            Py_INCREF(tb);
            tstate->curexc_traceback = tb;
            Py_XDECREF(tmp_tb);
        }
#endif
    }
bad:
    Py_XDECREF(owned_instance);
    return;
}
#endif

/* RaiseTooManyValuesToUnpack */
      static CYTHON_INLINE void __Pyx_RaiseTooManyValuesError(Py_ssize_t expected) {
    PyErr_Format(PyExc_ValueError,
                 "too many values to unpack (expected %" CYTHON_FORMAT_SSIZE_T "d)", expected);
}

/* RaiseNeedMoreValuesToUnpack */
      static CYTHON_INLINE void __Pyx_RaiseNeedMoreValuesError(Py_ssize_t index) {
    PyErr_Format(PyExc_ValueError,
                 "need more than %" CYTHON_FORMAT_SSIZE_T "d value%.1s to unpack",
                 index, (index == 1) ? "" : "s");
}

/* RaiseNoneIterError */
      static CYTHON_INLINE void __Pyx_RaiseNoneNotIterableError(void) {
    PyErr_SetString(PyExc_TypeError, "'NoneType' object is not iterable");
}

/* Import */
      static PyObject *__Pyx_Import(PyObject *name, PyObject *from_list, int level) {
    PyObject *empty_list = 0;
    PyObject *module = 0;
    PyObject *global_dict = 0;
    PyObject *empty_dict = 0;
    PyObject *list;
    #if PY_VERSION_HEX < 0x03030000
    PyObject *py_import;
    py_import = __Pyx_PyObject_GetAttrStr(__pyx_b, __pyx_n_s_import);
    if (!py_import)
        goto bad;
    #endif
    if (from_list)
        list = from_list;
    else {
        empty_list = PyList_New(0);
        if (!empty_list)
            goto bad;
        list = empty_list;
    }
    global_dict = PyModule_GetDict(__pyx_m);
    if (!global_dict)
        goto bad;
    empty_dict = PyDict_New();
    if (!empty_dict)
        goto bad;
    {
        #if PY_MAJOR_VERSION >= 3
        if (level == -1) {
            if (strchr(__Pyx_MODULE_NAME, '.')) {
                #if PY_VERSION_HEX < 0x03030000
                PyObject *py_level = PyInt_FromLong(1);
                if (!py_level)
                    goto bad;
                module = PyObject_CallFunctionObjArgs(py_import,
                    name, global_dict, empty_dict, list, py_level, NULL);
                Py_DECREF(py_level);
                #else
                module = PyImport_ImportModuleLevelObject(
                    name, global_dict, empty_dict, list, 1);
                #endif
                if (!module) {
                    if (!PyErr_ExceptionMatches(PyExc_ImportError))
                        goto bad;
                    PyErr_Clear();
                }
            }
            level = 0;
        }
        #endif
        if (!module) {
            #if PY_VERSION_HEX < 0x03030000
            PyObject *py_level = PyInt_FromLong(level);
            if (!py_level)
                goto bad;
            module = PyObject_CallFunctionObjArgs(py_import,
                name, global_dict, empty_dict, list, py_level, NULL);
            Py_DECREF(py_level);
            #else
            module = PyImport_ImportModuleLevelObject(
                name, global_dict, empty_dict, list, level);
            #endif
        }
    }
bad:
    #if PY_VERSION_HEX < 0x03030000
    Py_XDECREF(py_import);
    #endif
    Py_XDECREF(empty_list);
    Py_XDECREF(empty_dict);
    return module;
}

/* CodeObjectCache */
      static int __pyx_bisect_code_objects(__Pyx_CodeObjectCacheEntry* entries, int count, int code_line) {
    int start = 0, mid = 0, end = count - 1;
    if (end >= 0 && code_line > entries[end].code_line) {
        return count;
    }
    while (start < end) {
        mid = start + (end - start) / 2;
        if (code_line < entries[mid].code_line) {
            end = mid;
        } else if (code_line > entries[mid].code_line) {
             start = mid + 1;
        } else {
            return mid;
        }
    }
    if (code_line <= entries[mid].code_line) {
        return mid;
    } else {
        return mid + 1;
    }
}
static PyCodeObject *__pyx_find_code_object(int code_line) {
    PyCodeObject* code_object;
    int pos;
    if (unlikely(!code_line) || unlikely(!__pyx_code_cache.entries)) {
        return NULL;
    }
    pos = __pyx_bisect_code_objects(__pyx_code_cache.entries, __pyx_code_cache.count, code_line);
    if (unlikely(pos >= __pyx_code_cache.count) || unlikely(__pyx_code_cache.entries[pos].code_line != code_line)) {
        return NULL;
    }
    code_object = __pyx_code_cache.entries[pos].code_object;
    Py_INCREF(code_object);
    return code_object;
}
static void __pyx_insert_code_object(int code_line, PyCodeObject* code_object) {
    int pos, i;
    __Pyx_CodeObjectCacheEntry* entries = __pyx_code_cache.entries;
    if (unlikely(!code_line)) {
        return;
    }
    if (unlikely(!entries)) {
        entries = (__Pyx_CodeObjectCacheEntry*)PyMem_Malloc(64*sizeof(__Pyx_CodeObjectCacheEntry));
        if (likely(entries)) {
            __pyx_code_cache.entries = entries;
            __pyx_code_cache.max_count = 64;
            __pyx_code_cache.count = 1;
            entries[0].code_line = code_line;
            entries[0].code_object = code_object;
            Py_INCREF(code_object);
        }
        return;
    }
    pos = __pyx_bisect_code_objects(__pyx_code_cache.entries, __pyx_code_cache.count, code_line);
    if ((pos < __pyx_code_cache.count) && unlikely(__pyx_code_cache.entries[pos].code_line == code_line)) {
        PyCodeObject* tmp = entries[pos].code_object;
        entries[pos].code_object = code_object;
        Py_DECREF(tmp);
        return;
    }
    if (__pyx_code_cache.count == __pyx_code_cache.max_count) {
        int new_max = __pyx_code_cache.max_count + 64;
        entries = (__Pyx_CodeObjectCacheEntry*)PyMem_Realloc(
            __pyx_code_cache.entries, (size_t)new_max*sizeof(__Pyx_CodeObjectCacheEntry));
        if (unlikely(!entries)) {
            return;
        }
        __pyx_code_cache.entries = entries;
        __pyx_code_cache.max_count = new_max;
    }
    for (i=__pyx_code_cache.count; i>pos; i--) {
        entries[i] = entries[i-1];
    }
    entries[pos].code_line = code_line;
    entries[pos].code_object = code_object;
    __pyx_code_cache.count++;
    Py_INCREF(code_object);
}

/* AddTraceback */
      #include "compile.h"
#include "frameobject.h"
#include "traceback.h"
static PyCodeObject* __Pyx_CreateCodeObjectForTraceback(
            const char *funcname, int c_line,
            int py_line, const char *filename) {
    PyCodeObject *py_code = 0;
    PyObject *py_srcfile = 0;
    PyObject *py_funcname = 0;
    #if PY_MAJOR_VERSION < 3
    py_srcfile = PyString_FromString(filename);
    #else
    py_srcfile = PyUnicode_FromString(filename);
    #endif
    if (!py_srcfile) goto bad;
    if (c_line) {
        #if PY_MAJOR_VERSION < 3
        py_funcname = PyString_FromFormat( "%s (%s:%d)", funcname, __pyx_cfilenm, c_line);
        #else
        py_funcname = PyUnicode_FromFormat( "%s (%s:%d)", funcname, __pyx_cfilenm, c_line);
        #endif
    }
    else {
        #if PY_MAJOR_VERSION < 3
        py_funcname = PyString_FromString(funcname);
        #else
        py_funcname = PyUnicode_FromString(funcname);
        #endif
    }
    if (!py_funcname) goto bad;
    py_code = __Pyx_PyCode_New(
        0,
        0,
        0,
        0,
        0,
        __pyx_empty_bytes, /*PyObject *code,*/
        __pyx_empty_tuple, /*PyObject *consts,*/
        __pyx_empty_tuple, /*PyObject *names,*/
        __pyx_empty_tuple, /*PyObject *varnames,*/
        __pyx_empty_tuple, /*PyObject *freevars,*/
        __pyx_empty_tuple, /*PyObject *cellvars,*/
        py_srcfile,   /*PyObject *filename,*/
        py_funcname,  /*PyObject *name,*/
        py_line,
        __pyx_empty_bytes  /*PyObject *lnotab*/
    );
    Py_DECREF(py_srcfile);
    Py_DECREF(py_funcname);
    return py_code;
bad:
    Py_XDECREF(py_srcfile);
    Py_XDECREF(py_funcname);
    return NULL;
}
static void __Pyx_AddTraceback(const char *funcname, int c_line,
                               int py_line, const char *filename) {
    PyCodeObject *py_code = 0;
    PyFrameObject *py_frame = 0;
    py_code = __pyx_find_code_object(c_line ? c_line : py_line);
    if (!py_code) {
        py_code = __Pyx_CreateCodeObjectForTraceback(
            funcname, c_line, py_line, filename);
        if (!py_code) goto bad;
        __pyx_insert_code_object(c_line ? c_line : py_line, py_code);
    }
    py_frame = PyFrame_New(
        PyThreadState_GET(), /*PyThreadState *tstate,*/
        py_code,             /*PyCodeObject *code,*/
        __pyx_d,      /*PyObject *globals,*/
        0                    /*PyObject *locals*/
    );
    if (!py_frame) goto bad;
    py_frame->f_lineno = py_line;
    PyTraceBack_Here(py_frame);
bad:
    Py_XDECREF(py_code);
    Py_XDECREF(py_frame);
}

#if PY_MAJOR_VERSION < 3
static int __Pyx_GetBuffer(PyObject *obj, Py_buffer *view, int flags) {
    if (PyObject_CheckBuffer(obj)) return PyObject_GetBuffer(obj, view, flags);
        if (PyObject_TypeCheck(obj, __pyx_ptype_5numpy_ndarray)) return __pyx_pw_5numpy_7ndarray_1__getbuffer__(obj, view, flags);
    PyErr_Format(PyExc_TypeError, "'%.200s' does not have the buffer interface", Py_TYPE(obj)->tp_name);
    return -1;
}
static void __Pyx_ReleaseBuffer(Py_buffer *view) {
    PyObject *obj = view->obj;
    if (!obj) return;
    if (PyObject_CheckBuffer(obj)) {
        PyBuffer_Release(view);
        return;
    }
        if (PyObject_TypeCheck(obj, __pyx_ptype_5numpy_ndarray)) { __pyx_pw_5numpy_7ndarray_3__releasebuffer__(obj, view); return; }
    Py_DECREF(obj);
    view->obj = NULL;
}
#endif


      /* CIntFromPyVerify */
      #define __PYX_VERIFY_RETURN_INT(target_type, func_type, func_value)\
    __PYX__VERIFY_RETURN_INT(target_type, func_type, func_value, 0)
#define __PYX_VERIFY_RETURN_INT_EXC(target_type, func_type, func_value)\
    __PYX__VERIFY_RETURN_INT(target_type, func_type, func_value, 1)
#define __PYX__VERIFY_RETURN_INT(target_type, func_type, func_value, exc)\
    {\
        func_type value = func_value;\
        if (sizeof(target_type) < sizeof(func_type)) {\
            if (unlikely(value != (func_type) (target_type) value)) {\
                func_type zero = 0;\
                if (exc && unlikely(value == (func_type)-1 && PyErr_Occurred()))\
                    return (target_type) -1;\
                if (is_unsigned && unlikely(value < zero))\
                    goto raise_neg_overflow;\
                else\
                    goto raise_overflow;\
            }\
        }\
        return (target_type) value;\
    }

/* CIntToPy */
      static CYTHON_INLINE PyObject* __Pyx_PyInt_From_int(int value) {
    const int neg_one = (int) -1, const_zero = (int) 0;
    const int is_unsigned = neg_one > const_zero;
    if (is_unsigned) {
        if (sizeof(int) < sizeof(long)) {
            return PyInt_FromLong((long) value);
        } else if (sizeof(int) <= sizeof(unsigned long)) {
            return PyLong_FromUnsignedLong((unsigned long) value);
        } else if (sizeof(int) <= sizeof(unsigned PY_LONG_LONG)) {
            return PyLong_FromUnsignedLongLong((unsigned PY_LONG_LONG) value);
        }
    } else {
        if (sizeof(int) <= sizeof(long)) {
            return PyInt_FromLong((long) value);
        } else if (sizeof(int) <= sizeof(PY_LONG_LONG)) {
            return PyLong_FromLongLong((PY_LONG_LONG) value);
        }
    }
    {
        int one = 1; int little = (int)*(unsigned char *)&one;
        unsigned char *bytes = (unsigned char *)&value;
        return _PyLong_FromByteArray(bytes, sizeof(int),
                                     little, !is_unsigned);
    }
}

/* CIntToPy */
      static CYTHON_INLINE PyObject* __Pyx_PyInt_From_Py_intptr_t(Py_intptr_t value) {
    const Py_intptr_t neg_one = (Py_intptr_t) -1, const_zero = (Py_intptr_t) 0;
    const int is_unsigned = neg_one > const_zero;
    if (is_unsigned) {
        if (sizeof(Py_intptr_t) < sizeof(long)) {
            return PyInt_FromLong((long) value);
        } else if (sizeof(Py_intptr_t) <= sizeof(unsigned long)) {
            return PyLong_FromUnsignedLong((unsigned long) value);
        } else if (sizeof(Py_intptr_t) <= sizeof(unsigned PY_LONG_LONG)) {
            return PyLong_FromUnsignedLongLong((unsigned PY_LONG_LONG) value);
        }
    } else {
        if (sizeof(Py_intptr_t) <= sizeof(long)) {
            return PyInt_FromLong((long) value);
        } else if (sizeof(Py_intptr_t) <= sizeof(PY_LONG_LONG)) {
            return PyLong_FromLongLong((PY_LONG_LONG) value);
        }
    }
    {
        int one = 1; int little = (int)*(unsigned char *)&one;
        unsigned char *bytes = (unsigned char *)&value;
        return _PyLong_FromByteArray(bytes, sizeof(Py_intptr_t),
                                     little, !is_unsigned);
    }
}

/* None */
      #if CYTHON_CCOMPLEX
  #ifdef __cplusplus
    static CYTHON_INLINE __pyx_t_float_complex __pyx_t_float_complex_from_parts(float x, float y) {
      return ::std::complex< float >(x, y);
    }
  #else
    static CYTHON_INLINE __pyx_t_float_complex __pyx_t_float_complex_from_parts(float x, float y) {
      return x + y*(__pyx_t_float_complex)_Complex_I;
    }
  #endif
#else
    static CYTHON_INLINE __pyx_t_float_complex __pyx_t_float_complex_from_parts(float x, float y) {
      __pyx_t_float_complex z;
      z.real = x;
      z.imag = y;
      return z;
    }
#endif

/* None */
      #if CYTHON_CCOMPLEX
#else
    static CYTHON_INLINE int __Pyx_c_eqf(__pyx_t_float_complex a, __pyx_t_float_complex b) {
       return (a.real == b.real) && (a.imag == b.imag);
    }
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_sumf(__pyx_t_float_complex a, __pyx_t_float_complex b) {
        __pyx_t_float_complex z;
        z.real = a.real + b.real;
        z.imag = a.imag + b.imag;
        return z;
    }
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_difff(__pyx_t_float_complex a, __pyx_t_float_complex b) {
        __pyx_t_float_complex z;
        z.real = a.real - b.real;
        z.imag = a.imag - b.imag;
        return z;
    }
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_prodf(__pyx_t_float_complex a, __pyx_t_float_complex b) {
        __pyx_t_float_complex z;
        z.real = a.real * b.real - a.imag * b.imag;
        z.imag = a.real * b.imag + a.imag * b.real;
        return z;
    }
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_quotf(__pyx_t_float_complex a, __pyx_t_float_complex b) {
        __pyx_t_float_complex z;
        float denom = b.real * b.real + b.imag * b.imag;
        z.real = (a.real * b.real + a.imag * b.imag) / denom;
        z.imag = (a.imag * b.real - a.real * b.imag) / denom;
        return z;
    }
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_negf(__pyx_t_float_complex a) {
        __pyx_t_float_complex z;
        z.real = -a.real;
        z.imag = -a.imag;
        return z;
    }
    static CYTHON_INLINE int __Pyx_c_is_zerof(__pyx_t_float_complex a) {
       return (a.real == 0) && (a.imag == 0);
    }
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_conjf(__pyx_t_float_complex a) {
        __pyx_t_float_complex z;
        z.real =  a.real;
        z.imag = -a.imag;
        return z;
    }
    #if 1
        static CYTHON_INLINE float __Pyx_c_absf(__pyx_t_float_complex z) {
          #if !defined(HAVE_HYPOT) || defined(_MSC_VER)
            return sqrtf(z.real*z.real + z.imag*z.imag);
          #else
            return hypotf(z.real, z.imag);
          #endif
        }
        static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_powf(__pyx_t_float_complex a, __pyx_t_float_complex b) {
            __pyx_t_float_complex z;
            float r, lnr, theta, z_r, z_theta;
            if (b.imag == 0 && b.real == (int)b.real) {
                if (b.real < 0) {
                    float denom = a.real * a.real + a.imag * a.imag;
                    a.real = a.real / denom;
                    a.imag = -a.imag / denom;
                    b.real = -b.real;
                }
                switch ((int)b.real) {
                    case 0:
                        z.real = 1;
                        z.imag = 0;
                        return z;
                    case 1:
                        return a;
                    case 2:
                        z = __Pyx_c_prodf(a, a);
                        return __Pyx_c_prodf(a, a);
                    case 3:
                        z = __Pyx_c_prodf(a, a);
                        return __Pyx_c_prodf(z, a);
                    case 4:
                        z = __Pyx_c_prodf(a, a);
                        return __Pyx_c_prodf(z, z);
                }
            }
            if (a.imag == 0) {
                if (a.real == 0) {
                    return a;
                }
                r = a.real;
                theta = 0;
            } else {
                r = __Pyx_c_absf(a);
                theta = atan2f(a.imag, a.real);
            }
            lnr = logf(r);
            z_r = expf(lnr * b.real - theta * b.imag);
            z_theta = theta * b.real + lnr * b.imag;
            z.real = z_r * cosf(z_theta);
            z.imag = z_r * sinf(z_theta);
            return z;
        }
    #endif
#endif

/* None */
      #if CYTHON_CCOMPLEX
  #ifdef __cplusplus
    static CYTHON_INLINE __pyx_t_double_complex __pyx_t_double_complex_from_parts(double x, double y) {
      return ::std::complex< double >(x, y);
    }
  #else
    static CYTHON_INLINE __pyx_t_double_complex __pyx_t_double_complex_from_parts(double x, double y) {
      return x + y*(__pyx_t_double_complex)_Complex_I;
    }
  #endif
#else
    static CYTHON_INLINE __pyx_t_double_complex __pyx_t_double_complex_from_parts(double x, double y) {
      __pyx_t_double_complex z;
      z.real = x;
      z.imag = y;
      return z;
    }
#endif

/* None */
      #if CYTHON_CCOMPLEX
#else
    static CYTHON_INLINE int __Pyx_c_eq(__pyx_t_double_complex a, __pyx_t_double_complex b) {
       return (a.real == b.real) && (a.imag == b.imag);
    }
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_sum(__pyx_t_double_complex a, __pyx_t_double_complex b) {
        __pyx_t_double_complex z;
        z.real = a.real + b.real;
        z.imag = a.imag + b.imag;
        return z;
    }
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_diff(__pyx_t_double_complex a, __pyx_t_double_complex b) {
        __pyx_t_double_complex z;
        z.real = a.real - b.real;
        z.imag = a.imag - b.imag;
        return z;
    }
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_prod(__pyx_t_double_complex a, __pyx_t_double_complex b) {
        __pyx_t_double_complex z;
        z.real = a.real * b.real - a.imag * b.imag;
        z.imag = a.real * b.imag + a.imag * b.real;
        return z;
    }
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_quot(__pyx_t_double_complex a, __pyx_t_double_complex b) {
        __pyx_t_double_complex z;
        double denom = b.real * b.real + b.imag * b.imag;
        z.real = (a.real * b.real + a.imag * b.imag) / denom;
        z.imag = (a.imag * b.real - a.real * b.imag) / denom;
        return z;
    }
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_neg(__pyx_t_double_complex a) {
        __pyx_t_double_complex z;
        z.real = -a.real;
        z.imag = -a.imag;
        return z;
    }
    static CYTHON_INLINE int __Pyx_c_is_zero(__pyx_t_double_complex a) {
       return (a.real == 0) && (a.imag == 0);
    }
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_conj(__pyx_t_double_complex a) {
        __pyx_t_double_complex z;
        z.real =  a.real;
        z.imag = -a.imag;
        return z;
    }
    #if 1
        static CYTHON_INLINE double __Pyx_c_abs(__pyx_t_double_complex z) {
          #if !defined(HAVE_HYPOT) || defined(_MSC_VER)
            return sqrt(z.real*z.real + z.imag*z.imag);
          #else
            return hypot(z.real, z.imag);
          #endif
        }
        static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_pow(__pyx_t_double_complex a, __pyx_t_double_complex b) {
            __pyx_t_double_complex z;
            double r, lnr, theta, z_r, z_theta;
            if (b.imag == 0 && b.real == (int)b.real) {
                if (b.real < 0) {
                    double denom = a.real * a.real + a.imag * a.imag;
                    a.real = a.real / denom;
                    a.imag = -a.imag / denom;
                    b.real = -b.real;
                }
                switch ((int)b.real) {
                    case 0:
                        z.real = 1;
                        z.imag = 0;
                        return z;
                    case 1:
                        return a;
                    case 2:
                        z = __Pyx_c_prod(a, a);
                        return __Pyx_c_prod(a, a);
                    case 3:
                        z = __Pyx_c_prod(a, a);
                        return __Pyx_c_prod(z, a);
                    case 4:
                        z = __Pyx_c_prod(a, a);
                        return __Pyx_c_prod(z, z);
                }
            }
            if (a.imag == 0) {
                if (a.real == 0) {
                    return a;
                }
                r = a.real;
                theta = 0;
            } else {
                r = __Pyx_c_abs(a);
                theta = atan2(a.imag, a.real);
            }
            lnr = log(r);
            z_r = exp(lnr * b.real - theta * b.imag);
            z_theta = theta * b.real + lnr * b.imag;
            z.real = z_r * cos(z_theta);
            z.imag = z_r * sin(z_theta);
            return z;
        }
    #endif
#endif

/* CIntToPy */
      static CYTHON_INLINE PyObject* __Pyx_PyInt_From_enum__NPY_TYPES(enum NPY_TYPES value) {
    const enum NPY_TYPES neg_one = (enum NPY_TYPES) -1, const_zero = (enum NPY_TYPES) 0;
    const int is_unsigned = neg_one > const_zero;
    if (is_unsigned) {
        if (sizeof(enum NPY_TYPES) < sizeof(long)) {
            return PyInt_FromLong((long) value);
        } else if (sizeof(enum NPY_TYPES) <= sizeof(unsigned long)) {
            return PyLong_FromUnsignedLong((unsigned long) value);
        } else if (sizeof(enum NPY_TYPES) <= sizeof(unsigned PY_LONG_LONG)) {
            return PyLong_FromUnsignedLongLong((unsigned PY_LONG_LONG) value);
        }
    } else {
        if (sizeof(enum NPY_TYPES) <= sizeof(long)) {
            return PyInt_FromLong((long) value);
        } else if (sizeof(enum NPY_TYPES) <= sizeof(PY_LONG_LONG)) {
            return PyLong_FromLongLong((PY_LONG_LONG) value);
        }
    }
    {
        int one = 1; int little = (int)*(unsigned char *)&one;
        unsigned char *bytes = (unsigned char *)&value;
        return _PyLong_FromByteArray(bytes, sizeof(enum NPY_TYPES),
                                     little, !is_unsigned);
    }
}

/* CIntFromPy */
      static CYTHON_INLINE npy_int32 __Pyx_PyInt_As_npy_int32(PyObject *x) {
    const npy_int32 neg_one = (npy_int32) -1, const_zero = (npy_int32) 0;
    const int is_unsigned = neg_one > const_zero;
#if PY_MAJOR_VERSION < 3
    if (likely(PyInt_Check(x))) {
        if (sizeof(npy_int32) < sizeof(long)) {
            __PYX_VERIFY_RETURN_INT(npy_int32, long, PyInt_AS_LONG(x))
        } else {
            long val = PyInt_AS_LONG(x);
            if (is_unsigned && unlikely(val < 0)) {
                goto raise_neg_overflow;
            }
            return (npy_int32) val;
        }
    } else
#endif
    if (likely(PyLong_Check(x))) {
        if (is_unsigned) {
#if CYTHON_USE_PYLONG_INTERNALS
            const digit* digits = ((PyLongObject*)x)->ob_digit;
            switch (Py_SIZE(x)) {
                case  0: return (npy_int32) 0;
                case  1: __PYX_VERIFY_RETURN_INT(npy_int32, digit, digits[0])
                case 2:
                    if (8 * sizeof(npy_int32) > 1 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 2 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(npy_int32, unsigned long, (((((unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(npy_int32) >= 2 * PyLong_SHIFT) {
                            return (npy_int32) (((((npy_int32)digits[1]) << PyLong_SHIFT) | (npy_int32)digits[0]));
                        }
                    }
                    break;
                case 3:
                    if (8 * sizeof(npy_int32) > 2 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 3 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(npy_int32, unsigned long, (((((((unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(npy_int32) >= 3 * PyLong_SHIFT) {
                            return (npy_int32) (((((((npy_int32)digits[2]) << PyLong_SHIFT) | (npy_int32)digits[1]) << PyLong_SHIFT) | (npy_int32)digits[0]));
                        }
                    }
                    break;
                case 4:
                    if (8 * sizeof(npy_int32) > 3 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 4 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(npy_int32, unsigned long, (((((((((unsigned long)digits[3]) << PyLong_SHIFT) | (unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(npy_int32) >= 4 * PyLong_SHIFT) {
                            return (npy_int32) (((((((((npy_int32)digits[3]) << PyLong_SHIFT) | (npy_int32)digits[2]) << PyLong_SHIFT) | (npy_int32)digits[1]) << PyLong_SHIFT) | (npy_int32)digits[0]));
                        }
                    }
                    break;
            }
#endif
#if CYTHON_COMPILING_IN_CPYTHON
            if (unlikely(Py_SIZE(x) < 0)) {
                goto raise_neg_overflow;
            }
#else
            {
                int result = PyObject_RichCompareBool(x, Py_False, Py_LT);
                if (unlikely(result < 0))
                    return (npy_int32) -1;
                if (unlikely(result == 1))
                    goto raise_neg_overflow;
            }
#endif
            if (sizeof(npy_int32) <= sizeof(unsigned long)) {
                __PYX_VERIFY_RETURN_INT_EXC(npy_int32, unsigned long, PyLong_AsUnsignedLong(x))
            } else if (sizeof(npy_int32) <= sizeof(unsigned PY_LONG_LONG)) {
                __PYX_VERIFY_RETURN_INT_EXC(npy_int32, unsigned PY_LONG_LONG, PyLong_AsUnsignedLongLong(x))
            }
        } else {
#if CYTHON_USE_PYLONG_INTERNALS
            const digit* digits = ((PyLongObject*)x)->ob_digit;
            switch (Py_SIZE(x)) {
                case  0: return (npy_int32) 0;
                case -1: __PYX_VERIFY_RETURN_INT(npy_int32, sdigit, (sdigit) (-(sdigit)digits[0]))
                case  1: __PYX_VERIFY_RETURN_INT(npy_int32,  digit, +digits[0])
                case -2:
                    if (8 * sizeof(npy_int32) - 1 > 1 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 2 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(npy_int32, long, -(long) (((((unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(npy_int32) - 1 > 2 * PyLong_SHIFT) {
                            return (npy_int32) (((npy_int32)-1)*(((((npy_int32)digits[1]) << PyLong_SHIFT) | (npy_int32)digits[0])));
                        }
                    }
                    break;
                case 2:
                    if (8 * sizeof(npy_int32) > 1 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 2 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(npy_int32, unsigned long, (((((unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(npy_int32) - 1 > 2 * PyLong_SHIFT) {
                            return (npy_int32) ((((((npy_int32)digits[1]) << PyLong_SHIFT) | (npy_int32)digits[0])));
                        }
                    }
                    break;
                case -3:
                    if (8 * sizeof(npy_int32) - 1 > 2 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 3 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(npy_int32, long, -(long) (((((((unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(npy_int32) - 1 > 3 * PyLong_SHIFT) {
                            return (npy_int32) (((npy_int32)-1)*(((((((npy_int32)digits[2]) << PyLong_SHIFT) | (npy_int32)digits[1]) << PyLong_SHIFT) | (npy_int32)digits[0])));
                        }
                    }
                    break;
                case 3:
                    if (8 * sizeof(npy_int32) > 2 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 3 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(npy_int32, unsigned long, (((((((unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(npy_int32) - 1 > 3 * PyLong_SHIFT) {
                            return (npy_int32) ((((((((npy_int32)digits[2]) << PyLong_SHIFT) | (npy_int32)digits[1]) << PyLong_SHIFT) | (npy_int32)digits[0])));
                        }
                    }
                    break;
                case -4:
                    if (8 * sizeof(npy_int32) - 1 > 3 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 4 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(npy_int32, long, -(long) (((((((((unsigned long)digits[3]) << PyLong_SHIFT) | (unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(npy_int32) - 1 > 4 * PyLong_SHIFT) {
                            return (npy_int32) (((npy_int32)-1)*(((((((((npy_int32)digits[3]) << PyLong_SHIFT) | (npy_int32)digits[2]) << PyLong_SHIFT) | (npy_int32)digits[1]) << PyLong_SHIFT) | (npy_int32)digits[0])));
                        }
                    }
                    break;
                case 4:
                    if (8 * sizeof(npy_int32) > 3 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 4 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(npy_int32, unsigned long, (((((((((unsigned long)digits[3]) << PyLong_SHIFT) | (unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(npy_int32) - 1 > 4 * PyLong_SHIFT) {
                            return (npy_int32) ((((((((((npy_int32)digits[3]) << PyLong_SHIFT) | (npy_int32)digits[2]) << PyLong_SHIFT) | (npy_int32)digits[1]) << PyLong_SHIFT) | (npy_int32)digits[0])));
                        }
                    }
                    break;
            }
#endif
            if (sizeof(npy_int32) <= sizeof(long)) {
                __PYX_VERIFY_RETURN_INT_EXC(npy_int32, long, PyLong_AsLong(x))
            } else if (sizeof(npy_int32) <= sizeof(PY_LONG_LONG)) {
                __PYX_VERIFY_RETURN_INT_EXC(npy_int32, PY_LONG_LONG, PyLong_AsLongLong(x))
            }
        }
        {
#if CYTHON_COMPILING_IN_PYPY && !defined(_PyLong_AsByteArray)
            PyErr_SetString(PyExc_RuntimeError,
                            "_PyLong_AsByteArray() not available in PyPy, cannot convert large numbers");
#else
            npy_int32 val;
            PyObject *v = __Pyx_PyNumber_IntOrLong(x);
 #if PY_MAJOR_VERSION < 3
            if (likely(v) && !PyLong_Check(v)) {
                PyObject *tmp = v;
                v = PyNumber_Long(tmp);
                Py_DECREF(tmp);
            }
 #endif
            if (likely(v)) {
                int one = 1; int is_little = (int)*(unsigned char *)&one;
                unsigned char *bytes = (unsigned char *)&val;
                int ret = _PyLong_AsByteArray((PyLongObject *)v,
                                              bytes, sizeof(val),
                                              is_little, !is_unsigned);
                Py_DECREF(v);
                if (likely(!ret))
                    return val;
            }
#endif
            return (npy_int32) -1;
        }
    } else {
        npy_int32 val;
        PyObject *tmp = __Pyx_PyNumber_IntOrLong(x);
        if (!tmp) return (npy_int32) -1;
        val = __Pyx_PyInt_As_npy_int32(tmp);
        Py_DECREF(tmp);
        return val;
    }
raise_overflow:
    PyErr_SetString(PyExc_OverflowError,
        "value too large to convert to npy_int32");
    return (npy_int32) -1;
raise_neg_overflow:
    PyErr_SetString(PyExc_OverflowError,
        "can't convert negative value to npy_int32");
    return (npy_int32) -1;
}

/* CIntFromPy */
      static CYTHON_INLINE int __Pyx_PyInt_As_int(PyObject *x) {
    const int neg_one = (int) -1, const_zero = (int) 0;
    const int is_unsigned = neg_one > const_zero;
#if PY_MAJOR_VERSION < 3
    if (likely(PyInt_Check(x))) {
        if (sizeof(int) < sizeof(long)) {
            __PYX_VERIFY_RETURN_INT(int, long, PyInt_AS_LONG(x))
        } else {
            long val = PyInt_AS_LONG(x);
            if (is_unsigned && unlikely(val < 0)) {
                goto raise_neg_overflow;
            }
            return (int) val;
        }
    } else
#endif
    if (likely(PyLong_Check(x))) {
        if (is_unsigned) {
#if CYTHON_USE_PYLONG_INTERNALS
            const digit* digits = ((PyLongObject*)x)->ob_digit;
            switch (Py_SIZE(x)) {
                case  0: return (int) 0;
                case  1: __PYX_VERIFY_RETURN_INT(int, digit, digits[0])
                case 2:
                    if (8 * sizeof(int) > 1 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 2 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(int, unsigned long, (((((unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(int) >= 2 * PyLong_SHIFT) {
                            return (int) (((((int)digits[1]) << PyLong_SHIFT) | (int)digits[0]));
                        }
                    }
                    break;
                case 3:
                    if (8 * sizeof(int) > 2 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 3 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(int, unsigned long, (((((((unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(int) >= 3 * PyLong_SHIFT) {
                            return (int) (((((((int)digits[2]) << PyLong_SHIFT) | (int)digits[1]) << PyLong_SHIFT) | (int)digits[0]));
                        }
                    }
                    break;
                case 4:
                    if (8 * sizeof(int) > 3 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 4 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(int, unsigned long, (((((((((unsigned long)digits[3]) << PyLong_SHIFT) | (unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(int) >= 4 * PyLong_SHIFT) {
                            return (int) (((((((((int)digits[3]) << PyLong_SHIFT) | (int)digits[2]) << PyLong_SHIFT) | (int)digits[1]) << PyLong_SHIFT) | (int)digits[0]));
                        }
                    }
                    break;
            }
#endif
#if CYTHON_COMPILING_IN_CPYTHON
            if (unlikely(Py_SIZE(x) < 0)) {
                goto raise_neg_overflow;
            }
#else
            {
                int result = PyObject_RichCompareBool(x, Py_False, Py_LT);
                if (unlikely(result < 0))
                    return (int) -1;
                if (unlikely(result == 1))
                    goto raise_neg_overflow;
            }
#endif
            if (sizeof(int) <= sizeof(unsigned long)) {
                __PYX_VERIFY_RETURN_INT_EXC(int, unsigned long, PyLong_AsUnsignedLong(x))
            } else if (sizeof(int) <= sizeof(unsigned PY_LONG_LONG)) {
                __PYX_VERIFY_RETURN_INT_EXC(int, unsigned PY_LONG_LONG, PyLong_AsUnsignedLongLong(x))
            }
        } else {
#if CYTHON_USE_PYLONG_INTERNALS
            const digit* digits = ((PyLongObject*)x)->ob_digit;
            switch (Py_SIZE(x)) {
                case  0: return (int) 0;
                case -1: __PYX_VERIFY_RETURN_INT(int, sdigit, (sdigit) (-(sdigit)digits[0]))
                case  1: __PYX_VERIFY_RETURN_INT(int,  digit, +digits[0])
                case -2:
                    if (8 * sizeof(int) - 1 > 1 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 2 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(int, long, -(long) (((((unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(int) - 1 > 2 * PyLong_SHIFT) {
                            return (int) (((int)-1)*(((((int)digits[1]) << PyLong_SHIFT) | (int)digits[0])));
                        }
                    }
                    break;
                case 2:
                    if (8 * sizeof(int) > 1 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 2 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(int, unsigned long, (((((unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(int) - 1 > 2 * PyLong_SHIFT) {
                            return (int) ((((((int)digits[1]) << PyLong_SHIFT) | (int)digits[0])));
                        }
                    }
                    break;
                case -3:
                    if (8 * sizeof(int) - 1 > 2 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 3 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(int, long, -(long) (((((((unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(int) - 1 > 3 * PyLong_SHIFT) {
                            return (int) (((int)-1)*(((((((int)digits[2]) << PyLong_SHIFT) | (int)digits[1]) << PyLong_SHIFT) | (int)digits[0])));
                        }
                    }
                    break;
                case 3:
                    if (8 * sizeof(int) > 2 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 3 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(int, unsigned long, (((((((unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(int) - 1 > 3 * PyLong_SHIFT) {
                            return (int) ((((((((int)digits[2]) << PyLong_SHIFT) | (int)digits[1]) << PyLong_SHIFT) | (int)digits[0])));
                        }
                    }
                    break;
                case -4:
                    if (8 * sizeof(int) - 1 > 3 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 4 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(int, long, -(long) (((((((((unsigned long)digits[3]) << PyLong_SHIFT) | (unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(int) - 1 > 4 * PyLong_SHIFT) {
                            return (int) (((int)-1)*(((((((((int)digits[3]) << PyLong_SHIFT) | (int)digits[2]) << PyLong_SHIFT) | (int)digits[1]) << PyLong_SHIFT) | (int)digits[0])));
                        }
                    }
                    break;
                case 4:
                    if (8 * sizeof(int) > 3 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 4 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(int, unsigned long, (((((((((unsigned long)digits[3]) << PyLong_SHIFT) | (unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(int) - 1 > 4 * PyLong_SHIFT) {
                            return (int) ((((((((((int)digits[3]) << PyLong_SHIFT) | (int)digits[2]) << PyLong_SHIFT) | (int)digits[1]) << PyLong_SHIFT) | (int)digits[0])));
                        }
                    }
                    break;
            }
#endif
            if (sizeof(int) <= sizeof(long)) {
                __PYX_VERIFY_RETURN_INT_EXC(int, long, PyLong_AsLong(x))
            } else if (sizeof(int) <= sizeof(PY_LONG_LONG)) {
                __PYX_VERIFY_RETURN_INT_EXC(int, PY_LONG_LONG, PyLong_AsLongLong(x))
            }
        }
        {
#if CYTHON_COMPILING_IN_PYPY && !defined(_PyLong_AsByteArray)
            PyErr_SetString(PyExc_RuntimeError,
                            "_PyLong_AsByteArray() not available in PyPy, cannot convert large numbers");
#else
            int val;
            PyObject *v = __Pyx_PyNumber_IntOrLong(x);
 #if PY_MAJOR_VERSION < 3
            if (likely(v) && !PyLong_Check(v)) {
                PyObject *tmp = v;
                v = PyNumber_Long(tmp);
                Py_DECREF(tmp);
            }
 #endif
            if (likely(v)) {
                int one = 1; int is_little = (int)*(unsigned char *)&one;
                unsigned char *bytes = (unsigned char *)&val;
                int ret = _PyLong_AsByteArray((PyLongObject *)v,
                                              bytes, sizeof(val),
                                              is_little, !is_unsigned);
                Py_DECREF(v);
                if (likely(!ret))
                    return val;
            }
#endif
            return (int) -1;
        }
    } else {
        int val;
        PyObject *tmp = __Pyx_PyNumber_IntOrLong(x);
        if (!tmp) return (int) -1;
        val = __Pyx_PyInt_As_int(tmp);
        Py_DECREF(tmp);
        return val;
    }
raise_overflow:
    PyErr_SetString(PyExc_OverflowError,
        "value too large to convert to int");
    return (int) -1;
raise_neg_overflow:
    PyErr_SetString(PyExc_OverflowError,
        "can't convert negative value to int");
    return (int) -1;
}

/* CIntToPy */
      static CYTHON_INLINE PyObject* __Pyx_PyInt_From_long(long value) {
    const long neg_one = (long) -1, const_zero = (long) 0;
    const int is_unsigned = neg_one > const_zero;
    if (is_unsigned) {
        if (sizeof(long) < sizeof(long)) {
            return PyInt_FromLong((long) value);
        } else if (sizeof(long) <= sizeof(unsigned long)) {
            return PyLong_FromUnsignedLong((unsigned long) value);
        } else if (sizeof(long) <= sizeof(unsigned PY_LONG_LONG)) {
            return PyLong_FromUnsignedLongLong((unsigned PY_LONG_LONG) value);
        }
    } else {
        if (sizeof(long) <= sizeof(long)) {
            return PyInt_FromLong((long) value);
        } else if (sizeof(long) <= sizeof(PY_LONG_LONG)) {
            return PyLong_FromLongLong((PY_LONG_LONG) value);
        }
    }
    {
        int one = 1; int little = (int)*(unsigned char *)&one;
        unsigned char *bytes = (unsigned char *)&value;
        return _PyLong_FromByteArray(bytes, sizeof(long),
                                     little, !is_unsigned);
    }
}

/* CIntFromPy */
      static CYTHON_INLINE long __Pyx_PyInt_As_long(PyObject *x) {
    const long neg_one = (long) -1, const_zero = (long) 0;
    const int is_unsigned = neg_one > const_zero;
#if PY_MAJOR_VERSION < 3
    if (likely(PyInt_Check(x))) {
        if (sizeof(long) < sizeof(long)) {
            __PYX_VERIFY_RETURN_INT(long, long, PyInt_AS_LONG(x))
        } else {
            long val = PyInt_AS_LONG(x);
            if (is_unsigned && unlikely(val < 0)) {
                goto raise_neg_overflow;
            }
            return (long) val;
        }
    } else
#endif
    if (likely(PyLong_Check(x))) {
        if (is_unsigned) {
#if CYTHON_USE_PYLONG_INTERNALS
            const digit* digits = ((PyLongObject*)x)->ob_digit;
            switch (Py_SIZE(x)) {
                case  0: return (long) 0;
                case  1: __PYX_VERIFY_RETURN_INT(long, digit, digits[0])
                case 2:
                    if (8 * sizeof(long) > 1 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 2 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(long, unsigned long, (((((unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(long) >= 2 * PyLong_SHIFT) {
                            return (long) (((((long)digits[1]) << PyLong_SHIFT) | (long)digits[0]));
                        }
                    }
                    break;
                case 3:
                    if (8 * sizeof(long) > 2 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 3 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(long, unsigned long, (((((((unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(long) >= 3 * PyLong_SHIFT) {
                            return (long) (((((((long)digits[2]) << PyLong_SHIFT) | (long)digits[1]) << PyLong_SHIFT) | (long)digits[0]));
                        }
                    }
                    break;
                case 4:
                    if (8 * sizeof(long) > 3 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 4 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(long, unsigned long, (((((((((unsigned long)digits[3]) << PyLong_SHIFT) | (unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(long) >= 4 * PyLong_SHIFT) {
                            return (long) (((((((((long)digits[3]) << PyLong_SHIFT) | (long)digits[2]) << PyLong_SHIFT) | (long)digits[1]) << PyLong_SHIFT) | (long)digits[0]));
                        }
                    }
                    break;
            }
#endif
#if CYTHON_COMPILING_IN_CPYTHON
            if (unlikely(Py_SIZE(x) < 0)) {
                goto raise_neg_overflow;
            }
#else
            {
                int result = PyObject_RichCompareBool(x, Py_False, Py_LT);
                if (unlikely(result < 0))
                    return (long) -1;
                if (unlikely(result == 1))
                    goto raise_neg_overflow;
            }
#endif
            if (sizeof(long) <= sizeof(unsigned long)) {
                __PYX_VERIFY_RETURN_INT_EXC(long, unsigned long, PyLong_AsUnsignedLong(x))
            } else if (sizeof(long) <= sizeof(unsigned PY_LONG_LONG)) {
                __PYX_VERIFY_RETURN_INT_EXC(long, unsigned PY_LONG_LONG, PyLong_AsUnsignedLongLong(x))
            }
        } else {
#if CYTHON_USE_PYLONG_INTERNALS
            const digit* digits = ((PyLongObject*)x)->ob_digit;
            switch (Py_SIZE(x)) {
                case  0: return (long) 0;
                case -1: __PYX_VERIFY_RETURN_INT(long, sdigit, (sdigit) (-(sdigit)digits[0]))
                case  1: __PYX_VERIFY_RETURN_INT(long,  digit, +digits[0])
                case -2:
                    if (8 * sizeof(long) - 1 > 1 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 2 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(long, long, -(long) (((((unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(long) - 1 > 2 * PyLong_SHIFT) {
                            return (long) (((long)-1)*(((((long)digits[1]) << PyLong_SHIFT) | (long)digits[0])));
                        }
                    }
                    break;
                case 2:
                    if (8 * sizeof(long) > 1 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 2 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(long, unsigned long, (((((unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(long) - 1 > 2 * PyLong_SHIFT) {
                            return (long) ((((((long)digits[1]) << PyLong_SHIFT) | (long)digits[0])));
                        }
                    }
                    break;
                case -3:
                    if (8 * sizeof(long) - 1 > 2 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 3 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(long, long, -(long) (((((((unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(long) - 1 > 3 * PyLong_SHIFT) {
                            return (long) (((long)-1)*(((((((long)digits[2]) << PyLong_SHIFT) | (long)digits[1]) << PyLong_SHIFT) | (long)digits[0])));
                        }
                    }
                    break;
                case 3:
                    if (8 * sizeof(long) > 2 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 3 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(long, unsigned long, (((((((unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(long) - 1 > 3 * PyLong_SHIFT) {
                            return (long) ((((((((long)digits[2]) << PyLong_SHIFT) | (long)digits[1]) << PyLong_SHIFT) | (long)digits[0])));
                        }
                    }
                    break;
                case -4:
                    if (8 * sizeof(long) - 1 > 3 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 4 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(long, long, -(long) (((((((((unsigned long)digits[3]) << PyLong_SHIFT) | (unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(long) - 1 > 4 * PyLong_SHIFT) {
                            return (long) (((long)-1)*(((((((((long)digits[3]) << PyLong_SHIFT) | (long)digits[2]) << PyLong_SHIFT) | (long)digits[1]) << PyLong_SHIFT) | (long)digits[0])));
                        }
                    }
                    break;
                case 4:
                    if (8 * sizeof(long) > 3 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 4 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(long, unsigned long, (((((((((unsigned long)digits[3]) << PyLong_SHIFT) | (unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(long) - 1 > 4 * PyLong_SHIFT) {
                            return (long) ((((((((((long)digits[3]) << PyLong_SHIFT) | (long)digits[2]) << PyLong_SHIFT) | (long)digits[1]) << PyLong_SHIFT) | (long)digits[0])));
                        }
                    }
                    break;
            }
#endif
            if (sizeof(long) <= sizeof(long)) {
                __PYX_VERIFY_RETURN_INT_EXC(long, long, PyLong_AsLong(x))
            } else if (sizeof(long) <= sizeof(PY_LONG_LONG)) {
                __PYX_VERIFY_RETURN_INT_EXC(long, PY_LONG_LONG, PyLong_AsLongLong(x))
            }
        }
        {
#if CYTHON_COMPILING_IN_PYPY && !defined(_PyLong_AsByteArray)
            PyErr_SetString(PyExc_RuntimeError,
                            "_PyLong_AsByteArray() not available in PyPy, cannot convert large numbers");
#else
            long val;
            PyObject *v = __Pyx_PyNumber_IntOrLong(x);
 #if PY_MAJOR_VERSION < 3
            if (likely(v) && !PyLong_Check(v)) {
                PyObject *tmp = v;
                v = PyNumber_Long(tmp);
                Py_DECREF(tmp);
            }
 #endif
            if (likely(v)) {
                int one = 1; int is_little = (int)*(unsigned char *)&one;
                unsigned char *bytes = (unsigned char *)&val;
                int ret = _PyLong_AsByteArray((PyLongObject *)v,
                                              bytes, sizeof(val),
                                              is_little, !is_unsigned);
                Py_DECREF(v);
                if (likely(!ret))
                    return val;
            }
#endif
            return (long) -1;
        }
    } else {
        long val;
        PyObject *tmp = __Pyx_PyNumber_IntOrLong(x);
        if (!tmp) return (long) -1;
        val = __Pyx_PyInt_As_long(tmp);
        Py_DECREF(tmp);
        return val;
    }
raise_overflow:
    PyErr_SetString(PyExc_OverflowError,
        "value too large to convert to long");
    return (long) -1;
raise_neg_overflow:
    PyErr_SetString(PyExc_OverflowError,
        "can't convert negative value to long");
    return (long) -1;
}

/* CheckBinaryVersion */
      static int __Pyx_check_binary_version(void) {
    char ctversion[4], rtversion[4];
    PyOS_snprintf(ctversion, 4, "%d.%d", PY_MAJOR_VERSION, PY_MINOR_VERSION);
    PyOS_snprintf(rtversion, 4, "%s", Py_GetVersion());
    if (ctversion[0] != rtversion[0] || ctversion[2] != rtversion[2]) {
        char message[200];
        PyOS_snprintf(message, sizeof(message),
                      "compiletime version %s of module '%.100s' "
                      "does not match runtime version %s",
                      ctversion, __Pyx_MODULE_NAME, rtversion);
        return PyErr_WarnEx(NULL, message, 1);
    }
    return 0;
}

/* ModuleImport */
      #ifndef __PYX_HAVE_RT_ImportModule
#define __PYX_HAVE_RT_ImportModule
static PyObject *__Pyx_ImportModule(const char *name) {
    PyObject *py_name = 0;
    PyObject *py_module = 0;
    py_name = __Pyx_PyIdentifier_FromString(name);
    if (!py_name)
        goto bad;
    py_module = PyImport_Import(py_name);
    Py_DECREF(py_name);
    return py_module;
bad:
    Py_XDECREF(py_name);
    return 0;
}
#endif

/* TypeImport */
      #ifndef __PYX_HAVE_RT_ImportType
#define __PYX_HAVE_RT_ImportType
static PyTypeObject *__Pyx_ImportType(const char *module_name, const char *class_name,
    size_t size, int strict)
{
    PyObject *py_module = 0;
    PyObject *result = 0;
    PyObject *py_name = 0;
    char warning[200];
    Py_ssize_t basicsize;
#ifdef Py_LIMITED_API
    PyObject *py_basicsize;
#endif
    py_module = __Pyx_ImportModule(module_name);
    if (!py_module)
        goto bad;
    py_name = __Pyx_PyIdentifier_FromString(class_name);
    if (!py_name)
        goto bad;
    result = PyObject_GetAttr(py_module, py_name);
    Py_DECREF(py_name);
    py_name = 0;
    Py_DECREF(py_module);
    py_module = 0;
    if (!result)
        goto bad;
    if (!PyType_Check(result)) {
        PyErr_Format(PyExc_TypeError,
            "%.200s.%.200s is not a type object",
            module_name, class_name);
        goto bad;
    }
#ifndef Py_LIMITED_API
    basicsize = ((PyTypeObject *)result)->tp_basicsize;
#else
    py_basicsize = PyObject_GetAttrString(result, "__basicsize__");
    if (!py_basicsize)
        goto bad;
    basicsize = PyLong_AsSsize_t(py_basicsize);
    Py_DECREF(py_basicsize);
    py_basicsize = 0;
    if (basicsize == (Py_ssize_t)-1 && PyErr_Occurred())
        goto bad;
#endif
    if (!strict && (size_t)basicsize > size) {
        PyOS_snprintf(warning, sizeof(warning),
            "%s.%s size changed, may indicate binary incompatibility. Expected %zd, got %zd",
            module_name, class_name, basicsize, size);
        if (PyErr_WarnEx(NULL, warning, 0) < 0) goto bad;
    }
    else if ((size_t)basicsize != size) {
        PyErr_Format(PyExc_ValueError,
            "%.200s.%.200s has the wrong size, try recompiling. Expected %zd, got %zd",
            module_name, class_name, basicsize, size);
        goto bad;
    }
    return (PyTypeObject *)result;
bad:
    Py_XDECREF(py_module);
    Py_XDECREF(result);
    return NULL;
}
#endif

/* InitStrings */
      static int __Pyx_InitStrings(__Pyx_StringTabEntry *t) {
    while (t->p) {
        #if PY_MAJOR_VERSION < 3
        if (t->is_unicode) {
            *t->p = PyUnicode_DecodeUTF8(t->s, t->n - 1, NULL);
        } else if (t->intern) {
            *t->p = PyString_InternFromString(t->s);
        } else {
            *t->p = PyString_FromStringAndSize(t->s, t->n - 1);
        }
        #else
        if (t->is_unicode | t->is_str) {
            if (t->intern) {
                *t->p = PyUnicode_InternFromString(t->s);
            } else if (t->encoding) {
                *t->p = PyUnicode_Decode(t->s, t->n - 1, t->encoding, NULL);
            } else {
                *t->p = PyUnicode_FromStringAndSize(t->s, t->n - 1);
            }
        } else {
            *t->p = PyBytes_FromStringAndSize(t->s, t->n - 1);
        }
        #endif
        if (!*t->p)
            return -1;
        ++t;
    }
    return 0;
}

static CYTHON_INLINE PyObject* __Pyx_PyUnicode_FromString(const char* c_str) {
    return __Pyx_PyUnicode_FromStringAndSize(c_str, (Py_ssize_t)strlen(c_str));
}
static CYTHON_INLINE char* __Pyx_PyObject_AsString(PyObject* o) {
    Py_ssize_t ignore;
    return __Pyx_PyObject_AsStringAndSize(o, &ignore);
}
static CYTHON_INLINE char* __Pyx_PyObject_AsStringAndSize(PyObject* o, Py_ssize_t *length) {
#if CYTHON_COMPILING_IN_CPYTHON && (__PYX_DEFAULT_STRING_ENCODING_IS_ASCII || __PYX_DEFAULT_STRING_ENCODING_IS_DEFAULT)
    if (
#if PY_MAJOR_VERSION < 3 && __PYX_DEFAULT_STRING_ENCODING_IS_ASCII
            __Pyx_sys_getdefaultencoding_not_ascii &&
#endif
            PyUnicode_Check(o)) {
#if PY_VERSION_HEX < 0x03030000
        char* defenc_c;
        PyObject* defenc = _PyUnicode_AsDefaultEncodedString(o, NULL);
        if (!defenc) return NULL;
        defenc_c = PyBytes_AS_STRING(defenc);
#if __PYX_DEFAULT_STRING_ENCODING_IS_ASCII
        {
            char* end = defenc_c + PyBytes_GET_SIZE(defenc);
            char* c;
            for (c = defenc_c; c < end; c++) {
                if ((unsigned char) (*c) >= 128) {
                    PyUnicode_AsASCIIString(o);
                    return NULL;
                }
            }
        }
#endif
        *length = PyBytes_GET_SIZE(defenc);
        return defenc_c;
#else
        if (__Pyx_PyUnicode_READY(o) == -1) return NULL;
#if __PYX_DEFAULT_STRING_ENCODING_IS_ASCII
        if (PyUnicode_IS_ASCII(o)) {
            *length = PyUnicode_GET_LENGTH(o);
            return PyUnicode_AsUTF8(o);
        } else {
            PyUnicode_AsASCIIString(o);
            return NULL;
        }
#else
        return PyUnicode_AsUTF8AndSize(o, length);
#endif
#endif
    } else
#endif
#if (!CYTHON_COMPILING_IN_PYPY) || (defined(PyByteArray_AS_STRING) && defined(PyByteArray_GET_SIZE))
    if (PyByteArray_Check(o)) {
        *length = PyByteArray_GET_SIZE(o);
        return PyByteArray_AS_STRING(o);
    } else
#endif
    {
        char* result;
        int r = PyBytes_AsStringAndSize(o, &result, length);
        if (unlikely(r < 0)) {
            return NULL;
        } else {
            return result;
        }
    }
}
static CYTHON_INLINE int __Pyx_PyObject_IsTrue(PyObject* x) {
   int is_true = x == Py_True;
   if (is_true | (x == Py_False) | (x == Py_None)) return is_true;
   else return PyObject_IsTrue(x);
}
static CYTHON_INLINE PyObject* __Pyx_PyNumber_IntOrLong(PyObject* x) {
  PyNumberMethods *m;
  const char *name = NULL;
  PyObject *res = NULL;
#if PY_MAJOR_VERSION < 3
  if (PyInt_Check(x) || PyLong_Check(x))
#else
  if (PyLong_Check(x))
#endif
    return __Pyx_NewRef(x);
  m = Py_TYPE(x)->tp_as_number;
#if PY_MAJOR_VERSION < 3
  if (m && m->nb_int) {
    name = "int";
    res = PyNumber_Int(x);
  }
  else if (m && m->nb_long) {
    name = "long";
    res = PyNumber_Long(x);
  }
#else
  if (m && m->nb_int) {
    name = "int";
    res = PyNumber_Long(x);
  }
#endif
  if (res) {
#if PY_MAJOR_VERSION < 3
    if (!PyInt_Check(res) && !PyLong_Check(res)) {
#else
    if (!PyLong_Check(res)) {
#endif
      PyErr_Format(PyExc_TypeError,
                   "__%.4s__ returned non-%.4s (type %.200s)",
                   name, name, Py_TYPE(res)->tp_name);
      Py_DECREF(res);
      return NULL;
    }
  }
  else if (!PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError,
                    "an integer is required");
  }
  return res;
}
static CYTHON_INLINE Py_ssize_t __Pyx_PyIndex_AsSsize_t(PyObject* b) {
  Py_ssize_t ival;
  PyObject *x;
#if PY_MAJOR_VERSION < 3
  if (likely(PyInt_CheckExact(b))) {
    if (sizeof(Py_ssize_t) >= sizeof(long))
        return PyInt_AS_LONG(b);
    else
        return PyInt_AsSsize_t(x);
  }
#endif
  if (likely(PyLong_CheckExact(b))) {
    #if CYTHON_USE_PYLONG_INTERNALS
    const digit* digits = ((PyLongObject*)b)->ob_digit;
    const Py_ssize_t size = Py_SIZE(b);
    if (likely(__Pyx_sst_abs(size) <= 1)) {
        ival = likely(size) ? digits[0] : 0;
        if (size == -1) ival = -ival;
        return ival;
    } else {
      switch (size) {
         case 2:
           if (8 * sizeof(Py_ssize_t) > 2 * PyLong_SHIFT) {
             return (Py_ssize_t) (((((size_t)digits[1]) << PyLong_SHIFT) | (size_t)digits[0]));
           }
           break;
         case -2:
           if (8 * sizeof(Py_ssize_t) > 2 * PyLong_SHIFT) {
             return -(Py_ssize_t) (((((size_t)digits[1]) << PyLong_SHIFT) | (size_t)digits[0]));
           }
           break;
         case 3:
           if (8 * sizeof(Py_ssize_t) > 3 * PyLong_SHIFT) {
             return (Py_ssize_t) (((((((size_t)digits[2]) << PyLong_SHIFT) | (size_t)digits[1]) << PyLong_SHIFT) | (size_t)digits[0]));
           }
           break;
         case -3:
           if (8 * sizeof(Py_ssize_t) > 3 * PyLong_SHIFT) {
             return -(Py_ssize_t) (((((((size_t)digits[2]) << PyLong_SHIFT) | (size_t)digits[1]) << PyLong_SHIFT) | (size_t)digits[0]));
           }
           break;
         case 4:
           if (8 * sizeof(Py_ssize_t) > 4 * PyLong_SHIFT) {
             return (Py_ssize_t) (((((((((size_t)digits[3]) << PyLong_SHIFT) | (size_t)digits[2]) << PyLong_SHIFT) | (size_t)digits[1]) << PyLong_SHIFT) | (size_t)digits[0]));
           }
           break;
         case -4:
           if (8 * sizeof(Py_ssize_t) > 4 * PyLong_SHIFT) {
             return -(Py_ssize_t) (((((((((size_t)digits[3]) << PyLong_SHIFT) | (size_t)digits[2]) << PyLong_SHIFT) | (size_t)digits[1]) << PyLong_SHIFT) | (size_t)digits[0]));
           }
           break;
      }
    }
    #endif
    return PyLong_AsSsize_t(b);
  }
  x = PyNumber_Index(b);
  if (!x) return -1;
  ival = PyInt_AsSsize_t(x);
  Py_DECREF(x);
  return ival;
}
static CYTHON_INLINE PyObject * __Pyx_PyInt_FromSize_t(size_t ival) {
    return PyInt_FromSize_t(ival);
}


#endif /* Py_PYTHON_H */

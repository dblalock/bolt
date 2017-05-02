/*
 * The Biomechanical ToolKit
 * Copyright (c) 2009-2013, Arnaud Barré
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *     * Redistributions of source code must retain the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer.
 *     * Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name(s) of the copyright holders nor the names
 *       of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

 /* Additional credit to Martin Felis:
  * https://bitbucket.org/MartinFelis/eigen3swig/src
  *
  * and Robin Deits:
  * https://github.com/rdeits/swig-eigen-numpy/blob/master/swigmake/swig/python/eigen.i
  *
  * Plus some minor modifications of my own (Davis Blalock, 2016-3-7)
  */

%{
  #define SWIG_FILE_WITH_INIT
  #include "eigen/Core"
%}

%include "numpy.i"

%init
%{
  import_array();
%}

%fragment("Eigen_Fragments", "header",  fragment="NumPy_Fragments")
%{
  // these funcs define the mapping between c types and numpy types;
  // add more as needed
  template <typename T> int NumPyType() { return -1; };

  template<> int NumPyType<float>() {return NPY_FLOAT;};
  template<> int NumPyType<double>() {return NPY_DOUBLE;};

  template<> int NumPyType<signed char>() {return NPY_BYTE;};
  template<> int NumPyType<unsigned char>() {return NPY_UBYTE;};
  template<> int NumPyType<short>() {return NPY_INT;};
  template<> int NumPyType<unsigned short>() {return NPY_USHORT;};
  template<> int NumPyType<int>() {return NPY_INT;};
  template<> int NumPyType<unsigned int>() {return NPY_UINT;};
  template<> int NumPyType<long>() {return NPY_LONG;};
  template<> int NumPyType<unsigned long>() {return NPY_ULONG;};
  template<> int NumPyType<long long>() {return NPY_LONGLONG;};
  template<> int NumPyType<unsigned long long>() {return NPY_ULONGLONG;};

  // template<> int NumPyType<int8_t>() {return NPY_BYTE;};
  // template<> int NumPyType<int16_t>() {return NPY_INT;};
  // template<> int NumPyType<int32_t>() {return NPY_INT;};
  // template<> int NumPyType<int64_t>() {return NPY_LONGLONG;};
  // template<> int NumPyType<uint8_t>() {return NPY_BYTE;};
  // template<> int NumPyType<uint16_t>() {return NPY_UINT;};
  // template<> int NumPyType<uint32_t>() {return NPY_UINT;};
  // template<> int NumPyType<uint64_t>() {return NPY_ULONGLONG;};


  template <class Derived>
  bool ConvertFromNumpyToEigenMatrix(Eigen::DenseBase<Derived>* out, PyObject* in)
  {
    typedef typename Derived::Scalar Scalar;
    int rows = 0;
    int cols = 0;
    // Check object type
    if (!is_array(in))
    {
      PyErr_SetString(PyExc_ValueError, "The given input is not known as a NumPy array or matrix.");
      return false;
    }
    // Check data type
    else if (array_type(in) != NumPyType<Scalar>())
    {
      PyErr_SetString(PyExc_ValueError, "Type mismatch between NumPy and Eigen objects.");
      return false;
    }
    // Check dimensions
    else if (array_numdims(in) > 2)
    {
      PyErr_SetString(PyExc_ValueError, "Eigen only support 1D or 2D array.");
      return false;
    }
    else if (array_numdims(in) == 1)
    {
      rows = array_size(in,0);
      cols = 1;
      if ((Derived::RowsAtCompileTime != Eigen::Dynamic) && (Derived::RowsAtCompileTime != rows))
      {
        PyErr_SetString(PyExc_ValueError, "Row dimension mismatch between NumPy and Eigen objects (1D).");
        return false;
      }
      else if ((Derived::ColsAtCompileTime != Eigen::Dynamic) && (Derived::ColsAtCompileTime != 1))
      {
        PyErr_SetString(PyExc_ValueError, "Column dimension mismatch between NumPy and Eigen objects (1D).");
        return false;
      }
    }
    else if (array_numdims(in) == 2)
    {
      rows = array_size(in,0);
      cols = array_size(in,1);
      if ((Derived::RowsAtCompileTime != Eigen::Dynamic) && (Derived::RowsAtCompileTime != array_size(in,0)))
      {
        PyErr_SetString(PyExc_ValueError, "Row dimension mismatch between NumPy and Eigen objects (2D).");
        return false;
      }
      else if ((Derived::ColsAtCompileTime != Eigen::Dynamic) && (Derived::ColsAtCompileTime != array_size(in,1)))
      {
        PyErr_SetString(PyExc_ValueError, "Column dimension mismatch between NumPy and Eigen objects (2D).");
        return false;
      }
    }
    // Extract data
    int isNewObject = 0;
    PyArrayObject* temp = obj_to_array_contiguous_allow_conversion(in, array_type(in), &isNewObject);
    if (temp == NULL)
    {
      PyErr_SetString(PyExc_ValueError, "Impossible to convert the input into a Python array object.");
      return false;
    }
    out->derived().setZero(rows, cols);
    Scalar* data = static_cast<Scalar*>(array_data(temp));
    if (array_is_fortran(temp)) { // column-major
      for (int j = 0; j != cols; ++j) {
        for (int i = 0; i != rows; ++i) {
          out->coeffRef(i,j) = data[j*rows + i];
        }
      }
    } else { // row-major
      for (int i = 0; i != rows; ++i) {
        for (int j = 0; j != cols; ++j) {
          out->coeffRef(i,j) = data[i*cols + j];
        }
      }
    }

    return true;
  };

  // Copies values from Eigen type into an existing NumPy type
  template <class Derived>
  bool CopyFromEigenToNumPyMatrix(PyObject* out, Eigen::DenseBase<Derived>* in)
  {
    typedef typename Derived::Scalar Scalar;
    int numpy_scalar_t = NumPyType<Scalar>();
    int rows = 0;
    int cols = 0;
    // Check object type
    if (!is_array(out))
    {
      PyErr_SetString(PyExc_ValueError, "The given input is not known as a NumPy array or matrix.");
      return false;
    }
    // Check data type
    else if (array_type(out) != numpy_scalar_t)
    {
      PyErr_SetString(PyExc_ValueError, "Type mismatch between NumPy and Eigen objects.");
      return false;
    }
    // Check dimensions
    else if (array_numdims(out) > 2)
    {
      PyErr_SetString(PyExc_ValueError, "Eigen only supports 1D or 2D array.");
      return false;
    }
    else if (array_numdims(out) == 1)
    {
      rows = array_size(out,0);
      cols = 1;
      if ((Derived::RowsAtCompileTime != Eigen::Dynamic) && (Derived::RowsAtCompileTime != rows))
      {
        PyErr_SetString(PyExc_ValueError, "Row dimension mismatch between NumPy and Eigen objects (1D).");
        return false;
      }
      else if ((Derived::ColsAtCompileTime != Eigen::Dynamic) && (Derived::ColsAtCompileTime != 1))
      {
        PyErr_SetString(PyExc_ValueError, "Column dimension mismatch between NumPy and Eigen objects (1D).");
        return false;
      }
    }
    else if (array_numdims(out) == 2)
    {
      rows = array_size(out,0);
      cols = array_size(out,1);
    }

    if (in->cols() != cols || in->rows() != rows) {
      /// TODO: be forgiving and simply create or resize the array
      PyErr_SetString(PyExc_ValueError, "Dimension mismatch between NumPy and Eigen object (return argument).");
      return false;
    }

    // Extract data
    int isNewObject = 0;
    PyArrayObject* temp = obj_to_array_contiguous_allow_conversion(out, array_type(out), &isNewObject);
    if (temp == NULL)
    {
      PyErr_SetString(PyExc_ValueError, "Impossible to convert the input into a Python array object.");
      return false;
    }

    Scalar* data = static_cast<Scalar*>(array_data(out));
    if (array_is_fortran(out)) { // column-major
      for (int j = 0; j != in->cols(); ++j) {
        for (int i = 0; i != in->rows(); ++i) {
          data[j*in->rows() + i] = in->coeff(i,j);
        }
      }
    } else { // row-major
      for (int i = 0; i != in->rows(); ++i) {
        for (int j = 0; j != in->cols(); ++j) {
          data[i*in->cols() + j] = in->coeff(i,j);
        }
      }
    }
    return true;
  };

  template <class Derived>
  bool ConvertFromEigenToNumPyMatrix(PyObject** out, Eigen::DenseBase<Derived>* in)
  {
    typedef typename Derived::Scalar Scalar;
    int numpy_scalar_t = NumPyType<Scalar>();

    if (numpy_scalar_t == -1) {
      PyErr_SetString(PyExc_ValueError, "No numpy type known for Eigen object's scalar type");
      return false;
    }

    // vector (1D)
    if (in->cols() == 1 || in->rows() == 1) {
      npy_intp dims[1] = {in->size()};
      *out = PyArray_SimpleNew(1, dims, numpy_scalar_t);
      if (!out) {
        return false;
      }
      Scalar* data = static_cast<Scalar*>(array_data(*out));
      for (int i = 0; i != dims[0]; ++i) {
        data[i] = in->coeff(i);
      }
      return true;
    }
    // matrix (2D)
    npy_intp dims[2] = {in->rows(), in->cols()};
    *out = PyArray_SimpleNew(2, dims, numpy_scalar_t);
    if (!out) {
      return false;
    }
    Scalar* data = static_cast<Scalar*>(array_data(*out));
    npy_intp rows = dims[0];
    npy_intp cols = dims[1];
    if (array_is_fortran(out)) { // column-major
      for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
          data[j*rows + i] = in->coeff(i,j);
        }
      }
    } else { // row-major
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          data[i*cols + j] = in->coeff(i,j);
        }
      }
    }
    return true;
  };
%}

// ----------------------------------------------------------------------------
// Macro to create the typemap for Eigen classes
// ----------------------------------------------------------------------------
%define %eigen_typemaps(CLASS)

// Argout: const & (Disabled and prevents calling of the non-const typemap)
%typemap(argout, fragment="Eigen_Fragments") const CLASS & ""

// Argout: & (for returning values to in-out arguments)
%typemap(argout, fragment="Eigen_Fragments") CLASS &
{
  // Argout: &
  if (!CopyFromEigenToNumPyMatrix<CLASS >($input, $1)) {
    SWIG_fail;
  }
}

// In: (nothing: no constness)
%typemap(in, fragment="Eigen_Fragments") CLASS (CLASS temp)
{
  if (!ConvertFromNumpyToEigenMatrix<CLASS>(&temp, $input)) {
    SWIG_fail;
  }
  $1 = temp;
}
// In: const&
%typemap(in, fragment="Eigen_Fragments") CLASS const& (CLASS temp)
{
  // In: const&
  if (!ConvertFromNumpyToEigenMatrix<CLASS >(&temp, $input)) {
    SWIG_fail;
  }
  $1 = &temp;
}
// In: &
%typemap(in, fragment="Eigen_Fragments") CLASS & (CLASS temp)
{
  // In: non-const&
  if (!ConvertFromNumpyToEigenMatrix<CLASS >(&temp, $input)) {
    SWIG_fail;
  }
  $1 = &temp;
}
// In: const* (not yet implemented)
%typemap(in, fragment="Eigen_Fragments") CLASS const*
{
  PyErr_SetString(PyExc_ValueError, "The input typemap for const pointer is not yet implemented. Please report this problem to the developer.");
  SWIG_fail;
}
// In: * (not yet implemented)
%typemap(in, fragment="Eigen_Fragments") CLASS *
{
  PyErr_SetString(PyExc_ValueError, "The input typemap for non-const pointer is not yet implemented. Please report this problem to the developer.");
  SWIG_fail;
}

// Out: (nothing: no constness)
%typemap(out, fragment="Eigen_Fragments") CLASS
{
  ConvertFromEigenToNumPyMatrix<CLASS>(&$result, &$1);
}
// Out: const
%typemap(out, fragment="Eigen_Fragments") CLASS const
{
  ConvertFromEigenToNumPyMatrix<CLASS>(&$result, &$1);
}
// Out: const&
%typemap(out, fragment="Eigen_Fragments") CLASS const&
{
  ConvertFromEigenToNumPyMatrix<CLASS>(&$result, $1);
}
// Out: & (not yet implemented)
%typemap(out, fragment="Eigen_Fragments") CLASS &
{
  PyErr_SetString(PyExc_ValueError, "The output typemap for non-const reference is not yet implemented. Please report this problem to the developer.");
}
// Out: const* (not yet implemented)
%typemap(out, fragment="Eigen_Fragments") CLASS const*
{
  PyErr_SetString(PyExc_ValueError, "The output typemap for const pointer is not yet implemented. Please report this problem to the developer.");
}
// Out: * (not yet implemented)
%typemap(out, fragment="Eigen_Fragments") CLASS *
{
  PyErr_SetString(PyExc_ValueError, "The output typemap for non-const pointer is not yet implemented. Please report this problem to the developer.");
}

%typemap(out, fragment="Eigen_Fragments") std::vector<CLASS >
{
  $result = PyList_New($1.size());
  if (!$result)
    SWIG_fail;
  for (size_t i=0; i != $1.size(); ++i) {
    PyObject *out;
    if (!ConvertFromEigenToNumPyMatrix(&out, &$1[i]))
      SWIG_fail;
    if (PyList_SetItem($result, i, out) == -1)
      SWIG_fail;
  }
}

// ------------------------ Rdeits typemaps for vectors of arrays/matrices

// %typemap(in, fragment="Eigen_Fragments") std::vector<CLASS > (std::vector<CLASS > temp)
// {
//   if (!PyList_Check($input))
//     SWIG_fail;
//   temp.resize(PyList_Size($input));
//   for (size_t i=0; i != PyList_Size($input); ++i) {
//     if (!ConvertFromNumpyToEigenMatrix<CLASS >(&(temp[i]), PyList_GetItem($input, i)))
//       SWIG_fail;
//   }
//   $1 = temp;
// }

// %typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY)
//     CLASS,
//     const CLASS &,
//     CLASS const &,
//     Eigen::DenseBase< CLASS >,
//     const Eigen::DenseBase< CLASS > &,
//     CLASS &
//   {
//     $1 = is_array($input);
//   }

// %typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY)
//   std::vector<CLASS >
//   {
//     $1 = PyList_Check($input) && ((PyList_Size($input) == 0) || is_array(PyList_GetItem($input, 0)));
//   }

// %typemap(in, fragment="Eigen_Fragments") const Eigen::Ref<const CLASS >& (CLASS temp)
// {
//   if (!ConvertFromNumpyToEigenMatrix<CLASS >(&temp, $input))
//     SWIG_fail;
//   Eigen::Ref<const CLASS > temp_ref(temp);
//   $1 = &temp_ref;
// }

%enddef

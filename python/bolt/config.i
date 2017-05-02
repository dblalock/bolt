

// include numpy swig stuff
%include "numpy.i"
%init %{
import_array();
%}
%include <eigen.i>
%include <np_vector.i>

// ================================================================
// stl vector typemaps
// ================================================================

// pairings taken from the bottom of numpy.i
%np_vector_typemaps(signed char       , NPY_BYTE     )
%np_vector_typemaps(unsigned char     , NPY_UBYTE    )
%np_vector_typemaps(short             , NPY_SHORT    )
%np_vector_typemaps(unsigned short    , NPY_USHORT   )
%np_vector_typemaps(int               , NPY_INT      )
%np_vector_typemaps(unsigned int      , NPY_UINT     )
%np_vector_typemaps(long              , NPY_LONG     )
%np_vector_typemaps(unsigned long     , NPY_ULONG    )
%np_vector_typemaps(long long         , NPY_LONGLONG )
%np_vector_typemaps(unsigned long long, NPY_ULONGLONG)
%np_vector_typemaps(float             , NPY_FLOAT    )
%np_vector_typemaps(double            , NPY_DOUBLE   )

// apparently these are also necessary...
%np_vector_typemaps(int16_t, NPY_SHORT)
%np_vector_typemaps(int32_t, NPY_INT)
%np_vector_typemaps(int64_t, NPY_LONGLONG)
%np_vector_typemaps(uint16_t, NPY_USHORT)
%np_vector_typemaps(uint32_t, NPY_UINT)
%np_vector_typemaps(uint64_t, NPY_ULONGLONG)

%np_vector_typemaps(length_t, NPY_LONGLONG)

// ================================================================
// eigen typemaps
// ================================================================

// ------------------------ matrices

using Eigen::Matrix;
using Eigen::Dynamic;
using Eigen::RowMajor;
using Eigen::MatrixXi;

template<class T>
using Eigen::Matrix<T, Dynamic, Dynamic> = ColMatrix<T>;
template<class T>
using Eigen::Matrix<T, Dynamic, Dynamic, RowMajor> = RowMatrix<T>;

// XXX these have to match typedefs in code exactly
// TODO just include a shared header with these in it?
typedef Matrix<double, Dynamic, Dynamic> ColMatrixXd;
typedef Matrix<double, Dynamic, Dynamic, RowMajor> RowMatrixXd;
typedef Matrix<float, Dynamic, Dynamic> ColMatrixXf;
typedef Matrix<float, Dynamic, Dynamic, RowMajor> RowMatrixXf;
typedef Matrix<long long, Dynamic, Dynamic> ColMatrixXi;
typedef Matrix<long long, Dynamic, Dynamic, RowMajor> RowMatrixXi;

%eigen_typemaps(ColMatrixXd);
%eigen_typemaps(RowMatrixXd);
%eigen_typemaps(ColMatrixXf);
%eigen_typemaps(RowMatrixXf);
%eigen_typemaps(ColMatrixXi);
%eigen_typemaps(RowMatrixXi);

%eigen_typemaps(MatrixXd);
%eigen_typemaps(VectorXd);
%eigen_typemaps(RowVectorXd);

%eigen_typemaps(MatrixXf);
%eigen_typemaps(VectorXf);
%eigen_typemaps(RowVectorXf);

%eigen_typemaps(MatrixXi);

typedef Array<double, Dynamic, Dynamic> ColArrayXXd;
typedef Array<double, Dynamic, Dynamic, RowMajor> RowArrayXXd;
%eigen_typemaps(ColArrayXXd);
%eigen_typemaps(RowArrayXXd);
%eigen_typemaps(ArrayXd);  // 1d array
%eigen_typemaps(ArrayXXd); // 2d array


%eigen_typemaps(RowVector<uint16_t>);
%eigen_typemaps(RowVector<float>);
%eigen_typemaps(RowMatrix<float>);
%eigen_typemaps(ColMatrix<float>);
%eigen_typemaps(RowMatrix<uint8_t>);
%eigen_typemaps(ColMatrix<uint8_t>);

// ================================================================
// raw c array typemaps
// ================================================================

// ================================
// in-place modification of arrays
// ================================

// ------------------------------- 1D arrays
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* inVec, int len)};

// ================================
// read-only input arrays
// ================================

// ------------------------------- 1D arrays
// float
%apply (float* IN_ARRAY1, int DIM1) {(const float* ar, int len)};
%apply (float* IN_ARRAY1, int DIM1) {(const float* buff, int n)};
%apply (float* IN_ARRAY1, int DIM1) {(const float* buff, int len)};
%apply (float* IN_ARRAY1, int DIM1) {(const float* buff, int buffLen)};
%apply (float* IN_ARRAY1, int DIM1) {(const float* x, int m)};
%apply (float* IN_ARRAY1, int DIM1) {(const float* x, int len)};
%apply (float* IN_ARRAY1, int DIM1) {(const float* x, int xLen)};
%apply (float* IN_ARRAY1, int DIM1) {(const float* q, int m)};
%apply (float* IN_ARRAY1, int DIM1) {(const float* q, int len)};
%apply (float* IN_ARRAY1, int DIM1) {(const float* q, int qLen)};
%apply (float* IN_ARRAY1, int DIM1) {(const float* query, int qLen)};
%apply (float* IN_ARRAY1, int DIM1) {(const float* v, int len)};
%apply (float* IN_ARRAY1, int DIM1) {(const float* v, int inLen)};
%apply (float* IN_ARRAY1, int DIM1) {(const float* v1, int m)};
%apply (float* IN_ARRAY1, int DIM1) {(const float* v1, int len1)};
%apply (float* IN_ARRAY1, int DIM1) {(const float* v2, int m)};
%apply (float* IN_ARRAY1, int DIM1) {(const float* v2, int n)};
%apply (float* IN_ARRAY1, int DIM1) {(const float* v2, int len2)};
%apply (float* IN_ARRAY1, int DIM1) {(const float* seq, int seqLen)};
// double
%apply (double* IN_ARRAY1, int DIM1) {(const double* ar, int len)};
%apply (double* IN_ARRAY1, int DIM1) {(const double* buff, int n)};
%apply (double* IN_ARRAY1, int DIM1) {(const double* buff, int len)};
%apply (double* IN_ARRAY1, int DIM1) {(const double* buff, int buffLen)};
%apply (double* IN_ARRAY1, int DIM1) {(const double* x, int m)};
%apply (double* IN_ARRAY1, int DIM1) {(const double* x, int len)};
%apply (double* IN_ARRAY1, int DIM1) {(const double* x, int xLen)};
%apply (double* IN_ARRAY1, int DIM1) {(const double* q, int m)};
%apply (double* IN_ARRAY1, int DIM1) {(const double* q, int len)};
%apply (double* IN_ARRAY1, int DIM1) {(const double* q, int qLen)};
%apply (double* IN_ARRAY1, int DIM1) {(const double* query, int qLen)};
%apply (double* IN_ARRAY1, int DIM1) {(const double* v, int len)};
%apply (double* IN_ARRAY1, int DIM1) {(const double* v, int inLen)};
%apply (double* IN_ARRAY1, int DIM1) {(const double* v1, int m)};
%apply (double* IN_ARRAY1, int DIM1) {(const double* v1, int len1)};
%apply (double* IN_ARRAY1, int DIM1) {(const double* v2, int m)};
%apply (double* IN_ARRAY1, int DIM1) {(const double* v2, int n)};
%apply (double* IN_ARRAY1, int DIM1) {(const double* v2, int len2)};
%apply (double* IN_ARRAY1, int DIM1) {(const double* seq, int seqLen)};

// ------------------------------- 2D arrays
// float
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(const float* A, int m, int n)};
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(const float* X, int m, int n)};
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(const float* X, int d, int n)};
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(const float* X, int n, int d)};
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float* A, int m, int n)};
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float* X, int m, int n)};
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float* X, int d, int n)};
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float* X, int n, int d)};
// double
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(const double* A, int m, int n)};
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(const double* X, int m, int n)};
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(const double* X, int d, int n)};
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(const double* X, int n, int d)};
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* A, int m, int n)};
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* X, int m, int n)};
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* X, int d, int n)};
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* X, int n, int d)};

// ================================
// returned arrays
// ================================

// ------------------------------- 1D arrays
%apply (int* ARGOUT_ARRAY1, int DIM1) {(int* outVec, int len)};
%apply (int* ARGOUT_ARRAY1, int DIM1) {(int* outVec, int outLen)};
%apply (double* ARGOUT_ARRAY1, int DIM1) {(double* outVec, int len)};
%apply (double* ARGOUT_ARRAY1, int DIM1) {(double* outVec, int outLen)};

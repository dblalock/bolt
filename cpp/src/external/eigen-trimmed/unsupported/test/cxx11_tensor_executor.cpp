// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2018 Eugene Zhulenev <ezhulenev@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_USE_THREADS

#include "main.h"

#include <Eigen/CXX11/Tensor>

using Eigen::Tensor;
using Eigen::RowMajor;
using Eigen::ColMajor;

// A set of tests to verify that different TensorExecutor strategies yields the
// same results for all the ops, supporting tiled evaluation.

template <int NumDims>
static array<Index, NumDims> RandomDims(int min_dim = 1, int max_dim = 20) {
  array<Index, NumDims> dims;
  for (int i = 0; i < NumDims; ++i) {
    dims[i] = internal::random<int>(min_dim, max_dim);
  }
  return dims;
}

template <typename T, int NumDims, typename Device, bool Vectorizable,
          bool Tileable, int Layout>
static void test_execute_unary_expr(Device d)
{
  static constexpr int Options = 0 | Layout;

  // Pick a large enough tensor size to bypass small tensor block evaluation
  // optimization.
  auto dims = RandomDims<NumDims>(50 / NumDims, 100 / NumDims);

  Tensor<T, NumDims, Options, Index> src(dims);
  Tensor<T, NumDims, Options, Index> dst(dims);

  src.setRandom();
  const auto expr = src.square();

  using Assign = TensorAssignOp<decltype(dst), const decltype(expr)>;
  using Executor =
      internal::TensorExecutor<const Assign, Device, Vectorizable, Tileable>;

  Executor::run(Assign(dst, expr), d);

  for (Index i = 0; i < dst.dimensions().TotalSize(); ++i) {
    T square = src.coeff(i) * src.coeff(i);
    VERIFY_IS_EQUAL(square, dst.coeff(i));
  }
}

template <typename T, int NumDims, typename Device, bool Vectorizable,
          bool Tileable, int Layout>
static void test_execute_binary_expr(Device d)
{
  static constexpr int Options = 0 | Layout;

  // Pick a large enough tensor size to bypass small tensor block evaluation
  // optimization.
  auto dims = RandomDims<NumDims>(50 / NumDims, 100 / NumDims);

  Tensor<T, NumDims, Options, Index> lhs(dims);
  Tensor<T, NumDims, Options, Index> rhs(dims);
  Tensor<T, NumDims, Options, Index> dst(dims);

  lhs.setRandom();
  rhs.setRandom();

  const auto expr = lhs + rhs;

  using Assign = TensorAssignOp<decltype(dst), const decltype(expr)>;
  using Executor =
      internal::TensorExecutor<const Assign, Device, Vectorizable, Tileable>;

  Executor::run(Assign(dst, expr), d);

  for (Index i = 0; i < dst.dimensions().TotalSize(); ++i) {
    T sum = lhs.coeff(i) + rhs.coeff(i);
    VERIFY_IS_EQUAL(sum, dst.coeff(i));
  }
}

template <typename T, int NumDims, typename Device, bool Vectorizable,
          bool Tileable, int Layout>
static void test_execute_broadcasting(Device d)
{
  static constexpr int Options = 0 | Layout;

  auto dims = RandomDims<NumDims>(1, 10);
  Tensor<T, NumDims, Options, Index> src(dims);
  src.setRandom();

  const auto broadcasts = RandomDims<NumDims>(1, 7);
  const auto expr = src.broadcast(broadcasts);

  // We assume that broadcasting on a default device is tested and correct, so
  // we can rely on it to verify correctness of tensor executor and tiling.
  Tensor<T, NumDims, Options, Index> golden;
  golden = expr;

  // Now do the broadcasting using configured tensor executor.
  Tensor<T, NumDims, Options, Index> dst(golden.dimensions());

  using Assign = TensorAssignOp<decltype(dst), const decltype(expr)>;
  using Executor =
      internal::TensorExecutor<const Assign, Device, Vectorizable, Tileable>;

  Executor::run(Assign(dst, expr), d);

  for (Index i = 0; i < dst.dimensions().TotalSize(); ++i) {
    VERIFY_IS_EQUAL(dst.coeff(i), golden.coeff(i));
  }
}

template <typename T, int NumDims, typename Device, bool Vectorizable,
          bool Tileable, int Layout>
static void test_execute_chipping_rvalue(Device d)
{
  auto dims = RandomDims<NumDims>(1, 10);
  Tensor<T, NumDims, Layout, Index> src(dims);
  src.setRandom();

#define TEST_CHIPPING(CHIP_DIM)                                           \
  if (NumDims > (CHIP_DIM)) {                                             \
    const auto offset = internal::random<Index>(0, dims[(CHIP_DIM)] - 1); \
    const auto expr = src.template chip<(CHIP_DIM)>(offset);              \
                                                                          \
    Tensor<T, NumDims - 1, Layout, Index> golden;                         \
    golden = expr;                                                        \
                                                                          \
    Tensor<T, NumDims - 1, Layout, Index> dst(golden.dimensions());       \
                                                                          \
    using Assign = TensorAssignOp<decltype(dst), const decltype(expr)>;   \
    using Executor = internal::TensorExecutor<const Assign, Device,       \
                                              Vectorizable, Tileable>;    \
                                                                          \
    Executor::run(Assign(dst, expr), d);                                  \
                                                                          \
    for (Index i = 0; i < dst.dimensions().TotalSize(); ++i) {            \
      VERIFY_IS_EQUAL(dst.coeff(i), golden.coeff(i));                     \
    }                                                                     \
  }

  TEST_CHIPPING(0)
  TEST_CHIPPING(1)
  TEST_CHIPPING(2)
  TEST_CHIPPING(3)
  TEST_CHIPPING(4)
  TEST_CHIPPING(5)

#undef TEST_CHIPPING
}

template <typename T, int NumDims, typename Device, bool Vectorizable,
    bool Tileable, int Layout>
static void test_execute_chipping_lvalue(Device d)
{
  auto dims = RandomDims<NumDims>(1, 10);

#define TEST_CHIPPING(CHIP_DIM)                                             \
  if (NumDims > (CHIP_DIM)) {                                               \
    /* Generate random data that we'll assign to the chipped tensor dim. */ \
    array<Index, NumDims - 1> src_dims;                                     \
    for (int i = 0; i < NumDims - 1; ++i) {                                 \
      int dim = i < (CHIP_DIM) ? i : i + 1;                                 \
      src_dims[i] = dims[dim];                                              \
    }                                                                       \
                                                                            \
    Tensor<T, NumDims - 1, Layout, Index> src(src_dims);                    \
    src.setRandom();                                                        \
                                                                            \
    const auto offset = internal::random<Index>(0, dims[(CHIP_DIM)] - 1);   \
                                                                            \
    /* Generate random data to fill non-chipped dimensions*/                \
    Tensor<T, NumDims, Layout, Index> random(dims);                         \
    random.setRandom();                                                     \
                                                                            \
    Tensor<T, NumDims, Layout, Index> golden(dims);                         \
    golden = random;                                                        \
    golden.template chip<(CHIP_DIM)>(offset) = src;                         \
                                                                            \
    Tensor<T, NumDims, Layout, Index> dst(dims);                            \
    dst = random;                                                           \
    auto expr = dst.template chip<(CHIP_DIM)>(offset);                      \
                                                                            \
    using Assign = TensorAssignOp<decltype(expr), const decltype(src)>;     \
    using Executor = internal::TensorExecutor<const Assign, Device,         \
                                              Vectorizable, Tileable>;      \
                                                                            \
    Executor::run(Assign(expr, src), d);                                    \
                                                                            \
    for (Index i = 0; i < dst.dimensions().TotalSize(); ++i) {              \
      VERIFY_IS_EQUAL(dst.coeff(i), golden.coeff(i));                       \
    }                                                                       \
  }

  TEST_CHIPPING(0)
  TEST_CHIPPING(1)
  TEST_CHIPPING(2)
  TEST_CHIPPING(3)
  TEST_CHIPPING(4)
  TEST_CHIPPING(5)

#undef TEST_CHIPPING
}

template <typename T, int NumDims, typename Device, bool Vectorizable,
          bool Tileable, int Layout>
static void test_execute_shuffle_rvalue(Device d)
{
  static constexpr int Options = 0 | Layout;

  auto dims = RandomDims<NumDims>(1, 10);
  Tensor<T, NumDims, Options, Index> src(dims);
  src.setRandom();

  // Create a random dimension re-ordering/shuffle.
  std::vector<Index> shuffle;
  for (int i = 0; i < NumDims; ++i) shuffle.push_back(i);
  std::shuffle(shuffle.begin(), shuffle.end(), std::mt19937());

  const auto expr = src.shuffle(shuffle);

  // We assume that shuffling on a default device is tested and correct, so
  // we can rely on it to verify correctness of tensor executor and tiling.
  Tensor<T, NumDims, Options, Index> golden;
  golden = expr;

  // Now do the shuffling using configured tensor executor.
  Tensor<T, NumDims, Options, Index> dst(golden.dimensions());

  using Assign = TensorAssignOp<decltype(dst), const decltype(expr)>;
  using Executor =
      internal::TensorExecutor<const Assign, Device, Vectorizable, Tileable>;

  Executor::run(Assign(dst, expr), d);

  for (Index i = 0; i < dst.dimensions().TotalSize(); ++i) {
    VERIFY_IS_EQUAL(dst.coeff(i), golden.coeff(i));
  }
}

template <typename T, int NumDims, typename Device, bool Vectorizable,
          bool Tileable, int Layout>
static void test_execute_shuffle_lvalue(Device d)
{
  static constexpr int Options = 0 | Layout;

  auto dims = RandomDims<NumDims>(5, 10);
  Tensor<T, NumDims, Options, Index> src(dims);
  src.setRandom();

  // Create a random dimension re-ordering/shuffle.
  std::vector<Index> shuffle;
  for (int i = 0; i < NumDims; ++i) shuffle.push_back(i);
  std::shuffle(shuffle.begin(), shuffle.end(), std::mt19937());

  array<Index, NumDims> shuffled_dims;
  for (int i = 0; i < NumDims; ++i) shuffled_dims[shuffle[i]] = dims[i];

  // We assume that shuffling on a default device is tested and correct, so
  // we can rely on it to verify correctness of tensor executor and tiling.
  Tensor<T, NumDims, Options, Index> golden(shuffled_dims);
  golden.shuffle(shuffle) = src;

  // Now do the shuffling using configured tensor executor.
  Tensor<T, NumDims, Options, Index> dst(shuffled_dims);

  auto expr = dst.shuffle(shuffle);

  using Assign = TensorAssignOp<decltype(expr), const decltype(src)>;
  using Executor =
      internal::TensorExecutor<const Assign, Device, Vectorizable, Tileable>;

  Executor::run(Assign(expr, src), d);

  for (Index i = 0; i < dst.dimensions().TotalSize(); ++i) {
    VERIFY_IS_EQUAL(dst.coeff(i), golden.coeff(i));
  }
}

template <typename T, int NumDims, typename Device, bool Vectorizable,
          bool Tileable, int Layout>
static void test_execute_reduction(Device d)
{
  static_assert(NumDims >= 2, "NumDims must be greater or equal than 2");

  static constexpr int ReducedDims = NumDims - 2;
  static constexpr int Options = 0 | Layout;

  auto dims = RandomDims<NumDims>(5, 10);
  Tensor<T, NumDims, Options, Index> src(dims);
  src.setRandom();

  // Pick two random and unique reduction dimensions.
  int reduction0 = internal::random<int>(0, NumDims - 1);
  int reduction1 = internal::random<int>(0, NumDims - 1);
  while (reduction0 == reduction1) {
    reduction1 = internal::random<int>(0, NumDims - 1);
  }

  DSizes<Index, 2> reduction_axis;
  reduction_axis[0] = reduction0;
  reduction_axis[1] = reduction1;

  Tensor<T, ReducedDims, Options, Index> golden = src.sum(reduction_axis);

  // Now do the reduction using configured tensor executor.
  Tensor<T, ReducedDims, Options, Index> dst(golden.dimensions());

  auto expr = src.sum(reduction_axis);

  using Assign = TensorAssignOp<decltype(dst), const decltype(expr)>;
  using Executor =
      internal::TensorExecutor<const Assign, Device, Vectorizable, Tileable>;

  Executor::run(Assign(dst, expr), d);

  for (Index i = 0; i < dst.dimensions().TotalSize(); ++i) {
    VERIFY_IS_EQUAL(dst.coeff(i), golden.coeff(i));
  }
}

template <typename T, int NumDims, typename Device, bool Vectorizable,
    bool Tileable, int Layout>
static void test_execute_reshape(Device d)
{
  static_assert(NumDims >= 2, "NumDims must be greater or equal than 2");

  static constexpr int ReshapedDims = NumDims - 1;
  static constexpr int Options = 0 | Layout;

  auto dims = RandomDims<NumDims>(5, 10);
  Tensor<T, NumDims, Options, Index> src(dims);
  src.setRandom();

  // Multiple 0th dimension and then shuffle.
  std::vector<Index> shuffle;
  for (int i = 0; i < ReshapedDims; ++i) shuffle.push_back(i);
  std::shuffle(shuffle.begin(), shuffle.end(), std::mt19937());

  DSizes<Index, ReshapedDims> reshaped_dims;
  reshaped_dims[shuffle[0]] = dims[0] * dims[1];
  for (int i = 1; i < ReshapedDims; ++i) reshaped_dims[shuffle[i]] = dims[i + 1];

  Tensor<T, ReshapedDims, Options, Index> golden = src.reshape(reshaped_dims);

  // Now reshape using configured tensor executor.
  Tensor<T, ReshapedDims, Options, Index> dst(golden.dimensions());

  auto expr = src.reshape(reshaped_dims);

  using Assign = TensorAssignOp<decltype(dst), const decltype(expr)>;
  using Executor =
      internal::TensorExecutor<const Assign, Device, Vectorizable, Tileable>;

  Executor::run(Assign(dst, expr), d);

  for (Index i = 0; i < dst.dimensions().TotalSize(); ++i) {
    VERIFY_IS_EQUAL(dst.coeff(i), golden.coeff(i));
  }
}

template <typename T, int NumDims, typename Device, bool Vectorizable,
          bool Tileable, int Layout>
static void test_execute_slice_rvalue(Device d)
{
  static_assert(NumDims >= 2, "NumDims must be greater or equal than 2");
  static constexpr int Options = 0 | Layout;

  auto dims = RandomDims<NumDims>(5, 10);
  Tensor<T, NumDims, Options, Index> src(dims);
  src.setRandom();

  // Pick a random slice of src tensor.
  auto slice_start = DSizes<Index, NumDims>(RandomDims<NumDims>());
  auto slice_size = DSizes<Index, NumDims>(RandomDims<NumDims>());

  // Make sure that slice start + size do not overflow tensor dims.
  for (int i = 0; i < NumDims; ++i) {
    slice_start[i] = numext::mini(dims[i] - 1, slice_start[i]);
    slice_size[i] = numext::mini(slice_size[i], dims[i] - slice_start[i]);
  }

  Tensor<T, NumDims, Options, Index> golden =
      src.slice(slice_start, slice_size);

  // Now reshape using configured tensor executor.
  Tensor<T, NumDims, Options, Index> dst(golden.dimensions());

  auto expr = src.slice(slice_start, slice_size);

  using Assign = TensorAssignOp<decltype(dst), const decltype(expr)>;
  using Executor =
      internal::TensorExecutor<const Assign, Device, Vectorizable, Tileable>;

  Executor::run(Assign(dst, expr), d);

  for (Index i = 0; i < dst.dimensions().TotalSize(); ++i) {
    VERIFY_IS_EQUAL(dst.coeff(i), golden.coeff(i));
  }
}

template <typename T, int NumDims, typename Device, bool Vectorizable,
    bool Tileable, int Layout>
static void test_execute_slice_lvalue(Device d)
{
  static_assert(NumDims >= 2, "NumDims must be greater or equal than 2");
  static constexpr int Options = 0 | Layout;

  auto dims = RandomDims<NumDims>(5, 10);
  Tensor<T, NumDims, Options, Index> src(dims);
  src.setRandom();

  // Pick a random slice of src tensor.
  auto slice_start = DSizes<Index, NumDims>(RandomDims<NumDims>(1, 10));
  auto slice_size = DSizes<Index, NumDims>(RandomDims<NumDims>(1, 10));

  // Make sure that slice start + size do not overflow tensor dims.
  for (int i = 0; i < NumDims; ++i) {
    slice_start[i] = numext::mini(dims[i] - 1, slice_start[i]);
    slice_size[i] = numext::mini(slice_size[i], dims[i] - slice_start[i]);
  }

  Tensor<T, NumDims, Options, Index> slice(slice_size);
  slice.setRandom();

  // Assign a slice using default executor.
  Tensor<T, NumDims, Options, Index> golden = src;
  golden.slice(slice_start, slice_size) = slice;

  // And using configured execution strategy.
  Tensor<T, NumDims, Options, Index> dst = src;
  auto expr = dst.slice(slice_start, slice_size);

  using Assign = TensorAssignOp<decltype(expr), const decltype(slice)>;
  using Executor =
      internal::TensorExecutor<const Assign, Device, Vectorizable, Tileable>;

  Executor::run(Assign(expr, slice), d);

  for (Index i = 0; i < dst.dimensions().TotalSize(); ++i) {
    VERIFY_IS_EQUAL(dst.coeff(i), golden.coeff(i));
  }
}

#define CALL_SUBTEST_PART(PART) \
  CALL_SUBTEST_##PART

#define CALL_SUBTEST_COMBINATIONS(PART, NAME, T, NUM_DIMS)                                                \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, DefaultDevice,    false, false, ColMajor>(default_device))); \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, DefaultDevice,    false, true,  ColMajor>(default_device))); \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, DefaultDevice,    true,  false, ColMajor>(default_device))); \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, DefaultDevice,    true,  true,  ColMajor>(default_device))); \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, DefaultDevice,    false, false, RowMajor>(default_device))); \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, DefaultDevice,    false, true,  RowMajor>(default_device))); \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, DefaultDevice,    true,  false, RowMajor>(default_device))); \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, DefaultDevice,    true,  true,  RowMajor>(default_device))); \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, ThreadPoolDevice, false, false, ColMajor>(tp_device)));      \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, ThreadPoolDevice, false, true,  ColMajor>(tp_device)));      \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, ThreadPoolDevice, true,  false, ColMajor>(tp_device)));      \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, ThreadPoolDevice, true,  true,  ColMajor>(tp_device)));      \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, ThreadPoolDevice, false, false, RowMajor>(tp_device)));      \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, ThreadPoolDevice, false, true,  RowMajor>(tp_device)));      \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, ThreadPoolDevice, true,  false, RowMajor>(tp_device)));      \
  CALL_SUBTEST_PART(PART)((NAME<T, NUM_DIMS, ThreadPoolDevice, true,  true,  RowMajor>(tp_device)))

EIGEN_DECLARE_TEST(cxx11_tensor_executor) {
  Eigen::DefaultDevice default_device;

  const auto num_threads = internal::random<int>(1, 24);
  Eigen::ThreadPool tp(num_threads);
  Eigen::ThreadPoolDevice tp_device(&tp, num_threads);

  CALL_SUBTEST_COMBINATIONS(1, test_execute_unary_expr, float, 3);
  CALL_SUBTEST_COMBINATIONS(1, test_execute_unary_expr, float, 4);
  CALL_SUBTEST_COMBINATIONS(1, test_execute_unary_expr, float, 5);

  CALL_SUBTEST_COMBINATIONS(2, test_execute_binary_expr, float, 3);
  CALL_SUBTEST_COMBINATIONS(2, test_execute_binary_expr, float, 4);
  CALL_SUBTEST_COMBINATIONS(2, test_execute_binary_expr, float, 5);

  CALL_SUBTEST_COMBINATIONS(3, test_execute_broadcasting, float, 3);
  CALL_SUBTEST_COMBINATIONS(3, test_execute_broadcasting, float, 4);
  CALL_SUBTEST_COMBINATIONS(3, test_execute_broadcasting, float, 5);

  CALL_SUBTEST_COMBINATIONS(4, test_execute_chipping_rvalue, float, 3);
  CALL_SUBTEST_COMBINATIONS(4, test_execute_chipping_rvalue, float, 4);
  CALL_SUBTEST_COMBINATIONS(4, test_execute_chipping_rvalue, float, 5);

  CALL_SUBTEST_COMBINATIONS(5, test_execute_chipping_lvalue, float, 3);
  CALL_SUBTEST_COMBINATIONS(5, test_execute_chipping_lvalue, float, 4);
  CALL_SUBTEST_COMBINATIONS(5, test_execute_chipping_lvalue, float, 5);

  CALL_SUBTEST_COMBINATIONS(6, test_execute_shuffle_rvalue, float, 3);
  CALL_SUBTEST_COMBINATIONS(6, test_execute_shuffle_rvalue, float, 4);
  CALL_SUBTEST_COMBINATIONS(6, test_execute_shuffle_rvalue, float, 5);

  CALL_SUBTEST_COMBINATIONS(7, test_execute_shuffle_lvalue, float, 3);
  CALL_SUBTEST_COMBINATIONS(7, test_execute_shuffle_lvalue, float, 4);
  CALL_SUBTEST_COMBINATIONS(7, test_execute_shuffle_lvalue, float, 5);

  CALL_SUBTEST_COMBINATIONS(8, test_execute_reduction, float, 2);
  CALL_SUBTEST_COMBINATIONS(8, test_execute_reduction, float, 3);
  CALL_SUBTEST_COMBINATIONS(8, test_execute_reduction, float, 4);
  CALL_SUBTEST_COMBINATIONS(8, test_execute_reduction, float, 5);

  CALL_SUBTEST_COMBINATIONS(9, test_execute_reshape, float, 2);
  CALL_SUBTEST_COMBINATIONS(9, test_execute_reshape, float, 3);
  CALL_SUBTEST_COMBINATIONS(9, test_execute_reshape, float, 4);
  CALL_SUBTEST_COMBINATIONS(9, test_execute_reshape, float, 5);

  CALL_SUBTEST_COMBINATIONS(10, test_execute_slice_rvalue, float, 2);
  CALL_SUBTEST_COMBINATIONS(10, test_execute_slice_rvalue, float, 3);
  CALL_SUBTEST_COMBINATIONS(10, test_execute_slice_rvalue, float, 4);
  CALL_SUBTEST_COMBINATIONS(10, test_execute_slice_rvalue, float, 5);

  CALL_SUBTEST_COMBINATIONS(11, test_execute_slice_lvalue, float, 2);
  CALL_SUBTEST_COMBINATIONS(11, test_execute_slice_lvalue, float, 3);
  CALL_SUBTEST_COMBINATIONS(11, test_execute_slice_lvalue, float, 4);
  CALL_SUBTEST_COMBINATIONS(11, test_execute_slice_lvalue, float, 5);

  // Force CMake to split this test.
  // EIGEN_SUFFIXES;1;2;3;4;5;6;7;8;9;10;11
}

#undef CALL_SUBTEST_COMBINATIONS

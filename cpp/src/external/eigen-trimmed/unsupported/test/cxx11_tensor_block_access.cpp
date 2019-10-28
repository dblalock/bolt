// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2018 Andy Davis <andydavis@google.com>
// Copyright (C) 2018 Eugene Zhulenev <ezhulenev@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <algorithm>
#include <set>

#include <Eigen/CXX11/Tensor>

using Eigen::Tensor;
using Eigen::Index;
using Eigen::RowMajor;
using Eigen::ColMajor;


template<typename T>
static const T& choose(int layout, const T& col, const T& row) {
  return layout == ColMajor ? col : row;
}

static internal::TensorBlockShapeType RandomShape() {
  return internal::random<bool>()
             ? internal::kUniformAllDims
             : internal::kSkewedInnerDims;
}

template <int NumDims>
static Index RandomTargetSize(const DSizes<Index, NumDims>& dims) {
  return internal::random<Index>(1, dims.TotalSize());
}

template <int NumDims>
static DSizes<Index, NumDims> RandomDims() {
  array<Index, NumDims> dims;
  for (int i = 0; i < NumDims; ++i) {
    dims[i] = internal::random<int>(1, 20);
  }
  return DSizes<Index, NumDims>(dims);
}

/** Dummy data type to test TensorBlock copy ops. */
struct Data {
  Data() : value(0) {}
  explicit Data(int v) : value(v) { }
  int value;
};

bool operator==(const Data& lhs, const Data& rhs) {
  return lhs.value == rhs.value;
}

std::ostream& operator<<(std::ostream& os, const Data& d) {
  os << "Data: value=" << d.value;
  return os;
}

template <typename T>
static T* GenerateRandomData(const Index& size) {
  T* data = new T[size];
  for (int i = 0; i < size; ++i) {
    data[i] = internal::random<T>();
  }
  return data;
}

template <>
Data* GenerateRandomData(const Index& size) {
  Data* data = new Data[size];
  for (int i = 0; i < size; ++i) {
    data[i] = Data(internal::random<int>(1, 100));
  }
  return data;
}

template <int NumDims>
static void Debug(DSizes<Index, NumDims> dims) {
  for (int i = 0; i < NumDims; ++i) {
    std::cout << dims[i] << "; ";
  }
  std::cout << std::endl;
}

template <int Layout>
static void test_block_mapper_sanity()
{
  typedef internal::TensorBlockMapper<int, Index, 2, Layout> TensorBlockMapper;

  DSizes<Index, 2> tensor_dims(100, 100);

  // Test uniform blocks.
  TensorBlockMapper uniform_block_mapper(
      tensor_dims, internal::kUniformAllDims, 100);

  VERIFY_IS_EQUAL(uniform_block_mapper.total_block_count(), 100);
  VERIFY_IS_EQUAL(uniform_block_mapper.block_dims_total_size(), 100);

  // 10x10 blocks
  typename TensorBlockMapper::Block uniform_b0 = uniform_block_mapper.GetBlockForIndex(0, NULL);
  VERIFY_IS_EQUAL(uniform_b0.block_sizes().at(0), 10);
  VERIFY_IS_EQUAL(uniform_b0.block_sizes().at(1), 10);
  // Depending on a layout we stride by cols rows.
  VERIFY_IS_EQUAL(uniform_b0.block_strides().at(0), choose(Layout, 1, 10));
  VERIFY_IS_EQUAL(uniform_b0.block_strides().at(1), choose(Layout, 10, 1));
  // Tensor strides depend only on a layout and not on the block size.
  VERIFY_IS_EQUAL(uniform_b0.tensor_strides().at(0), choose(Layout, 1, 100));
  VERIFY_IS_EQUAL(uniform_b0.tensor_strides().at(1), choose(Layout, 100, 1));

  // Test skewed to inner dims blocks.
  TensorBlockMapper skewed_block_mapper(
      tensor_dims, internal::kSkewedInnerDims, 100);

  VERIFY_IS_EQUAL(skewed_block_mapper.total_block_count(), 100);
  VERIFY_IS_EQUAL(skewed_block_mapper.block_dims_total_size(), 100);

  // 1x100 (100x1) rows/cols depending on a tensor layout.
  typename TensorBlockMapper::Block skewed_b0 = skewed_block_mapper.GetBlockForIndex(0, NULL);
  VERIFY_IS_EQUAL(skewed_b0.block_sizes().at(0), choose(Layout, 100, 1));
  VERIFY_IS_EQUAL(skewed_b0.block_sizes().at(1), choose(Layout, 1, 100));
  // Depending on a layout we stride by cols rows.
  VERIFY_IS_EQUAL(skewed_b0.block_strides().at(0), choose(Layout, 1, 100));
  VERIFY_IS_EQUAL(skewed_b0.block_strides().at(1), choose(Layout, 100, 1));
  // Tensor strides depend only on a layout and not on the block size.
  VERIFY_IS_EQUAL(skewed_b0.tensor_strides().at(0), choose(Layout, 1, 100));
  VERIFY_IS_EQUAL(skewed_b0.tensor_strides().at(1), choose(Layout, 100, 1));
}

// Given a TensorBlock "visit" every element accessible though it, and a keep an
// index in the visited set. Verify that every coeff accessed only once.
template <typename T, int Layout, int NumDims>
static void UpdateCoeffSet(
    const internal::TensorBlock<T, Index, NumDims, Layout>& block,
    Index first_coeff_index, int dim_index, std::set<Index>* visited_coeffs) {
  const DSizes<Index, NumDims>& block_sizes = block.block_sizes();
  const DSizes<Index, NumDims>& tensor_strides = block.tensor_strides();

  for (int i = 0; i < block_sizes[dim_index]; ++i) {
    if (tensor_strides[dim_index] == 1) {
      typedef std::pair<std::set<Index>::iterator, bool> ReturnType;
      ReturnType inserted = visited_coeffs->insert(first_coeff_index + i);
      VERIFY_IS_EQUAL(inserted.second, true);
    } else {
      int next_dim_index = dim_index + choose(Layout, -1, 1);
      UpdateCoeffSet<T, Layout, NumDims>(block, first_coeff_index,
                                         next_dim_index, visited_coeffs);
      first_coeff_index += tensor_strides[dim_index];
    }
  }
}

template <typename T, int NumDims, int Layout>
static void test_block_mapper_maps_every_element() {
  typedef internal::TensorBlock<T, Index, NumDims, Layout> TensorBlock;
  typedef internal::TensorBlockMapper<T, Index, NumDims, Layout> TensorBlockMapper;

  DSizes<Index, NumDims> dims = RandomDims<NumDims>();

  // Keep track of elements indices available via block access.
  std::set<Index> coeff_set;

  // Try different combinations of block types and sizes.
  TensorBlockMapper block_mapper(dims, RandomShape(), RandomTargetSize(dims));

  for (int i = 0; i < block_mapper.total_block_count(); ++i) {
    TensorBlock block = block_mapper.GetBlockForIndex(i, NULL);
    UpdateCoeffSet<T, Layout, NumDims>(block, block.first_coeff_index(),
                                       choose(Layout, NumDims - 1, 0),
                                       &coeff_set);
  }

  // Verify that every coefficient in the original Tensor is accessible through
  // TensorBlock only once.
  Index total_coeffs = dims.TotalSize();
  VERIFY_IS_EQUAL(Index(coeff_set.size()), total_coeffs);
  VERIFY_IS_EQUAL(*coeff_set.begin(), 0);
  VERIFY_IS_EQUAL(*coeff_set.rbegin(), total_coeffs - 1);
}

template <typename T, int NumDims, int Layout>
static void test_slice_block_mapper_maps_every_element() {
  typedef internal::TensorBlock<T, Index, NumDims, Layout> TensorBlock;
  typedef internal::TensorSliceBlockMapper<T, Index, NumDims, Layout> TensorSliceBlockMapper;

  DSizes<Index, NumDims> tensor_dims = RandomDims<NumDims>();
  DSizes<Index, NumDims> tensor_slice_offsets = RandomDims<NumDims>();
  DSizes<Index, NumDims> tensor_slice_extents = RandomDims<NumDims>();

  // Make sure that tensor offsets + extents do not overflow.
  for (int i = 0; i < NumDims; ++i) {
    tensor_slice_offsets[i] =
        numext::mini(tensor_dims[i] - 1, tensor_slice_offsets[i]);
    tensor_slice_extents[i] = numext::mini(
        tensor_slice_extents[i], tensor_dims[i] - tensor_slice_offsets[i]);
  }

  // Keep track of elements indices available via block access.
  std::set<Index> coeff_set;

  int total_coeffs = static_cast<int>(tensor_slice_extents.TotalSize());

  // Pick a random dimension sizes for the tensor blocks.
  DSizes<Index, NumDims> block_sizes;
  for (int i = 0; i < NumDims; ++i) {
    block_sizes[i] = internal::random<Index>(1, tensor_slice_extents[i]);
  }

  TensorSliceBlockMapper block_mapper(tensor_dims, tensor_slice_offsets,
                                      tensor_slice_extents, block_sizes,
                                      DimensionList<Index, NumDims>());

  for (int i = 0; i < block_mapper.total_block_count(); ++i) {
    TensorBlock block = block_mapper.GetBlockForIndex(i, NULL);
    UpdateCoeffSet<T, Layout, NumDims>(block, block.first_coeff_index(),
                                       choose(Layout, NumDims - 1, 0),
                                       &coeff_set);
  }

  VERIFY_IS_EQUAL(Index(coeff_set.size()), total_coeffs);
}

template <typename T, int NumDims, int Layout>
static void test_block_io_copy_data_from_source_to_target() {
  typedef internal::TensorBlock<T, Index, NumDims, Layout> TensorBlock;
  typedef internal::TensorBlockMapper<T, Index, NumDims, Layout>
      TensorBlockMapper;

  typedef internal::TensorBlockReader<T, Index, NumDims, Layout>
      TensorBlockReader;
  typedef internal::TensorBlockWriter<T, Index, NumDims, Layout>
      TensorBlockWriter;

  DSizes<Index, NumDims> input_tensor_dims = RandomDims<NumDims>();
  const Index input_tensor_size = input_tensor_dims.TotalSize();

  T* input_data = GenerateRandomData<T>(input_tensor_size);
  T* output_data = new T[input_tensor_size];

  TensorBlockMapper block_mapper(input_tensor_dims, RandomShape(),
                                 RandomTargetSize(input_tensor_dims));
  T* block_data = new T[block_mapper.block_dims_total_size()];

  for (int i = 0; i < block_mapper.total_block_count(); ++i) {
    TensorBlock block = block_mapper.GetBlockForIndex(i, block_data);
    TensorBlockReader::Run(&block, input_data);
    TensorBlockWriter::Run(block, output_data);
  }

  for (int i = 0; i < input_tensor_size; ++i) {
    VERIFY_IS_EQUAL(input_data[i], output_data[i]);
  }

  delete[] input_data;
  delete[] output_data;
  delete[] block_data;
}

template <int Layout, int NumDims>
static Index GetInputIndex(Index output_index,
                         const array<Index, NumDims>& output_to_input_dim_map,
                         const array<Index, NumDims>& input_strides,
                         const array<Index, NumDims>& output_strides) {
  int input_index = 0;
  if (Layout == ColMajor) {
    for (int i = NumDims - 1; i > 0; --i) {
      const Index idx = output_index / output_strides[i];
      input_index += idx * input_strides[output_to_input_dim_map[i]];
      output_index -= idx * output_strides[i];
    }
    return input_index +
           output_index * input_strides[output_to_input_dim_map[0]];
  } else {
    for (int i = 0; i < NumDims - 1; ++i) {
      const Index idx = output_index / output_strides[i];
      input_index += idx * input_strides[output_to_input_dim_map[i]];
      output_index -= idx * output_strides[i];
    }
    return input_index +
           output_index * input_strides[output_to_input_dim_map[NumDims - 1]];
  }
}

template <int Layout, int NumDims>
static array<Index, NumDims> ComputeStrides(
    const array<Index, NumDims>& sizes) {
  array<Index, NumDims> strides;
  if (Layout == ColMajor) {
    strides[0] = 1;
    for (int i = 1; i < NumDims; ++i) {
      strides[i] = strides[i - 1] * sizes[i - 1];
    }
  } else {
    strides[NumDims - 1] = 1;
    for (int i = NumDims - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * sizes[i + 1];
    }
  }
  return strides;
}

template <typename T, int NumDims, int Layout>
static void test_block_io_copy_using_reordered_dimensions() {
  typedef internal::TensorBlock<T, Index, NumDims, Layout> TensorBlock;
  typedef internal::TensorBlockMapper<T, Index, NumDims, Layout>
      TensorBlockMapper;

  typedef internal::TensorBlockReader<T, Index, NumDims, Layout>
      TensorBlockReader;
  typedef internal::TensorBlockWriter<T, Index, NumDims, Layout>
      TensorBlockWriter;

  DSizes<Index, NumDims> input_tensor_dims = RandomDims<NumDims>();
  const Index input_tensor_size = input_tensor_dims.TotalSize();

  // Create a random input tensor.
  T* input_data = GenerateRandomData<T>(input_tensor_size);

  // Create a random dimension re-ordering/shuffle.
  std::vector<Index> shuffle;
  for (int i = 0; i < NumDims; ++i) shuffle.push_back(i);
  std::random_shuffle(shuffle.begin(), shuffle.end());

  DSizes<Index, NumDims> output_tensor_dims;
  array<Index, NumDims> input_to_output_dim_map;
  array<Index, NumDims> output_to_input_dim_map;
  for (Index i = 0; i < NumDims; ++i) {
    output_tensor_dims[shuffle[i]] = input_tensor_dims[i];
    input_to_output_dim_map[i] = shuffle[i];
    output_to_input_dim_map[shuffle[i]] = i;
  }

  // Random block shape and size.
  TensorBlockMapper block_mapper(output_tensor_dims, RandomShape(),
                                 RandomTargetSize(input_tensor_dims));

  T* block_data = new T[block_mapper.block_dims_total_size()];
  T* output_data = new T[input_tensor_size];

  array<Index, NumDims> input_tensor_strides =
      ComputeStrides<Layout, NumDims>(input_tensor_dims);
  array<Index, NumDims> output_tensor_strides =
      ComputeStrides<Layout, NumDims>(output_tensor_dims);

  for (Index i = 0; i < block_mapper.total_block_count(); ++i) {
    TensorBlock block = block_mapper.GetBlockForIndex(i, block_data);
    const Index first_coeff_index = GetInputIndex<Layout, NumDims>(
        block.first_coeff_index(), output_to_input_dim_map,
        input_tensor_strides, output_tensor_strides);
    TensorBlockReader::Run(&block, first_coeff_index, input_to_output_dim_map,
                           input_tensor_strides, input_data);
    TensorBlockWriter::Run(block, first_coeff_index, input_to_output_dim_map,
                           input_tensor_strides, output_data);
  }

  for (int i = 0; i < input_tensor_size; ++i) {
    VERIFY_IS_EQUAL(input_data[i], output_data[i]);
  }

  delete[] input_data;
  delete[] block_data;
  delete[] output_data;
}

// This is the special case for reading data with reordering, when dimensions
// before/after reordering are the same. Squeezing reads along inner dimensions
// in this case is illegal, because we reorder innermost dimension.
template <int Layout>
static void test_block_io_copy_using_reordered_dimensions_do_not_squeeze()
{
  typedef internal::TensorBlock<float, Index, 3, Layout> TensorBlock;
  typedef internal::TensorBlockReader<float, Index, 3, Layout>
      TensorBlockReader;

  DSizes<Index, 3> tensor_dims;
  tensor_dims[0] = 7;
  tensor_dims[1] = 9;
  tensor_dims[2] = 7;

  DSizes<Index, 3> block_dims = tensor_dims;

  DSizes<Index, 3> tensor_to_block_dim_map;
  tensor_to_block_dim_map[0] = 2;
  tensor_to_block_dim_map[1] = 1;
  tensor_to_block_dim_map[2] = 0;

  DSizes<Index, 3> tensor_strides(ComputeStrides<Layout, 3>(tensor_dims));
  DSizes<Index, 3> block_strides(ComputeStrides<Layout, 3>(block_dims));

  const Index tensor_size = tensor_dims.TotalSize();
  float* tensor_data = GenerateRandomData<float>(tensor_size);
  float* block_data = new float[tensor_size];

  TensorBlock block(0, block_dims, block_strides, tensor_strides, block_data);
  TensorBlockReader::Run(&block,
                         0,
                         tensor_to_block_dim_map,
                         tensor_strides,
                         tensor_data);

  TensorMap<Tensor<float, 3, Layout> > block_tensor(block_data, block_dims);
  TensorMap<Tensor<float, 3, Layout> > tensor_tensor(tensor_data, tensor_dims);

  for (Index d0 = 0; d0 < tensor_dims[0]; ++d0) {
    for (Index d1 = 0; d1 < tensor_dims[1]; ++d1) {
      for (Index d2 = 0; d2 < tensor_dims[2]; ++d2) {
        float block_value = block_tensor(d2, d1, d0);
        float tensor_value = tensor_tensor(d0, d1, d2);
        VERIFY_IS_EQUAL(block_value, tensor_value);
      }
    }
  }

  delete[] block_data;
  delete[] tensor_data;
}

// This is the special case for reading data with reordering, when dimensions
// before/after reordering are the same. Squeezing reads in this case is allowed
// because we reorder outer dimensions.
template <int Layout>
static void test_block_io_copy_using_reordered_dimensions_squeeze()
{
  typedef internal::TensorBlock<float, Index, 4, Layout> TensorBlock;
  typedef internal::TensorBlockReader<float, Index, 4, Layout>
      TensorBlockReader;

  DSizes<Index, 4> tensor_dims;
  tensor_dims[0] = 7;
  tensor_dims[1] = 5;
  tensor_dims[2] = 9;
  tensor_dims[3] = 9;

  DSizes<Index, 4> block_dims = tensor_dims;

  DSizes<Index, 4> tensor_to_block_dim_map;
  tensor_to_block_dim_map[0] = 0;
  tensor_to_block_dim_map[1] = 1;
  tensor_to_block_dim_map[2] = 3;
  tensor_to_block_dim_map[3] = 2;

  DSizes<Index, 4> tensor_strides(ComputeStrides<Layout, 4>(tensor_dims));
  DSizes<Index, 4> block_strides(ComputeStrides<Layout, 4>(block_dims));

  const Index tensor_size = tensor_dims.TotalSize();
  float* tensor_data = GenerateRandomData<float>(tensor_size);
  float* block_data = new float[tensor_size];

  TensorBlock block(0, block_dims, block_strides, tensor_strides, block_data);
  TensorBlockReader::Run(&block,
                         0,
                         tensor_to_block_dim_map,
                         tensor_strides,
                         tensor_data);

  TensorMap<Tensor<float, 4, Layout> > block_tensor(block_data, block_dims);
  TensorMap<Tensor<float, 4, Layout> > tensor_tensor(tensor_data, tensor_dims);

  for (Index d0 = 0; d0 < tensor_dims[0]; ++d0) {
    for (Index d1 = 0; d1 < tensor_dims[1]; ++d1) {
      for (Index d2 = 0; d2 < tensor_dims[2]; ++d2) {
        for (Index d3 = 0; d3 < tensor_dims[3]; ++d3) {
          float block_value = block_tensor(d0, d1, d3, d2);
          float tensor_value = tensor_tensor(d0, d1, d2, d3);
          VERIFY_IS_EQUAL(block_value, tensor_value);
        }
      }
    }
  }

  delete[] block_data;
  delete[] tensor_data;
}

template<typename Scalar, typename StorageIndex, int Dim>
class EqualityChecker
{
    const Scalar* input_data;
    const DSizes<StorageIndex, Dim> &input_dims, &input_strides, &output_dims, &output_strides;
    void check_recursive(const Scalar* input, const Scalar* output, int depth=0) const
    {
        if(depth==Dim)
        {
            VERIFY_IS_EQUAL(*input, *output);
            return;
        }

        for(int i=0; i<output_dims[depth]; ++i)
        {
            check_recursive(input + i % input_dims[depth] * input_strides[depth], output + i*output_strides[depth], depth+1);
        }
    }
public:
    EqualityChecker(const Scalar* input_data_,
            const DSizes<StorageIndex, Dim> &input_dims_, const DSizes<StorageIndex, Dim> &input_strides_,
            const DSizes<StorageIndex, Dim> &output_dims_, const DSizes<StorageIndex, Dim> &output_strides_)
        : input_data(input_data_)
        , input_dims(input_dims_), input_strides(input_strides_)
        , output_dims(output_dims_), output_strides(output_strides_)
        {}

    void operator()(const Scalar* output_data) const
    {
        check_recursive(input_data, output_data);
    }
};

template <int Layout>
static void test_block_io_zero_stride()
{
  typedef internal::TensorBlock<float, Index, 5, Layout> TensorBlock;
  typedef internal::TensorBlockReader<float, Index, 5, Layout>
      TensorBlockReader;
  typedef internal::TensorBlockWriter<float, Index, 5, Layout>
      TensorBlockWriter;

  DSizes<Index, 5> rnd_dims = RandomDims<5>();

  DSizes<Index, 5> input_tensor_dims = rnd_dims;
  input_tensor_dims[0] = 1;
  input_tensor_dims[2] = 1;
  input_tensor_dims[4] = 1;
  const Index input_tensor_size = input_tensor_dims.TotalSize();
  float* input_data = GenerateRandomData<float>(input_tensor_size);

  DSizes<Index, 5> output_tensor_dims = rnd_dims;

  DSizes<Index, 5> input_tensor_strides(
      ComputeStrides<Layout, 5>(input_tensor_dims));
  DSizes<Index, 5> output_tensor_strides(
      ComputeStrides<Layout, 5>(output_tensor_dims));

  DSizes<Index, 5> input_tensor_strides_with_zeros(input_tensor_strides);
  input_tensor_strides_with_zeros[0] = 0;
  input_tensor_strides_with_zeros[2] = 0;
  input_tensor_strides_with_zeros[4] = 0;

  // Verify that data was correctly read/written from/into the block.
  const EqualityChecker<float, Index, 5> verify_is_equal(input_data, input_tensor_dims, input_tensor_strides, output_tensor_dims, output_tensor_strides);

  {
    float* output_data = new float[output_tensor_dims.TotalSize()];
    TensorBlock read_block(0, output_tensor_dims, output_tensor_strides,
                           input_tensor_strides_with_zeros, output_data);
    TensorBlockReader::Run(&read_block, input_data);
    verify_is_equal(output_data);
    delete[] output_data;
  }

  {
    float* output_data = new float[output_tensor_dims.TotalSize()];
    TensorBlock write_block(0, output_tensor_dims,
                            input_tensor_strides_with_zeros,
                            output_tensor_strides, input_data);
    TensorBlockWriter::Run(write_block, output_data);
    verify_is_equal(output_data);
    delete[] output_data;
  }

  delete[] input_data;
}

template <int Layout>
static void test_block_io_squeeze_ones() {
  typedef internal::TensorBlock<float, Index, 5, Layout> TensorBlock;
  typedef internal::TensorBlockReader<float, Index, 5, Layout>
      TensorBlockReader;
  typedef internal::TensorBlockWriter<float, Index, 5, Layout>
      TensorBlockWriter;

  // Total size > 1.
  {
    DSizes<Index, 5> block_sizes(1, 2, 1, 2, 1);
    const Index total_size = block_sizes.TotalSize();

    // Create a random input tensor.
    float* input_data = GenerateRandomData<float>(total_size);
    DSizes<Index, 5> strides(ComputeStrides<Layout, 5>(block_sizes));

    {
      float* output_data = new float[block_sizes.TotalSize()];
      TensorBlock read_block(0, block_sizes, strides, strides, output_data);
      TensorBlockReader::Run(&read_block, input_data);
      for (int i = 0; i < total_size; ++i) {
        VERIFY_IS_EQUAL(output_data[i], input_data[i]);
      }
      delete[] output_data;
    }

    {
      float* output_data = new float[block_sizes.TotalSize()];
      TensorBlock write_block(0, block_sizes, strides, strides, input_data);
      TensorBlockWriter::Run(write_block, output_data);
      for (int i = 0; i < total_size; ++i) {
        VERIFY_IS_EQUAL(output_data[i], input_data[i]);
      }
      delete[] output_data;
    }
  }

  // Total size == 1.
  {
    DSizes<Index, 5> block_sizes(1, 1, 1, 1, 1);
    const Index total_size = block_sizes.TotalSize();

    // Create a random input tensor.
    float* input_data = GenerateRandomData<float>(total_size);
    DSizes<Index, 5> strides(ComputeStrides<Layout, 5>(block_sizes));

    {
      float* output_data = new float[block_sizes.TotalSize()];
      TensorBlock read_block(0, block_sizes, strides, strides, output_data);
      TensorBlockReader::Run(&read_block, input_data);
      for (int i = 0; i < total_size; ++i) {
        VERIFY_IS_EQUAL(output_data[i], input_data[i]);
      }
      delete[] output_data;
    }

    {
      float* output_data = new float[block_sizes.TotalSize()];
      TensorBlock write_block(0, block_sizes, strides, strides, input_data);
      TensorBlockWriter::Run(write_block, output_data);
      for (int i = 0; i < total_size; ++i) {
        VERIFY_IS_EQUAL(output_data[i], input_data[i]);
      }
      delete[] output_data;
    }
  }
}

template <typename T, int NumDims, int Layout>
static void test_block_cwise_unary_io_basic() {
  typedef internal::scalar_square_op<T> UnaryFunctor;
  typedef internal::TensorBlockCwiseUnaryIO<UnaryFunctor, Index, T, NumDims,
                                            Layout>
      TensorBlockCwiseUnaryIO;

  DSizes<Index, NumDims> block_sizes = RandomDims<NumDims>();
  DSizes<Index, NumDims> strides(ComputeStrides<Layout, NumDims>(block_sizes));

  const Index total_size = block_sizes.TotalSize();

  // Create a random input tensors.
  T* input_data = GenerateRandomData<T>(total_size);

  T* output_data = new T[total_size];
  UnaryFunctor functor;
  TensorBlockCwiseUnaryIO::Run(functor, block_sizes, strides, output_data,
                               strides, input_data);
  for (int i = 0; i < total_size; ++i) {
    VERIFY_IS_EQUAL(output_data[i], functor(input_data[i]));
  }

  delete[] input_data;
  delete[] output_data;
}

template <int Layout>
static void test_block_cwise_unary_io_squeeze_ones() {
  typedef internal::scalar_square_op<float> UnaryFunctor;
  typedef internal::TensorBlockCwiseUnaryIO<UnaryFunctor, Index, float, 5,
                                            Layout>
      TensorBlockCwiseUnaryIO;

  DSizes<Index, 5> block_sizes(1, 2, 1, 3, 1);
  DSizes<Index, 5> strides(ComputeStrides<Layout, 5>(block_sizes));

  const Index total_size = block_sizes.TotalSize();

  // Create a random input tensors.
  float* input_data = GenerateRandomData<float>(total_size);

  float* output_data = new float[total_size];
  UnaryFunctor functor;
  TensorBlockCwiseUnaryIO::Run(functor, block_sizes, strides, output_data,
                               strides, input_data);
  for (int i = 0; i < total_size; ++i) {
    VERIFY_IS_EQUAL(output_data[i], functor(input_data[i]));
  }

  delete[] input_data;
  delete[] output_data;
}

template <int Layout>
static void test_block_cwise_unary_io_zero_strides() {
  typedef internal::scalar_square_op<float> UnaryFunctor;
  typedef internal::TensorBlockCwiseUnaryIO<UnaryFunctor, Index, float, 5,
                                            Layout>
      TensorBlockCwiseUnaryIO;

  DSizes<Index, 5> rnd_dims = RandomDims<5>();

  DSizes<Index, 5> input_sizes = rnd_dims;
  input_sizes[0] = 1;
  input_sizes[2] = 1;
  input_sizes[4] = 1;

  DSizes<Index, 5> input_strides(ComputeStrides<Layout, 5>(input_sizes));
  input_strides[0] = 0;
  input_strides[2] = 0;
  input_strides[4] = 0;

  // Generate random data.
  float* input_data = GenerateRandomData<float>(input_sizes.TotalSize());

  DSizes<Index, 5> output_sizes = rnd_dims;
  DSizes<Index, 5> output_strides(ComputeStrides<Layout, 5>(output_sizes));

  const Index output_total_size = output_sizes.TotalSize();
  float* output_data = new float[output_total_size];

  UnaryFunctor functor;
  TensorBlockCwiseUnaryIO::Run(functor, output_sizes, output_strides,
                               output_data, input_strides, input_data);
  for (int i = 0; i < rnd_dims[0]; ++i) {
    for (int j = 0; j < rnd_dims[1]; ++j) {
      for (int k = 0; k < rnd_dims[2]; ++k) {
        for (int l = 0; l < rnd_dims[3]; ++l) {
          for (int m = 0; m < rnd_dims[4]; ++m) {
            Index output_index = i * output_strides[0] + j * output_strides[1] +
                                 k * output_strides[2] + l * output_strides[3] +
                                 m * output_strides[4];
            Index input_index = i * input_strides[0] + j * input_strides[1] +
                                k * input_strides[2] + l * input_strides[3] +
                                m * input_strides[4];
            VERIFY_IS_EQUAL(output_data[output_index],
                            functor(input_data[input_index]));
          }
        }
      }
    }
  }

  delete[] input_data;
  delete[] output_data;
}

template <typename T, int NumDims, int Layout>
static void test_block_cwise_binary_io_basic() {
  typedef internal::scalar_sum_op<T> BinaryFunctor;
  typedef internal::TensorBlockCwiseBinaryIO<BinaryFunctor, Index, T, NumDims,
                                             Layout>
      TensorBlockCwiseBinaryIO;

  DSizes<Index, NumDims> block_sizes = RandomDims<NumDims>();
  DSizes<Index, NumDims> strides(ComputeStrides<Layout, NumDims>(block_sizes));

  const Index total_size = block_sizes.TotalSize();

  // Create a random input tensors.
  T* left_data = GenerateRandomData<T>(total_size);
  T* right_data = GenerateRandomData<T>(total_size);

  T* output_data = new T[total_size];
  BinaryFunctor functor;
  TensorBlockCwiseBinaryIO::Run(functor, block_sizes, strides, output_data,
                                strides, left_data, strides, right_data);
  for (int i = 0; i < total_size; ++i) {
    VERIFY_IS_EQUAL(output_data[i], functor(left_data[i], right_data[i]));
  }

  delete[] left_data;
  delete[] right_data;
  delete[] output_data;
}

template <int Layout>
static void test_block_cwise_binary_io_squeeze_ones() {
  typedef internal::scalar_sum_op<float> BinaryFunctor;
  typedef internal::TensorBlockCwiseBinaryIO<BinaryFunctor, Index, float, 5,
                                             Layout>
      TensorBlockCwiseBinaryIO;

  DSizes<Index, 5> block_sizes(1, 2, 1, 3, 1);
  DSizes<Index, 5> strides(ComputeStrides<Layout, 5>(block_sizes));

  const Index total_size = block_sizes.TotalSize();

  // Create a random input tensors.
  float* left_data = GenerateRandomData<float>(total_size);
  float* right_data = GenerateRandomData<float>(total_size);

  float* output_data = new float[total_size];
  BinaryFunctor functor;
  TensorBlockCwiseBinaryIO::Run(functor, block_sizes, strides, output_data,
                                strides, left_data, strides, right_data);
  for (int i = 0; i < total_size; ++i) {
    VERIFY_IS_EQUAL(output_data[i], functor(left_data[i], right_data[i]));
  }

  delete[] left_data;
  delete[] right_data;
  delete[] output_data;
}

template <int Layout>
static void test_block_cwise_binary_io_zero_strides() {
  typedef internal::scalar_sum_op<float> BinaryFunctor;
  typedef internal::TensorBlockCwiseBinaryIO<BinaryFunctor, Index, float, 5,
                                             Layout>
      TensorBlockCwiseBinaryIO;

  DSizes<Index, 5> rnd_dims = RandomDims<5>();

  DSizes<Index, 5> left_sizes = rnd_dims;
  left_sizes[0] = 1;
  left_sizes[2] = 1;
  left_sizes[4] = 1;

  DSizes<Index, 5> left_strides(ComputeStrides<Layout, 5>(left_sizes));
  left_strides[0] = 0;
  left_strides[2] = 0;
  left_strides[4] = 0;

  DSizes<Index, 5> right_sizes = rnd_dims;
  right_sizes[1] = 1;
  right_sizes[3] = 1;

  DSizes<Index, 5> right_strides(ComputeStrides<Layout, 5>(right_sizes));
  right_strides[1] = 0;
  right_strides[3] = 0;

  // Generate random data.
  float* left_data = GenerateRandomData<float>(left_sizes.TotalSize());
  float* right_data = GenerateRandomData<float>(right_sizes.TotalSize());

  DSizes<Index, 5> output_sizes = rnd_dims;
  DSizes<Index, 5> output_strides(ComputeStrides<Layout, 5>(output_sizes));

  const Index output_total_size = output_sizes.TotalSize();
  float* output_data = new float[output_total_size];

  BinaryFunctor functor;
  TensorBlockCwiseBinaryIO::Run(functor, output_sizes, output_strides,
                                output_data, left_strides, left_data,
                                right_strides, right_data);
  for (int i = 0; i < rnd_dims[0]; ++i) {
    for (int j = 0; j < rnd_dims[1]; ++j) {
      for (int k = 0; k < rnd_dims[2]; ++k) {
        for (int l = 0; l < rnd_dims[3]; ++l) {
          for (int m = 0; m < rnd_dims[4]; ++m) {
            Index output_index = i * output_strides[0] + j * output_strides[1] +
                                 k * output_strides[2] + l * output_strides[3] +
                                 m * output_strides[4];
            Index left_index = i * left_strides[0] + j * left_strides[1] +
                               k * left_strides[2] + l * left_strides[3] +
                               m * left_strides[4];
            Index right_index = i * right_strides[0] + j * right_strides[1] +
                                k * right_strides[2] + l * right_strides[3] +
                                m * right_strides[4];
            VERIFY_IS_EQUAL(
                output_data[output_index],
                functor(left_data[left_index], right_data[right_index]));
          }
        }
      }
    }
  }

  delete[] left_data;
  delete[] right_data;
  delete[] output_data;
}

template <int Layout>
static void test_uniform_block_shape()
{
  typedef internal::TensorBlock<int, Index, 5, Layout> TensorBlock;
  typedef internal::TensorBlockMapper<int, Index, 5, Layout> TensorBlockMapper;

  {
    // Test shape 'UniformAllDims' with uniform 'max_coeff count'.
    DSizes<Index, 5> dims(11, 5, 6, 17, 7);
    const Index max_coeff_count = 5 * 5 * 5 * 5 * 5;
    TensorBlockMapper block_mapper(dims, internal::kUniformAllDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    for (int i = 0; i < 5; ++i) {
      VERIFY_IS_EQUAL(5, block.block_sizes()[i]);
    }
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  }

  // Test shape 'UniformAllDims' with larger 'max_coeff count' which spills
  // partially into first inner-most dimension.
  if (Layout == ColMajor) {
    DSizes<Index, 5> dims(11, 5, 6, 17, 7);
    const Index max_coeff_count = 7 * 5 * 5 * 5 * 5;
    TensorBlockMapper block_mapper(dims, internal::kUniformAllDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    VERIFY_IS_EQUAL(7, block.block_sizes()[0]);
    for (int i = 1; i < 5; ++i) {
      VERIFY_IS_EQUAL(5, block.block_sizes()[i]);
    }
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  } else {
    DSizes<Index, 5> dims(11, 5, 6, 17, 7);
    const Index max_coeff_count = 5 * 5 * 5 * 5 * 6;
    TensorBlockMapper block_mapper(dims, internal::kUniformAllDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    VERIFY_IS_EQUAL(6, block.block_sizes()[4]);
    for (int i = 3; i >= 0; --i) {
      VERIFY_IS_EQUAL(5, block.block_sizes()[i]);
    }
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  }

  // Test shape 'UniformAllDims' with larger 'max_coeff count' which spills
  // fully into first inner-most dimension.
  if (Layout == ColMajor) {
    DSizes<Index, 5> dims(11, 5, 6, 17, 7);
    const Index max_coeff_count = 11 * 5 * 5 * 5 * 5;
    TensorBlockMapper block_mapper(dims, internal::kUniformAllDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    VERIFY_IS_EQUAL(11, block.block_sizes()[0]);
    for (int i = 1; i < 5; ++i) {
      VERIFY_IS_EQUAL(5, block.block_sizes()[i]);
    }
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  } else {
    DSizes<Index, 5> dims(11, 5, 6, 17, 7);
    const Index max_coeff_count = 5 * 5 * 5 * 5 * 7;
    TensorBlockMapper block_mapper(dims, internal::kUniformAllDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    VERIFY_IS_EQUAL(7, block.block_sizes()[4]);
    for (int i = 3; i >= 0; --i) {
      VERIFY_IS_EQUAL(5, block.block_sizes()[i]);
    }
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  }

  // Test shape 'UniformAllDims' with larger 'max_coeff count' which spills
  // fully into first few inner-most dimensions.
  if (Layout == ColMajor) {
    DSizes<Index, 5> dims(7, 5, 6, 17, 7);
    const Index max_coeff_count = 7 * 5 * 6 * 7 * 5;
    TensorBlockMapper block_mapper(dims, internal::kUniformAllDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    VERIFY_IS_EQUAL(7, block.block_sizes()[0]);
    VERIFY_IS_EQUAL(5, block.block_sizes()[1]);
    VERIFY_IS_EQUAL(6, block.block_sizes()[2]);
    VERIFY_IS_EQUAL(7, block.block_sizes()[3]);
    VERIFY_IS_EQUAL(5, block.block_sizes()[4]);
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  } else {
    DSizes<Index, 5> dims(7, 5, 6, 9, 7);
    const Index max_coeff_count = 5 * 5 * 5 * 6 * 7;
    TensorBlockMapper block_mapper(dims, internal::kUniformAllDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    VERIFY_IS_EQUAL(7, block.block_sizes()[4]);
    VERIFY_IS_EQUAL(6, block.block_sizes()[3]);
    VERIFY_IS_EQUAL(5, block.block_sizes()[2]);
    VERIFY_IS_EQUAL(5, block.block_sizes()[1]);
    VERIFY_IS_EQUAL(5, block.block_sizes()[0]);
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  }

  // Test shape 'UniformAllDims' with full allocation to all dims.
  if (Layout == ColMajor) {
    DSizes<Index, 5> dims(7, 5, 6, 17, 7);
    const Index max_coeff_count = 7 * 5 * 6 * 17 * 7;
    TensorBlockMapper block_mapper(dims, internal::kUniformAllDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    VERIFY_IS_EQUAL(7, block.block_sizes()[0]);
    VERIFY_IS_EQUAL(5, block.block_sizes()[1]);
    VERIFY_IS_EQUAL(6, block.block_sizes()[2]);
    VERIFY_IS_EQUAL(17, block.block_sizes()[3]);
    VERIFY_IS_EQUAL(7, block.block_sizes()[4]);
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  } else {
    DSizes<Index, 5> dims(7, 5, 6, 9, 7);
    const Index max_coeff_count = 7 * 5 * 6 * 9 * 7;
    TensorBlockMapper block_mapper(dims, internal::kUniformAllDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    VERIFY_IS_EQUAL(7, block.block_sizes()[4]);
    VERIFY_IS_EQUAL(9, block.block_sizes()[3]);
    VERIFY_IS_EQUAL(6, block.block_sizes()[2]);
    VERIFY_IS_EQUAL(5, block.block_sizes()[1]);
    VERIFY_IS_EQUAL(7, block.block_sizes()[0]);
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  }
}

template <int Layout>
static void test_skewed_inner_dim_block_shape()
{
  typedef internal::TensorBlock<int, Index, 5, Layout> TensorBlock;
  typedef internal::TensorBlockMapper<int, Index, 5, Layout> TensorBlockMapper;

  // Test shape 'SkewedInnerDims' with partial allocation to inner-most dim.
  if (Layout == ColMajor) {
    DSizes<Index, 5> dims(11, 5, 6, 17, 7);
    const Index max_coeff_count = 10 * 1 * 1 * 1 * 1;
    TensorBlockMapper block_mapper(dims, internal::kSkewedInnerDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    VERIFY_IS_EQUAL(10, block.block_sizes()[0]);
    for (int i = 1; i < 5; ++i) {
      VERIFY_IS_EQUAL(1, block.block_sizes()[i]);
    }
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  } else {
    DSizes<Index, 5> dims(11, 5, 6, 17, 7);
    const Index max_coeff_count = 1 * 1 * 1 * 1 * 6;
    TensorBlockMapper block_mapper(dims, internal::kSkewedInnerDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    VERIFY_IS_EQUAL(6, block.block_sizes()[4]);
    for (int i = 3; i >= 0; --i) {
      VERIFY_IS_EQUAL(1, block.block_sizes()[i]);
    }
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  }

  // Test shape 'SkewedInnerDims' with full allocation to inner-most dim.
  if (Layout == ColMajor) {
    DSizes<Index, 5> dims(11, 5, 6, 17, 7);
    const Index max_coeff_count = 11 * 1 * 1 * 1 * 1;
    TensorBlockMapper block_mapper(dims, internal::kSkewedInnerDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    VERIFY_IS_EQUAL(11, block.block_sizes()[0]);
    for (int i = 1; i < 5; ++i) {
      VERIFY_IS_EQUAL(1, block.block_sizes()[i]);
    }
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  } else {
    DSizes<Index, 5> dims(11, 5, 6, 17, 7);
    const Index max_coeff_count = 1 * 1 * 1 * 1 * 7;
    TensorBlockMapper block_mapper(dims, internal::kSkewedInnerDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    VERIFY_IS_EQUAL(7, block.block_sizes()[4]);
    for (int i = 3; i >= 0; --i) {
      VERIFY_IS_EQUAL(1, block.block_sizes()[i]);
    }
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  }

  // Test shape 'SkewedInnerDims' with full allocation to inner-most dim,
  // and partial allocation to second inner-dim.
  if (Layout == ColMajor) {
    DSizes<Index, 5> dims(11, 5, 6, 17, 7);
    const Index max_coeff_count = 11 * 3 * 1 * 1 * 1;
    TensorBlockMapper block_mapper(dims, internal::kSkewedInnerDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    VERIFY_IS_EQUAL(11, block.block_sizes()[0]);
    VERIFY_IS_EQUAL(3, block.block_sizes()[1]);
    for (int i = 2; i < 5; ++i) {
      VERIFY_IS_EQUAL(1, block.block_sizes()[i]);
    }
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  } else {
    DSizes<Index, 5> dims(11, 5, 6, 17, 7);
    const Index max_coeff_count = 1 * 1 * 1 * 15 * 7;
    TensorBlockMapper block_mapper(dims, internal::kSkewedInnerDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    VERIFY_IS_EQUAL(7, block.block_sizes()[4]);
    VERIFY_IS_EQUAL(15, block.block_sizes()[3]);
    for (int i = 2; i >= 0; --i) {
      VERIFY_IS_EQUAL(1, block.block_sizes()[i]);
    }
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  }

  // Test shape 'SkewedInnerDims' with full allocation to inner-most dim,
  // and partial allocation to third inner-dim.
  if (Layout == ColMajor) {
    DSizes<Index, 5> dims(11, 5, 6, 17, 7);
    const Index max_coeff_count = 11 * 5 * 5 * 1 * 1;
    TensorBlockMapper block_mapper(dims, internal::kSkewedInnerDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    VERIFY_IS_EQUAL(11, block.block_sizes()[0]);
    VERIFY_IS_EQUAL(5, block.block_sizes()[1]);
    VERIFY_IS_EQUAL(5, block.block_sizes()[2]);
    for (int i = 3; i < 5; ++i) {
      VERIFY_IS_EQUAL(1, block.block_sizes()[i]);
    }
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  } else {
    DSizes<Index, 5> dims(11, 5, 6, 17, 7);
    const Index max_coeff_count = 1 * 1 * 5 * 17 * 7;
    TensorBlockMapper block_mapper(dims, internal::kSkewedInnerDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    VERIFY_IS_EQUAL(7, block.block_sizes()[4]);
    VERIFY_IS_EQUAL(17, block.block_sizes()[3]);
    VERIFY_IS_EQUAL(5, block.block_sizes()[2]);
    for (int i = 1; i >= 0; --i) {
      VERIFY_IS_EQUAL(1, block.block_sizes()[i]);
    }
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  }

  // Test shape 'SkewedInnerDims' with full allocation to all dims.
  if (Layout == ColMajor) {
    DSizes<Index, 5> dims(11, 5, 6, 17, 7);
    const Index max_coeff_count = 11 * 5 * 6 * 17 * 7;
    TensorBlockMapper block_mapper(dims, internal::kSkewedInnerDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    VERIFY_IS_EQUAL(11, block.block_sizes()[0]);
    VERIFY_IS_EQUAL(5, block.block_sizes()[1]);
    VERIFY_IS_EQUAL(6, block.block_sizes()[2]);
    VERIFY_IS_EQUAL(17, block.block_sizes()[3]);
    VERIFY_IS_EQUAL(7, block.block_sizes()[4]);
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  } else {
    DSizes<Index, 5> dims(11, 5, 6, 17, 7);
    const Index max_coeff_count = 11 * 5 * 6 * 17 * 7;
    TensorBlockMapper block_mapper(dims, internal::kSkewedInnerDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    VERIFY_IS_EQUAL(7, block.block_sizes()[4]);
    VERIFY_IS_EQUAL(17, block.block_sizes()[3]);
    VERIFY_IS_EQUAL(6, block.block_sizes()[2]);
    VERIFY_IS_EQUAL(5, block.block_sizes()[1]);
    VERIFY_IS_EQUAL(11, block.block_sizes()[0]);
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  }
}

template <int Layout>
static void test_empty_dims(const internal::TensorBlockShapeType block_shape)
{
  // Test blocking of tensors with zero dimensions:
  //  - we must not crash on asserts and divisions by zero
  //  - we must not return block with zero dimensions
  //    (recipe for overflows/underflows, divisions by zero and NaNs later)
  //  - total block count must be zero
  {
    typedef internal::TensorBlockMapper<int, Index, 1, Layout> TensorBlockMapper;
    DSizes<Index, 1> dims(0);
    for (int max_coeff_count = 0; max_coeff_count < 2; ++max_coeff_count) {
      TensorBlockMapper block_mapper(dims, block_shape, max_coeff_count);
      VERIFY_IS_EQUAL(block_mapper.total_block_count(), 0);
      VERIFY(block_mapper.block_dims_total_size() >= 1);
    }
  }

  {
    typedef internal::TensorBlockMapper<int, Index, 2, Layout> TensorBlockMapper;
    for (int dim1 = 0; dim1 < 3; ++dim1) {
      for (int dim2 = 0; dim2 < 3; ++dim2) {
        DSizes<Index, 2> dims(dim1, dim2);
        for (int max_coeff_count = 0; max_coeff_count < 2; ++max_coeff_count) {
          TensorBlockMapper block_mapper(dims, block_shape, max_coeff_count);
          if (dim1 * dim2 == 0) {
            VERIFY_IS_EQUAL(block_mapper.total_block_count(), 0);
          }
          VERIFY(block_mapper.block_dims_total_size() >= 1);
        }
      }
    }
  }
}

#define TEST_LAYOUTS(NAME) \
  CALL_SUBTEST(NAME<ColMajor>()); \
  CALL_SUBTEST(NAME<RowMajor>())

#define TEST_LAYOUTS_AND_DIMS(TYPE, NAME)    \
  CALL_SUBTEST((NAME<TYPE, 1, ColMajor>())); \
  CALL_SUBTEST((NAME<TYPE, 1, RowMajor>())); \
  CALL_SUBTEST((NAME<TYPE, 2, ColMajor>())); \
  CALL_SUBTEST((NAME<TYPE, 2, RowMajor>())); \
  CALL_SUBTEST((NAME<TYPE, 3, ColMajor>())); \
  CALL_SUBTEST((NAME<TYPE, 3, RowMajor>())); \
  CALL_SUBTEST((NAME<TYPE, 4, ColMajor>())); \
  CALL_SUBTEST((NAME<TYPE, 4, RowMajor>())); \
  CALL_SUBTEST((NAME<TYPE, 5, ColMajor>())); \
  CALL_SUBTEST((NAME<TYPE, 5, RowMajor>()))

#define TEST_LAYOUTS_WITH_ARG(NAME, ARG) \
  CALL_SUBTEST(NAME<ColMajor>(ARG)); \
  CALL_SUBTEST(NAME<RowMajor>(ARG))

EIGEN_DECLARE_TEST(cxx11_tensor_block_access) {
  TEST_LAYOUTS(test_block_mapper_sanity);
  TEST_LAYOUTS_AND_DIMS(float, test_block_mapper_maps_every_element);
  TEST_LAYOUTS_AND_DIMS(float, test_slice_block_mapper_maps_every_element);
  TEST_LAYOUTS_AND_DIMS(float, test_block_io_copy_data_from_source_to_target);
  TEST_LAYOUTS_AND_DIMS(Data, test_block_io_copy_data_from_source_to_target);
  TEST_LAYOUTS_AND_DIMS(float, test_block_io_copy_using_reordered_dimensions);
  TEST_LAYOUTS_AND_DIMS(Data, test_block_io_copy_using_reordered_dimensions);
  TEST_LAYOUTS(test_block_io_copy_using_reordered_dimensions_do_not_squeeze);
  TEST_LAYOUTS(test_block_io_copy_using_reordered_dimensions_squeeze);
  TEST_LAYOUTS(test_block_io_zero_stride);
  TEST_LAYOUTS(test_block_io_squeeze_ones);
  TEST_LAYOUTS_AND_DIMS(float, test_block_cwise_unary_io_basic);
  TEST_LAYOUTS(test_block_cwise_unary_io_squeeze_ones);
  TEST_LAYOUTS(test_block_cwise_unary_io_zero_strides);
  TEST_LAYOUTS_AND_DIMS(float, test_block_cwise_binary_io_basic);
  TEST_LAYOUTS(test_block_cwise_binary_io_squeeze_ones);
  TEST_LAYOUTS(test_block_cwise_binary_io_zero_strides);
  TEST_LAYOUTS(test_uniform_block_shape);
  TEST_LAYOUTS(test_skewed_inner_dim_block_shape);
  TEST_LAYOUTS_WITH_ARG(test_empty_dims, internal::kUniformAllDims);
  TEST_LAYOUTS_WITH_ARG(test_empty_dims, internal::kSkewedInnerDims);
}

#undef TEST_LAYOUTS
#undef TEST_LAYOUTS_WITH_ARG

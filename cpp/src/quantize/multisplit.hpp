//
//  multisplit.hpp
//  Bolt
//
//  Created by DB on 10/6/19.
//  Copyright © 2019 D Blalock. All rights reserved.
//

#ifndef multisplit_h
#define multisplit_h

#include <assert.h>
#include <stdint.h>
#include <sys/types.h>
#include <math.h>
#include "immintrin.h" // this is what defines all the simd funcs + _MM_SHUFFLE

#ifdef BLAZE
    #include "src/utils/avx_utils.hpp"
#else
    #include "avx_utils.hpp"
#endif
namespace {

void split_encode_8b_colmajor(const float* X, int64_t nrows, int ncols,
    const uint32_t* splitdims, const int8_t* splitvals,
    const float* scales, const float* offsets,
    int ncodebooks, int splits_per_codebook, uint8_t* out)
{
    static constexpr int block_rows = 32;
    const int64_t nblocks = ceil(nrows / (double)block_rows);
    assert(splits_per_codebook <= 8); // code assumes we don't overflow bytes
    assert(nrows % block_rows == 0); // TODO remove this constraint

    int split_idx = 0;
    for (int c = 0; c < ncodebooks; c++) {
        // zero out this column of the output; we do this so that we can
        // use it as working memory to store encodings-so-far as we go
        auto out_col_start = out + (c * nrows);
        for (int64_t i = 0; i < nrows; i++) {
            out_col_start[i] = 0;
        }

        for (int s = 0; s < splits_per_codebook; s++) {
            auto splitdim = splitdims[split_idx];
            auto splitval = splitvals[split_idx];
            auto splitvals_i8 = _mm256_set1_epi8(splitval);

            // load f32->u8 quantization params for this dim
            auto vscales = _mm256_set1_ps(scales[split_idx]);
            auto voffsets = _mm256_set1_ps(offsets[split_idx]);

            // iterate through this col of X
            auto x_col_start = X + (nrows * splitdim);  // X colmajor contiguous
            auto x_ptr = x_col_start;
            auto out_ptr = out_col_start;
            for (int b = 0; b < nblocks; b++) { // for each block
                // true = signed saturation; better because cmp instructions
                // exist for epi8 but not epu8
                auto x_i8 = load_4xf32_as_32xepi8_or_epu8<true>(
                    x_ptr, vscales, voffsets);
                auto codes_so_far = load_si256i(out_ptr);

                auto masks = _mm256_cmpgt_epi8(x_i8, splitvals_i8);
                // shift left by multiplying by 2, by adding to itself
                auto codes = _mm256_add_epi8(codes_so_far, codes_so_far);
                // map -1 -> 1; 0 stays the same
                auto masks_0_or_1 = _mm256_sign_epi8(masks, masks);
                // OR in new low bit by adding
                codes = _mm256_add_epi8(codes, masks_0_or_1);

                _mm256_storeu_si256((__m256i*)out_ptr, codes);
                out_ptr += block_rows;
                x_ptr += block_rows;
            }
            split_idx++;  // increment for each split in each codebook
        }
    }
}

void multisplit_encode_8b_colmajor(const float* X, int64_t nrows, int ncols,
    const uint32_t* splitdims, const int8_t* all_splitvals,
    const float* scales, const float* offsets,
    int ncodebooks, int splits_per_codebook, uint8_t* out)
{
    static const int low_bits_used_in_shuffle = 4;
    static const int group_id_nbits = low_bits_used_in_shuffle;
    // static_assert(GroupIdNumBits <= low_bits_used_in_shuffle,
    //     "Can vectorize at most 16 groups");
    static constexpr int block_rows = 32;
    static const __m256i low_4bits_mask = _mm256_set1_epi8(0x0F);
    const int64_t nblocks = ceil(nrows / (double)block_rows);
    const int vals_per_split = 1 << group_id_nbits; // usually 16
    assert(group_id_nbits <= low_bits_used_in_shuffle);
    assert(splits_per_codebook <= 8); // code assumes we don't overflow bytes
    assert(nrows % block_rows == 0); // TODO remove this constraint

    int split_idx = 0;
    for (int c = 0; c < ncodebooks; c++) {
        // zero out this column of the output; we do this so that we can
        // use it as working memory to store encodings-so-far as we go
        auto out_col_start = out + (c * nrows);
        for (int64_t i = 0; i < nrows; i++) {
            out_col_start[i] = 0;
        }

        for (int s = 0; s < splits_per_codebook; s++) {
            auto splitdim = splitdims[split_idx];
            auto splitvals_ptr = all_splitvals + (vals_per_split * split_idx);

            // assumes exactly 16B of i8 split vals for each dim we split on
            auto splitvals_lut = _mm256_broadcastsi128_si256(
                load_si128i((const __m128i*)splitvals_ptr));

            // load f32->u8 quantization params for this dim
            auto vscales = _mm256_set1_ps(scales[split_idx]);
            auto voffsets = _mm256_set1_ps(offsets[split_idx]);

            // iterate through this col of X
            // printf("splitdim: %d\n", splitdim);
            auto x_col_start = X + (nrows * splitdim);  // X colmajor contiguous
            auto x_ptr = x_col_start;
            auto out_ptr = out_col_start;
            if (s < low_bits_used_in_shuffle) {
                for (int b = 0; b < nblocks; b++) { // for each block
                    // true = signed saturation; better because cmp instructions
                    // exist for epi8 but not epu8
                    auto x_i8 = load_4xf32_as_32xepi8_or_epu8<true>(
                        x_ptr, vscales, voffsets);
                    auto codes_so_far = load_si256i(out_ptr);

                    // look up splits to use based on low bits of codes so far
                    auto vsplitvals = _mm256_shuffle_epi8(
                        splitvals_lut, codes_so_far); // codes = group_ids

                    auto masks = _mm256_cmpgt_epi8(x_i8, vsplitvals);
                    // shift left by multiplying by 2, by adding to itself
                    auto codes = _mm256_add_epi8(codes_so_far, codes_so_far);
                    // map -1 -> 1; 0 stays the same
                    auto masks_0_or_1 = _mm256_sign_epi8(masks, masks);
                    // OR in new low bit by adding
                    codes = _mm256_add_epi8(codes, masks_0_or_1);

                    _mm256_storeu_si256((__m256i*)out_ptr, codes);
                    out_ptr += block_rows;
                    x_ptr += block_rows;
                }
            } else {  // group_id is no longer in the low 4 bits
                // assert(false); // TODO rm
                int shift_amount = (s + 1) - low_bits_used_in_shuffle;
                for (int b = 0; b < nblocks; b++) { // for each block
                    auto x_i8 = load_4xf32_as_32xepi8_or_epu8<true>(
                        x_ptr, vscales, voffsets);
                    auto codes_so_far = load_si256i(out_ptr);

                    // this is the part that's different from the above loop;
                    // we could just have an if statement in the previous one,
                    // but not a great idea in the hot loop; note that the
                    // size of the words we shift doesn't matter since we
                    // zero out the upper bits regardless (which is necessary
                    // since vpshufb will return 0, instead of the value at
                    // the associated index, if the MSB of the index is 1)
                    auto group_ids =  _mm256_srli_epi16(
                        codes_so_far, shift_amount);
                    group_ids = _mm256_and_si256(group_ids, low_4bits_mask);
                    auto vsplitvals = _mm256_shuffle_epi8(
                        splitvals_lut, group_ids);

                    auto masks = _mm256_cmpgt_epi8(x_i8, vsplitvals);
                    auto codes = _mm256_add_epi8(codes_so_far, codes_so_far);
                    auto masks_0_or_1 = _mm256_sign_epi8(masks, masks);
                    codes = _mm256_add_epi8(codes, masks_0_or_1);

                    _mm256_storeu_si256((__m256i*)out_ptr, codes);
                    out_ptr += block_rows;
                    x_ptr += block_rows;
                }
            }
            split_idx++;  // increment for each split in each codebook
        }
    }
}

} // anon namespace
#endif /* multisplit_h */

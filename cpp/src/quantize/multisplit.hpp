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
    int ncodebooks, int nsplits_per_codebook, uint8_t* out)
{
    static constexpr int block_nrows = 32;
    const int64_t nblocks = ceil(nrows / (double)block_nrows);
    assert(nsplits_per_codebook <= 8); // code assumes we don't overflow bytes
    assert(nrows % block_nrows == 0); // TODO remove this constraint

    int split_idx = 0;
    for (int c = 0; c < ncodebooks; c++) {
        // zero out this column of the output; we do this so that we can
        // use it as working memory to store encodings-so-far as we go
        auto out_col_start = out + (c * nrows);
        for (int64_t i = 0; i < nrows; i++) {
            out_col_start[i] = 0;
        }

        for (int s = 0; s < nsplits_per_codebook; s++) {
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
                out_ptr += block_nrows;
                x_ptr += block_nrows;
            }
            split_idx++;  // increment for each split in each codebook
        }
    }
}

/* https://godbolt.org/z/_MPkKn
 * here's the inner loop (all 4 iters get unrolled); completely dominated
 * by data loading + scaling/offseting + converting to ints
 * vmovups      ymm4, ymmword ptr [rdx + 4*r9 - 96]
 * vmovups      ymm5, ymmword ptr [rdx + 4*r9 - 64]
 * vmovups      ymm6, ymmword ptr [rdx + 4*r9 - 32]
 * vmovups      ymm7, ymmword ptr [rdx + 4*r9]
 * vfmadd231ps  ymm4, ymm0, ymm4 # ymm4 = (ymm0 * ymm4) + ymm4
 * vfmadd231ps  ymm5, ymm0, ymm5 # ymm5 = (ymm0 * ymm5) + ymm5
 * vfmadd231ps  ymm6, ymm0, ymm6 # ymm6 = (ymm0 * ymm6) + ymm6
 * vfmadd231ps  ymm7, ymm0, ymm7 # ymm7 = (ymm0 * ymm7) + ymm7
 * vcvtps2dq    ymm4, ymm4
 * vcvtps2dq    ymm5, ymm5
 * vpackssdw    ymm4, ymm4, ymm5
 * vcvtps2dq    ymm5, ymm6
 * vcvtps2dq    ymm6, ymm7
 * vpackssdw    ymm5, ymm5, ymm6
 * vpacksswb    ymm4, ymm4, ymm5
 * vpermd       ymm4, ymm0, ymm4
 * vpcmpgtb     ymm4, ymm4, ymm2
 * vpsignb      ymm4, ymm4, ymm4
 * vpaddb       ymm3, ymm3, ymm3
 * vpor         ymm5, ymm4, ymm3
 *
 * I think if we were willing to be really sneaky, we could set the scales
 * such that the exponents were all 0s (or some other const?) so that
 * we can just directly pack the floats without converting them to ints;
 * probably just a matter of preprocessing the splitvals the same way
 */
void split_encode_4b_colmajor(const float* X, int64_t nrows, int ncols,
    const uint32_t* splitdims, const int8_t* splitvals,
    const float* scales, const float* offsets,
    int ncodebooks, uint8_t* out)
{
    static constexpr int block_nrows = 32;
    static constexpr int nsplits_per_codebook = 4;
    const int64_t nblocks = ceil(nrows / (double)block_nrows);
    assert(nsplits_per_codebook <= 8); // code assumes we don't overflow bytes
    assert(nrows % block_nrows == 0); // TODO remove this constraint

    size_t x_col_stride = nrows;
    size_t out_col_stride = nrows;
    const float* x_ptrs[nsplits_per_codebook];
    __m256i current_vsplitvals[nsplits_per_codebook];
    __m256 current_vscales[nsplits_per_codebook];
    __m256 current_voffsets[nsplits_per_codebook];

    int split_idx = 0;
    for (int c = 0; c < ncodebooks; c++) {
        // compute input and output column starts
        auto out_ptr = out + (out_col_stride * c);
        for (int s = 0; s < nsplits_per_codebook; s++) {
            auto splitdim = splitdims[split_idx + s];
            auto splitval = splitvals[split_idx + s];
            x_ptrs[s] = X + (x_col_stride * splitdim);
            current_vsplitvals[s] = _mm256_set1_epi8(splitval);
            current_vscales[s] = _mm256_set1_ps(scales[split_idx + s]);
            current_voffsets[s] = _mm256_set1_ps(offsets[split_idx + s]);
        }
        split_idx += nsplits_per_codebook;

        for (int b = 0; b < nblocks; b++) { // for each block
            __m256i codes = _mm256_setzero_si256();
            #pragma unroll
            for (int s = 0; s < nsplits_per_codebook; s++) {
                auto vscales = current_vscales[s];
                auto voffsets = current_voffsets[s];
                auto vsplitvals = current_vsplitvals[s];

                auto x_ptr = x_ptrs[s];
                x_ptrs[s] += block_nrows;

                // true = signed saturation; better because cmp instructions
                // exist for epi8 but not epu8
                auto x_i8 = load_4xf32_as_32xepi8_or_epu8<true>(
                    x_ptr, vscales, voffsets);

                auto masks = _mm256_cmpgt_epi8(x_i8, vsplitvals);
                // map -1 -> 1; 0 stays the same
                auto masks_0_or_1 = _mm256_sign_epi8(masks, masks);

                if (s > 0) {
                    // shift left by multiplying by 2, by adding to itself
                    codes = _mm256_add_epi8(codes, codes);
                }

                // OR in new low bit
                codes = _mm256_or_si256(codes, masks_0_or_1);
            }
            _mm256_storeu_si256((__m256i*)out_ptr, codes);
            out_ptr += block_nrows;
        }
    }
}

/* https://godbolt.org/z/Hd2Ynm
 * inner loop (unrolled, with interleaved stuff for other iters removed; also
 * 2 ymm register loads total (though no stores), so doesn't quite all fit)):
 *
 * vcmpltps        ymm3, ymm1, ymmword ptr [rbx + 4*r9 - 96]
 * vcmpltps        ymm4, ymm1, ymmword ptr [rbx + 4*r9 - 64]
 * vcmpltps        ymm5, ymm1, ymmword ptr [rbx + 4*r9 - 32]
 * vcmpltps        ymm6, ymm1, ymmword ptr [rbx + 4*r9]
 * vpackssdw       ymm3, ymm3, ymm4
 * vpackssdw       ymm4, ymm5, ymm6
 * vpacksswb       ymm3, ymm3, ymm4
 * vpermd          ymm3, ymm0, ymm3
 * vpsignb         ymm3, ymm3, ymm3
 * vpaddb          ymm3, ymm3, ymm3
 * vpor            ymm3, ymm4, ymm3
 */
void split_encode_4b_colmajor_alt(const float* X, int64_t nrows, int ncols,
    const uint32_t* splitdims, const float* splitvals, int ncodebooks,
    uint8_t* out)
{
    static constexpr bool DeferPerm = true;
    static constexpr int block_nrows = 32;
    // static constexpr int stripe_rows = 8;
    // static constexpr int nstripes = block_nrows / stripe_rows;
    static constexpr int nsplits_per_codebook = 4;
    const int64_t nblocks = ceil(nrows / (double)block_nrows);
    assert(nsplits_per_codebook <= 8); // code assumes we don't overflow bytes
    assert(nrows % block_nrows == 0); // TODO remove this constraint

    size_t x_col_stride = nrows;
    size_t out_col_stride = nrows;
    const float* x_ptrs[nsplits_per_codebook];
    __m256 current_vsplitvals[nsplits_per_codebook];
    // __m256 current_vscales[nsplits_per_codebook];
    // __m256 current_voffsets[nsplits_per_codebook];

    int split_idx = 0;
    for (int c = 0; c < ncodebooks; c++) {
        // compute input and output column starts
        auto out_ptr = out + (out_col_stride * c);
        for (int s = 0; s < nsplits_per_codebook; s++) {
            auto splitdim = splitdims[split_idx + s];
            x_ptrs[s] = X + (x_col_stride * splitdim);
            current_vsplitvals[s] = _mm256_set1_ps(splitvals[split_idx + s]);
        }
        split_idx += nsplits_per_codebook;

        for (int b = 0; b < nblocks; b++) { // for each block
            __m256i codes = _mm256_setzero_si256();
            #pragma unroll
            for (int s = 0; s < nsplits_per_codebook; s++) {
                auto vsplitvals = current_vsplitvals[s];

                auto x_ptr = x_ptrs[s];
                auto x0 = _mm256_loadu_ps(x_ptr + 0);
                auto x1 = _mm256_loadu_ps(x_ptr + 8);
                auto x2 = _mm256_loadu_ps(x_ptr + 16);
                auto x3 = _mm256_loadu_ps(x_ptr + 24);
                x_ptrs[s] += block_nrows;

                // see https://stackoverflow.com/questions/16988199/
                // how-to-choose-avx-compare-predicate-variants for how to
                // choose predicate constant
                auto a = (__m256i)_mm256_cmp_ps(x0, vsplitvals, _CMP_GT_OQ);
                auto b = (__m256i)_mm256_cmp_ps(x1, vsplitvals, _CMP_GT_OQ);
                auto c = (__m256i)_mm256_cmp_ps(x2, vsplitvals, _CMP_GT_OQ);
                auto d = (__m256i)_mm256_cmp_ps(x3, vsplitvals, _CMP_GT_OQ);

                __m256i ab = _mm256_packs_epi32(a, b);
                __m256i cd = _mm256_packs_epi32(c, d);
                __m256i masks = _mm256_packs_epi16(ab, cd);
                if (!DeferPerm) {
                    masks = _mm256_permutevar8x32_epi32(
                        masks, _mm256_setr_epi32(0,4, 1,5, 2,6, 3,7));
                }

                // map -1 -> 1; 0 stays the same
                auto masks_0_or_1 = _mm256_sign_epi8(masks, masks);

                if (s > 0) {
                    // shift left by multiplying by 2, by adding to itself
                    codes = _mm256_add_epi8(codes, codes);
                }

                // OR in new low bit
                codes = _mm256_or_si256(codes, masks_0_or_1);
            }
            if (DeferPerm) {
                codes = _mm256_permutevar8x32_epi32(
                        codes, _mm256_setr_epi32(0,4, 1,5, 2,6, 3,7));
            }
            _mm256_storeu_si256((__m256i*)out_ptr, codes);
            out_ptr += block_nrows;
        }
    }
}

void multisplit_encode_8b_colmajor(const float* X, int64_t nrows, int ncols,
    const uint32_t* splitdims, const int8_t* all_splitvals,
    const float* scales, const float* offsets,
    int ncodebooks, int nsplits_per_codebook, uint8_t* out)
{
    static const int low_bits_used_in_shuffle = 4;
    static const int group_id_nbits = low_bits_used_in_shuffle;
    // static_assert(GroupIdNumBits <= low_bits_used_in_shuffle,
    //     "Can vectorize at most 16 groups");
    static constexpr int block_nrows = 32;
    static const __m256i low_4bits_mask = _mm256_set1_epi8(0x0F);
    const int64_t nblocks = ceil(nrows / (double)block_nrows);
    const int vals_per_split = 1 << group_id_nbits; // usually 16
    assert(group_id_nbits <= low_bits_used_in_shuffle);
    assert(nsplits_per_codebook <= 8); // code assumes we don't overflow bytes
    assert(nrows % block_nrows == 0); // TODO remove this constraint

    int split_idx = 0;
    for (int c = 0; c < ncodebooks; c++) {
        // zero out this column of the output; we do this so that we can
        // use it as working memory to store encodings-so-far as we go
        auto out_col_start = out + (c * nrows);
        for (int64_t i = 0; i < nrows; i++) {
            out_col_start[i] = 0;
        }

        for (int s = 0; s < nsplits_per_codebook; s++) {
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
                    out_ptr += block_nrows;
                    x_ptr += block_nrows;
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
                    out_ptr += block_nrows;
                    x_ptr += block_nrows;
                }
            }
            split_idx++;  // increment for each split in each codebook
        }
    }
}

/* https://godbolt.org/z/OEqSHD (without voffsets):
 * vmulps       ymm11, ymm8, ymmword ptr [rdx + 4*r8 - 96]
 * vmulps       ymm12, ymm8, ymmword ptr [rdx + 4*r8 - 64]
 * vmulps       ymm13, ymm8, ymmword ptr [rdx + 4*r8 - 32]
 * vmulps       ymm8, ymm8, ymmword ptr [rdx + 4*r8]
 * vpshufb      ymm10, ymm10, ymm9
 * vcvtps2dq    ymm11, ymm11
 * vcvtps2dq    ymm12, ymm12
 * vpackssdw    ymm11, ymm11, ymm12
 * vcvtps2dq    ymm12, ymm13
 * vcvtps2dq    ymm8, ymm8
 * vpackssdw    ymm8, ymm12, ymm8
 * vpacksswb    ymm8, ymm11, ymm8
 * vpermd       ymm8, ymm1, ymm8
 * vpcmpgtb     ymm8, ymm8, ymm10
 * vpsignb      ymm8, ymm8, ymm8
 * vpaddb       ymm9, ymm9, ymm9
 * vpor         ymm8, ymm9, ymm8
 */
// template<bool DeferPerm=true>
void multisplit_encode_4b_colmajor(
    const float* X, int64_t nrows, int ncols,
    const uint32_t* splitdims, const int8_t* all_splitvals,
    const float* scales, const float* offsets, int ncodebooks, uint8_t* out)
    // const float* scales, int ncodebooks, uint8_t* out)
{
    static constexpr bool DeferPerm = true;
    static constexpr int block_nrows = 32;
    static constexpr int nsplits_per_codebook = 4;
    static constexpr int vals_per_split = 1 << nsplits_per_codebook; // 16
    const int64_t nblocks = ceil(nrows / (double)block_nrows);
    assert(nrows % block_nrows == 0); // TODO remove this constraint

    size_t x_col_stride = nrows;
    size_t out_col_stride = nrows;
    size_t splitval_luts_stride = vals_per_split;
    const float* x_ptrs[nsplits_per_codebook];
    __m256i current_vsplitval_luts[nsplits_per_codebook];
    __m256 current_vscales[nsplits_per_codebook];
    __m256 current_voffsets[nsplits_per_codebook];

    int split_idx = 0;
    for (int c = 0; c < ncodebooks; c++) {
        // compute input and output column starts
        auto out_ptr = out + (out_col_stride * c);
        for (int s = 0; s < nsplits_per_codebook; s++) {
            auto splitdim = splitdims[split_idx + s];
            x_ptrs[s] = X + (x_col_stride * splitdim);
            auto splitvals_ptr = all_splitvals + (vals_per_split * split_idx);
            current_vsplitval_luts[s] = _mm256_broadcastsi128_si256(
                load_si128i((const __m128i*)splitvals_ptr));
            current_vscales[s] = _mm256_set1_ps(scales[split_idx + s]);
            current_voffsets[s] = _mm256_set1_ps(offsets[split_idx + s]);
        }
        split_idx += nsplits_per_codebook;

        for (int b = 0; b < nblocks; b++) { // for each block
            __m256i codes = _mm256_setzero_si256();
            #pragma unroll
            for (int s = 0; s < nsplits_per_codebook; s++) {
                auto vscales = current_vscales[s];
                auto voffsets = current_voffsets[s];
                // auto voffsets = _mm256_setzero_si256();
                auto vsplitvals_lut = current_vsplitval_luts[s];
                auto vsplitvals = _mm256_shuffle_epi8(
                        vsplitvals_lut, codes); // codes = group_ids

                auto x_ptr = x_ptrs[s];
                x_ptrs[s] += block_nrows;

                // true = signed saturation; better because cmp instructions
                // exist for epi8 but not epu8
                auto x_i8 = load_4xf32_as_32xepi8_or_epu8<true, !DeferPerm>(
                    // x_ptr, vscales);
                    x_ptr, vscales, voffsets);

                auto masks = _mm256_cmpgt_epi8(x_i8, vsplitvals);
                // map -1 -> 1; 0 stays the same
                auto masks_0_or_1 = _mm256_sign_epi8(masks, masks);

                if (s > 0) {
                    // shift left by multiplying by 2, by adding to itself
                    codes = _mm256_add_epi8(codes, codes);
                }

                // OR in new low bit
                codes = _mm256_or_si256(codes, masks_0_or_1);
            }
            if (DeferPerm) {
                codes = _mm256_permutevar8x32_epi32(
                    codes, _mm256_setr_epi32(0,4, 1,5, 2,6, 3,7));
            }
            _mm256_storeu_si256((__m256i*)out_ptr, codes);
            out_ptr += block_nrows;
        }
    }
}

struct Layouts {
    enum {
        ColMajorNoPack = 0,
        ColMajorPack2 = 1,
        ColMajorPack4 = 2,
        BoltNoPack = 3,
        BoltPack2 = 4,
        BoltPack4 = 5
    };
};

// static constexpr int kLayoutColMajor = 0

// version with int8 data
template<int Layout=Layouts::ColMajorNoPack>
void multisplit_encode_4b_colmajor(const int8_t* X, int64_t nrows, int ncols,
    const uint32_t* splitdims, const int8_t* all_splitvals,
    int ncodebooks, uint8_t* out)
    // const float* scales, int ncodebooks, uint8_t* out)
{
    static_assert(Layout == Layouts::ColMajorNoPack ||
                  Layout == Layouts::BoltNoPack, "Unsupported Layout");
    static constexpr int block_nrows = 32;
    static constexpr int simd_vec_sz = 32;
    static constexpr int nsplits_per_codebook = 4;
    // static constexpr int ncodebooks_per_group = 2;
    static constexpr int vals_per_split = 1 << nsplits_per_codebook; // 16
    const int64_t nblocks = ceil(nrows / (double)block_nrows);
    assert(nrows % block_nrows == 0); // TODO remove this constraint
    // assert(ncodebooks % ncodebooks_per_group == 0);
    // int ncolgroups = ncodebooks / ncodebooks_per_group;

    size_t x_col_stride = nrows;
    size_t out_col_stride = nrows;
    size_t splitval_luts_stride = vals_per_split;
    const int8_t* x_ptrs[nsplits_per_codebook];
    __m256i current_vsplitval_luts[nsplits_per_codebook];
    // const int8_t* x_ptrs[ncodebooks_per_group][nsplits_per_codebook];
    // __m256i current_vsplitval_luts[ncodebooks_per_group][nsplits_per_codebook];

    int split_idx = 0;
    // for (int g = 0; g < ncolgroups; g++) {
    //     for (int gg = 0; gg < ncodebooks_per_group; gg++) {
    //         auto c = 8 * ncodebooks_per_group + gg;  // codebook
    for (int c = 0; c < ncodebooks; c++) {
        // compute input and output column starts
        uint8_t* out_ptr;
        size_t out_block_stride;
        if (Layout == Layouts::BoltNoPack) {
            out_ptr = out + (simd_vec_sz * c);
            /// XXX this will be off by a factor of 2 with a packed layout
            out_block_stride = block_nrows * ncodebooks;
        } else {
            out_ptr = out + (out_col_stride * c);
            out_block_stride = block_nrows;
        }
        for (int s = 0; s < nsplits_per_codebook; s++) {
            auto splitdim = splitdims[split_idx + s];
            x_ptrs[s] = X + (x_col_stride * splitdim);
            auto splitvals_ptr = all_splitvals + (vals_per_split * split_idx);
            current_vsplitval_luts[s] = _mm256_broadcastsi128_si256(
                load_si128i((const __m128i*)splitvals_ptr));
        }
        split_idx += nsplits_per_codebook;

        for (int b = 0; b < nblocks; b++) { // for each block
            __m256i codes = _mm256_setzero_si256();
            #pragma unroll
            for (int s = 0; s < nsplits_per_codebook; s++) {
                auto vsplitvals_lut = current_vsplitval_luts[s];
                auto vsplitvals = _mm256_shuffle_epi8(
                        vsplitvals_lut, codes); // codes = group_ids

                auto x_i8 = load_si256i(x_ptrs[s]);
                x_ptrs[s] += block_nrows;

                auto masks = _mm256_cmpgt_epi8(x_i8, vsplitvals);
                // map -1 -> 1; 0 stays the same
                auto masks_0_or_1 = _mm256_sign_epi8(masks, masks);

                if (s > 0) {
                    // shift left by multiplying by 2, by adding to itself
                    codes = _mm256_add_epi8(codes, codes);
                }

                // OR in new low bit
                codes = _mm256_or_si256(codes, masks_0_or_1);
            }
            _mm256_storeu_si256((__m256i*)out_ptr, codes);
            out_ptr += out_block_stride;
            // if (Layout == Layouts::BoltNoPack) { // doesn't help
            //     __builtin_prefetch(out_ptr + 16 * out_block_stride);
            // }
        }
    }
    // }
}

// wrapper for int8 version that can deal with scales and offsets provided
void multisplit_encode_4b_colmajor(const int8_t* X, int64_t nrows, int ncols,
    const uint32_t* splitdims, const int8_t* all_splitvals,
    const void* shifts_unused, const void* offsets_unused,
    int ncodebooks, uint8_t* out)
{
    multisplit_encode_4b_colmajor(
        X, nrows, ncols, splitdims, all_splitvals, ncodebooks, out);
}

// version with int16 data
template<int Layout=Layouts::ColMajorNoPack>
void multisplit_encode_4b_colmajor(const int16_t* X, int64_t nrows, int ncols,
    const uint32_t* splitdims, const int8_t* all_splitvals,
    const uint8_t* shifts, const int16_t* offsets,
    int ncodebooks, uint8_t* out)
    // const float* scales, int ncodebooks, uint8_t* out)
{
    static constexpr int block_nrows = 32;
    static constexpr int simd_vec_sz = 32;
    static constexpr int nsplits_per_codebook = 4;
    static constexpr int vals_per_split = 1 << nsplits_per_codebook; // 16
    const int64_t nblocks = ceil(nrows / (double)block_nrows);
    assert(nrows % block_nrows == 0); // TODO remove this constraint

    size_t x_col_stride = nrows;
    size_t out_col_stride = nrows;
    size_t splitval_luts_stride = vals_per_split;
    const int16_t* x_ptrs[nsplits_per_codebook];
    __m256i current_vsplitval_luts[nsplits_per_codebook];
    uint8_t current_shifts[nsplits_per_codebook];
    __m256i current_voffsets[nsplits_per_codebook];

    int split_idx = 0;
    for (int c = 0; c < ncodebooks; c++) {
        // compute input and output column starts
        uint8_t* out_ptr;
        size_t out_block_stride;
        if (Layout == Layouts::BoltNoPack) {
            out_ptr = out + (simd_vec_sz * c);
            /// XXX this will be off by a factor of 2 with a packed layout
            out_block_stride = block_nrows * ncodebooks;
        } else {
            out_ptr = out + (out_col_stride * c);
            out_block_stride = block_nrows;
        }
        for (int s = 0; s < nsplits_per_codebook; s++) {
            auto splitdim = splitdims[split_idx + s];
            x_ptrs[s] = X + (x_col_stride * splitdim);
            auto splitvals_ptr = all_splitvals + (vals_per_split * split_idx);
            current_vsplitval_luts[s] = _mm256_broadcastsi128_si256(
                load_si128i((const __m128i*)splitvals_ptr));
            current_shifts[s] = shifts[split_idx + s];
            current_voffsets[s] = _mm256_set1_epi16(offsets[split_idx + s]);
        }
        split_idx += nsplits_per_codebook;

        for (int b = 0; b < nblocks; b++) { // for each block
            __m256i codes = _mm256_setzero_si256();
            #pragma unroll
            for (int s = 0; s < nsplits_per_codebook; s++) {
                auto shift = current_shifts[s];
                auto voffsets = current_voffsets[s];

                auto vsplitvals_lut = current_vsplitval_luts[s];
                auto vsplitvals = _mm256_shuffle_epi8(
                        vsplitvals_lut, codes); // codes = group_ids

                auto x_i16_0_15 = load_si256i(x_ptrs[s]);
                auto x_i16_16_31 = load_si256i(x_ptrs[s] + 16);
                x_ptrs[s] += 32;

                // offset and shift to get to i8 range
                x_i16_0_15 = _mm256_adds_epi16(x_i16_0_15, voffsets);
                x_i16_16_31 = _mm256_adds_epi16(x_i16_16_31, voffsets);
                x_i16_0_15 = _mm256_srai_epi16(x_i16_0_15, shift);
                x_i16_16_31 = _mm256_srai_epi16(x_i16_16_31, shift);

                // convert i16 to i8; note that this puts it in a weird
                // order
                auto x_i8 = _mm256_packs_epi16(x_i16_0_15, x_i16_16_31);

                auto masks = _mm256_cmpgt_epi8(x_i8, vsplitvals);
                // map -1 -> 1; 0 stays the same
                auto masks_0_or_1 = _mm256_sign_epi8(masks, masks);

                if (s > 0) {
                    // shift left by multiplying by 2, by adding to itself
                    codes = _mm256_add_epi8(codes, codes);
                }

                // OR in new low bit
                codes = _mm256_or_si256(codes, masks_0_or_1);
            }
            // undo weird permutation from packing i16 -> i8
            codes = _mm256_permute4x64_epi64(codes, _MM_SHUFFLE(3,1,2,0));
            _mm256_storeu_si256((__m256i*)out_ptr, codes);
            out_ptr += out_block_stride;
        }
    }
}

/// XXX scales and offsets need to have one element per col, not split
template<bool DeferPerm=false>
void multisplit_encode_4b_colmajor_v2(const float* X, int64_t nrows, int ncols,
    const uint32_t* splitdims, const int8_t* all_splitvals,
    const float* scales, const float* offsets,
    int ncodebooks, uint8_t* out, int8_t* X_tmp)
    // const float* scales, int ncodebooks, uint8_t* out)
{
    static constexpr int block_nrows = 32;
    static constexpr int nsplits_per_codebook = 4;
    static constexpr int vals_per_split = 1 << nsplits_per_codebook; // 16
    const int64_t nblocks = ceil(nrows / (double)block_nrows);
    assert(nrows % block_nrows == 0); // TODO remove this constraint

    size_t x_col_stride = nrows;
    size_t out_col_stride = nrows;

    // ------------------------ convert needed cols of X to i8

    int total_nsplits = ncodebooks * nsplits_per_codebook;

    // for reusing quantized cols
    int quantized_col_starts[ncols];
    for (int i = 0; i < ncols; i++) {
        quantized_col_starts[i] = -1;
    }
    auto nuniq_cols = 0;
    int uniq_cols[ncols];
    for (int s = 0; s < total_nsplits; s++) {
        auto dim = splitdims[s];
        if (quantized_col_starts[dim] < 0) {
            quantized_col_starts[dim] = nuniq_cols;
            uniq_cols[nuniq_cols] = dim;
            nuniq_cols++;
        }
    }

    // actually quantize all the uniq cols
    for (int c = 0; c < nuniq_cols; c++) {
        auto in_col = uniq_cols[c];
        auto out_col = c;
        auto in_ptr = X + (in_col * x_col_stride);
        auto out_ptr = X_tmp + (out_col * x_col_stride);

        // load f32->u8 quantization params for this dim
        auto vscales = _mm256_set1_ps(scales[in_col]);
        auto voffsets = _mm256_set1_ps(offsets[in_col]);

        for (int b = 0; b < nblocks; b++) {
            auto x_i8 = load_4xf32_as_32xepi8_or_epu8<true, !DeferPerm>(
                in_ptr, vscales, voffsets);
            _mm256_store_si256((__m256i*)out_ptr, x_i8);
            in_ptr += block_nrows;
            out_ptr += block_nrows;
        }
    }

    // ------------------------ everything is i8 now

    size_t splitval_luts_stride = vals_per_split;
    const int8_t* x_ptrs[nsplits_per_codebook];
    __m256i current_vsplitval_luts[nsplits_per_codebook];

    int split_idx = 0;
    for (int c = 0; c < ncodebooks; c++) {
        // compute input and output column starts
        auto out_ptr = out + (out_col_stride * c);
        for (int s = 0; s < nsplits_per_codebook; s++) {
            auto splitdim = splitdims[split_idx + s];
            auto i8_dim = quantized_col_starts[splitdim];
            assert(i8_dim >= 0);
            x_ptrs[s] = X_tmp + (x_col_stride * i8_dim);
            auto splitvals_ptr = all_splitvals + (vals_per_split * split_idx);
            current_vsplitval_luts[s] = _mm256_broadcastsi128_si256(
                load_si128i((const __m128i*)splitvals_ptr));
        }
        split_idx += nsplits_per_codebook;

        for (int b = 0; b < nblocks; b++) { // for each block
            __m256i codes = _mm256_setzero_si256();
            for (int s = 0; s < nsplits_per_codebook; s++) {
                auto vsplitvals_lut = current_vsplitval_luts[s];
                auto vsplitvals = _mm256_shuffle_epi8(
                        vsplitvals_lut, codes); // codes = group_ids

                auto x_i8 = load_si256i(x_ptrs[s]);
                x_ptrs[s] += block_nrows;

                auto masks = _mm256_cmpgt_epi8(x_i8, vsplitvals);
                // map -1 -> 1; 0 stays the same
                auto masks_0_or_1 = _mm256_sign_epi8(masks, masks);

                if (s > 0) {
                    // shift left by multiplying by 2, by adding to itself
                    codes = _mm256_add_epi8(codes, codes);
                }

                // OR in new low bit
                codes = _mm256_or_si256(codes, masks_0_or_1);
            }
            if (DeferPerm) {
                codes = _mm256_permutevar8x32_epi32(
                    codes, _mm256_setr_epi32(0,4, 1,5, 2,6, 3,7));
            }
            _mm256_storeu_si256((__m256i*)out_ptr, codes);
            out_ptr += block_nrows;
        }
    }
}

} // anon namespace
#endif // multisplit_h

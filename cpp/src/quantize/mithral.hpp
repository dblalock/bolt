//
//  mithral.hpp
//
//  Created by DB on 2019-10-28
//  Copyright (c) 2019 DB. All rights reserved.
//
#ifndef __MITHRAL_HPP
#define __MITHRAL_HPP

#include <assert.h>
#include <stdint.h>
#include <sys/types.h>
#include <math.h>
#include <type_traits>
#include "immintrin.h"

#include "debug_utils.hpp" // TODO rm

#ifdef BLAZE
    #include "src/utils/avx_utils.hpp"
#else
    #include "avx_utils.hpp"
#endif

namespace {

/** 4 cols of 4b codes in the low 4 bits -> 1 col, twice as
 * long, of packed 4b codes in a particular layout.
 *
 * Layout is from the Quicker-ADC paper, and is basically designed
 * so that each 16B LUT is one lane of a ymm register; this is
 * better than broadcasting it to both lanes because you can fit
 * more LUTs in registers at once, and so either compute bigger
 * chunks of one output sum (many codebooks) or multiple outputs
 * at once (fewer codebooks). I.e., less register pressure, so do
 * more compute per read, and fewer writes.
 *
 * 0a   0b   0c   0d
 * 0a   0b   0c   0d
 * 0a   0b   0c   0d
 * 0a   0b   0c   0d
 * 0e   0f   0g   0h
 * 0e   0f   0g   0h
 * 0e   0f   0g   0h
 * 0e   0f   0g   0h
 *
 * -> packing
 *
 * ac
 * ac
 * ac
 * ac
 * eg
 * eg
 * eg
 * eg
 *
 * bd
 * bd
 * bd
 * bd
 * fh
 * fh
 * fh
 * fh
 *
 * -> perms
 *
 * ac
 * ac
 * ac
 * ac
 * bd
 * bd
 * bd
 * bd
 *
 * eg
 * eg
 * eg
 * eg
 * fh
 * fh
 * fh
 * fh
 */
inline void zip4_4b_colmajor(const uint8_t* codes_in, int64_t nrows,
                             uint32_t ncodebooks, uint8_t* codes_out)
{
    static constexpr int in_block_sz = 32;      // read 32 codes at once
    static constexpr int out_block_sz = 64;     // 32 x 4 cols -> 64 rows
    static constexpr int ncodebooks_per_group = 4;
    static constexpr int64_t chunk_sz = 1 << 29;
    // static constexpr int chunk_sz = 4096;       // one page
    // static constexpr int chunk_sz = 512;       // one page
    // static constexpr int chunk_sz = 64;       // one page
    assert(ncodebooks % ncodebooks_per_group == 0);
    assert(nrows % in_block_sz == 0);
    int ncolgroups = ncodebooks / ncodebooks_per_group;

    auto in_col_stride = nrows;
    auto out_col_stride = nrows * 2;
    auto nchunks = (nrows + chunk_sz - 1) / chunk_sz;

    for (int chunk = 0; chunk < nchunks; chunk++) {
        auto nrows_done_so_far = chunk * chunk_sz;
        auto N = MIN(chunk_sz, nrows - nrows_done_so_far);
        auto nblocks = N / in_block_sz;

        for (int c = 0; c < ncolgroups; c++) {
            // initialize col starts
            auto initial_col = c * ncodebooks_per_group;
            auto initial_col_ptr = codes_in + (initial_col * in_col_stride);
            auto out_col_ptr = codes_out + (c * out_col_stride);
            // for each block
            for (int b = 0; b < nblocks; b++) {
                // okay, forget being generic here; load from all 4 cols
                auto x0 = load_si256i(initial_col_ptr + 0 * in_col_stride);
                auto x1 = load_si256i(initial_col_ptr + 1 * in_col_stride);
                auto x2 = load_si256i(initial_col_ptr + 2 * in_col_stride);
                auto x3 = load_si256i(initial_col_ptr + 3 * in_col_stride);
                initial_col_ptr += 32;

                // pack bits
                auto x02 = _mm256_or_si256(x0, _mm256_slli_epi16(x2, 4));
                auto x13 = _mm256_or_si256(x1, _mm256_slli_epi16(x3, 4));

                // put lower 128b lanes together and higher 128b lanes together;
                // this corresponds to having y0 store 4 codebooks of codes for
                // rows 0-15, and y1 store same 4 codebooks of codes for rows 16-31
                auto y0 = _mm256_permute2x128_si256(x02, x13, 0 | (2 << 4));
                auto y1 = _mm256_permute2x128_si256(x02, x13, 1 | (3 << 4));

                // _mm256_store_si256((__m256i*)out_col_ptr, y0);
                _mm256_stream_si256((__m256i*)out_col_ptr, y0);
                out_col_ptr += 32;
                // _mm256_store_si256((__m256i*)out_col_ptr, y1);
                _mm256_stream_si256((__m256i*)out_col_ptr, y1);
                out_col_ptr += 32;
            }
        }
        codes_in += chunk_sz;
        codes_out += 2 * chunk_sz;
    }
}

// just pack low 4b from 2 cols into one col by using upper 8b; note that
// we assume that upper 4b are originally 0
inline void zip2_4b_colmajor(const uint8_t* codes_in, int64_t nrows,
                             uint32_t ncodebooks, uint8_t* codes_out)
{
    static constexpr int in_block_sz = 32;
    static constexpr int out_block_sz = 32;
    static constexpr int ncodebooks_per_group = 2;
    static constexpr int64_t chunk_sz = 1 << 29;
    // static constexpr int chunk_sz = 4096;  // one page
    // static constexpr int chunk_sz = 1024;
    // static constexpr int chunk_sz = 512;
    // static constexpr int chunk_sz = 256;
    assert(ncodebooks % ncodebooks_per_group == 0);
    assert(nrows % in_block_sz == 0);
    int ncolgroups = ncodebooks / ncodebooks_per_group;

    auto in_col_stride = nrows;
    auto out_col_stride = nrows;
    auto nchunks = (nrows + chunk_sz - 1) / chunk_sz;

    for (int chunk = 0; chunk < nchunks; chunk++) {
        auto nrows_done_so_far = chunk * chunk_sz;
        auto N = MIN(chunk_sz, nrows - nrows_done_so_far);
        auto nblocks = N / in_block_sz;

        for (int c = 0; c < ncolgroups; c++) {
            // initialize col starts
            auto initial_col = c * ncodebooks_per_group;
            auto initial_col_ptr = codes_in + (initial_col * in_col_stride);
            auto out_col_ptr = codes_out + (c * out_col_stride);
            // for each block
            for (int b = 0; b < nblocks; b++) {
                auto initial_col = c * ncodebooks_per_group;
                auto x0 = load_si256i(initial_col_ptr + 0 * in_col_stride);
                auto x1 = load_si256i(initial_col_ptr + 1 * in_col_stride);
                initial_col_ptr += 32;

                // pack bits and store result
                auto x01 = _mm256_or_si256(x0, _mm256_slli_epi16(x1, 4));
                // _mm256_store_si256((__m256i*)out_col_ptr, x01);
                _mm256_stream_si256((__m256i*)out_col_ptr, x01);
                out_col_ptr += 32;
            }
        }
        codes_in += chunk_sz;
        codes_out += chunk_sz;
    }
}

// inline void zip_bolt_colmajor(const uint8_t* codes_in, int64_t nrows,
inline void zip_bolt_colmajor_v1(const uint8_t* codes_in, int64_t nrows,
                              uint32_t ncodebooks, uint8_t* codes_out)
{
    static constexpr int in_block_sz = 32;      // read 32 codes at once
    static constexpr int simd_vec_sz = 32;     // 32 x 4 cols -> 64 rows
    static constexpr int ncodebooks_per_group = 2;
    assert(ncodebooks % ncodebooks_per_group == 0);
    assert(nrows % in_block_sz == 0);
    int ncolgroups = ncodebooks / ncodebooks_per_group;
    auto nblocks = nrows / in_block_sz;

    // auto in_col_stride = in_block_sz * nblocks;
    auto in_col_stride = nrows;
    // auto out_col_stride = out_block_sz * nblocks;

    for (int c = 0; c < ncolgroups; c++) {
        // initialize col starts
        auto initial_col = c * ncodebooks_per_group;
        auto initial_col_ptr = codes_in + (initial_col * in_col_stride);
        auto out_col_ptr = codes_out + (c * simd_vec_sz);
        // for each block
        for (int b = 0; b < nblocks; b++) {
            // auto x0 = load_si256i(initial_col_ptr + 0 * in_col_stride);
            // auto x1 = load_si256i(initial_col_ptr + 1 * in_col_stride);
            auto x0 = stream_load_si256i(initial_col_ptr + 0 * in_col_stride);
            auto x1 = stream_load_si256i(initial_col_ptr + 1 * in_col_stride);
            initial_col_ptr += simd_vec_sz;

            // pack bits and store result
            auto x01 = _mm256_or_si256(x0, _mm256_slli_epi16(x1, 4));
            _mm256_store_si256((__m256i*)out_col_ptr, x01);
            // _mm256_stream_si256((__m256i*)out_col_ptr, x01);
            out_col_ptr += simd_vec_sz * ncolgroups;

            // __builtin_prefetch(out_col_ptr + 2 * simd_vec_sz * ncolgroups);
            // __builtin_prefetch(out_col_ptr + 4096); // one page ahead
            // __builtin_prefetch(out_col_ptr + 128);
        }
    }
}

// https://godbolt.org/z/BMx6D7 (also includes zip2_4b_colmajor to compare)
// inline void zip_bolt_colmajor(const uint8_t* codes_in, int64_t nrows,
inline void zip_bolt_colmajor_v2(const uint8_t* codes_in, int64_t nrows,
                              uint32_t ncodebooks, uint8_t* codes_out)
{
    static constexpr int in_block_sz = 32;
    static constexpr int simd_vec_sz = 32;
    // static constexpr int ncodebooks_per_group = 2;
    static constexpr int ncols_in_per_group = 4;
    static constexpr int ncols_out_per_group = 2;
    // static constexpr int ncols_in_per_group = 2;
    // static constexpr int ncols_out_per_group = 1;
    // static constexpr int chunk_sz = 4096;  // one page
    // static constexpr int chunk_sz = 2048;  // half a page
    static constexpr int chunk_sz = 1024;  // quarter of a page
    // static constexpr int chunk_sz = 512;
    // static constexpr int chunk_sz = 256;
    // static constexpr int chunk_sz = 128;
    // static constexpr int chunk_sz = 64;
    assert(ncodebooks % ncols_in_per_group == 0);
    assert(nrows % in_block_sz == 0);
    int ncolgroups = ncodebooks / ncols_in_per_group;
    auto nchunks = (nrows + chunk_sz - 1) / chunk_sz;
    auto in_col_stride = nrows;

    for (int chunk = 0; chunk < nchunks; chunk++) {
        auto nrows_done_so_far = chunk * chunk_sz;
        auto nblocks = chunk_sz / in_block_sz;
        if (chunk == nchunks - 1) {
            auto N = MIN(chunk_sz, nrows - nrows_done_so_far);
            nblocks = N / in_block_sz;
        }

        for (int g = 0; g < ncolgroups; g++) {
            for (int gg = 0; gg < ncols_out_per_group; gg++) {
                // initialize col starts
                auto initial_col_in = (g * ncols_in_per_group) + (2 * gg);
                auto col_out = initial_col_in / 2;
                auto initial_col_ptr = codes_in + (initial_col_in * in_col_stride);
                auto out_col_ptr = codes_out + (col_out * simd_vec_sz);
                // for each block
                #pragma unroll
                for (int b = 0; b < nblocks; b++) {
                    auto x0 = load_si256i(initial_col_ptr + 0 * in_col_stride);
                    auto x1 = load_si256i(initial_col_ptr + 1 * in_col_stride);
                    initial_col_ptr += simd_vec_sz;

                    // pack bits and store result
                    auto x01 = _mm256_or_si256(x0, _mm256_slli_epi16(x1, 4));
                    _mm256_store_si256((__m256i*)out_col_ptr, x01);
                    out_col_ptr += simd_vec_sz * ncolgroups;
                }
            }
        }
        codes_in += chunk_sz;
        codes_out += chunk_sz * ncodebooks / 2;
    }
}

// https://godbolt.org/z/BMx6D7 (also includes zip2_4b_colmajor to compare)
// inline void zip_bolt_colmajor_v2(const uint8_t* codes_in, int64_t nrows,
template<int NReadColsAtOnce=2>
inline void zip_bolt_colmajor(const uint8_t* codes_in, int64_t nrows,
                              uint32_t ncodebooks, uint8_t* codes_out)
{
    static constexpr int in_block_sz = 32;
    static constexpr int simd_vec_sz = 32;
    static constexpr int ncols_in_per_group = NReadColsAtOnce;
    // static constexpr int ncols_in_per_group = 16;
    // static constexpr int ncols_in_per_group = 4;
    // static constexpr int ncols_in_per_group = 2;
    static constexpr int ncols_in_per_col_out = 2;
    static constexpr int ncols_out_per_group =
        ncols_in_per_group / ncols_in_per_col_out;
    // static constexpr int ncols_in_per_group = 2;
    // static constexpr int ncols_out_per_group = 1;
    // static constexpr int chunk_sz = 4096;  // one page
    static constexpr int chunk_sz = 2048;  // half a page
    // static constexpr int chunk_sz = 1024;  // quarter of a page
    // static constexpr int chunk_sz = 512;  // quarter of a page
    // static constexpr int chunk_sz = 256;
    // static constexpr int chunk_sz = 128;
    // static constexpr int chunk_sz = 64;
    assert(ncodebooks % ncols_in_per_group == 0);
    assert(nrows % in_block_sz == 0);
    int ncolgroups = ncodebooks / ncols_in_per_group;

    // int chunk_sz = MAX(256, 4096 / (ncodebooks / 2));
    // int chunk_sz = 4096 / ncodebooks;
    auto nchunks = (nrows + chunk_sz - 1) / chunk_sz;

    auto in_col_stride = nrows;
    // auto out_stride = simd_vec_sz;

    // uint8_t* in_col_ptrs[ncols_in_per_group];
    // const uint8_t* in_col_ptrs[ncols_in_per_col_out];
    const uint8_t* in_col_ptrs[ncols_out_per_group];
    uint8_t* out_ptrs[ncols_out_per_group];

    for (int chunk = 0; chunk < nchunks; chunk++) {
        auto nrows_done_so_far = chunk * chunk_sz;
        auto nblocks = chunk_sz / in_block_sz;
        if (chunk == nchunks - 1) {
            auto N = MIN(chunk_sz, nrows - nrows_done_so_far);
            nblocks = N / in_block_sz;
        }

        for (int g = 0; g < ncolgroups; g++) {
            // initialize col starts
            for (int gg = 0; gg < ncols_out_per_group; gg++) {
                auto initial_col_in = (g * ncols_in_per_group) + (2 * gg);
                auto col_out = initial_col_in / 2;
                in_col_ptrs[gg] = codes_in + (initial_col_in * in_col_stride);
                out_ptrs[gg] = codes_out + (col_out * simd_vec_sz);
            }
            // for each block
            // #pragma unroll
            for (int b = 0; b < nblocks; b++) {
                #pragma unroll
                for (int gg = 0; gg < ncols_out_per_group; gg++) {
                    // // initialize col starts
                    // auto initial_col_in = (g * ncols_in_per_group) + (2 * gg);
                    // auto col_out = initial_col_in / 2;
                    // auto initial_col_ptr = codes_in + (initial_col_in * in_col_stride);
                    // auto out_col_ptr = codes_out + (col_out * simd_vec_sz);
                    // // for each block
                    // for (int b = 0; b < nblocks; b++) {
                    // auto x0 = load_si256i(initial_col_ptr + 0 * in_col_stride);
                    // auto x1 = load_si256i(initial_col_ptr + 1 * in_col_stride);
                    // initial_col_ptr += simd_vec_sz;
                    auto in_ptr = in_col_ptrs[gg];
                    auto x0 = load_si256i(in_ptr + 0 * in_col_stride);
                    auto x1 = load_si256i(in_ptr + 1 * in_col_stride);
                    // initial_col_ptr += simd_vec_sz;
                    in_col_ptrs[gg] += simd_vec_sz;

                    // pack bits and store result
                    auto x01 = _mm256_or_si256(x0, _mm256_slli_epi16(x1, 4));
                    _mm256_store_si256((__m256i*)(out_ptrs[gg]), x01);
                    out_ptrs[gg] += simd_vec_sz * ncolgroups;
                    // _mm256_store_si256((__m256i*)out_col_ptr, x01);
                    // _mm256_stream_si256((__m256i*)out_col_ptr, x01); // 4x slower
                    // out_col_ptr += simd_vec_sz * ncolgroups;
                    // __builtin_prefetch(out_col_ptr + 16 * simd_vec_sz * ncolgroups);
                    // __builtin_prefetch(out_col_ptr + 16 * simd_vec_sz * ncodebooks / 2);
                }
            }
        }
        codes_in += chunk_sz;
        codes_out += chunk_sz * ncodebooks / 2;
    }
}

inline void zip_bolt_colmajor(const uint8_t* codes_in, int64_t nrows,
                              uint32_t ncodebooks, uint8_t* codes_out)
{
    if (ncodebooks % 8 == 0) {
        zip_bolt_colmajor<8>(codes_in, nrows, ncodebooks, codes_out); return;
    }
    if (ncodebooks % 4 == 0) {
        zip_bolt_colmajor<4>(codes_in, nrows, ncodebooks, codes_out); return;
    }
    zip_bolt_colmajor<2>(codes_in, nrows, ncodebooks, codes_out);
}


// like the above, but we assume that codes are in colmajor order
// TODO version of this that uses packed 4b codes? along with packing func?
// TODO version of this that doesn't immediately upcast to u16?
template<bool NoOverflow=false, bool SignedLUTs=false>
inline void mithral_scan_unpacked_colmajor(const uint8_t* codes,
    int64_t nblocks, int ncodebooks, const uint8_t* luts, int16_t* out)
{
    static constexpr int lut_sz = 16;
    static constexpr int block_rows = 32;
    const int64_t nrows = nblocks * block_rows;

    // zero output to use as workmem
    for (int64_t i = 0; i < nrows; i++) { out[i] = 0; }

    for (int m = 0; m < ncodebooks; m++) {
        // load lut for this subspace
        auto lut_ptr = luts + (m * lut_sz);
        auto vlut = _mm256_broadcastsi128_si256(
                load_si128i((const __m128i*)lut_ptr));

        // iterate thru this whole column of codes
        auto codes_col_start = codes + (nrows * m);  // colmajor contiguous
        auto codes_ptr = codes_col_start;
        auto out_ptr = out;
        for (int64_t b = 0; b < nblocks; b++) {
            auto vcodes = load_si256i(codes_ptr);
            auto dists_so_far_0 = load_si256i(out_ptr);
            auto dists_so_far_1 = load_si256i(out_ptr + 16);

            auto dists = _mm256_shuffle_epi8(vlut, vcodes);

            auto dists0 = _mm256_cvtepi8_epi16(
                _mm256_extracti128_si256(dists, 0));
            auto dists1 = _mm256_cvtepi8_epi16(
                _mm256_extracti128_si256(dists, 1));

            auto new_dists0 = _mm256_undefined_si256();
            auto new_dists1 = _mm256_undefined_si256();
            if (SignedLUTs) {
                new_dists0 = _mm256_adds_epi16(dists0, dists_so_far_0);
                new_dists1 = _mm256_adds_epi16(dists1, dists_so_far_1);
            } else {
                new_dists0 = _mm256_adds_epu16(dists0, dists_so_far_0);
                new_dists1 = _mm256_adds_epu16(dists1, dists_so_far_1);
            }

            _mm256_store_si256((__m256i*)out_ptr, new_dists0);
            _mm256_store_si256((__m256i*)(out_ptr + 16), new_dists1);
            codes_ptr += block_rows;
            out_ptr += block_rows;
        }
    }
}


template<int UpcastEvery, bool SignedLUTs>
void _accumulate_8bit_si256(const __m256i& x0, const __m256i& x1,
                            const __m256i& x2, const __m256i& x3,
                            const __m256i& initial_0_15,
                            const __m256i& initial_16_31,
                            __m256i& out_0_15, __m256i& out_16_31)
{
    static_assert(UpcastEvery == 1 || UpcastEvery == 2 || UpcastEvery == 4,
        "UpcastEvery must be one of {1,2,4}");
    if (SignedLUTs) {
        if (UpcastEvery == 4) {
            // add all four, and only then upcast to 16b
            auto x01 = _mm256_adds_epi8(x0, x1);
            auto x23 = _mm256_adds_epi8(x2, x3);
            auto x = _mm256_adds_epi8(x01, x23);
            out_0_15 = _mm256_cvtepi8_epi16(
                _mm256_extracti128_si256(x, 0));
            out_16_31 = _mm256_cvtepi8_epi16(
                _mm256_extracti128_si256(x, 1));
        } else if (UpcastEvery == 2) {
            // pairwise sums, then upcast to 16 bits
            auto x01 = _mm256_adds_epi8(x0, x1);
            auto x23 = _mm256_adds_epi8(x2, x3);

            auto x01_0 = _mm256_cvtepi8_epi16(
                _mm256_extracti128_si256(x01, 0));
            auto x01_1 = _mm256_cvtepi8_epi16(
                _mm256_extracti128_si256(x01, 1));
            auto x23_0 = _mm256_cvtepi8_epi16(
                _mm256_extracti128_si256(x23, 0));
            auto x23_1 = _mm256_cvtepi8_epi16(
                _mm256_extracti128_si256(x23, 1));

            out_0_15 = _mm256_adds_epi16(x01_0, x23_0);
            out_16_31 = _mm256_adds_epi16(x01_1, x23_1);
        } else if (UpcastEvery == 1) {
            // convert everything to 16 bits immediately
            auto x0_0 = _mm256_cvtepi8_epi16(
                _mm256_extracti128_si256(x0, 0));
            auto x0_1 = _mm256_cvtepi8_epi16(
                _mm256_extracti128_si256(x0, 1));
            auto x1_0 = _mm256_cvtepi8_epi16(
                _mm256_extracti128_si256(x1, 0));
            auto x1_1 = _mm256_cvtepi8_epi16(
                _mm256_extracti128_si256(x1, 1));
            auto x2_0 = _mm256_cvtepi8_epi16(
                _mm256_extracti128_si256(x2, 0));
            auto x2_1 = _mm256_cvtepi8_epi16(
                _mm256_extracti128_si256(x2, 1));
            auto x3_0 = _mm256_cvtepi8_epi16(
                _mm256_extracti128_si256(x3, 0));
            auto x3_1 = _mm256_cvtepi8_epi16(
                _mm256_extracti128_si256(x3, 1));

            auto x01_0 = _mm256_adds_epi16(x0_0, x1_0);
            auto x01_1 = _mm256_adds_epi16(x0_1, x1_1);
            auto x23_0 = _mm256_adds_epi16(x2_0, x3_0);
            auto x23_1 = _mm256_adds_epi16(x2_1, x3_1);
            out_0_15 = _mm256_adds_epi16(x01_0, x23_0);
            out_16_31 = _mm256_adds_epi16(x01_1, x23_1);
        }
        out_0_15 = _mm256_adds_epi16(out_0_15, initial_0_15);
        out_16_31 = _mm256_adds_epi16(out_16_31, initial_16_31);
    } else {
        if (UpcastEvery == 4) {
            // add all four, and only then upcast to 16b
            auto x01 = _mm256_adds_epu8(x0, x1);
            auto x23 = _mm256_adds_epu8(x2, x3);
            auto x = _mm256_adds_epu8(x01, x23);
            out_0_15 = _mm256_cvtepu8_epi16(
                _mm256_extracti128_si256(x, 0));
            out_16_31 = _mm256_cvtepu8_epi16(
                _mm256_extracti128_si256(x, 1));
        } else if (UpcastEvery == 2) {
            // pairwise sums, then upcast to 16 bits
            auto x01 = _mm256_adds_epu8(x0, x1);
            auto x23 = _mm256_adds_epu8(x2, x3);

            auto x01_0 = _mm256_cvtepu8_epi16(
                _mm256_extracti128_si256(x01, 0));
            auto x01_1 = _mm256_cvtepu8_epi16(
                _mm256_extracti128_si256(x01, 1));
            auto x23_0 = _mm256_cvtepu8_epi16(
                _mm256_extracti128_si256(x23, 0));
            auto x23_1 = _mm256_cvtepu8_epi16(
                _mm256_extracti128_si256(x23, 1));

            out_0_15 = _mm256_adds_epi16(x01_0, x23_0);
            out_16_31 = _mm256_adds_epi16(x01_1, x23_1);
        } else if (UpcastEvery == 1) {
            // convert everything to 16 bits immediately
            auto x0_0 = _mm256_cvtepu8_epi16(
                _mm256_extracti128_si256(x0, 0));
            auto x0_1 = _mm256_cvtepu8_epi16(
                _mm256_extracti128_si256(x0, 1));
            auto x1_0 = _mm256_cvtepu8_epi16(
                _mm256_extracti128_si256(x1, 0));
            auto x1_1 = _mm256_cvtepu8_epi16(
                _mm256_extracti128_si256(x1, 1));
            auto x2_0 = _mm256_cvtepu8_epi16(
                _mm256_extracti128_si256(x2, 0));
            auto x2_1 = _mm256_cvtepu8_epi16(
                _mm256_extracti128_si256(x2, 1));
            auto x3_0 = _mm256_cvtepu8_epi16(
                _mm256_extracti128_si256(x3, 0));
            auto x3_1 = _mm256_cvtepu8_epi16(
                _mm256_extracti128_si256(x3, 1));

            auto x01_0 = _mm256_adds_epi16(x0_0, x1_0);
            auto x01_1 = _mm256_adds_epi16(x0_1, x1_1);
            auto x23_0 = _mm256_adds_epi16(x2_0, x3_0);
            auto x23_1 = _mm256_adds_epi16(x2_1, x3_1);
            out_0_15 = _mm256_adds_epi16(x01_0, x23_0);
            out_16_31 = _mm256_adds_epi16(x01_1, x23_1);
        }
        out_0_15 = _mm256_adds_epu16(out_0_15, initial_0_15);
        out_16_31 = _mm256_adds_epu16(out_16_31, initial_16_31);
    }
}

// for looking at assembly: https://godbolt.org/z/PEjpiz
// .LBB0_7:                                #   Parent Loop BB0_6 Depth=1
//         vmovdqa      ymm5, ymmword ptr [rdi + rcx]
//         vmovdqa      ymm6, ymmword ptr [rdx + rcx]
//         vpsrlw       ymm7, ymm5, 4
//         vpsrlw       ymm8, ymm6, 4
//         vpand        ymm5, ymm5, ymm0
//         vpshufb      ymm5, ymm1, ymm5
//         vpand        ymm7, ymm7, ymm0
//         vpshufb      ymm7, ymm2, ymm7
//         vpaddsb      ymm5, ymm5, ymm7
//         vpand        ymm6, ymm6, ymm0
//         vpshufb      ymm6, ymm4, ymm6
//         vpand        ymm7, ymm8, ymm0
//         vpshufb      ymm7, ymm3, ymm7
//         vpaddsb      ymm6, ymm6, ymm7
//         vpaddsb      ymm5, ymm5, ymm6
//         vpmovsxbw    ymm6, xmm5
//         vextracti128 xmm5, ymm5, 1
//         vpmovsxbw    ymm5, xmm5
//         vpaddsw      ymm6, ymm6, ymmword ptr [rbx + 2*rcx]
//         vpaddsw      ymm5, ymm5, ymmword ptr [rbx + 2*rcx + 32]
//         vmovdqa      ymmword ptr [rbx + 2*rcx], ymm6
//         vmovdqa      ymmword ptr [rbx + 2*rcx + 32], ymm5
//         add          rcx, 32
//         dec          rax
//         jne          .LBB0_7
template<int UpcastEvery=4, bool Packed=false, typename LutT=uint8_t>
inline void _mithral_scan_tile4(const uint8_t* codes,
    int64_t nblocks, int ncodebooks, const LutT* luts, int16_t* out)
{
    static const bool SignedLUTs = std::is_signed<LutT>::value;
    // static const bool SignedLUTs = false; // TODO rm
    static_assert(sizeof(LutT) == 1, "Lookup table entries must be 1 byte!");
    static constexpr int codes_per_byte = Packed ? 2 : 1;
    static constexpr int lut_sz = 16;
    static constexpr int block_rows = 32;
    static constexpr int tile_sz = 4;
    // static const __m256i low_4bits_mask = _mm256_set1_epi8(0x0F);
    const int64_t nrows = nblocks * block_rows;
    assert(ncodebooks % tile_sz == 0); // otherwise tiling gets nasty

    auto ntiles = ncodebooks / tile_sz;

    // zero output to use as workmem
    for (int64_t i = 0; i < nrows; i++) { out[i] = 0; }

    // for (int m = 0; m < ncodebooks; m++) {
    for (int t = 0; t < ntiles; t++) {
        auto m0 = t * tile_sz;
        auto m1 = m0 + 1;
        auto m2 = m0 + 2;
        auto m3 = m0 + 3;

        // load lut for this subspace
        auto lut_ptr0 = luts + (m0 * lut_sz);
        auto lut_ptr1 = luts + (m1 * lut_sz);
        auto lut_ptr2 = luts + (m2 * lut_sz);
        auto lut_ptr3 = luts + (m3 * lut_sz);
        auto vlut0 = _mm256_broadcastsi128_si256(load_si128i(lut_ptr0));
        auto vlut1 = _mm256_broadcastsi128_si256(load_si128i(lut_ptr1));
        auto vlut2 = _mm256_broadcastsi128_si256(load_si128i(lut_ptr2));
        auto vlut3 = _mm256_broadcastsi128_si256(load_si128i(lut_ptr3));

        // iterate thru tile_sz columns of codes at once

        auto codes_ptr0 = codes + (nrows * (m0 / codes_per_byte));
        auto codes_ptr1 = codes + (nrows * (m1 / codes_per_byte));
        auto codes_ptr2 = codes + (nrows * (m2 / codes_per_byte));
        auto codes_ptr3 = codes + (nrows * (m3 / codes_per_byte));
        auto out_ptr = out;
        auto low_4bits_mask = _mm256_set1_epi8(0x0F);
        for (int64_t b = 0; b < nblocks; b++) {
            auto vcodes0 = _mm256_undefined_si256();
            auto vcodes1 = _mm256_undefined_si256();
            auto vcodes2 = _mm256_undefined_si256();
            auto vcodes3 = _mm256_undefined_si256();
            if (Packed) {
                // just ignore codes_ptr 1 and 3; they'll point to the
                // same indices as 0 and 2 since we do integer division
                // auto vcodes01 = stream_load_si256i(codes_ptr0);
                // auto vcodes23 = stream_load_si256i(codes_ptr2);
                auto vcodes01 = load_si256i(codes_ptr0);
                auto vcodes23 = load_si256i(codes_ptr2);

                vcodes0 = _mm256_and_si256(vcodes01, low_4bits_mask);
                vcodes1 = _mm256_and_si256(
                    _mm256_srli_epi16(vcodes01, 4), low_4bits_mask);
                vcodes2 = _mm256_and_si256(vcodes23, low_4bits_mask);
                vcodes3 = _mm256_and_si256(
                    _mm256_srli_epi16(vcodes23, 4), low_4bits_mask);
            } else {
                vcodes0 = load_si256i(codes_ptr0);
                vcodes1 = load_si256i(codes_ptr1);
                vcodes2 = load_si256i(codes_ptr2);
                vcodes3 = load_si256i(codes_ptr3);
            }

            auto dists_so_far_0_15 = load_si256i(out_ptr);
            auto dists_so_far_16_31 = load_si256i(out_ptr + 16);

            auto dists0 = _mm256_shuffle_epi8(vlut0, vcodes0);
            auto dists1 = _mm256_shuffle_epi8(vlut1, vcodes1);
            auto dists2 = _mm256_shuffle_epi8(vlut2, vcodes2);
            auto dists3 = _mm256_shuffle_epi8(vlut3, vcodes3);

            auto new_dists_0_15 = _mm256_undefined_si256();
            auto new_dists_16_31 = _mm256_undefined_si256();
            _accumulate_8bit_si256<UpcastEvery, SignedLUTs>(
                dists0, dists1, dists2, dists3,
                dists_so_far_0_15, dists_so_far_16_31,
                new_dists_0_15, new_dists_16_31);

            // _mm256_stream_si256 is way slower factor of (4.5 / 3.4)
            _mm256_store_si256((__m256i*)out_ptr, new_dists_0_15);
            _mm256_store_si256((__m256i*)(out_ptr + 16), new_dists_16_31);
            codes_ptr0 += block_rows;
            codes_ptr1 += block_rows;
            codes_ptr2 += block_rows;
            codes_ptr3 += block_rows;
            out_ptr += block_rows;
        }
    }
}

template<int UpcastEvery=4, typename LutT=uint8_t>
inline void mithral_scan_tile4(const uint8_t* codes,
    int64_t nblocks, int ncodebooks, const LutT* luts, int16_t* out)
{
    _mithral_scan_tile4<UpcastEvery, true>(
        codes, nblocks, ncodebooks, luts, out);
}

/* https://godbolt.org/z/qeMdf0
 * inner loop with 5 input cols, one output (for one output):
 *
 * vmovdqa      ymm3, YMMWORD PTR [rcx-32]
 * mov          QWORD PTR [rsp+160], rcx
 * vpaddsb      ymm0, ymm0, ymm1
 * vpmovsxbw    ymm1, xmm0
 * vextracti128 xmm0, ymm0, 0x1
 * vpmovsxbw    ymm0, xmm0
 * vpaddsw      ymm1, ymm1, ymm0
 * vpsrlw       ymm0, ymm3, 4
 * vpand        ymm3, ymm3, ymm2
 * vpand        ymm0, ymm2, ymm0
 * vpshufb      ymm3, ymm9, ymm3
 * vpaddsw      ymm1, ymm1, YMMWORD PTR [r14-32]
 * vpshufb      ymm0, ymm10, ymm0
 * vpaddsb      ymm3, ymm3, ymm0
 */
template<int NReadCols, int NWriteCols=1, int UpcastEvery=4, int NCodebooks=-1, int ChunkSz=256>
inline void _mithral_scan(const uint8_t* codes,
    int nrows, int ncodebooks, int noutputs,
    const int8_t* luts, int16_t* out, bool add_to_output=false,
    int codes_col_stride=-1, int lut_col_stride=-1, int out_col_stride=-1,
    int nrows_per_chunk=512)
    // int nrows_per_chunk=256)
    // int nrows_per_chunk=(1 << 20))
    // int nrows_per_chunk=128)
{
    static constexpr int block_nrows = 16;
    static constexpr int simd_vec_sz = 32;
    static constexpr int ncodebooks_per_col = 4;
    static constexpr int nlutvecs_per_col = 2;
    static constexpr int out_elem_sz = sizeof(out[0]);
    static constexpr int nreadcodebooks = NReadCols * ncodebooks_per_col;
    static constexpr int UpcastEveryNCols = UpcastEvery / nlutvecs_per_col;
    static_assert(sizeof(luts[0]) == 1, "Lookup table entries must be 1 byte!");
    static_assert(UpcastEvery % 2 == 0, "UpcastEvery must be even");
    assert(nrows % block_nrows == 0);
    assert(ncodebooks % ncodebooks_per_col == 0);
    assert(ncodebooks % nreadcodebooks == 0);
    assert(noutputs % NWriteCols == 0);
    auto ncols = ncodebooks / ncodebooks_per_col;
    int nlutvecs_per_output = ncols * nlutvecs_per_col;
    // luts are (ncols / nlutvecs_per_col) x noutputs colmajor
    int default_lut_col_stride = nlutvecs_per_output * simd_vec_sz;

    ncodebooks = NCodebooks > 0 ? NCodebooks : ncodebooks;
    nrows_per_chunk = ChunkSz > 0 ? ChunkSz : nrows_per_chunk;

    auto N = nrows;
    auto N_orig = N;
    auto nchunks_N = (N + nrows_per_chunk - 1) / nrows_per_chunk;
    N = N < nrows_per_chunk ? N : nrows_per_chunk; // *after* setting strides
    auto codes_orig = codes;
    auto out_orig = out;
    codes_col_stride = codes_col_stride >= 1 ? codes_col_stride : 2 * N_orig;
    lut_col_stride = lut_col_stride   >= 1 ?
        lut_col_stride  : default_lut_col_stride;
    out_col_stride = out_col_stride >= 1 ? out_col_stride : N_orig;

    // costants derived from matrix / tiling sizes
    int ncolstripes_in = ncols / NReadCols;
    int nstripes_out = noutputs / NWriteCols;

    // arrays that will all get unrolled and not really exist
    int in_cols[NReadCols];
    const uint8_t* codes_col_starts[NReadCols];
    const uint8_t* codes_col_ptrs[NReadCols];
    // const int8_t* lut_col_starts[NWriteCols];
    const int8_t* lut_col_ptrs[NWriteCols];
    int16_t* out_col_starts[NWriteCols];
    int16_t* out_col_ptrs[NWriteCols];
    __m256i vluts[NReadCols][NWriteCols][nlutvecs_per_col];
    for (int i = 0; i < NReadCols; i++) {
        for (int o = 0; o < NWriteCols; o++) {
            for (int v = 0; v < nlutvecs_per_col; v++) {
                vluts[i][o][v] = _mm256_undefined_si256();
            }
        }
    }

    // // PRINT_VAR(N);
    // PRINT_VAR(ncols);
    // PRINT_VAR(NReadCols);
    // PRINT_VAR(NWriteCols);
    // // // PRINT_VAR(N_orig);
    // // // PRINT_VAR(nchunks_N);
    // // PRINT_VAR(ncolstripes_in);
    // // PRINT_VAR(codes_col_stride);
    // PRINT_VAR(nstripes_out);
    // PRINT_VAR(out_col_stride);
    // // PRINT_VAR(nlutvecs_per_col);
    // // PRINT_VAR(nlutvecs_per_output);
    // PRINT_VAR(lut_col_stride);

    for (int n = 0; n < nchunks_N; n++) {
        codes = codes_orig + (2 * n * nrows_per_chunk);
        out = out_orig + (n * nrows_per_chunk);
        if (n == (nchunks_N - 1)) { // handle last chunk
            auto N_done_so_far = n * nrows_per_chunk;
            N = N_orig - N_done_so_far;
        }
        int nblocks_N = N / block_nrows;

        for (int m = 0; m < nstripes_out; m++) { // for each group of outputs
            // set output and lut col start ptrs
            for (int mm = 0; mm < NWriteCols; mm++) {
                auto out_col = (m * NWriteCols) + mm;
                // lut_col_starts[mm] = luts + (lut_col_stride * out_col);
                lut_col_ptrs[mm] = luts + (lut_col_stride * out_col);
                out_col_starts[mm] = out + (out_col_stride * out_col);

                // if (mm == 0) {
                //     printf("out_col: %2d; lut offset = %d; initial lut vals:\n", out_col, (lut_col_stride * out_col));
                //     dump_m256i<int8_t>(load_si256i(lut_col_ptrs[mm]));
                // }

                if (!add_to_output && NCodebooks != 4) {  // zero this block of output buffer
                    for (int i = 0; i < N; i++) {
                        out_col_starts[mm][i] = 0;
                    }
                }
            }
            // printf("right after zeroing output buff; new lut vals:\n");
            // dump_m256i<int8_t>(load_si256i(lut_col_ptrs[0]));

            // for each group of input cols
            for (int j = 0; j < ncolstripes_in; j++) {
                // set col start ptrs and current ptrs for simplicity
                for (int jj = 0; jj < NReadCols; jj++) {
                    auto in_col = j * NReadCols + jj;
                    in_cols[jj] = in_col;
                    // printf("in_col = %2d, j=%2d, jj=%2d\n", in_col, j, jj);
                    codes_col_starts[jj] = codes + (codes_col_stride * in_col);
                    codes_col_ptrs[jj] = codes_col_starts[jj];
                }

                // printf("start of input group %d; new lut vals:\n", j);
                // dump_m256i<int8_t>(load_si256i(lut_col_ptrs[0]));

                for (int mm = 0; mm < NWriteCols; mm++) {
                    // reset output write positions to top of cols
                    out_col_ptrs[mm] = out_col_starts[mm];
                    // lut_col_ptrs[mm] = lut_col_starts[mm];

                    // load up coeffs for this group of input dims
                    // auto lut_col_start = lut_col_starts[mm];
                    // for (int jj = 0; jj < NReadCols; jj++) {
                        // auto in_col = in_cols[jj];
                        // auto vlut_low_idx = in_col * nlutvecs_per_col;
                        // auto vlut_high_idx = vlut_low_idx + 1;
                        // auto vlut_low_ptr = lut_col_start + (simd_vec_sz * vlut_low_idx);
                        // auto vlut_high_ptr = lut_col_start + (simd_vec_sz * vlut_high_idx);
                        // vluts[jj][mm][0] = load_si256i(vlut_low_ptr);
                        // vluts[jj][mm][1] = load_si256i(vlut_high_ptr);

                    for (int jj = 0; jj < NReadCols; jj++) {
                        // if (mm == 0) {
                        //     printf("out_col: %2d; new lut vals:\n", m * NWriteCols + mm);
                        //     dump_m256i<int8_t>(load_si256i(lut_col_ptrs[mm]));
                        // }

                        vluts[jj][mm][0] = load_si256i(lut_col_ptrs[mm]);
                        lut_col_ptrs[mm] += simd_vec_sz;
                        vluts[jj][mm][1] = load_si256i(lut_col_ptrs[mm]);
                        lut_col_ptrs[mm] += simd_vec_sz;

                        // if (m == 0) {
                        //     printf("LUT for outcol=%2d, incol=%2d, mm=%2d, jj=%2d, lut_ab, lut_cd:\n", m * NWriteCols + mm, j * NReadCols + jj, mm, jj);
                        //     dump_m256i<int8_t>(vluts[jj][mm][0]);
                        //     dump_m256i<int8_t>(vluts[jj][mm][1]);
                        // }
                    }
                }

                // static constexpr bool evens_odds = true;
                static constexpr bool evens_odds = false;

                auto low_4bits_mask = _mm256_set1_epi8(0x0F);
                auto low_8bits_mask = _mm256_set1_epi16(0x00FF);
                for (int b = 0; b < nblocks_N; b++) {   // for each block of rows
                    // load up sums-so-far from current output
                    __m256i sums[NWriteCols]; // 16xepi16
                    __m256i sums_evens[NWriteCols]; // 16xepi16
                    __m256i sums_odds[NWriteCols]; // 16xepi16
                    __m256i sums_8b[NWriteCols]; // only used if upcast_every == 8
                    for (int mm = 0; mm < NWriteCols; mm++) {
                        if (NCodebooks != 4) {
                            sums[mm] = load_si256i(out_col_ptrs[mm]);
                        } else { // only one stripe
                            sums[mm] = _mm256_setzero_si256();
                        }
                        sums_8b[mm] = _mm256_setzero_si256();

                        if (evens_odds) {
                            sums_evens[mm] = _mm256_setzero_si256();
                            sums_odds[mm] = _mm256_setzero_si256();
                        }
                        // printf("sums[%d]:\n", mm); dump_m256i<int16_t>(sums[mm]);
                    }

                    // load input from each col, and update partial sums for
                    // each output
                    for (int jj = 0; jj < NReadCols; jj++) {  // for each in col
                        auto code_ptr = codes_col_ptrs[jj];
                        auto vcodes_packed = load_si256i(code_ptr);
                        codes_col_ptrs[jj] += simd_vec_sz;

                        auto vcodes_ab = _mm256_and_si256(
                            vcodes_packed, low_4bits_mask);
                        auto vcodes_cd = _mm256_and_si256(
                            _mm256_srli_epi16(vcodes_packed, 4), low_4bits_mask);

                        // if (j > 0) {
                        // if (m == 0) {
                        //     printf("codes ab, cd:\n");
                        //     dump_m256i(vcodes_ab);
                        //     dump_m256i(vcodes_cd);
                        // }

                        for (int mm = 0; mm < NWriteCols; mm++) { // each out col
                            auto vlut_ab = vluts[jj][mm][0];
                            auto vlut_cd = vluts[jj][mm][1];

                            auto prod_ab = _mm256_shuffle_epi8(vlut_ab, vcodes_ab);
                            auto prod_cd = _mm256_shuffle_epi8(vlut_cd, vcodes_cd);

                            // if (j > 0) {
                            // if (m == 0 && mm == 0) {
                            //     printf("jj=%2d, mm=%2d; prods ab, cd:\n", jj, mm);
                            //     dump_m256i<int8_t>(prod_ab);
                            //     dump_m256i<int8_t>(prod_cd);
                            // }

                            auto sum_ac_bd = _mm256_adds_epi8(prod_ab, prod_cd);

                            // static_assert(UpcastEvery == 2, ""); // TODO rm
                            if (UpcastEvery == 2) { // just immediately upcast
                                if (evens_odds) {
                                    // left then right shift to sign extend
                                    auto sum_ac_bd_evens = _mm256_srai_epi16(
                                        _mm256_slli_epi16(sum_ac_bd, 8), 8);
                                    auto sum_ac_bd_odds = _mm256_srai_epi16(sum_ac_bd, 8);
                                    sums_evens[mm] = _mm256_adds_epi16(sums_evens[mm], sum_ac_bd_evens);
                                    sums_odds[mm] = _mm256_adds_epi16(sums_odds[mm], sum_ac_bd_odds);
                                } else {
                                    auto sum_ac = _mm256_cvtepi8_epi16(
                                    _mm256_extracti128_si256(sum_ac_bd, 0));
                                    auto sum_bd = _mm256_cvtepi8_epi16(
                                        _mm256_extracti128_si256(sum_ac_bd, 1));
                                    // auto sum_abcd = _mm256_adds_epi16(sum_ac, sum_bd);
                                    // sums[mm] = _mm256_adds_epi16(sums[mm], sum_abcd);
                                    sums[mm] = _mm256_add_epi16(sums[mm], sum_ac);
                                    sums[mm] = _mm256_add_epi16(sums[mm], sum_bd);
                                }
                            } else {
                                sums_8b[mm] = _mm256_adds_epi8(
                                    sums_8b[mm], sum_ac_bd);
                            }
                        }
                        auto needs_upcast = (jj == NReadCols - 1) || // last col
                            ((jj + 1) % UpcastEveryNCols == 0); // last col in set
                        needs_upcast = needs_upcast && (UpcastEvery > 2);
                        if (needs_upcast) {
                            for (int mm = 0; mm < NWriteCols; mm++) { // out col
                                auto sum_ac_bd = sums_8b[mm];
                                sums_8b[mm] = _mm256_setzero_si256();

                                if (evens_odds) {
                                    auto sum_ac_bd_evens = _mm256_srai_epi16(
                                        _mm256_slli_epi16(sum_ac_bd, 8), 8);
                                    auto sum_ac_bd_odds = _mm256_srai_epi16(
                                        sum_ac_bd, 8);
                                    sums_evens[mm] = _mm256_adds_epi16(
                                        sums_evens[mm], sum_ac_bd_evens);
                                    sums_odds[mm] = _mm256_adds_epi16(
                                        sums_odds[mm], sum_ac_bd_odds);
                                } else {
                                    auto sum_ac = _mm256_cvtepi8_epi16(
                                    _mm256_extracti128_si256(sum_ac_bd, 0));
                                    auto sum_bd = _mm256_cvtepi8_epi16(
                                        _mm256_extracti128_si256(sum_ac_bd, 1));
                                    // auto sum_abcd = _mm256_adds_epi16(sum_ac, sum_bd);
                                    // sums[mm] = _mm256_adds_epi16(sums[mm], sum_abcd);
                                    sums[mm] = _mm256_add_epi16(sums[mm], sum_ac);
                                    sums[mm] = _mm256_add_epi16(sums[mm], sum_bd);
                                }
                            }
                        }
                    }

                    // write back partial sums and increment output
                    // if (n > 0) { PRINT_VAR(b); }
                    for (int mm = 0; mm < NWriteCols; mm++) {
                        auto out_ptr = out_col_ptrs[mm];
                        if (evens_odds) {
                            // sums for codebooks for {low, high} 128b
                            auto tmp_evens = _mm256_permute4x64_epi64(
                                    sums_evens[mm], _MM_SHUFFLE(3,1,2,0));
                            auto tmp_odds = _mm256_permute4x64_epi64(
                                    sums_odds[mm], _MM_SHUFFLE(3,1,2,0));
                            auto low_sums = _mm256_unpacklo_epi16(
                                        tmp_evens, tmp_odds);
                            auto high_sums = _mm256_unpackhi_epi16(
                                tmp_evens, tmp_odds);
                            sums[mm] = _mm256_adds_epi16(
                                sums[mm], low_sums);
                            sums[mm] = _mm256_adds_epi16(
                                sums[mm], high_sums);
                        }
                        _mm256_storeu_si256((__m256i*)out_ptr, sums[mm]);
                        out_col_ptrs[mm] += simd_vec_sz / out_elem_sz;
                    }
                }
            }
        }
    }
}


template<int UpcastEvery=4>
void mithral_scan(const uint8_t* codes, int nrows, int ncodebooks,
    int noutputs, const int8_t* luts, int16_t* out, bool add_to_output=false)
{
    // TODO figure out optimal amount of tiling and reduce to this case, then
    // clean up leftover input/output columns

    auto ncols = ncodebooks / 4;
    if (ncols == 1) {
        static constexpr int NCodebooks = 4;
        // with 1 input col 6 outputs: https://godbolt.org/z/dLLfV8
        // if (noutputs % 6 == 0) {
        //     _mithral_scan<1, 6, UpcastEvery, NCodebooks>(
        //         codes, nrows, ncodebooks, noutputs, luts, out); return;
        // } else if (noutputs % 5 == 0) {
        if (noutputs % 5 == 0 && UpcastEvery > 2) {
            _mithral_scan<1, 5, UpcastEvery, NCodebooks>(
                codes, nrows, ncodebooks, noutputs, luts, out); return;
        } else if (noutputs % 4 == 0) {
            _mithral_scan<1, 4, UpcastEvery, NCodebooks>(
                codes, nrows, ncodebooks, noutputs, luts, out); return;
        } else if (noutputs % 3 == 0) {
            _mithral_scan<1, 3, UpcastEvery, NCodebooks>(
                codes, nrows, ncodebooks, noutputs, luts, out); return;
        } else if (noutputs % 2 == 0) {
            _mithral_scan<1, 2, UpcastEvery, NCodebooks>(
                codes, nrows, ncodebooks, noutputs, luts, out); return;
        } else {
            _mithral_scan<1, 1, UpcastEvery, NCodebooks>(
                codes, nrows, ncodebooks, noutputs, luts, out); return;
        }
    } else if (ncols == 2 && (noutputs % 2) == 0) {
        static constexpr int NCodebooks = 8;
        _mithral_scan<2, 2, UpcastEvery, NCodebooks>(
                codes, nrows, ncodebooks, noutputs, luts, out); return;
    // } else if (ncols % 6 == 0) {
    //     _mithral_scan<6, 1, UpcastEvery>(
    //             codes, nrows, ncodebooks, noutputs, luts, out); return;
    } else if (ncols % 5 == 0 && UpcastEvery > 2) {
        _mithral_scan<5, 1, UpcastEvery>(
                codes, nrows, ncodebooks, noutputs, luts, out); return;
    } else if (ncols % 4 == 0 && UpcastEvery > 2) {
        _mithral_scan<4, 1, UpcastEvery>(
                codes, nrows, ncodebooks, noutputs, luts, out); return;
    } else if (ncols % 3 == 0 && UpcastEvery > 2) {
        _mithral_scan<3, 1, UpcastEvery>(
                codes, nrows, ncodebooks, noutputs, luts, out); return;
    } else if (ncols % 2 == 0) {
        _mithral_scan<2, 1, UpcastEvery>(
                codes, nrows, ncodebooks, noutputs, luts, out); return;
    } else {
        _mithral_scan<1, 1, UpcastEvery>(
                codes, nrows, ncodebooks, noutputs, luts, out); return;
    }
}

} // anon namespace
#endif // __MITHRAL_HPP

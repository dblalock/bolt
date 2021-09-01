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
#include <cmath>
#include <type_traits>
#include <limits>
#include "immintrin.h"

#ifdef BLAZE
    #include "src/utils/avx_utils.hpp"
#else
    #include "avx_utils.hpp"
#endif


// ================================================================ in cpp

void zip_bolt_colmajor(const uint8_t* codes_in, int64_t nrows,
                       uint32_t ncodebooks, uint8_t* codes_out);

void dense_lut_f32_fused(const float* Q, int nrows, int ncols, int ncodebooks,
    const float* centroids, float*__restrict__ out_offsets, float& out_offset_sum,
    float& out_scale, float*__restrict__ out);

void dense_lut_f32(const float* Q, int nrows, int ncols, int ncodebooks,
                 const float* centroids, float* out);

void sparse_lut_f32(const float* Q, int nrows, int ncols, int ncodebooks,
                    const float* centroids,
                    const int* idxs, int nnz_per_centroid, float* out);

void mithral_lut_dense(const float* Q, int nrows, int ncols, int ncodebooks,
    const float* centroids, float& out_offset_sum, float& out_scale,
    float*__restrict__ tmp_lut_f32, uint8_t* out);

void mithral_lut_sparse(const float* Q, int nrows, int ncols, int ncodebooks,
    const float* centroids, const int* idxs, int nnz_per_centroid,
    float& out_offset_sum, float& out_scale,
    float*__restrict__ tmp_lut_f32, uint8_t* out);

// ================================================================ here

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
        int64_t nblocks = chunk_sz / in_block_sz;
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
        int64_t nblocks = chunk_sz / in_block_sz;
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


// template<int ncodebooks>
void _compute_offsets_scale_from_mins_maxs(
    const __m256* mins, const __m256* maxs, int ncodebooks,
    float* out_offsets, float& out_offset_sum, float& out_scale)
{
    // we now have the mins and maxes for each codebook; compute offsets
    // for each codebook, then global offset, then largest value - offset
    // auto vmins = mins[0];
    // float offset = 0;
    out_offset_sum = 0;
    __m256 vmax = _mm256_set1_ps(std::numeric_limits<float>::min());
    for (int c = 0; c < ncodebooks; c++) {
        auto vmin = broadcast_min(mins[c]);
        auto offset = pfirst(vmin);  // minimum value
        out_offsets[c] = offset;
        out_offset_sum += offset;

        // update vector of max vals seen so far
        vmax = _mm256_max_ps(vmax, _mm256_sub_ps(maxs[c], vmin));
    }
    vmax = broadcast_max(vmax);
    out_scale = pfirst(vmax);
    if (out_scale <= 0.f) {
        out_scale = 0.;
        return; // data is constant; just return
    }

    // round scale up to nearest power of 2
    float exponent = std::ceil(std::log2f(out_scale));
    out_scale = std::exp2(-exponent);  // reciprocal so we can use fma
    out_scale *= (255.f - 1e-10f);  // so max val is at most just under 255

    // update offsets based on scale, so that one can incorporate offsets
    // in an fma (specifically, fmsub to create lut and fma to invert)
    for (int c = 0; c < ncodebooks; c++) {
        out_offsets[c] *= out_scale;
    }
}

// template<int CodebookTileSz=2, int RowTileSz=2>
// NOTE: ColTileSz has no effect on performance; already unrolled plenty
// template<int CodebookTileSz=2, int RowTileSz=2, int ColTileSz=1>
template<int CodebookTileSz=2, int RowTileSz=2>
void dense_lut_f32(const float* Q, int nrows, int ncols, int ncodebooks,
                 const float* centroids, float* out)
{
    static constexpr int ColTileSz = 1;
    static constexpr int ncentroids = 16;
    static constexpr int lut_sz = ncentroids;
    static constexpr int packet_width = 8; // objs per simd register
    static constexpr int nstripes = lut_sz / packet_width;
    assert(ncodebooks % CodebookTileSz == 0);
    assert(nrows % RowTileSz == 0);

    // __m256 accumulators[CodebookTileSz * RowTileSz * nstripes];
    __m256 accumulators[CodebookTileSz][RowTileSz][nstripes];
    __m256 vbroadcasted[RowTileSz];

    const float* queries_ptrs[RowTileSz];
    const float* centroids_ptrs[CodebookTileSz];
    float* out_ptrs[RowTileSz][CodebookTileSz];

    auto q_row_stride = ncols;
    auto centroids_codebook_stride = ncentroids * ncols;
    auto out_row_stride = ncodebooks * lut_sz;
    auto out_codebook_stride = lut_sz;

    auto ncodebook_blocks = ncodebooks / CodebookTileSz;
    auto nrow_blocks = nrows / RowTileSz;
    auto ncol_blocks_full = ncols / ColTileSz;

    for (int r = 0; r < nrow_blocks; r++) {
        for (int c = 0; c < ncodebook_blocks; c++) {
            for (int cc = 0; cc < CodebookTileSz; cc++) {
                // set centroid start ptrs for this codebook
                auto codebook = (c * CodebookTileSz) + cc;
                centroids_ptrs[cc] =
                    centroids + (centroids_codebook_stride * codebook);
                for (int rr = 0; rr < RowTileSz; rr++) {
                    // set output ptrs for this codebook
                    auto row = (r * RowTileSz) + rr;
                    out_ptrs[rr][cc] = out + (out_row_stride * row) +
                        (out_codebook_stride * codebook);

                    // zero accumulators
                    for (int s = 0; s < nstripes; s++) {
                        accumulators[cc][rr][s] = _mm256_setzero_ps();
                        // auto idx = cc * (RowTileSz + nstripes) + (rr * RowTileSz) + s;
                        // accumulators[idx] = _mm256_setzero_ps();
                    }
                }
            }
            for (int rr = 0; rr < RowTileSz; rr++) {
                auto row = (r * RowTileSz) + rr;
                queries_ptrs[rr] = Q + (q_row_stride * row);
            }

            // compute sums for each output row for this block of codebooks
            // for (int j = 0; j < ncols; j++) {
            for (int j = 0; j < ncol_blocks_full; j++) {
                for (int jj = 0; jj < ColTileSz; jj++) {
                    for (int rr = 0; rr < RowTileSz; rr++) {
                        auto qval = *queries_ptrs[rr];
                        vbroadcasted[rr] = _mm256_set1_ps(qval);
                        queries_ptrs[rr]++;
                    }

                    for (int cc = 0; cc < CodebookTileSz; cc++) {
                        for (int s = 0; s < nstripes; s++) {
                            auto centroids_col = _mm256_load_ps(centroids_ptrs[cc]);
                            centroids_ptrs[cc] += packet_width;
                            for (int rr = 0; rr < RowTileSz; rr++) {
                                accumulators[cc][rr][s] = fma(vbroadcasted[rr],
                                    centroids_col, accumulators[cc][rr][s]);
                                // auto idx = cc * (RowTileSz + nstripes) + (rr * RowTileSz) + s;
                                // accumulators[idx] = fma(vbroadcasted[rr],
                                //     centroids_col, accumulators[idx]);
                            }
                        }
                    }
                }
            }
            // handle trailing cols
            for (int jj = ncol_blocks_full * ColTileSz; jj < ncols; jj++) {
                for (int rr = 0; rr < RowTileSz; rr++) {
                    auto qval = *queries_ptrs[rr];
                    vbroadcasted[rr] = _mm256_set1_ps(qval);
                    queries_ptrs[rr]++;
                }
                for (int cc = 0; cc < CodebookTileSz; cc++) {
                    for (int s = 0; s < nstripes; s++) {
                        auto centroids_col = _mm256_load_ps(centroids_ptrs[cc]);
                        centroids_ptrs[cc] += packet_width;
                        for (int rr = 0; rr < RowTileSz; rr++) {
                            accumulators[cc][rr][s] = fma(vbroadcasted[rr],
                                centroids_col, accumulators[cc][rr][s]);
                        }
                    }
                }
            }
            // write out sums
            for (int rr = 0; rr < RowTileSz; rr++) {
                for (int cc = 0; cc < CodebookTileSz; cc++) {
                    for (int s = 0; s < nstripes; s++) {
                        _mm256_store_ps(out_ptrs[rr][cc], accumulators[cc][rr][s]);
                        // auto idx = cc * (RowTileSz + nstripes) + (rr * RowTileSz) + s;
                        // _mm256_store_ps(out_ptrs[rr][cc], accumulators[idx]);
                        out_ptrs[rr][cc] += packet_width;
                    }
                }
            }
        }
    }
}
// // force it to instantiate this template
// template void dense_lut_f32<2, 3>(const float* Q, int nrows, int ncols,
//     int ncodebooks, const float* centroids, float* out);


// this is basically just a dense matmul that also tracks the min/max
// product; Q = nrows, ncols; centroids = ncols, ncodebooks; but centroids.T
// is already in a block-colmajor layout, with block size of 16; also Q
// is rowmajor
template<int CodebookTileSz=2, int RowTileSz=2>
void _dense_lut_f32_fused(const float* Q, int nrows, int ncols, int ncodebooks,
    // const float* centroids, float* out)
    // SELF: above args are fast, while ones below make it like 2x slower
    __m256*__restrict__ mins, __m256*__restrict__ maxs,
    const float* centroids, float*__restrict__ out_offsets,
    float& out_offset_sum, float& out_scale, float*__restrict__ out)
{
    static constexpr int ncentroids = 16;
    static constexpr int lut_sz = ncentroids;
    static constexpr int packet_width = 8; // objs per simd register
    static constexpr int nstripes = lut_sz / packet_width;
    assert(ncodebooks % CodebookTileSz == 0);
    assert(nrows % RowTileSz == 0);

    __m256 accumulators[CodebookTileSz][RowTileSz][nstripes];
    __m256 vbroadcasted[RowTileSz];

    const float* queries_ptrs[RowTileSz];
    const float* centroids_ptrs[CodebookTileSz];
    float* out_ptrs[RowTileSz][CodebookTileSz];

    // __m256 mins[ncodebooks];
    // __m256 maxs[ncodebooks];
    // for (int c = 0; c < ncodebooks; c++) {
    //     mins[c] = _mm256_set1_ps(std::numeric_limits<float>::max());
    //     maxs[c] = _mm256_set1_ps(std::numeric_limits<float>::min());
    // }

    auto q_row_stride = ncols;
    auto centroids_codebook_stride = ncentroids * ncols;
    auto out_row_stride = ncodebooks * lut_sz;
    auto out_codebook_stride = lut_sz;

    auto ncodebook_blocks = ncodebooks / CodebookTileSz;
    auto nrow_blocks = nrows / RowTileSz;

    static constexpr int ColTileSz = 1;
    auto ncol_blocks_full = ncols / ColTileSz;

    for (int r = 0; r < nrow_blocks; r++) {
        for (int c = 0; c < ncodebook_blocks; c++) {
            for (int cc = 0; cc < CodebookTileSz; cc++) {
                // set centroid start ptrs for this codebook
                auto codebook = (c * CodebookTileSz) + cc;
                centroids_ptrs[cc] =
                    centroids + (centroids_codebook_stride * codebook);
                for (int rr = 0; rr < RowTileSz; rr++) {
                    // set output ptrs for this codebook
                    auto row = (r * RowTileSz) + rr;
                    out_ptrs[rr][cc] = out + (out_row_stride * row) +
                        (out_codebook_stride * codebook);
                    // zero accumulators
                    for (int s = 0; s < nstripes; s++) {
                        accumulators[cc][rr][s] = _mm256_setzero_ps();
                    }
                }
            }
            for (int rr = 0; rr < RowTileSz; rr++) {
                auto row = (r * RowTileSz) + rr;
                queries_ptrs[rr] = Q + (q_row_stride * row);
            }

            // compute sums for each output row for this block of codebooks
            // for (int j = 0; j < ncols; j++) {
            for (int j = 0; j < ncol_blocks_full; j++) {
                for (int jj = 0; jj < ColTileSz; jj++) {
                    for (int rr = 0; rr < RowTileSz; rr++) {
                        auto qval = *queries_ptrs[rr];
                        vbroadcasted[rr] = _mm256_set1_ps(qval);
                        queries_ptrs[rr]++;
                    }

                    for (int cc = 0; cc < CodebookTileSz; cc++) {
                        for (int s = 0; s < nstripes; s++) {
                            auto centroids_col = _mm256_load_ps(centroids_ptrs[cc]);
                            centroids_ptrs[cc] += packet_width;
                            for (int rr = 0; rr < RowTileSz; rr++) {
                                accumulators[cc][rr][s] = fma(vbroadcasted[rr],
                                    centroids_col, accumulators[cc][rr][s]);
                            }
                        }
                    }
                }
            }
            // write out sums
            for (int rr = 0; rr < RowTileSz; rr++) {
                for (int cc = 0; cc < CodebookTileSz; cc++) {
                    auto codebook = (c * CodebookTileSz) + cc;
                    for (int s = 0; s < nstripes; s++) {
                        auto half_lut = accumulators[cc][rr][s];
                        mins[codebook] = _mm256_min_ps(
                            mins[codebook], half_lut);
                        maxs[codebook] = _mm256_max_ps(
                            maxs[codebook], half_lut);

                        _mm256_store_ps(out_ptrs[rr][cc], half_lut);
                        out_ptrs[rr][cc] += packet_width;
                    }
                }
            }
        }
    }
}

template<int CodebookTileSz=2, int RowTileSz=2>
void dense_lut_f32_fused(const float* Q, int nrows, int ncols, int ncodebooks,
    // const float* centroids, float* out)
    // SELF: above args are fast, while ones below make it like 2x slower
    const float* centroids, float*__restrict__ out_offsets,
    float& out_offset_sum, float& out_scale, float*__restrict__ out)
{
    static constexpr int ncentroids = 16;
    static constexpr int lut_sz = ncentroids;
    static_assert(RowTileSz >= 1, "RowTileSz must be >= 1");
    static_assert(RowTileSz <= 4, "RowTileSz must be <= 4 for now");

    // initilize mins and maxes; note that it's okay for the two calls to
    // see different mins and maxes since these arrays aren't used except
    // to compute the offsets and scale at the very end
    __m256 mins[ncodebooks];
    __m256 maxs[ncodebooks];
    for (int c = 0; c < ncodebooks; c++) {
        mins[c] = _mm256_set1_ps(std::numeric_limits<float>::max());
        maxs[c] = _mm256_set1_ps(std::numeric_limits<float>::min());
    }
    // handle most rows
    auto nrows_trailing = nrows % RowTileSz;
    auto nrows_round = nrows - nrows_trailing;
    if (nrows_round > 0) {
        _dense_lut_f32_fused<CodebookTileSz, RowTileSz>(
            Q, nrows_round, ncols, ncodebooks, mins, maxs,
            centroids, out_offsets, out_offset_sum, out_scale, out);
    }
    // handle trailing rows
    auto q_row_stride = ncols;
    Q += q_row_stride * nrows_round;
    auto out_row_stride = ncodebooks * lut_sz;
    out += out_row_stride * nrows_round;

    // NOTE: if we hardcode this to 1 instead of having a switch, or just
    // rm handling of the trailing rows entirely, code is twice as fast
    _dense_lut_f32_fused<CodebookTileSz, 1>(
            Q, nrows_trailing, ncols, ncodebooks, mins, maxs,
            centroids, out_offsets, out_offset_sum, out_scale, out);

    // switch(nrows_trailing) {
    //     case 0: break;
    //     case 1: _dense_lut_f32_fused<CodebookTileSz, 1>(
    //         Q, nrows_trailing, ncols, ncodebooks, mins, maxs,
    //         centroids, out_offsets, out_offset_sum, out_scale, out); break;
    //     case 2: _dense_lut_f32_fused<CodebookTileSz, 2>(
    //         Q, nrows_trailing, ncols, ncodebooks, mins, maxs,
    //         centroids, out_offsets, out_offset_sum, out_scale, out); break;
    //     case 3: _dense_lut_f32_fused<CodebookTileSz, 3>(
    //         Q, nrows_trailing, ncols, ncodebooks, mins, maxs,
    //         centroids, out_offsets, out_offset_sum, out_scale, out); break;
    // }
    // write out stats using mins and maxs
    _compute_offsets_scale_from_mins_maxs(
        mins, maxs, ncodebooks, out_offsets, out_offset_sum, out_scale);
}

template<int CodebookTileSz=2, int RowTileSz=2>
void sparse_lut_f32(const float* Q, int nrows, int ncols, int ncodebooks,
                     const float* centroids,
                     const int* idxs, int nnz_per_centroid,
                     float* out)
{
    static constexpr int ncentroids = 16;
    static constexpr int lut_sz = ncentroids;
    static constexpr int packet_width = 8; // objs per simd register
    static constexpr int nstripes = lut_sz / packet_width;
    assert(ncodebooks % CodebookTileSz == 0);
    assert(nrows % RowTileSz == 0);

    __m256 accumulators[CodebookTileSz][RowTileSz][nstripes];
    __m256 vbroadcasted[RowTileSz];

    const float* query_start_ptrs[RowTileSz];
    const int* idx_ptrs[CodebookTileSz];
    const float* centroids_ptrs[CodebookTileSz];
    float* out_ptrs[RowTileSz][CodebookTileSz];

    auto q_row_stride = ncols;
    auto idxs_codebook_stride = nnz_per_centroid;
    auto centroids_codebook_stride = ncentroids * ncols;
    auto out_row_stride = ncodebooks * lut_sz;
    auto out_codebook_stride = lut_sz;

    auto ncodebook_blocks = ncodebooks / CodebookTileSz;
    auto nrow_blocks = nrows / RowTileSz;

    for (int r = 0; r < nrow_blocks; r++) {
        // prefetch contents of rows, so we don't end up with random access
        // EDIT: doesn't help; is actually slightly slower
        // static constexpr int cache_line_sz_bytes = 64;  // on almost everything
        // static constexpr int stride = cache_line_sz_bytes / sizeof(Q[0]);
        // for (int rr = 0; rr < RowTileSz; rr++) {
        //         auto row = (r * RowTileSz) + rr;
        //         query_start_ptrs[rr] = Q + (q_row_stride * row);
        //     }
        // for (int j = 0; j < ncols; j += stride) {
        //     for (int rr = 0; rr < RowTileSz; rr++) {
        //         __builtin_prefetch(query_start_ptrs[rr] + j);
        //     }
        // }
        // compute all luts for this block of rows
        for (int c = 0; c < ncodebook_blocks; c++) {
            for (int cc = 0; cc < CodebookTileSz; cc++) {
                auto codebook = (c * CodebookTileSz) + cc;
                // set centroid start ptrs for this codebook
                centroids_ptrs[cc] =
                    centroids + (centroids_codebook_stride * codebook);
                // set idxs start ptrs for this codebook
                idx_ptrs[cc] = idxs + (idxs_codebook_stride * codebook);

                for (int rr = 0; rr < RowTileSz; rr++) {
                    // set output ptrs for this codebook
                    auto row = (r * RowTileSz) + rr;
                    out_ptrs[rr][cc] = out + (out_row_stride * row) +
                        (out_codebook_stride * codebook);
                    // zero accumulators
                    for (int s = 0; s < nstripes; s++) {
                        accumulators[cc][rr][s] = _mm256_setzero_ps();
                    }
                }
            }
            for (int rr = 0; rr < RowTileSz; rr++) {
                auto row = (r * RowTileSz) + rr;
                query_start_ptrs[rr] = Q + (q_row_stride * row);
            }

            // compute sums for each output row for this block of codebooks
            for (int j = 0; j < nnz_per_centroid; j++) {
                for (int cc = 0; cc < CodebookTileSz; cc++) {
                    auto idx = *idx_ptrs[cc];
                    idx_ptrs[cc]++;
                    for (int rr = 0; rr < RowTileSz; rr++) {
                        auto row_start_ptr = query_start_ptrs[rr];
                        auto qval = row_start_ptr[idx];
                        vbroadcasted[rr] = _mm256_set1_ps(qval);
                    }
                    for (int s = 0; s < nstripes; s++) {
                        auto centroids_col = _mm256_load_ps(centroids_ptrs[cc]);
                        centroids_ptrs[cc] += packet_width;
                        for (int rr = 0; rr < RowTileSz; rr++) {
                            accumulators[cc][rr][s] = fma(vbroadcasted[rr],
                                centroids_col, accumulators[cc][rr][s]);
                        }
                    }
                }
            }
            // write out sums
            for (int rr = 0; rr < RowTileSz; rr++) {
                for (int cc = 0; cc < CodebookTileSz; cc++) {
                    for (int s = 0; s < nstripes; s++) {
                        _mm256_store_ps(out_ptrs[rr][cc], accumulators[cc][rr][s]);
                        out_ptrs[rr][cc] += packet_width;
                    }
                }
            }
        }
    }
}

// this is just so that we can profile this separately
// template<int ncodebooks, int RowTileSz=1>
template<int RowTileSz=1>
void mithral_learn_lut_offsets_scales(
    const float* luts_f32, int nrows, int ncodebooks,
    float* out_offsets, float& out_offset_sum, float& out_scale)
{
    static constexpr int lut_sz = 16;
    // static constexpr int codebook_group_sz = 2; // 4 f32 luts -> 1 epu8 lut
    static constexpr int packet_width = 8; // objs per simd register
    // static constexpr int nstripes = lut_sz / packet_width;
    // static constexpr int ncodebook_groups = ncodebooks / codebook_group_sz;
    // static_assert(ncodebooks % codebook_group_sz == 0,
    //     "Number of codebooks must be a multiple of 2");
    assert(nrows % RowTileSz == 0);

    auto row_stride = ncodebooks * lut_sz;
    auto nrow_blocks = RowTileSz > 1 ? nrows / RowTileSz : 0;
    auto nrows_round = nrow_blocks * RowTileSz;

    const float* in_ptrs[RowTileSz];
    const float* offset_ptrs[RowTileSz];
    uint8_t* out_ptrs[RowTileSz];

    __m256 mins[ncodebooks];
    __m256 maxs[ncodebooks];
    for (int c = 0; c < ncodebooks; c++) {
        mins[c] = _mm256_set1_ps(std::numeric_limits<float>::max());
        maxs[c] = _mm256_set1_ps(std::numeric_limits<float>::min());
    }

    /* .LBB1_3:  // using 8 codebooks; this loop is like 40% of the total time
        mov     esi, ecx
        and     esi, -128
        vmovaps ymm5, ymmword ptr [rbx + 4*rsi + 32]
        vmovaps ymm6, ymmword ptr [rbx + 4*rsi + 96]
        vmovaps ymm7, ymmword ptr [rbx + 4*rsi + 160]
        vmovaps ymm9, ymmword ptr [rbx + 4*rsi + 224]
        vminps  ymm0, ymm15, ymmword ptr [rbx + 4*rsi]
        vminps  ymm15, ymm0, ymm5
        vminps  ymm0, ymm8, ymmword ptr [rbx + 4*rsi + 64]
        vminps  ymm8, ymm0, ymm6
        vminps  ymm0, ymm14, ymmword ptr [rbx + 4*rsi + 128]
        vminps  ymm10, ymm10, ymmword ptr [rbx + 4*rsi + 192]
        vminps  ymm14, ymm0, ymm7
        vminps  ymm10, ymm10, ymm9
        vminps  ymm0, ymm4, ymmword ptr [rbx + 4*rsi + 256]
        vmovaps ymm11, ymmword ptr [rbx + 4*rsi + 288]
        vminps  ymm4, ymm0, ymm11
        vminps  ymm0, ymm3, ymmword ptr [rbx + 4*rsi + 320]
        vmovaps ymm12, ymmword ptr [rbx + 4*rsi + 352]
        vminps  ymm3, ymm0, ymm12
        vminps  ymm0, ymm2, ymmword ptr [rbx + 4*rsi + 384]
        vmovaps ymm13, ymmword ptr [rbx + 4*rsi + 416]
        vminps  ymm2, ymm0, ymm13
        vminps  ymm1, ymm1, ymmword ptr [rbx + 4*rsi + 448]
        vmovaps ymm0, ymmword ptr [rbx + 4*rsi + 480]
        vminps  ymm1, ymm1, ymm0
        sub     rcx, -128
        dec     rax
        jne     .LBB1_3
     */
    // compute min and max vals for each codebook
    for (int r = 0; r < nrow_blocks; r++) {
        // new set of rows; reset read ptrs
        for (int rr = 0; rr < RowTileSz; rr++) {
            auto row = (r * RowTileSz) + rr;
            in_ptrs[rr] = luts_f32 + (row * row_stride);
        }
        // update all the mins and maxes
        for (int c = 0; c < ncodebooks; c++) {
            for (int rr = 0; rr < RowTileSz; rr++) {
                auto vlut_stripe0 = _mm256_load_ps(in_ptrs[rr]);
                in_ptrs[rr] += packet_width;
                auto vlut_stripe1 = _mm256_load_ps(in_ptrs[rr]);
                in_ptrs[rr] += packet_width;

                mins[c] = _mm256_min_ps(mins[c], vlut_stripe0);
                mins[c] = _mm256_min_ps(mins[c], vlut_stripe1);
                maxs[c] = _mm256_max_ps(mins[c], vlut_stripe0);
                maxs[c] = _mm256_max_ps(mins[c], vlut_stripe1);
            }
        }
    }
    for (int row = nrows_round; row < nrows; row++) { // handle trailing rows
        auto in_ptr = luts_f32 + (row * row_stride);
        for (int c = 0; c < ncodebooks; c++) {
            auto vlut_stripe0 = _mm256_load_ps(in_ptr);
            in_ptr += packet_width;
            auto vlut_stripe1 = _mm256_load_ps(in_ptr);
            in_ptr += packet_width;

            mins[c] = _mm256_min_ps(mins[c], vlut_stripe0);
            mins[c] = _mm256_min_ps(mins[c], vlut_stripe1);
            maxs[c] = _mm256_max_ps(mins[c], vlut_stripe0);
            maxs[c] = _mm256_max_ps(mins[c], vlut_stripe1);
        }
    }
    _compute_offsets_scale_from_mins_maxs(
        mins, maxs, ncodebooks, out_offsets, out_offset_sum, out_scale);
}

template<int ncodebooks, int RowTileSz=1>
void quantize_luts(const float* luts_f32, int nrows,
                   const float* offsets,
                   float scaleby, uint8_t* out_luts)
{
    static constexpr int lut_sz = 16;
    static constexpr int codebook_group_sz = 2; // 4 f32 luts -> 1 epu8 lut
    static constexpr int packet_width = 8; // objs per simd register
    static constexpr int nstripes = lut_sz / packet_width;
    static constexpr int ncodebook_groups = ncodebooks / codebook_group_sz;
    static_assert(ncodebooks % codebook_group_sz == 0,
        "Number of codebooks must be a multiple of 2");
    assert(nrows % RowTileSz == 0);

    auto row_stride = ncodebooks * lut_sz;
    auto nrow_blocks = RowTileSz > 1 ? nrows / RowTileSz : 0;
    auto nrows_round = nrow_blocks * RowTileSz;

    const float* in_ptrs[RowTileSz];
    uint8_t* out_ptrs[RowTileSz];

    // if luts constant, just zero the output and return
    if (scaleby <= 0.f) {
        size_t total_sz = nrows * ncodebooks * lut_sz;
        for (size_t i = 0; i < total_sz; i++) {
            *out_luts++ = 0;
        }
        return;
    }

    /* inner loop gets unrolled to this:
     vmovaps        ymm10, ymmword ptr [rbx + 4*rdx + 384]
     vfmadd132ps    ymm10, ymm4, ymm1 # ymm10 = (ymm10 * ymm1) + ymm4
     vmovaps        ymm11, ymmword ptr [rbx + 4*rdx + 416]
     vfmadd132ps    ymm11, ymm4, ymm1 # ymm11 = (ymm11 * ymm1) + ymm4
     vcvtps2dq      ymm10, ymm10
     vmovaps        ymm12, ymmword ptr [rbx + 4*rdx + 448]
     vfmadd132ps    ymm12, ymm5, ymm1 # ymm12 = (ymm12 * ymm1) + ymm5
     vcvtps2dq      ymm11, ymm11
     vmovaps        ymm13, ymmword ptr [rbx + 4*rdx + 480]
     vfmadd132ps    ymm13, ymm5, ymm1 # ymm13 = (ymm13 * ymm1) + ymm5
     vcvtps2dq      ymm12, ymm12
     vpackssdw      ymm10, ymm10, ymm11
     vcvtps2dq      ymm11, ymm13
     vpackssdw      ymm11, ymm12, ymm11
     vpackuswb      ymm10, ymm10, ymm11
     vpermd         ymm10, ymm2, ymm10
     vmovdqa        ymmword ptr [r14 + rdx + 96], ymm10
     */
    // given offsets and overall scale, actually quantize luts; this is
    // basically the same as the first 2 loops that pull out the offsets
    __m256i luts_epi16[RowTileSz][codebook_group_sz];
    auto vmulby = _mm256_set1_ps(scaleby);
    for (int r = 0; r < nrow_blocks; r++) {
        // new set of rows; reset read and write ptrs
        for (int rr = 0; rr < RowTileSz; rr++) {
            auto row = (r * RowTileSz) + rr;
            in_ptrs[rr] = luts_f32 + (row * row_stride);
            out_ptrs[rr] = out_luts + (row * row_stride);
        }
        // for each column group, col in group, row in rowgroup
        for (int g = 0; g < ncodebook_groups; g++) {
            for (int gg = 0; gg < codebook_group_sz; gg++) {
                auto c = (g * codebook_group_sz) + gg;
                // auto fma_offset = offsets[c] * scaleby;
                auto fma_offset = offsets[c];
                auto voffset = _mm256_set1_ps(fma_offset);  // p5

                for (int rr = 0; rr < RowTileSz; rr++) {
                    auto vlut_f32_0 = _mm256_load_ps(in_ptrs[rr]);
                    in_ptrs[rr] += packet_width;
                    auto vlut_f32_1 = _mm256_load_ps(in_ptrs[rr]);
                    in_ptrs[rr] += packet_width;

                    // fmas on p01; cvtps on p1
                    vlut_f32_0 = _mm256_fmsub_ps(vlut_f32_0, vmulby, voffset);
                    vlut_f32_1 = _mm256_fmsub_ps(vlut_f32_1, vmulby, voffset);
                    auto vlut_epi32_0 = _mm256_cvtps_epi32(vlut_f32_0);
                    auto vlut_epi32_1 = _mm256_cvtps_epi32(vlut_f32_1);

                    // the tricky part here is that we have to buffer luts from
                    // two consecutive columns to get a full epi32 vector
                    luts_epi16[rr][gg] = _mm256_packs_epi32( // p5
                        vlut_epi32_0, vlut_epi32_1);
                }
            }
            // combine epi16 luts from the 2 cols into 1 epu8 lut and store it
            for (int rr = 0; rr < RowTileSz; rr++) {
                auto lut0 = luts_epi16[rr][0];
                auto lut1 = luts_epi16[rr][1];
                auto lut = _mm256_packus_epi16(lut0, lut1); // p5
                lut = _mm256_permutevar8x32_epi32(  // p5
                    lut, _mm256_setr_epi32(0,4, 1,5, 2,6, 3,7)); // p5
                _mm256_store_si256((__m256i*)out_ptrs[rr], lut);
                out_ptrs[rr] += 32;
            }
        }
    }
    for (int row = nrows_round; row < nrows; row++) { // handle trailing rows
        auto in_ptr = luts_f32 + (row * row_stride);
        auto out_ptr = out_luts + (row * row_stride);
        for (int g = 0; g < ncodebook_groups; g++) {
            for (int gg = 0; gg < codebook_group_sz; gg++) {
                auto c = (g * codebook_group_sz) + gg;
                // auto fma_offset = offsets[c] * scaleby;
                auto fma_offset = offsets[c];
                auto voffset = _mm256_set1_ps(fma_offset);

                auto vlut_f32_0 = _mm256_load_ps(in_ptr);
                in_ptr += packet_width;
                auto vlut_f32_1 = _mm256_load_ps(in_ptr);
                in_ptr += packet_width;

                vlut_f32_0 = _mm256_fmsub_ps(vlut_f32_0, vmulby, voffset);
                vlut_f32_1 = _mm256_fmsub_ps(vlut_f32_1, vmulby, voffset);
                // vlut_f32_0 = fma(vlut_f32_0, vmulby, voffset);
                // vlut_f32_1 = fma(vlut_f32_1, vmulby, voffset);
                auto vlut_epi32_0 = _mm256_cvtps_epi32(vlut_f32_0);
                auto vlut_epi32_1 = _mm256_cvtps_epi32(vlut_f32_1);

                luts_epi16[0][gg] = _mm256_packs_epi32(
                    vlut_epi32_0, vlut_epi32_1);
            }
            auto lut0 = luts_epi16[0][0];
            auto lut1 = luts_epi16[0][1];
            auto lut = _mm256_packus_epi16(lut0, lut1);
            lut = _mm256_permutevar8x32_epi32(
                lut, _mm256_setr_epi32(0,4, 1,5, 2,6, 3,7));
            _mm256_store_si256((__m256i*)out_ptr, lut);
            out_ptr += 32;
        }
    }
}
template<int RowTileSz=1>
void quantize_luts(const float* luts_f32, int nrows, int ncodebooks,
                   const float* offsets, float scaleby, uint8_t* out_luts)
{
    switch (ncodebooks) {
        case 2: quantize_luts<2, RowTileSz>(
            luts_f32, nrows, offsets, scaleby, out_luts); break;
        case 4: quantize_luts<4, RowTileSz>(
            luts_f32, nrows, offsets, scaleby, out_luts); break;
        case 8: quantize_luts<8, RowTileSz>(
            luts_f32, nrows, offsets, scaleby, out_luts); break;
        case 16: quantize_luts<16, RowTileSz>(
            luts_f32, nrows, offsets, scaleby, out_luts); break;
        case 32: quantize_luts<32, RowTileSz>(
            luts_f32, nrows, offsets, scaleby, out_luts); break;
        case 64: quantize_luts<64, RowTileSz>(
            luts_f32, nrows, offsets, scaleby, out_luts); break;
        case 128: quantize_luts<128, RowTileSz>(
            luts_f32, nrows, offsets, scaleby, out_luts); break;
    }
}


/* https://godbolt.org/z/5zEJ6q
 * ya, inner loop is crap right now:
 *
 * vpbroadcastd ymm2, dword ptr [r15 + 4*rax]
 * vmovaps      ymm3, ymmword ptr [rbx]
 * vmovaps      ymm4, ymmword ptr [rbx + 32]
 * vfmadd231ps  ymm1, ymm3, ymm2 # ymm1 = (ymm3 * ymm2) + ymm1
 * add          rbx, 64
 * vfmadd231ps  ymm0, ymm4, ymm2 # ymm0 = (ymm4 * ymm2) + ymm0
 * inc          rax
 * cmp          rbp, rax
 * jne          .LBB0_7
 */
void mithral_lut_v1(const float* q, int len, int ncodebooks,
                 const float* centroids, uint8_t* out)
{
    static constexpr int lut_sz = 16;
    static constexpr int packet_width = 8; // objs per simd register
    static constexpr int nstripes = lut_sz / packet_width;
    static constexpr int j_tile_sz = 4;

    __m256 accumulators[nstripes];
    __m256i dists_uint16_0 = _mm256_undefined_si256();

    for (int m = 0; m < ncodebooks; m++) { // for each codebook
        for (int i = 0; i < nstripes; i++) {
            accumulators[i] = _mm256_setzero_ps();
        }
        auto round_len = len - (len % j_tile_sz);
        for (int j = 0; j < round_len; j += j_tile_sz) {
            // for (int jj = j; jj < MIN(len, j + j_tile_sz); jj++) { // for each dim in subvect
            for (int jj = j; jj < j + j_tile_sz; jj++) { // for each dim in subvect
                auto q_broadcast = _mm256_set1_ps(q[jj]);
                for (int i = 0; i < nstripes; i++) { // for upper 8, lower 8 floats
                    auto centroids_col = _mm256_load_ps(centroids);
                    centroids += packet_width;

                    accumulators[i] = fma(
                        q_broadcast, centroids_col, accumulators[i]);
                }
            }
        }
        if (round_len != len) { // so that above loop can unroll
            for (int jj = round_len; jj < len; jj++) {
                auto q_broadcast = _mm256_set1_ps(q[jj]);
                for (int i = 0; i < nstripes; i++) { // for upper 8, lower 8 floats
                    auto centroids_col = _mm256_load_ps(centroids);
                    centroids += packet_width;

                    accumulators[i] = fma(
                        q_broadcast, centroids_col, accumulators[i]);
                }
            }
        }


        // TODO write out float vals into a tmp array, then come up with
        // quantization params based on actual values; will also have to
        // return offset and scale used


        // convert the floats to ints
        auto dists_int32_low = _mm256_cvtps_epi32(accumulators[0]);
        auto dists_int32_high = _mm256_cvtps_epi32(accumulators[1]);

        // because we saturate to uint8s, we only get 32 objs to write after
        // two 16-element codebooks
        auto dists_uint16 = _mm256_packus_epi32(dists_int32_low, dists_int32_high);
        if (m % 2) {
            // if odd-numbered codebook, combine dists from previous codebook

            // undo the weird shuffling caused by the pack operations
            auto dists = packed_epu16_to_unpacked_epu8(
                dists_uint16_0, dists_uint16);

            _mm256_store_si256((__m256i*)out, dists);
            out += 32;
        } else {
            // if even-numbered codebook, just store these dists to be combined
            // when we look at the next codebook
            dists_uint16_0 = dists_uint16;
        }
    }
}



void mithral_lut_v1(const float* Q, int nrows, int ncols, int ncodebooks,
                 const float* centroids, uint8_t* out)
{
    auto in_ptr = Q;
    uint8_t* lut_out_ptr = (uint8_t*)out;
    for (int i = 0; i < nrows; i++) {
        mithral_lut_v1(in_ptr, ncols, ncodebooks, centroids, lut_out_ptr);
        in_ptr += ncols;
        lut_out_ptr += 16 * ncodebooks;
    }
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
inline void _mithral_scan_tiled(const uint8_t* codes,
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
void mithral_scan_tiled(const uint8_t* codes, int nrows, int ncodebooks,
    int noutputs, const int8_t* luts, int16_t* out, bool add_to_output=false)
{
    // TODO figure out optimal amount of tiling and reduce to this case, then
    // clean up leftover input/output columns

    auto ncols = ncodebooks / 4;
    if (ncols == 1) {
        static constexpr int NCodebooks = 4;
        // with 1 input col 6 outputs: https://godbolt.org/z/dLLfV8
        // if (noutputs % 6 == 0) {
        //     _mithral_scan_tiled<1, 6, UpcastEvery, NCodebooks>(
        //         codes, nrows, ncodebooks, noutputs, luts, out); return;
        // } else if (noutputs % 5 == 0) {
        if (noutputs % 5 == 0 && UpcastEvery > 2) {
            _mithral_scan_tiled<1, 5, UpcastEvery, NCodebooks>(
                codes, nrows, ncodebooks, noutputs, luts, out); return;
        } else if (noutputs % 4 == 0) {
            _mithral_scan_tiled<1, 4, UpcastEvery, NCodebooks>(
                codes, nrows, ncodebooks, noutputs, luts, out); return;
        } else if (noutputs % 3 == 0) {
            _mithral_scan_tiled<1, 3, UpcastEvery, NCodebooks>(
                codes, nrows, ncodebooks, noutputs, luts, out); return;
        } else if (noutputs % 2 == 0) {
            _mithral_scan_tiled<1, 2, UpcastEvery, NCodebooks>(
                codes, nrows, ncodebooks, noutputs, luts, out); return;
        } else {
            _mithral_scan_tiled<1, 1, UpcastEvery, NCodebooks>(
                codes, nrows, ncodebooks, noutputs, luts, out); return;
        }
    } else if (ncols == 2 && (noutputs % 2) == 0) {
        static constexpr int NCodebooks = 8;
        _mithral_scan_tiled<2, 2, UpcastEvery, NCodebooks>(
                codes, nrows, ncodebooks, noutputs, luts, out); return;
    // } else if (ncols % 6 == 0) {
    //     _mithral_scan_tiled<6, 1, UpcastEvery>(
    //             codes, nrows, ncodebooks, noutputs, luts, out); return;
    } else if (ncols % 5 == 0 && UpcastEvery > 2) {
        _mithral_scan_tiled<5, 1, UpcastEvery>(
                codes, nrows, ncodebooks, noutputs, luts, out); return;
    } else if (ncols % 4 == 0 && UpcastEvery > 2) {
        _mithral_scan_tiled<4, 1, UpcastEvery>(
                codes, nrows, ncodebooks, noutputs, luts, out); return;
    } else if (ncols % 3 == 0 && UpcastEvery > 2) {
        _mithral_scan_tiled<3, 1, UpcastEvery>(
                codes, nrows, ncodebooks, noutputs, luts, out); return;
    } else if (ncols % 2 == 0) {
        _mithral_scan_tiled<2, 1, UpcastEvery>(
                codes, nrows, ncodebooks, noutputs, luts, out); return;
    } else {
        _mithral_scan_tiled<1, 1, UpcastEvery>(
                codes, nrows, ncodebooks, noutputs, luts, out); return;
    }
}

template<int x> struct _log2_of_power_of_2 {
    // static_assert(x <= 512, "Only x <= 512 supported because I'm being lazy");
    static constexpr uint8_t value = (x == 1   ? 0 :
                                      x == 2   ? 1 :
                                      x == 4   ? 2 :
                                      x == 8   ? 3 :
                                      x == 16  ? 4 :
                                      x == 32  ? 5 :
                                      x == 64  ? 6 :
                                      x == 128 ? 7 :
                                      x == 256 ? 8 :
                                      x == 512 ? 9 : 255);
    static_assert(value != 255, "x must be one of 2^{1, 2, 4, 8, ..., 512}");
};

static constexpr bool is_power_of_2(int64_t x) {
    return (x & (x - 1)) == 0 && x > 0;
}


// TODO have output type be templated and intelligent upcast or
// whatever based on the type
//
// https://godbolt.org/z/cP80FF
template<int NBytes, int UpcastEvery=16, bool Force16BitOutput=true>
void mithral_scan(const uint8_t* codes, int64_t nblocks, const uint8_t* luts,
                   uint8_t* dists_out)
{
    static_assert(NBytes > 0, "Code length <= 0 is not valid");
    static_assert(UpcastEvery % 2 == 0, "UpcastEvery must be even");
    static_assert(UpcastEvery >= 2, "UpcastEvery must be >= 2");
    static_assert(UpcastEvery <= 16, "UpcastEvery must be <= 16");
    // static_assert(UpcastEvery == 2 || UpcastEvery == 4 || UpcastEvery == 8, "UpcastEvery must be <= 16");
    static constexpr int ncodebooks = 2 * NBytes;
    static constexpr int ncols = NBytes;
    static constexpr int actually_upcast_every = MIN(UpcastEvery, ncodebooks);
    // static_assert(actually_upcast_every == 2 || UpcastEvery == 4 || UpcastEvery == 8, "UpcastEvery must be <= 16");
    static constexpr int colgroup_sz = actually_upcast_every / 2;
    // static_assert(_log2_of_power_of_2<colgroup_sz>::value != 255,
    static_assert(is_power_of_2(colgroup_sz),
        "Invalid number of columns to unroll at once");
    static constexpr int ncolgroups = ncols / colgroup_sz;
    static_assert(colgroup_sz <= ncodebooks, "WTF, did some math wrong");
    static_assert(ncols % colgroup_sz == 0,
        "Size of column group must evenly number of columns");

    // unpack 16B luts into 32B registers
    __m256i luts_ar[ncodebooks];
    auto lut_ptr = luts;
    for (uint8_t j = 0; j < NBytes; j++) {
        auto both_luts = load_si256i(lut_ptr);
        lut_ptr += 32;
        auto lut0 = _mm256_permute2x128_si256(both_luts, both_luts, 0 + (0 << 4));
        auto lut1 = _mm256_permute2x128_si256(both_luts, both_luts, 1 + (1 << 4));
        luts_ar[2 * j] = lut0;
        luts_ar[2 * j + 1] = lut1;
    }

    for (int64_t i = 0; i < nblocks; i++) {
        // used if ncolgroups > 1, in which case we have to upcast
//        auto totals_dbg = _mm256_setzero_si256();
        auto totals_0_15 = _mm256_setzero_si256();
        auto totals_16_31 = _mm256_setzero_si256();

        auto low_4bits_mask = _mm256_set1_epi8(0x0F); // not static so sits in reg

        for (int g = 0; g < ncolgroups; g++) {

            __m256i avg_prev1 = _mm256_undefined_si256();
            __m256i avg_prev2 = _mm256_undefined_si256();
            __m256i avg_prev4 = _mm256_undefined_si256();
            __m256i avg_prev8 = _mm256_undefined_si256();
            __m256i avg_prev16 = _mm256_undefined_si256();

            #pragma unroll
            for (int gg = 0; gg < colgroup_sz; gg++) {
                auto j = (g * colgroup_sz) + gg;

                auto x_col = load_si256i(codes);
                // auto x_col = stream_load_si256i(codes);
                codes += 32;

                auto lut_low = luts_ar[2 * j];
                auto lut_high = luts_ar[2 * j + 1];

                auto x_low = _mm256_and_si256(x_col, low_4bits_mask);
                auto x_shft = _mm256_srli_epi16(x_col, 4);
                auto x_high = _mm256_and_si256(x_shft, low_4bits_mask);

                auto dists_low = _mm256_shuffle_epi8(lut_low, x_low);
                auto dists_high = _mm256_shuffle_epi8(lut_high, x_high);

                auto avgs = _mm256_avg_epu8(dists_low, dists_high);

                // update running averages; this is messy because if you
                // need to current and previous average to be over the same
                // number of values, or else it's a weird weighted average
                // instead of a true average
                // note that we need to use inline asm to get the right
                // instruction here on my machine for unclear reasons
                if (gg % 16 == 15) {
                    auto new_avg_prev2 = avg_epu8(avg_prev1, avgs);
                    auto new_avg_prev4 = avg_epu8(avg_prev2, new_avg_prev2);
                    auto new_avg_prev8 = avg_epu8(avg_prev4, new_avg_prev4);
                    avg_prev16 = avg_epu8(avg_prev8, new_avg_prev8);
                }
                // if ((gg + 1) % 8 == 0) {
                if (gg % 8 == 7) {
                    auto new_avg_prev2 = avg_epu8(avg_prev1, avgs);
                    auto new_avg_prev4 = avg_epu8(avg_prev2, new_avg_prev2);
                    avg_prev8 = avg_epu8(avg_prev4, new_avg_prev4);
                }
                // if ((gg + 1) % 4 == 0) {
                if (gg % 4 == 3) {
                    auto new_avg_prev2 = avg_epu8(avg_prev1, avgs);
                    avg_prev4 = avg_epu8(avg_prev2, new_avg_prev2);
                }
                // if ((gg + 1) % 2 == 0) {
                if (gg % 2 == 1) {
                    avg_prev2 = avg_epu8(avg_prev1, avgs);
                } else {
                    avg_prev1 = avgs;
                }
            }
            auto group_avg = colgroup_sz == 1  ? avg_prev1 :
                             colgroup_sz == 2  ? avg_prev2 :
                             colgroup_sz == 4  ? avg_prev4 :
                             colgroup_sz == 8  ? avg_prev8 :
                             avg_prev16;

            if (ncolgroups == 1 && !Force16BitOutput) { // write out 8b values
                _mm256_store_si256((__m256i*)dists_out, group_avg);
                dists_out += 32;
            } else {
                auto avgs_0_15 = _mm256_cvtepi8_epi16(
                _mm256_extracti128_si256(group_avg, 0));
                auto avgs_16_31 = _mm256_cvtepi8_epi16(
                    _mm256_extracti128_si256(group_avg, 1));
                totals_0_15 = _mm256_add_epi16(totals_0_15, avgs_0_15);
                totals_16_31 = _mm256_add_epi16(totals_16_31, avgs_16_31);
            }
        }
        // if (true) {
        if (ncolgroups > 1 || Force16BitOutput) {
            _mm256_store_si256((__m256i*)(dists_out + 0), totals_0_15);
            _mm256_store_si256((__m256i*)(dists_out + 32), totals_16_31);
            // _mm256_stream_si256((__m256i*)(dists_out + 0), totals_0_15);
            // _mm256_stream_si256((__m256i*)(dists_out + 32), totals_16_31);
            dists_out += 64;
        }
    }
}

template<int UpcastEvery=8>
void mithral_scan(const uint8_t* codes, int64_t nblocks, int ncodebooks,
                  const uint8_t* luts, uint8_t* dists_out)
{
    uint8_t* out = dists_out;
    switch(ncodebooks) {
        case 4: mithral_scan<2, UpcastEvery>(codes, nblocks, luts, out); break;
        case 8: mithral_scan<4, UpcastEvery>(codes, nblocks, luts, out); break;
        case 16: mithral_scan<8, UpcastEvery>(codes, nblocks, luts, out); break;
        case 32: mithral_scan<16, UpcastEvery>(codes, nblocks, luts, out); break;
        case 64: mithral_scan<32, UpcastEvery>(codes, nblocks, luts, out); break;
        case 128: mithral_scan<64, UpcastEvery>(codes, nblocks, luts, out); break;
    }
}

void mithral_scan(const uint8_t* codes, int64_t nblocks, int ncodebooks,
                  int noutputs, const uint8_t* luts, uint8_t* dists_out)
{
    static constexpr int block_nrows = 32;
    static constexpr int lut_sz = 16;
    auto out_ptr = dists_out;
    auto out_stride = nblocks * block_nrows;
    auto lut_ptr = luts;
    auto lut_stride = ncodebooks * lut_sz;

    for (int i = 0; i < noutputs; i++) {
        mithral_scan(codes, nblocks, ncodebooks, lut_ptr, out_ptr);
        out_ptr += out_stride;
        lut_ptr += lut_stride;
    }
}

} // anon namespace
#endif // __MITHRAL_HPP

//
//  mithral.cpp
//  Bolt
//
//  Created by DB on 12/3/19.
//  Copyright © 2019 D Blalock. All rights reserved.
//

#include "mithral.hpp"


// ================================================================ encode

void mithral_encode(
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

    // sanity check splits
    auto total_nsplits = ncodebooks * nsplits_per_codebook;
    auto maxdim = splitdims[0];
    auto mindim = splitdims[0];
    for (int i = 1; i < total_nsplits; i++) {
        maxdim = MAX(maxdim, splitdims[i]);
        mindim = MIN(maxdim, splitdims[i]);
    }
    assert(mindim >= 0);
    assert(maxdim < ncols);

    size_t x_col_stride = nrows;
    size_t out_col_stride = nrows;
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

// version with int16 data
void mithral_encode(const int16_t* X, int64_t nrows, int ncols,
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
    const int16_t* x_ptrs[nsplits_per_codebook];
    __m256i current_vsplitval_luts[nsplits_per_codebook];
    uint8_t current_shifts[nsplits_per_codebook];
    __m256i current_voffsets[nsplits_per_codebook];

    int split_idx = 0;
    for (int c = 0; c < ncodebooks; c++) {
        // compute input and output column starts
        uint8_t* out_ptr;
        size_t out_block_stride;
        out_ptr = out + (simd_vec_sz * c);
        /// XXX this will be off by a factor of 2 with a packed layout
        out_block_stride = block_nrows * ncodebooks;
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

// version with int8 data
void mithral_encode(const int8_t* X, int64_t nrows, int ncols,
    const uint32_t* splitdims, const int8_t* all_splitvals,
    int ncodebooks, uint8_t* out)
    // const float* scales, int ncodebooks, uint8_t* out)
{
    static constexpr int block_nrows = 32;
    static constexpr int simd_vec_sz = 32;
    static constexpr int nsplits_per_codebook = 4;
    // static constexpr int ncodebooks_per_group = 2;
    static constexpr int vals_per_split = 1 << nsplits_per_codebook; // 16
    const int64_t nblocks = ceil(nrows / (double)block_nrows);
    assert(nrows % block_nrows == 0); // TODO remove this constraint

    size_t x_col_stride = nrows;
    size_t out_col_stride = nrows;
    size_t splitval_luts_stride = vals_per_split;
    const int8_t* x_ptrs[nsplits_per_codebook];
    __m256i current_vsplitval_luts[nsplits_per_codebook];

    int split_idx = 0;
    for (int c = 0; c < ncodebooks; c++) {
        // compute input and output column starts
        uint8_t* out_ptr;
        size_t out_block_stride;
        out_ptr = out + (simd_vec_sz * c);
        out_block_stride = block_nrows * ncodebooks;
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
void mithral_encode(const int8_t* X, int64_t nrows, int ncols,
    const uint32_t* splitdims, const int8_t* all_splitvals,
    const void* shifts_unused, const void* offsets_unused,
    int ncodebooks, uint8_t* out)
{
    mithral_encode(
        X, nrows, ncols, splitdims, all_splitvals, ncodebooks, out);
}

void zip_bolt_colmajor(const uint8_t* codes_in, int64_t nrows,
                       uint32_t ncodebooks, uint8_t* codes_out)
{
    // if (ncodebooks % 64 == 0) {
    //     zip_bolt_colmajor<64>(codes_in, nrows, ncodebooks, codes_out); return;
    // }
    // if (ncodebooks % 16 == 0) {
    //     zip_bolt_colmajor<16>(codes_in, nrows, ncodebooks, codes_out); return;
    // }
    if (ncodebooks % 8 == 0) {
        zip_bolt_colmajor<8>(codes_in, nrows, ncodebooks, codes_out); return;
    }
    if (ncodebooks % 4 == 0) {
        zip_bolt_colmajor<4>(codes_in, nrows, ncodebooks, codes_out); return;
    }
    zip_bolt_colmajor<2>(codes_in, nrows, ncodebooks, codes_out);
}

// ================================================================ lut

void dense_lut_f32_fused(const float* Q, int nrows, int ncols, int ncodebooks,
    // const float* centroids, float* out)
    // SELF: above args are fast, while ones below make it like 2x slower
    const float* centroids, float*__restrict__ out_offsets, float& out_offset_sum,
    float& out_scale, float*__restrict__ out)
{
    static constexpr int codebook_tile_sz = 2;
    static constexpr int row_tile_sz = 2;
    // assert(ncodebooks % codebook_tile_sz == 0);
    // assert(nrows % row_tile_sz == 0);  // TODO handle trailing rows in fused func
    // // ^ has to be handled in fused func so that mins/maxs include trailing
    // // rows; for now, just require padding of query with zero rows; way to
    // // do this is by just having below func take in mins and maxes

    dense_lut_f32_fused<codebook_tile_sz, row_tile_sz>(
            Q, nrows, ncols, ncodebooks, centroids,
            out_offsets, out_offset_sum, out_scale, out);
}


void dense_lut_f32(const float* Q, int nrows, int ncols, int ncodebooks,
                 const float* centroids, float* out)
{
    static constexpr int lut_sz = 16;
    static constexpr int CodebookTileSz = 2;
    static constexpr int RowTileSz = 2;
    assert(ncodebooks % CodebookTileSz == 0);

    // handle most rows
    auto nrows_trailing = nrows % RowTileSz;
    auto nrows_round = nrows - nrows_trailing;
    if (nrows_round > 0) {
        dense_lut_f32<CodebookTileSz, RowTileSz>(
            Q, nrows_round, ncols, ncodebooks,
            centroids, out);
    }
    // handle trailing rows
    auto q_row_stride = ncols;
    Q += q_row_stride * nrows_round;
    auto out_row_stride = ncodebooks * lut_sz;
    out += out_row_stride * nrows_round;

    // NOTE: if we hardcode this to 1 instead of having a switch, or just
    // rm handling of the trailing rows entirely, code is twice as fast
    if (nrows_trailing > 0) {
        dense_lut_f32<CodebookTileSz, 1>(
                Q, nrows_trailing, ncols, ncodebooks, centroids, out);
    }

    // switch(nrows_trailing) {
    //     case 0: break;
    //     case 1: dense_lut_f32<CodebookTileSz, 1>(
    //         Q, nrows_trailing, ncols, ncodebooks, centroids, out); break;
    //     case 2: dense_lut_f32<CodebookTileSz, 2>(
    //         Q, nrows_trailing, ncols, ncodebooks, centroids, out); break;
    //     case 3: dense_lut_f32<CodebookTileSz, 3>(
    //         Q, nrows_trailing, ncols, ncodebooks, centroids, out); break;
    // }
}

void sparse_lut_f32(const float* Q, int nrows, int ncols, int ncodebooks,
                    const float* centroids,
                    const int* idxs, int nnz_per_centroid, float* out)
{
    static constexpr int lut_sz = 16;
    static constexpr int CodebookTileSz = 2;
    static constexpr int RowTileSz = 2;
    assert(ncodebooks % CodebookTileSz == 0);

    // handle most rows
    auto nrows_trailing = nrows % RowTileSz;
    auto nrows_round = nrows - nrows_trailing;
    if (nrows_round > 0) {
        sparse_lut_f32<CodebookTileSz, RowTileSz>(
            Q, nrows_round, ncols, ncodebooks,
            centroids, idxs, nnz_per_centroid, out);
    }
    // handle trailing rows
    auto q_row_stride = ncols;
    Q += q_row_stride * nrows_round;
    auto out_row_stride = ncodebooks * lut_sz;
    out += out_row_stride * nrows_round;

    // NOTE: if we hardcode this to 1 instead of having a switch, or just
    // rm handling of the trailing rows entirely, code is twice as fast
    if (nrows_trailing > 0) {
        sparse_lut_f32<CodebookTileSz, 1>(
                Q, nrows_trailing, ncols, ncodebooks, centroids,
                idxs, nnz_per_centroid, out);
    }

    // switch(nrows_trailing) {
    //     case 0: break;
    //     case 1: dense_lut_f32<CodebookTileSz, 1>(
    //         Q, nrows_trailing, ncols, ncodebooks, centroids, out); break;
    //     case 2: dense_lut_f32<CodebookTileSz, 2>(
    //         Q, nrows_trailing, ncols, ncodebooks, centroids, out); break;
    //     case 3: dense_lut_f32<CodebookTileSz, 3>(
    //         Q, nrows_trailing, ncols, ncodebooks, centroids, out); break;
    // }
}


void mithral_lut_dense(const float* Q, int nrows, int ncols, int ncodebooks,
    const float* centroids, float& out_offset_sum, float& out_scale,
    float*__restrict__ tmp_lut_f32, uint8_t* out)
{
    float tmp_offsets[ncodebooks];
    //
    // unfused stats computation
    //
    // dense_lut_f32(Q, nrows, ncols, ncodebooks, centroids, tmp_lut_f32);
    // mithral_learn_lut_offsets_scales(tmp_lut_f32, nrows, ncodebooks,
    //     tmp_offsets, out_offset_sum, out_scale);

    // fusing is like 3% faster with D=128,C=16, and D=32,C=16; so might
    // as well fuse, but sparse lut funcs don't need to worry about fusing
    // in mins/maxs computation; EDIT, well for N=C=8, about 15% faster
    dense_lut_f32_fused(
        Q, nrows, ncols, ncodebooks, centroids,
        tmp_offsets, out_offset_sum, out_scale, tmp_lut_f32);

    quantize_luts(tmp_lut_f32, nrows, ncodebooks, tmp_offsets, out_scale, out);
}

void mithral_lut_sparse(const float* Q, int nrows, int ncols, int ncodebooks,
    const float* centroids, const int* idxs, int nnz_per_centroid,
    float& out_offset_sum, float& out_scale,
    float*__restrict__ tmp_lut_f32, uint8_t* out)
{
    //
    // unfused stats computation
    //
    sparse_lut_f32(Q, nrows, ncols, ncodebooks, centroids,
                   idxs, nnz_per_centroid, tmp_lut_f32);
    float tmp_offsets[ncodebooks];
    mithral_learn_lut_offsets_scales(tmp_lut_f32, nrows, ncodebooks,
        tmp_offsets, out_offset_sum, out_scale);
    quantize_luts(tmp_lut_f32, nrows, ncodebooks, tmp_offsets, out_scale, out);
}

// ================================================================ scan

void mithral_scan(const uint8_t* codes, int64_t nblocks, int ncodebooks,
                  int noutputs, const uint8_t* luts, uint8_t* dists_out)
{
    // mithral_scan<128, 2>(codes, nblocks, ncodebooks, noutputs, luts, dists_out);
    mithral_scan<16, 2>(codes, nblocks, ncodebooks, noutputs, luts, dists_out);
    // if (ncodebooks >= 4) {
    //     mithral_scan<128, 2>(codes, nblocks, ncodebooks, noutputs, luts, dists_out);
    // } else {
    //     mithral_scan<128, 1>(codes, nblocks, ncodebooks, noutputs, luts, dists_out);
    // }
}

// void mithral_scan_notile(const uint8_t* codes, int64_t nblocks, int ncodebooks,
// // void mithral_scan(const uint8_t* codes, int64_t nblocks, int ncodebooks,
//                   int noutputs, const uint8_t* luts, uint8_t* dists_out)
// {
//     static constexpr int block_nrows = 32;
//     static constexpr int lut_sz = 16;
//     auto out_ptr = dists_out;
//     auto out_stride = nblocks * block_nrows;
//     auto lut_ptr = luts;
//     auto lut_stride = ncodebooks * lut_sz;

//     for (int i = 0; i < noutputs; i++) {
//         mithral_scan(codes, nblocks, ncodebooks, lut_ptr, out_ptr);
//         out_ptr += out_stride;
//         lut_ptr += lut_stride;
//     }
// }

// // void mithral_scan_tiled(const uint8_t* codes, int64_t nblocks, int ncodebooks,
// void mithral_scan(const uint8_t* codes, int64_t nblocks, int ncodebooks,
//                   int noutputs, const uint8_t* luts, uint8_t* dists_out)
// {
//     printf("called tiled mithral scan!\n");
//     static constexpr int block_nrows = 32;
//     static constexpr int lut_sz = 16;
//     // static constexpr int chunk_nrows = 999999;  // no chunking
//     static constexpr int chunk_nrows = 512;
//     static constexpr int chunk_nblocks = chunk_nrows / block_nrows;

//     // having chunk size adapt to ncodebooks doesn't help either;
//     // conclusion: scan is just not read-bound
//     // static constexpr int target_chunk_nbytes = 1 << 14;  // 16kiB for L1 cache
//     // static constexpr int target_chunk_nbytes = 24 * 1024;  // 24kiB for L1 cache
//     // static constexpr int target_chunk_nbytes = 31 * 1024;  // 31kiB for L1 cache
//     // int codes_row_nbytes = ncodebooks / 2;
//     // int codes_block_nbytes = codes_row_nbytes * block_nrows;
//     // int chunk_nblocks = target_chunk_nbytes / codes_block_nbytes;
//     // int chunk_nrows = chunk_nblocks * block_nrows;

//     auto codes_row_stride = ncodebooks / 2;
//     auto codes_chunk_stride = codes_row_stride * chunk_nrows;
//     auto out_chunk_stride = chunk_nrows;
//     auto out_col_stride = nblocks * block_nrows;
//     auto lut_chunk_stride = 0;
//     auto lut_col_stride = ncodebooks * lut_sz;

//     auto nchunks = (nblocks + chunk_nblocks - 1) / chunk_nblocks;
//     for (int chunk = 0; chunk < nchunks; chunk++) { // for each chunk of input rows
//         int64_t use_nblocks = chunk_nblocks;
//         if (chunk == (nchunks - 1)) { // handle last chunk
//             auto nblocks_done = chunk * chunk_nblocks;
//             use_nblocks = nblocks - nblocks_done;
//         }
//         auto codes_ptr = codes + (chunk * codes_chunk_stride);
//         auto out_ptr = dists_out + (chunk * out_chunk_stride);
//         auto lut_ptr = luts + (chunk * lut_chunk_stride);

//         for (int i = 0; i < noutputs; i++) {
//             mithral_scan(codes_ptr, use_nblocks, ncodebooks, lut_ptr, out_ptr);
//             out_ptr += out_col_stride;
//             lut_ptr += lut_col_stride;
//         }
//     }
// }

//
//  mithral.cpp
//  Bolt
//
//  Created by DB on 12/3/19.
//  Copyright © 2019 D Blalock. All rights reserved.
//

#include "mithral.hpp"


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


//
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

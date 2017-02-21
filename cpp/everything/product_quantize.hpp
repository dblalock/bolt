//
//  product_quantize.hpp
//  Dig
//
//  Created by DB on 2017-2-7
//  Copyright (c) 2016 DB. All rights reserved.
//


#ifndef __PRODUCT_QUANTIZE_HPP
#define __PRODUCT_QUANTIZE_HPP

#include <assert.h>
#include <sys/types.h>
#include <type_traits>

#include "avx_utils.hpp"
#include "eigen_utils.hpp" // for opq rotations


template<int NBytes>
void pq_encode_8b(const float* X, int64_t nrows, int64_t ncols,
                  const float* centroids, uint8_t* out)
{
    static constexpr int lut_sz = 256;
    static constexpr int packet_width = 8; // objs per simd register
    static constexpr int nstripes = lut_sz / packet_width;
    static constexpr int ncodebooks = NBytes;
    static_assert(NBytes > 0, "Code length <= 0 is not valid");
    const int subvect_len = static_cast<int>(ncols / ncodebooks);
    const int trailing_subvect_len = ncols % ncodebooks;
    assert(trailing_subvect_len == 0); // TODO remove this constraint

    __m256 accumulators[nstripes];

    for (int64_t n = 0; n < nrows; n++) { // for each row of X
        auto x_ptr = X + n * ncols;

        auto centroids_ptr = centroids;
        for (int m = 0; m < NBytes; m++) { // for each codebook
            for (int i = 0; i < nstripes; i++) {
                accumulators[i] = _mm256_setzero_ps();
            }
            // compute distances to each of the centroids, which we assume
            // are in column major order; this takes 256/8 = 32 packets per col
            for (int j = 0; j < subvect_len; j++) { // for each encoded dim
                auto x_j_broadcast = _mm256_set1_ps(*x_ptr);
                for (int i = 0; i < nstripes; i++) { // for upper and lower 8
                    auto centroids_half_col = _mm256_load_ps((float*)centroids_ptr);
                    centroids_ptr += packet_width;

                    auto diff = _mm256_sub_ps(x_j_broadcast, centroids_half_col);
                    accumulators[i] = fma(diff, diff, accumulators[i]);
                }
                x_ptr++;
            }

            // find argmin dist among all centroids
            // we do this by finding the minimum values in all groups of 16,
            // and then taking the min val, and associated index within
            // that group

            // find the group of 16 centroids containing the lowest min
            __m256i best_min_broadcast = _mm256_undefined_si256();
            int32_t min_val = std::numeric_limits<int32_t>::max();
            // uint8_t best_s = -1;
            uint32_t indicators = 0;
            for (int s = 0; s < nstripes; s += 2) {
                // convert the floats to ints
                // XXX distances *must* be >> 0 for this to preserve correctness
                auto dists_int32_low = _mm256_cvtps_epi32(accumulators[s]);
                auto dists_int32_high = _mm256_cvtps_epi32(accumulators[s+1]);

                // find the minimum value (and broadcast it to whole vector)
                auto dists = _mm256_min_epi32(dists_int32_low, dists_int32_high);
                auto min_broadcast = broadcast_min(dists);

                // ------------------------ if new best min found, store mins

                best_min_broadcast = _mm256_min_epi32(
                    best_min_broadcast, min_broadcast);

                int32_t val = pfirst(min_broadcast);
                bool less = val < min_val;
                min_val = less ? val : min_val;
                indicators = indicators | (static_cast<uint32_t>(less) << s);
                // the 3 lines above this, along with the msb extraction below,
                // are equivalent to the following:
                // if (val < min_val) {
                //     min_val = val;
                //     best_s = s;
                // }
            }
            uint8_t best_s = msb_idx_u32(indicators);


            // ------------------------ now find min idx within best group
            auto dists_int32_low = _mm256_cvtps_epi32(accumulators[best_s]);
            auto dists_int32_high = _mm256_cvtps_epi32(accumulators[best_s+1]);
            // mask where the minimum happens
            auto mask_low = _mm256_cmpeq_epi32(dists_int32_low, best_min_broadcast);
            auto mask_high = _mm256_cmpeq_epi32(dists_int32_high, best_min_broadcast);

            // find first int where mask is set
            uint32_t mask0 = _mm256_movemask_epi8(mask_low); // extracts MSBs
            uint32_t mask1 = _mm256_movemask_epi8(mask_high);
            uint64_t mask = mask0 + (static_cast<uint64_t>(mask1) << 32);
            uint8_t min_idx = __tzcnt_u64(mask) >> 2; // div by 4 since 4B objs

            // offset min_idx based on which group of 16 it was in
            min_idx += 16 * best_s;

            out[m] = min_idx;
        } // m
        out += NBytes;
    } // n
}

// static constexpr kReductionL2 = 0;
// static constexpr kReductionL2 = 0;

// enum class Reductions { DistL2, DotProd };

template<int NBytes, int Reduction=Reductions::DistL2, class dist_t>
void pq_lut_8b(const float* q, int64_t len, const float* centroids, dist_t* out)
{
    static constexpr int lut_sz = 256;
    static constexpr int packet_width = 8; // objs per simd register
    static constexpr int nstripes = lut_sz / packet_width;
    static constexpr int ncodebooks = NBytes;
    static constexpr bool u8_dists = std::is_same<dist_t, uint8_t>::value;
    static constexpr bool u16_dists = std::is_same<dist_t, uint16_t>::value;
    static constexpr bool float_dists = std::is_same<dist_t, float>::value;
    static_assert(NBytes > 0, "Code length <= 0 is not valid");
    static_assert(u8_dists || u16_dists || float_dists,
        "Distance type must be one of {float, uint16_t, uint8_t}.");
    static_assert(
        Reduction == Reductions::DistL2 ||
        Reduction == Reductions::DotProd,
        "Only reductions {DistL2, DotProd} are supported.");
    const int subvect_len = static_cast<int>(len / ncodebooks);
    const int trailing_subvect_len = len % ncodebooks;
    assert(trailing_subvect_len == 0); // TODO remove this constraint

    __m256 accumulators[nstripes];

    for (int m = 0; m < ncodebooks; m++) { // for each codebook
        for (int i = 0; i < nstripes; i++) {
            accumulators[i] = _mm256_setzero_ps();
        }
        for (int j = 0; j < subvect_len; j++) { // for each dim in subvect
            auto q_broadcast = _mm256_set1_ps(q[(m * subvect_len) + j]);
            for (int i = 0; i < nstripes; i++) { // for upper 8, lower 8 floats
                auto centroids_col = _mm256_load_ps(centroids);
                centroids += packet_width;

                if (Reduction == Reductions::DotProd) {
                    accumulators[i] = fma(q_broadcast, centroids_col, accumulators[i]);
                } else if (Reduction == Reductions::DistL2) {
                    auto diff = _mm256_sub_ps(q_broadcast, centroids_col);
                    accumulators[i] = fma(diff, diff, accumulators[i]);
                }
            }
        }
        // write out dists in this col of the lut
        if (float_dists) {
            for (uint8_t i = 0; i < nstripes; i++) {
                _mm256_store_ps((float*)out, accumulators[i]);
                out += packet_width;
            }
        } else if (u16_dists) {
            for (int s = 0; s < nstripes; s += 2) {
                auto d_i32_0 = _mm256_cvtps_epi32(accumulators[s]);
                auto d_i32_1 = _mm256_cvtps_epi32(accumulators[s+1]);
                auto d_u16 = _mm256_packus_epi32(d_i32_0, d_i32_1);

                auto dists = _mm256_permute4x64_epi64(d_u16, _MM_SHUFFLE(3,1,2,0));
                _mm256_store_si256((__m256i*)out, dists);
                out += 16;
            }
        } else if (u8_dists) {
            for (int s = 0; s < nstripes; s += 4) {
                auto d_i32_0 = _mm256_cvtps_epi32(accumulators[s]);
                auto d_i32_1 = _mm256_cvtps_epi32(accumulators[s+1]);
                auto d_i32_2 = _mm256_cvtps_epi32(accumulators[s+2]);
                auto d_i32_3 = _mm256_cvtps_epi32(accumulators[s+3]);
                auto d_u16_01 = _mm256_packus_epi32(d_i32_0, d_i32_1);
                auto d_u16_23 = _mm256_packus_epi32(d_i32_2, d_i32_3);
                auto dists = packed_epu16_to_unpacked_epu8(d_u16_01, d_u16_23);
                _mm256_store_si256((__m256i*)out, dists);
                out += 32;
            }
        }
    }
}

// XXX: confusingly, ncols refers to ncols in the original data, not
// in the row-major centroids mat; latter needs to have subvect_len cols
// and ncodebooks * lut_sz rows
template<int NBytes, class data_t>
void pq_encode_centroids_8b(const data_t* centroids, int ncols, data_t* out) {
    static constexpr int lut_sz = 256;
    static constexpr int ncodebooks = NBytes;
    static_assert(NBytes > 0, "Code length <= 0 is not valid");
    const int subvect_len = ncols / ncodebooks;
    const int trailing_subvect_len = ncols % ncodebooks;
    assert(trailing_subvect_len == 0); // TODO remove this constraint

    for (int m = 0; m < ncodebooks; m++) {
        // for each codebook, just copy rowmajor to colmajor, then offset
        // the starts of the centroids and out array
        for (int i = 0; i < lut_sz; i++) { // for each centroid
            auto in_row_ptr = centroids + subvect_len * i;
            for (int j = 0; j < subvect_len; j++) { // for each dim
                auto in_ptr = in_row_ptr + j;
                auto out_ptr = out + (lut_sz * j) + i;
                *out_ptr = *in_ptr;
            }
        }
        centroids += lut_sz * subvect_len;
        out += lut_sz * subvect_len;
    }
}

// non-vectorized NBytes-byte codes, where each byte is one codeword
template<int NBytes, class dist_t>
inline void pq_scan_8b(const uint8_t* codes, const dist_t* luts,
    dist_t* dists_out, int64_t N)
{
    static_assert(NBytes > 0, "Code length <= 0 is not valid");
    static constexpr int ncentroids = 256;
    for (int64_t i = 0; i < N; i++) {
        dists_out[i] = 0;
        for (int j = 0; j < NBytes; j++) {
            auto lut_ptr = luts + ncentroids * j;
            dists_out[i] += lut_ptr[codes[j]];
        }
        codes += NBytes;
    }
}

// ================================================================ OPQ

template<int NBytes, class MatrixT1, class MatrixT2> // R is a rotation mat
void opq_encode_8b(const MatrixT1& X, const float* centroids, const MatrixT2& R,
    RowMatrix<float>& X_out, uint8_t* out)
{
    // apply rotation and forward to pq func
    assert(X.rows() == X_out.rows());
    assert(X.cols() == X_out.cols());
    assert(X.cols() == R.rows());
    X_out = X * R;
    return pq_encode_8b<NBytes>(X_out.data(), X_out.rows(), X_out.cols(),
                                centroids, out);
}

template<int NBytes, int Reduction=Reductions::DistL2,
    class MatrixT, class dist_t>
void opq_lut_8b(const RowVector<float>& q, const float* centroids,
    const MatrixT& R, RowVector<float>& q_out, dist_t* out)
{
    // apply rotation and forward to pq func
    assert(q.cols() == q_out.cols());
    assert(q.cols() == R.rows());
    q_out = q * R;
    return pq_lut_8b<NBytes, Reduction>(q_out.data(), q_out.cols(), centroids, out);
}

#endif // include guard

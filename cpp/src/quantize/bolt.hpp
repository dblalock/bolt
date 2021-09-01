//
//  bolt.hpp
//  Dig
//
//  Created by DB on 2017-2-3
//  Copyright (c) 2016 DB. All rights reserved.
//
#ifndef __BOLT_HPP
#define __BOLT_HPP

#include <assert.h>
#include <stdint.h>
#include <sys/types.h>
#include <math.h>
#include <type_traits>
#include "immintrin.h" // this is what defines all the simd funcs + _MM_SHUFFLE

#ifdef BLAZE
    #include "src/utils/avx_utils.hpp"
#else
    #include "avx_utils.hpp"
#endif

// =============================================================== in cpp file

void bolt_encode(const float* X, int64_t nrows, int ncols, int ncodebooks,
    const float* centroids, uint8_t* out); // defined in cpp file

// =============================================================== defined here

namespace {

/**
 * @brief Encode a matrix of floats using Bolt.
 *
 * @param X Contiguous, row-major matrix whose rows are the vectors to encode
 * @param nrows Number of rows in X. Must be a multiple of 32.
 * @param ncols Number of columns in X; must be divisible by 2 * NBytes.
 * @param centroids A set of 16 * 2 * NBytes centroids in contiguous vertical
 *  layout, as returned by bolt_encode_centroids.
 * @param out Array in which to store the codes; must be 32B-aligned and of
 *  length nrows * NBytes or more. The codes are stored as a contiguous
 *  array of "blocks", where each block is a [32 x NBytes] column-major
 *  array of uint8_t. This is necessary so that bolt_scan can read in the
 *  first byte of 32 codes at once, then the second byte, etc.
 * @tparam NBytes Byte length of Bolt encoding for each row
 */
template<int NBytes, bool RowMajor=false>
void bolt_encode(const float* X, int64_t nrows, int ncols,
    const float* centroids, uint8_t* out)
{
    static constexpr int lut_sz = 16;
    static constexpr int packet_width = 8; // objs per simd register
    static constexpr int nstripes = lut_sz / packet_width;
    static constexpr int ncodebooks = 2 * NBytes;
    static constexpr int block_rows = 32;
    static_assert(NBytes > 0, "Code length <= 0 is not valid");
    const int64_t nblocks = ceil(nrows / (float)block_rows);
    const int subvect_len = ncols / ncodebooks;
    assert(ncols % ncodebooks == 0); // TODO remove this constraint

    __m256 accumulators[lut_sz / packet_width];

    auto x_ptr = X;
    for (int b = 0; b < nblocks; b++) { // for each block
        // handle nrows not a multiple of 32
        int limit = (b == (nblocks - 1)) ? (nrows % 32) : block_rows;
        if (limit == 0) {
            limit = block_rows;
        }
        for (int n = 0; n < limit; n++) { // for each row in block

            auto centroids_ptr = centroids;
            for (int m = 0; m < ncodebooks; m++) { // for each codebook
                for (int i = 0; i < nstripes; i++) {
                    accumulators[i] = _mm256_setzero_ps();
                }
                // compute distances to each of the centroids, which we assume
                // are in column major order; this takes 2 packets per col
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

                // convert the floats to ints
                // XXX distances *must* be >> 0 for this to preserve accuracy
                auto dists_int32_low = _mm256_cvtps_epi32(accumulators[0]);
                auto dists_int32_high = _mm256_cvtps_epi32(accumulators[1]);

                // find the minimum value
                auto dists = _mm256_min_epi32(dists_int32_low, dists_int32_high);
                auto min_broadcast = broadcast_min(dists);

                // mask where the minimum happens
                auto mask_low = _mm256_cmpeq_epi32(dists_int32_low, min_broadcast);
                auto mask_high = _mm256_cmpeq_epi32(dists_int32_high, min_broadcast);

                // find first int where mask is set
                uint32_t mask0 = _mm256_movemask_epi8(mask_low); // extracts MSBs
                uint32_t mask1 = _mm256_movemask_epi8(mask_high);
                uint64_t mask = mask0 + (static_cast<uint64_t>(mask1) << 32);
                uint8_t min_idx = __tzcnt_u64(mask) >> 2; // div by 4 since 4B objs

                int out_idx;
                if (RowMajor) {
                    out_idx = m / 2;
                } else {
                    out_idx = block_rows * (m / 2) + n;
                }
                if (m % 2) {
                    // odds -> store in upper 4 bits
                    out[out_idx] |= min_idx << 4;
                } else {
                    // evens -> store in lower 4 bits; we don't actually need to
                    // mask because odd iter will clobber upper 4 bits anyway
                    out[out_idx] = min_idx;
                }
            } // m
            if (RowMajor) { out += NBytes; }
        } // n within block
        if (!RowMajor) { out += NBytes * block_rows; }
    } // block
}
// // force template instantiations
// template void bolt_encode<2>(const float* X, int64_t nrows, int ncols,
//     const float* centroids, uint8_t* out);
// template void bolt_encode<4>(const float* X, int64_t nrows, int ncols,
//     const float* centroids, uint8_t* out);
// template void bolt_encode<8>(const float* X, int64_t nrows, int ncols,
//     const float* centroids, uint8_t* out);
// template void bolt_encode<16>(const float* X, int64_t nrows, int ncols,
//     const float* centroids, uint8_t* out);
// template void bolt_encode<32>(const float* X, int64_t nrows, int ncols,
//     const float* centroids, uint8_t* out);


/**
 * like bolt_lut() below, but without offsets or scale factor.
 */
template<int NBytes, int Reduction=Reductions::DistL2>
void bolt_lut(const float* q, int len, const float* centroids, uint8_t* out) {
    static_assert(
        Reduction == Reductions::DistL2 ||
        Reduction == Reductions::DotProd,
        "Only reductions {DistL2, DotProd} supported");
    static constexpr int lut_sz = 16;
    static constexpr int packet_width = 8; // objs per simd register
    static constexpr int nstripes = lut_sz / packet_width;
    static constexpr int ncodebooks = 2 * NBytes;
    static_assert(NBytes > 0, "Code length <= 0 is not valid");
    const int subvect_len = len / ncodebooks;
    assert(len % ncodebooks == 0); // TODO remove this constraint
    // printf("calling bolt_lut; len = %d\n", len);

    __m256 accumulators[nstripes];
    __m256i dists_uint16_0 = _mm256_undefined_si256();

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

template<int Reduction=Reductions::DistL2>
void bolt_lut(const float* q, int len, const float* centroids,
    int ncodebooks, uint8_t* out)
{
    switch(ncodebooks) {
        case 2: bolt_lut<1, Reduction>(q, len, centroids, out); break;
        case 4: bolt_lut<2, Reduction>(q, len, centroids, out); break;
        case 8: bolt_lut<4, Reduction>(q, len, centroids, out); break;
        case 16: bolt_lut<8, Reduction>(q, len, centroids, out); break;
        case 32: bolt_lut<16, Reduction>(q, len, centroids, out); break;
        case 64: bolt_lut<32, Reduction>(q, len, centroids, out); break;
        case 128: bolt_lut<32, Reduction>(q, len, centroids, out); break;
        default: assert(false); // invalid ncodebooks
    }
}
template<int Reduction=Reductions::DistL2>
void bolt_lut(const float* Q, int nrows, int ncols, const float* centroids,
                  int ncodebooks, uint8_t* out)
{
    auto in_ptr = Q;
    uint8_t* lut_out_ptr = (uint8_t*)out;
    for (int i = 0; i < nrows; i++) {
        bolt_lut(in_ptr, ncols, centroids, ncodebooks, lut_out_ptr);
        in_ptr += ncols;
        lut_out_ptr += 16 * ncodebooks;
    }
}

/**
 * @brief Create a lookup table (LUT) containing the distances from the
 * query q to each of the centroids.
 *
 * @details Centroids must be in vertical layout, end-to-end, as returned by
 * bolt_encode_centroids. Eg, if we only had two centroids instead of 16, and
 * they were [1, 2, 3, 4], [11, 12, 13, 14], with two subvectors, memory should
 * be column-major and look like:
 *
 * 1    2   3   4
 * 11   12  13  14
 *
 * The LUT will also be written in column major order. If the query were
 * [0 0 0 0], the LUT resulting LUT would be (assuming 0 offsets and scaleby=1):
 *
 * (1^2 + 2^2)      (3^2 + 4^2)
 * (11^2 + 12^2)    (13^2 + 14^2)
 *
 * @param q The (not-necessarily aligned) query vector for which to compute
 *  the LUT. Elements of the query must be contiguous.
 * @param len The length of the query, measured as the number of elements.
 * @param centroids A set of 16 * 2 * NBytes centroids in contiguous vertical
 *  layout, as returned by bolt_encode_centroids.
 * @param offsets The values to add to from each LUT in order to shift
 *  the range of represented values towards the minimum possible
 * @param scaleby The amount by which to scale raw LUT entries after
 *  subtracting the offsets
 * @param out 32B-aligned storage in which to write the look-up table. Must
 *  be of length at least 16 * 2 * NBytes.
 * @tparam NBytes Byte length of Bolt encoding
 * @tparam Reduction The scalar reduction to use. At present, must be
 *  either Reductions::DistL2 or Reductions::DotProd.
 */
template<int NBytes, int Reduction=Reductions::DistL2>
void bolt_lut(const float* q, int len, const float* centroids,
    const float* offsets, float scaleby, uint8_t* out)
{
    static_assert(
        Reduction == Reductions::DistL2 ||
        Reduction == Reductions::DotProd,
        "Only reductions {DistL2, DotProd} supported");
    static constexpr int lut_sz = 16;
    static constexpr int packet_width = 8; // objs per simd register
    static constexpr int nstripes = lut_sz / packet_width;
    static constexpr int ncodebooks = 2 * NBytes;
    static_assert(NBytes > 0, "Code length <= 0 is not valid");
    const int subvect_len = len / ncodebooks;
    assert(len % ncodebooks == 0); // TODO remove this constraint

    __m256 accumulators[nstripes];
    __m256i dists_uint16_0 = _mm256_undefined_si256();

    __m256 scaleby_vect = _mm256_set1_ps(scaleby);

    // int permute_calls_per_row = 0;

    for (int m = 0; m < ncodebooks; m++) { // for each codebook
        for (int i = 0; i < nstripes; i++) {
            accumulators[i] = _mm256_setzero_ps();
        }
        for (int j = 0; j < subvect_len; j++) { // for each dim in subvect
            float q_j = q[(m * subvect_len) + j];
            // if (Reduction == Reductions::DotProd) {
            //     q_j = (q_j - offsets[j]) * scaleby;
            // }
            auto q_broadcast = _mm256_set1_ps(q_j);
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

        auto offset_vect = _mm256_set1_ps(offsets[m]);
        auto dist0 = fma(accumulators[0], scaleby_vect, offset_vect);
        auto dist1 = fma(accumulators[1], scaleby_vect, offset_vect);

        // convert the floats to ints
        __m256i dists_int32_low = _mm256_cvtps_epi32(dist0);
        __m256i dists_int32_high = _mm256_cvtps_epi32(dist1);



        // because we saturate to uint8s, we only get 32 objs to write after
        // two 16-element codebooks
        auto dists_uint16 = _mm256_packus_epi32(dists_int32_low, dists_int32_high);
        if (m % 2) {
            // if odd-numbered codebook, combine dists from previous codebook

            // undo the weird shuffling caused by the pack operations
            auto dists = packed_epu16_to_unpacked_epu8(
                dists_uint16_0, dists_uint16);
            // permute_calls_per_row++;
            _mm256_store_si256((__m256i*)out, dists);
            out += 32;
        } else {
            // if even-numbered codebook, just store these dists to be combined
            // when we look at the next codebook
            dists_uint16_0 = dists_uint16;
        }
    }
    // printf("permute calls per row: %d\n", permute_calls_per_row);
}

template<int Reduction=Reductions::DistL2>
void bolt_lut(const float* q, int len, const float* centroids, int ncodebooks,
    const float* offsets, float scaleby, uint8_t* out)
{
    switch(ncodebooks) {
        case 2: bolt_lut<1>(q, len, centroids, offsets, scaleby, out); break;
        case 4: bolt_lut<2>(q, len, centroids, offsets, scaleby, out); break;
        case 8: bolt_lut<4>(q, len, centroids, offsets, scaleby, out); break;
        case 16: bolt_lut<8>(q, len, centroids, offsets, scaleby, out); break;
        case 32: bolt_lut<16>(q, len, centroids, offsets, scaleby, out); break;
        case 64: bolt_lut<32>(q, len, centroids, offsets, scaleby, out); break;
        case 128: bolt_lut<64>(q, len, centroids, offsets, scaleby, out); break;
        default: assert(false);  // unsupported ncodebooks
    }
}
template<int Reduction=Reductions::DistL2>
void bolt_lut(const float* Q, int nrows, int ncols, const float* centroids,
              int ncodebooks, const float* offsets, float scaleby, uint8_t* out)
{
    auto in_ptr = Q;
    uint8_t* lut_out_ptr = (uint8_t*)out;
    for (int i = 0; i < nrows; i++) {
        bolt_lut(in_ptr, ncols, centroids, ncodebooks,
                 offsets, scaleby, lut_out_ptr);
        in_ptr += ncols;
        lut_out_ptr += 16 * ncodebooks;
    }
}

// basically just a transpose with known centroid sizes
// note that this doesn't have to be fast because we do it once after training
//
// XXX: confusingly, ncols refers to ncols in the original data, not
// in the row-major centroids mat; latter needs to have subvect_len cols
// and ncodebooks * lut_sz rows
/**
 * @brief Convert an array of row-major centroids to the layout Bolt needs
 *
 * @details Bolt requires the centroids to be stored in column-major order
 * within each codebook, and the centroids for all codebooks to be contiguous.
 *
 * @param centroids A contiguous array of codebooks, where each codebook is
 *  itself a contiguous, row-major array whose rows are centroids.
 * @param ncols The number of columns in the original data. This is assumed
 *  to be a multiple of the number of codebooks.
 * @param out 32B-aligned storage in which to write the centroids. Must be at
 *  least 2 * NBytes * 16 * sizeof(data_t) bytes.
 * @tparam NBytes Byte length of Bolt encoding. This determines how many
 *  codebooks this function expects to receive.
 */
template<int NBytes, class data_t>
void bolt_encode_centroids(const data_t* centroids, int ncols, data_t* out) {
    static constexpr int lut_sz = 16;
    static constexpr int ncodebooks = 2 * NBytes;
    static_assert(NBytes > 0, "Code length <= 0 is not valid");
    const int subvect_len = ncols / ncodebooks;
    assert(ncols % ncodebooks == 0); // TODO remove this constraint

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


// https://godbolt.org/z/wCQQoY
// inner loop gets unrolled, but here's what repeats:
//      vmovdqa     ymm4, YMMWORD PTR [rsp+128]
//      vpaddusb    ymm1, ymm0, ymm1
//      vpsrlw      ymm0, ymm2, 4
//      vpand       ymm2, ymm2, ymm3
//      vpshufb     ymm2, ymm4, ymm2
//      vmovdqa     ymm4, YMMWORD PTR [rsp+96]
//      vpand       ymm0, ymm3, ymm0
//      vpaddusb    ymm1, ymm1, ymm2
//      vpshufb     ymm0, ymm4, ymm0
/**
 * @brief Compute distances for an array of codes using a given LUT
 *
 * @param codes A contiguous 1D array of code blocks, where each block is a
 *  column-major array of 32 codes. The memory must be 32B-aligned and is
 *  assumed to be the output of bolt_encode().
 * @param luts The lookup tables for each codebook, created for a single query.
 *  This is assumed to be the output of bolt_lut().
 * @param dists_out A 32B-aligned array in which to store the computed
 *  distances / products for each code. Must be of length at least
 *  32 * nblocks bytes.
 * @param nblocks The number of 32-row blocks of codes to scan through.
 * @tparam NBytes Byte length of each code
 * @tparam _ Placeholder so that bolt_scan<M, false> always works (see
 *  overload of bolt_scan below)
 * @tparam SignedLUTs Whether to store lookup table entries as int8_t instead
 *  of uint8_t
 */
template<int NBytes, bool _=false, bool SignedLUTs=false> // 2nd arg is so bolt<M, false> always works
inline void bolt_scan(const uint8_t* codes,
    const uint8_t* luts, uint8_t* dists_out, int64_t nblocks)
{
    static_assert(NBytes > 0, "Code length <= 0 is not valid");
    static const __m256i low_4bits_mask = _mm256_set1_epi8(0x0F);

    // unpack 16B luts into 32B registers
    __m256i luts_ar[NBytes * 2];
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
        auto totals = _mm256_setzero_si256();
        #pragma unroll
        for (uint8_t j = 0; j < NBytes; j++) {
            // auto x_col = stream_load_si256i(codes);
            auto x_col = load_si256i(codes);
            codes += 32;

            auto lut_low = luts_ar[2 * j];
            auto lut_high = luts_ar[2 * j + 1];

            // compute distances via lookups; we have one table for the upper
            // 4 bits of each byte in x, and one for the lower 4 bits; the
            // shuffle instruction always looks at the lower 4 bits, so we
            // have to shift x to look at its upper 4 bits; also note that
            // we have to mask out the upper bit because the shuffle
            // instruction will zero the corresponding byte if this bit is set
            auto x_low = _mm256_and_si256(x_col, low_4bits_mask);
            auto x_shft = _mm256_srli_epi16(x_col, 4);
            auto x_high = _mm256_and_si256(x_shft, low_4bits_mask);

            auto dists_low = _mm256_shuffle_epi8(lut_low, x_low);
            auto dists_high = _mm256_shuffle_epi8(lut_high, x_high);

            if (SignedLUTs) {
                totals = _mm256_adds_epi8(totals, dists_low);
                totals = _mm256_adds_epi8(totals, dists_high);
            } else {
                totals = _mm256_adds_epu8(totals, dists_low);
                totals = _mm256_adds_epu8(totals, dists_high);
            }
        }
        _mm256_stream_si256((__m256i*)dists_out, totals);
        // _mm256_store_si256((__m256i*)dists_out, totals);
        dists_out += 32;
    }
}

// https://godbolt.org/z/MIxYFF
// unrolls the whole inner loop since NBytes is a const; basically just repeats
// this block a bunch of times (note that that 16b insts are from the prev iter;
// it puts the load into ymm3 way before ymm3 is used):
//      vmovdqa     ymm3, YMMWORD PTR [rsp+288]
//      vpaddusb    ymm15, ymm14, ymm15
//      vpsrlw      ymm14, ymm15, 8
//      vpand       ymm15, ymm15, YMMWORD PTR <stuff>::low_8bits_mask[rip]
//      vpaddusw    ymm13, ymm13, ymm14
//      vmovntdqa   ymm14, YMMWORD PTR [rbx-320]
//      vpaddusw    ymm1, ymm1, ymm15
//      vpsrlw      ymm15, ymm14, 4
//      vpand       ymm14, ymm14, ymm0
//      vpshufb     ymm14, ymm3, ymm14
//      vpand       ymm15, ymm0, ymm15
//      vmovdqa     ymm3, YMMWORD PTR [rsp+256]
//      vpshufb     ymm15, ymm3, ymm15
// overload of above with uint16_t dists_out (as used in the paper); also
// has the option to immediately upcast uint8s from the LUTs to uint16_t, which
// is also what we did in the paper. Bolt is even faster if we don't do this.
template<int NBytes, bool NoOverflow=false, bool SignedLUTs=false>
inline void bolt_scan(const uint8_t* codes,
    const uint8_t* luts, uint16_t* dists_out, int64_t nblocks)
{
    static_assert(NBytes > 0, "Code length <= 0 is not valid");
    static const __m256i low_4bits_mask = _mm256_set1_epi8(0x0F);
    static const __m256i low_8bits_mask = _mm256_set1_epi16(0x00FF);

    // unpack 16B luts into 32B registers; faster than just storing them
    // unpacked for some reason
    __m256i luts_ar[NBytes * 2];
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
        // auto totals = _mm256_setzero_si256();
        auto totals_evens = _mm256_setzero_si256();
        auto totals_odds = _mm256_setzero_si256();
        #pragma unroll
        for (uint8_t j = 0; j < NBytes; j++) {
            // auto x_col = stream_load_si256i(codes);
            auto x_col = load_si256i(codes);
            codes += 32;

            auto lut_low = luts_ar[2 * j];
            auto lut_high = luts_ar[2 * j + 1];

            // compute distances via lookups; we have one table for the upper
            // 4 bits of each byte in x, and one for the lower 4 bits; the
            // shuffle instruction always looks at the lower 4 bits, so we
            // have to shift x to look at its upper 4 bits; also note that
            // we have to mask out the upper bit because the shuffle
            // instruction will zero the corresponding byte if this bit is set
            auto x_low = _mm256_and_si256(x_col, low_4bits_mask);
            auto x_shft = _mm256_srli_epi16(x_col, 4);
            auto x_high = _mm256_and_si256(x_shft, low_4bits_mask);

            auto dists_low = _mm256_shuffle_epi8(lut_low, x_low);
            auto dists_high = _mm256_shuffle_epi8(lut_high, x_high);

            // convert dists to epi16 by masking or shifting; we convert
            // 32 uint8s to a pair of uint16s by masking the low 8 bits to
            // get the even-numbered uint8s as the first vector of uint16s,
            // and shifting down by 8 bits to get the odd-numbered ones as
            // the second vector of uint16s
            if (NoOverflow) { // convert to epu16s before doing any adds
                if (SignedLUTs) {
                    auto dists16_low_odds = _mm256_srai_epi16(dists_low, 8);
                    auto dists16_high_odds = _mm256_srai_epi16(dists_high, 8);
                    // need to sign extend upper bit of low 8b
                    auto dists16_low_evens = _mm256_srai_epi16(_mm256_srai_epi16(dists_low, 8), 8);
                    auto dists16_high_evens = _mm256_srai_epi16(_mm256_srai_epi16(dists_high, 8), 8);
                    totals_evens = _mm256_adds_epi16(totals_evens, dists16_low_evens);
                    totals_evens = _mm256_adds_epi16(totals_evens, dists16_high_evens);
                    totals_odds = _mm256_adds_epi16(totals_odds, dists16_low_odds);
                    totals_odds = _mm256_adds_epi16(totals_odds, dists16_high_odds);
                } else {
                    auto dists16_low_odds = _mm256_srli_epi16(dists_low, 8);
                    auto dists16_high_odds = _mm256_srli_epi16(dists_high, 8);
                    auto dists16_low_evens = _mm256_and_si256(dists_low, low_8bits_mask);
                    auto dists16_high_evens = _mm256_and_si256(dists_high, low_8bits_mask);
                    totals_evens = _mm256_adds_epu16(totals_evens, dists16_low_evens);
                    totals_evens = _mm256_adds_epu16(totals_evens, dists16_high_evens);
                    totals_odds = _mm256_adds_epu16(totals_odds, dists16_low_odds);
                    totals_odds = _mm256_adds_epu16(totals_odds, dists16_high_odds);
                }

            } else { // add pairs as epu8s, then use pair sums as epu16s
                if (SignedLUTs) {
                    auto dists = _mm256_adds_epi8(dists_low, dists_high);
                    auto dists16_evens = _mm256_srai_epi16(_mm256_srai_epi16(dists, 8), 8);
                    auto dists16_odds = _mm256_srai_epi16(dists, 8);
                    totals_evens = _mm256_adds_epi16(totals_evens, dists16_evens);
                    totals_odds = _mm256_adds_epi16(totals_odds, dists16_odds);
                } else {
                    auto dists = _mm256_adds_epu8(dists_low, dists_high);
                    auto dists16_evens = _mm256_and_si256(dists, low_8bits_mask);
                    auto dists16_odds = _mm256_srli_epi16(dists, 8);
                    totals_evens = _mm256_adds_epu16(totals_evens, dists16_evens);
                    totals_odds = _mm256_adds_epu16(totals_odds, dists16_odds);
                }
            }
        }

        // unmix the interleaved 16bit dists and store them
        auto tmp_low = _mm256_permute4x64_epi64(
                totals_evens, _MM_SHUFFLE(3,1,2,0));
        auto tmp_high = _mm256_permute4x64_epi64(
                totals_odds, _MM_SHUFFLE(3,1,2,0));
        auto dists_out_0 = _mm256_unpacklo_epi16(tmp_low, tmp_high);
        auto dists_out_1 = _mm256_unpackhi_epi16(tmp_low, tmp_high);
        _mm256_stream_si256((__m256i*)dists_out, dists_out_0);
        dists_out += 16;
        _mm256_stream_si256((__m256i*)dists_out, dists_out_1);
        dists_out += 16;
    }
}

// wrapper that doesn't need ncodebooks at compile time
template<bool NoOverflow=true, bool SignedLUTs=false, class dist_t>
void bolt_scan(const uint8_t* codes, int64_t nblocks, int ncodebooks,
               const uint8_t* luts, dist_t* dists_out)
{
    switch(ncodebooks) {
        case 2: bolt_scan<1, NoOverflow, SignedLUTs>(
            codes, luts, dists_out, nblocks); break;
        case 4: bolt_scan<2, NoOverflow, SignedLUTs>(
            codes, luts, dists_out, nblocks); break;
        case 8: bolt_scan<4, NoOverflow, SignedLUTs>(
            codes, luts, dists_out, nblocks); break;
        case 16: bolt_scan<8, NoOverflow, SignedLUTs>(
            codes, luts, dists_out, nblocks); break;
        case 32: bolt_scan<16, NoOverflow, SignedLUTs>(
            codes, luts, dists_out, nblocks); break;
        case 64: bolt_scan<32, NoOverflow, SignedLUTs>(
            codes, luts, dists_out, nblocks); break;
        case 128: bolt_scan<64, NoOverflow, SignedLUTs>(
            codes, luts, dists_out, nblocks); break;
        default: assert(false);  // unsupported ncodebooks
    }
}

template<bool NoOverflow=true, bool SignedLUTs=false, bool tile=true, class dist_t>
void bolt_scan(const uint8_t* codes, int64_t nblocks, int ncodebooks,
                  int noutputs, const uint8_t* luts, dist_t* dists_out)
{
    static constexpr int block_nrows = 32;
    static constexpr int lut_sz = 16;

    int chunk_nblocks = (int)nblocks;
    int chunk_nrows = chunk_nblocks * block_nrows; // no tiling
    if (tile) {
        static constexpr int target_chunk_nbytes = 24 * 1024;  // most of L1 cache
        int codes_row_nbytes = ncodebooks / 2;
        int codes_block_nbytes = codes_row_nbytes * block_nrows;
        chunk_nblocks = target_chunk_nbytes / codes_block_nbytes;
        chunk_nrows = chunk_nblocks * block_nrows;
    }

    auto codes_row_stride = ncodebooks / 2;
    auto codes_chunk_stride = codes_row_stride * chunk_nrows;
    auto out_chunk_stride = chunk_nrows;
    auto out_col_stride = nblocks * block_nrows;
    auto lut_chunk_stride = 0;
    auto lut_col_stride = ncodebooks * lut_sz;

    auto nchunks = (nblocks + chunk_nblocks - 1) / chunk_nblocks;
    for (int chunk = 0; chunk < nchunks; chunk++) { // for each chunk of input rows
        int64_t use_nblocks = chunk_nblocks;
        if (chunk == (nchunks - 1)) { // handle last chunk
            auto nblocks_done = chunk * chunk_nblocks;
            use_nblocks = nblocks - nblocks_done;
        }
        auto codes_ptr = codes + (chunk * codes_chunk_stride);
        auto out_ptr = dists_out + (chunk * out_chunk_stride);
        auto lut_ptr = luts + (chunk * lut_chunk_stride);

        for (int i = 0; i < noutputs; i++) {
            bolt_scan<NoOverflow, SignedLUTs>(
                codes_ptr, use_nblocks, ncodebooks, lut_ptr, out_ptr);
            out_ptr += out_col_stride;
            lut_ptr += lut_col_stride;
        }
    }
}

} // anon namespace
#endif // __BOLT_HPP

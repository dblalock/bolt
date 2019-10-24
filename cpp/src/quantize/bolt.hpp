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

#include "debug_utils.hpp" // TODO rm

#ifdef BLAZE
    #include "src/utils/avx_utils.hpp"
#else
    #include "avx_utils.hpp"
#endif

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

            _mm256_store_si256((__m256i*)out, dists);
            out += 32;
        } else {
            // if even-numbered codebook, just store these dists to be combined
            // when we look at the next codebook
            dists_uint16_0 = dists_uint16;
        }
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
        for (uint8_t j = 0; j < NBytes; j++) {
            auto x_col = stream_load_si256i(codes);
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
        for (uint8_t j = 0; j < NBytes; j++) {
            auto x_col = stream_load_si256i(codes);
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
                auto dists16_low_evens = _mm256_and_si256(dists_low, low_8bits_mask);
                auto dists16_low_odds = _mm256_srli_epi16(dists_low, 8);
                auto dists16_high_evens = _mm256_and_si256(dists_high, low_8bits_mask);
                auto dists16_high_odds = _mm256_srli_epi16(dists_high, 8);

                if (SignedLUTs) {
                    totals_evens = _mm256_adds_epi16(totals_evens, dists16_low_evens);
                    totals_evens = _mm256_adds_epi16(totals_evens, dists16_high_evens);
                    totals_odds = _mm256_adds_epi16(totals_odds, dists16_low_odds);
                    totals_odds = _mm256_adds_epi16(totals_odds, dists16_high_odds);
                } else {
                    totals_evens = _mm256_adds_epu16(totals_evens, dists16_low_evens);
                    totals_evens = _mm256_adds_epu16(totals_evens, dists16_high_evens);
                    totals_odds = _mm256_adds_epu16(totals_odds, dists16_low_odds);
                    totals_odds = _mm256_adds_epu16(totals_odds, dists16_high_odds);
                }

            } else { // add pairs as epu8s, then use pair sums as epu16s
                if (SignedLUTs) {
                    auto dists = _mm256_adds_epi8(dists_low, dists_high);
                    auto dists16_evens = _mm256_and_si256(dists, low_8bits_mask);
                    auto dists16_odds = _mm256_srli_epi16(dists, 8);
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

// wrapper func so we can pass bolt_scan with immediate upcasting into
// our profiling macro (which can't deal with comma-separated template args)
template<int NBytes, bool SignedLUTs=false>
inline void bolt_scan_safe(const uint8_t* codes,
    const uint8_t* luts, uint16_t* dists_out, int64_t nblocks)
{
    bolt_scan<NBytes, true, SignedLUTs>(codes, luts, dists_out, nblocks);
}

// like the above, but we assume that codes are in colmajor order
// TODO version of this that uses packed 4b codes? along with packing func?
// TODO version of this that doesn't immediately upcast to u16?
template<bool NoOverflow=false, bool SignedLUTs=false>
inline void bolt_scan_unpacked_colmajor(const uint8_t* codes,
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

// for looking at assembly: https://godbolt.org/z/PEjpiz
// .LBB0_7:                                #   Parent Loop BB0_6 Depth=1
//         vmovdqa ymm5, ymmword ptr [rdi + rcx]
//         vmovdqa ymm6, ymmword ptr [rdx + rcx]
//         vpsrlw  ymm7, ymm5, 4
//         vpsrlw  ymm8, ymm6, 4
//         vpand   ymm5, ymm5, ymm0
//         vpshufb ymm5, ymm1, ymm5
//         vpand   ymm7, ymm7, ymm0
//         vpshufb ymm7, ymm2, ymm7
//         vpaddsb ymm5, ymm5, ymm7
//         vpand   ymm6, ymm6, ymm0
//         vpshufb ymm6, ymm4, ymm6
//         vpand   ymm7, ymm8, ymm0
//         vpshufb ymm7, ymm3, ymm7
//         vpaddsb ymm6, ymm6, ymm7
//         vpaddsb ymm5, ymm5, ymm6
//         vpmovsxbw       ymm6, xmm5
//         vextracti128    xmm5, ymm5, 1
//         vpmovsxbw       ymm5, xmm5
//         vpaddsw ymm6, ymm6, ymmword ptr [rbx + 2*rcx]
//         vpaddsw ymm5, ymm5, ymmword ptr [rbx + 2*rcx + 32]
//         vmovdqa ymmword ptr [rbx + 2*rcx], ymm6
//         vmovdqa ymmword ptr [rbx + 2*rcx + 32], ymm5
//         add     rcx, 32
//         dec     rax
//         jne     .LBB0_7
template<int UpcastEvery=4, bool Packed=false, typename LutT=uint8_t>
inline void bolt_scan_colmajor_tile4(const uint8_t* codes,
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
            // volatile __m256i dists_so_far_0_15 = load_si256i(out_ptr);
            // volatile __m256i dists_so_far_16_31 = load_si256i(out_ptr + 16);

            auto dists0 = _mm256_shuffle_epi8(vlut0, vcodes0);
            auto dists1 = _mm256_shuffle_epi8(vlut1, vcodes1);
            auto dists2 = _mm256_shuffle_epi8(vlut2, vcodes2);
            auto dists3 = _mm256_shuffle_epi8(vlut3, vcodes3);

            auto dists_0_15 = _mm256_undefined_si256();
            auto dists_16_31 = _mm256_undefined_si256();
            if (SignedLUTs) {
                if (UpcastEvery == 4) {
                    // add all four, and only then upcast to 16b
                    auto dists01 = _mm256_adds_epi8(dists0, dists1);
                    auto dists23 = _mm256_adds_epi8(dists2, dists3);
                    auto dists = _mm256_adds_epi8(dists01, dists23);
                    dists_0_15 = _mm256_cvtepi8_epi16(
                        _mm256_extracti128_si256(dists, 0));
                    dists_16_31 = _mm256_cvtepi8_epi16(
                        _mm256_extracti128_si256(dists, 1));
                } else if (UpcastEvery == 2) {
                    // pairwise sums, then upcast to 16 bits
                    auto dists01 = _mm256_adds_epi8(dists0, dists1);
                    auto dists23 = _mm256_adds_epi8(dists2, dists3);

                    auto dists01_0 = _mm256_cvtepi8_epi16(
                        _mm256_extracti128_si256(dists01, 0));
                    auto dists01_1 = _mm256_cvtepi8_epi16(
                        _mm256_extracti128_si256(dists01, 1));
                    auto dists23_0 = _mm256_cvtepi8_epi16(
                        _mm256_extracti128_si256(dists23, 0));
                    auto dists23_1 = _mm256_cvtepi8_epi16(
                        _mm256_extracti128_si256(dists23, 1));

                    dists_0_15 = _mm256_adds_epi16(dists01_0, dists23_0);
                    dists_16_31 = _mm256_adds_epi16(dists01_1, dists23_1);
                } else if (UpcastEvery == 1) {
                    // convert everything to 16 bits immediately
                    auto dists0_0 = _mm256_cvtepi8_epi16(
                        _mm256_extracti128_si256(dists0, 0));
                    auto dists0_1 = _mm256_cvtepi8_epi16(
                        _mm256_extracti128_si256(dists0, 1));
                    auto dists1_0 = _mm256_cvtepi8_epi16(
                        _mm256_extracti128_si256(dists1, 0));
                    auto dists1_1 = _mm256_cvtepi8_epi16(
                        _mm256_extracti128_si256(dists1, 1));
                    auto dists2_0 = _mm256_cvtepi8_epi16(
                        _mm256_extracti128_si256(dists2, 0));
                    auto dists2_1 = _mm256_cvtepi8_epi16(
                        _mm256_extracti128_si256(dists2, 1));
                    auto dists3_0 = _mm256_cvtepi8_epi16(
                        _mm256_extracti128_si256(dists3, 0));
                    auto dists3_1 = _mm256_cvtepi8_epi16(
                        _mm256_extracti128_si256(dists3, 1));

                    auto dists01_0 = _mm256_adds_epi16(dists0_0, dists1_0);
                    auto dists01_1 = _mm256_adds_epi16(dists0_1, dists1_1);
                    auto dists23_0 = _mm256_adds_epi16(dists2_0, dists3_0);
                    auto dists23_1 = _mm256_adds_epi16(dists2_1, dists3_1);
                    dists_0_15 = _mm256_adds_epi16(dists01_0, dists23_0);
                    dists_16_31 = _mm256_adds_epi16(dists01_1, dists23_1);
                }
            } else {
                if (UpcastEvery == 4) {
                    // add all four, and only then upcast to 16b
                    auto dists01 = _mm256_adds_epu8(dists0, dists1);
                    auto dists23 = _mm256_adds_epu8(dists2, dists3);
                    auto dists = _mm256_adds_epu8(dists01, dists23);
                    dists_0_15 = _mm256_cvtepu8_epi16(
                        _mm256_extracti128_si256(dists, 0));
                    dists_16_31 = _mm256_cvtepu8_epi16(
                        _mm256_extracti128_si256(dists, 1));
                } else if (UpcastEvery == 2) {
                    // pairwise sums, then upcast to 16 bits
                    auto dists01 = _mm256_adds_epu8(dists0, dists1);
                    auto dists23 = _mm256_adds_epu8(dists2, dists3);

                    auto dists01_0 = _mm256_cvtepu8_epi16(
                        _mm256_extracti128_si256(dists01, 0));
                    auto dists01_1 = _mm256_cvtepu8_epi16(
                        _mm256_extracti128_si256(dists01, 1));
                    auto dists23_0 = _mm256_cvtepu8_epi16(
                        _mm256_extracti128_si256(dists23, 0));
                    auto dists23_1 = _mm256_cvtepu8_epi16(
                        _mm256_extracti128_si256(dists23, 1));

                    dists_0_15 = _mm256_adds_epi16(dists01_0, dists23_0);
                    dists_16_31 = _mm256_adds_epi16(dists01_1, dists23_1);
                } else if (UpcastEvery == 1) {
                    // convert everything to 16 bits immediately
                    auto dists0_0 = _mm256_cvtepu8_epi16(
                        _mm256_extracti128_si256(dists0, 0));
                    auto dists0_1 = _mm256_cvtepu8_epi16(
                        _mm256_extracti128_si256(dists0, 1));
                    auto dists1_0 = _mm256_cvtepu8_epi16(
                        _mm256_extracti128_si256(dists1, 0));
                    auto dists1_1 = _mm256_cvtepu8_epi16(
                        _mm256_extracti128_si256(dists1, 1));
                    auto dists2_0 = _mm256_cvtepu8_epi16(
                        _mm256_extracti128_si256(dists2, 0));
                    auto dists2_1 = _mm256_cvtepu8_epi16(
                        _mm256_extracti128_si256(dists2, 1));
                    auto dists3_0 = _mm256_cvtepu8_epi16(
                        _mm256_extracti128_si256(dists3, 0));
                    auto dists3_1 = _mm256_cvtepu8_epi16(
                        _mm256_extracti128_si256(dists3, 1));

                    auto dists01_0 = _mm256_adds_epi16(dists0_0, dists1_0);
                    auto dists01_1 = _mm256_adds_epi16(dists0_1, dists1_1);
                    auto dists23_0 = _mm256_adds_epi16(dists2_0, dists3_0);
                    auto dists23_1 = _mm256_adds_epi16(dists2_1, dists3_1);
                    dists_0_15 = _mm256_adds_epi16(dists01_0, dists23_0);
                    dists_16_31 = _mm256_adds_epi16(dists01_1, dists23_1);
                }
            }

            auto new_dists_0_15 = _mm256_undefined_si256();
            auto new_dists_16_31 = _mm256_undefined_si256();
            if (SignedLUTs) {
                new_dists_0_15 = _mm256_adds_epi16(dists_0_15, dists_so_far_0_15);
                new_dists_16_31 = _mm256_adds_epi16(dists_16_31, dists_so_far_16_31);
            } else {
                new_dists_0_15 = _mm256_adds_epu16(dists_0_15, dists_so_far_0_15);
                new_dists_16_31 = _mm256_adds_epu16(dists_16_31, dists_so_far_16_31);
            }

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
inline void bolt_scan_colmajor_tile4_packed(const uint8_t* codes,
    int64_t nblocks, int ncodebooks, const LutT* luts, int16_t* out)
{
    bolt_scan_colmajor_tile4<UpcastEvery, true>(
        codes, nblocks, ncodebooks, luts, out);
}

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
inline void zip4_4b_colmajor(const uint8_t* codes_in, uint32_t ncodebooks,
                             int64_t nblocks, uint8_t* codes_out)
{
    static constexpr int in_block_sz = 32;      // read 32 codes at once
    static constexpr int out_block_sz = 64;     // 32 x 4 cols -> 64 rows
    static constexpr int ncodebooks_per_group = 4;
    assert(ncodebooks % ncodebooks_per_group == 0);
    int ncolgroups = ncodebooks / ncodebooks_per_group;

    auto in_col_stride = in_block_sz * nblocks;
    auto out_col_stride = out_block_sz * nblocks;

    // PRINT_VAR(ncolgroups);
    // PRINT_VAR(in_col_stride);
    // PRINT_VAR(out_col_stride);

    for (int c = 0; c < ncolgroups; c++) {
        // initialize col starts
        // for (int cc = 0; cc < ncodebooks_per_group; cc++) {
        //     auto col = initial_col + cc;
        //     in_col_ptrs[cc] = codes_in + col * in_col_stride;
        // }
        auto initial_col = c * ncodebooks_per_group;
        auto initial_col_ptr = codes_in + (initial_col * in_col_stride);
        auto out_col_ptr = codes_out + (c * out_col_stride);
        // for each block
        for (int b = 0; b < nblocks; b++) {
            // okay, forget being generic here; load from all 4 cols
            auto initial_col = c * ncodebooks_per_group;
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

            // printf("y0: \n");
            // dump_m256i(y0);

            _mm256_store_si256((__m256i*)out_col_ptr, y0);
            out_col_ptr += 32;
            _mm256_store_si256((__m256i*)out_col_ptr, y1);
            out_col_ptr += 32;
        }
    }
}

// just pack low 4b from 2 cols into one col by using upper 8b; note that
// we assume that upper 4b are originally 0
inline void zip2_4b_colmajor(const uint8_t* codes_in, uint32_t ncodebooks,
                             int64_t nblocks, uint8_t* codes_out)
{
    static constexpr int in_block_sz = 32;      // read 32 codes at once
    static constexpr int out_block_sz = 32;     // 32 x 4 cols -> 64 rows
    static constexpr int ncodebooks_per_group = 2;
    assert(ncodebooks % ncodebooks_per_group == 0);
    int ncolgroups = ncodebooks / ncodebooks_per_group;

    auto in_col_stride = in_block_sz * nblocks;
    auto out_col_stride = out_block_sz * nblocks;

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
            _mm256_store_si256((__m256i*)out_col_ptr, x01);
            out_col_ptr += 32;
        }
    }
}

// TODO impl boltscan_colmajor_tile8_packed; same idea as tile4, but with
// 8 dims at once instead of 4
// TODO impl encode funcs that just spit out encodings in regular bolt format


} // anon namespace
#endif // __BOLT_HPP

//
//  multi_codebook.hpp
//  Dig
//
//  Created by DB on 2017-1-22
//  Copyright (c) 2016 DB. All rights reserved.
//


#ifndef __MULTI_CODEBOOK_HPP
#define __MULTI_CODEBOOK_HPP

#include <assert.h>
#include <sys/types.h>

#ifdef BLAZE
    #include "src/utils/avx_utils.hpp"
#else
    #include "avx_utils.hpp"
#endif

namespace dist {

static const uint8_t mask_low4b = 0x0F;

// experimental version to see what bottlenecks are; this one assumes 4b codes
// are already unpacked (so no need to shift or mask); NOTE: this impl requires
// NBytes to be twice as large as other 4bit lut functions because we're only
// storing one code in each byte
template<int NBytes>
inline void incorrect_lut_dists_block32_4b_v2(const uint8_t* codes,
    const uint8_t* luts, uint8_t* dists_out, int64_t nblocks)
{
    static_assert(NBytes > 0, "Code length <= 0 is not valid");
    static_assert(NBytes % 2 == 0, "Only even NBytes supported!");
    // static const __m256i low_4bits_mask = _mm256_set1_epi8(0x0F);

    __m256i luts_ar[NBytes];
    auto lut_ptr = luts;
    for (uint8_t j = 0; j < NBytes; j+= 2) {
        auto both_luts = load_si256i(lut_ptr);
        lut_ptr += 32;
        auto lut0 = _mm256_permute2x128_si256(both_luts, both_luts, 0 + (0 << 4));
        auto lut1 = _mm256_permute2x128_si256(both_luts, both_luts, 1 + (1 << 4));
        luts_ar[j] = lut0;
        luts_ar[j+1] = lut1;
    }

    for (int64_t i = 0; i < nblocks; i++) {
        auto totals = _mm256_setzero_si256();
        for (uint8_t j = 0; j < NBytes; j+= 2) {
            // unpack lower and upper 16B into two 32B luts
            // auto both_luts = load_si256i(luts);
            // luts += 32;
            // auto lut0 = _mm256_permute2x128_si256(both_luts, both_luts, 0 + (0 << 4));
            // auto lut1 = _mm256_permute2x128_si256(both_luts, both_luts, 1 + (1 << 4));

            // dists from first byte
            // auto x_col0 = load_si256i(codes);
            auto lut0 = luts_ar[j];
            auto x_col0 = stream_load_si256i(codes);
            codes += 32;
            auto dists0 = _mm256_shuffle_epi8(lut0, x_col0);
            totals = _mm256_adds_epu8(totals, dists0);

            // dists from second byte
            // auto x_col1 = load_si256i(codes);
            auto lut1 = luts_ar[j+1];
            auto x_col1 = stream_load_si256i(codes);
            codes += 32;
            auto dists1 = _mm256_shuffle_epi8(lut1, x_col1);
            totals = _mm256_adds_epu8(totals, dists1);
        }
        // _mm256_store_si256((__m256i*)dists_out, totals);
        _mm256_stream_si256((__m256i*)dists_out, totals); // "non-temporal memory hint"
        // luts -= 8 * 64;
        luts -= NBytes / 2 * 32;
        dists_out += 32;
    }
}

// experimental version to see what bottlenecks are
template<int NBytes>
inline void incorrect_lut_dists_block32_4b(const uint8_t* codes, const uint8_t* luts,
    uint8_t* dists_out, int64_t nblocks)
{
    static_assert(NBytes > 0, "Code length <= 0 is not valid");
    static const __m256i low_4bits_mask = _mm256_set1_epi8(0x0F);

    for (int64_t i = 0; i < nblocks; i++) {
        auto totals = _mm256_setzero_si256();
        // auto x_col = load_si256i(codes);
        // auto both_luts = load_si256i(luts);
//        __m256i both_luts;
        __m256i four_luts = _mm256_undefined_si256();
        // __m256i luts0, luts1;
        // __m256i lut_low, lut_high;
        for (uint8_t j = 0; j < NBytes; j++) {
            // auto x_col = load_si256i(codes);
            auto x_col = stream_load_si256i(codes);

            __m256i both_luts;
            if (j % 2 == 0) {
                four_luts = load_si256i(luts);
                luts += 32;
                both_luts = _mm256_permute2x128_si256(four_luts, four_luts, 0 + (0 << 4));
            } else {
                both_luts = _mm256_permute2x128_si256(four_luts, four_luts, 1 + (1 << 4));
            }
            // unpack lower and upper 4 bits into luts for lower and upper 4
            // bits of codes
            auto lut_low = _mm256_and_si256(both_luts, low_4bits_mask);
            auto lut_high = _mm256_srli_epi16(both_luts, 4);
            lut_high = _mm256_and_si256(lut_high, low_4bits_mask);

            // auto both_luts = load_si256i(luts);
            // auto lut_low = _mm256_permute2x128_si256(both_luts, both_luts, 0 + (0 << 4));
            // auto lut_high = _mm256_permute2x128_si256(both_luts, both_luts, 1 + (1 << 4));

            // compute distances via lookups; we have one table for the upper
            // 4 bits of each byte in x, and one for the lower 4 bits; the
            // shuffle instruction always looks at the lower 4 bits, so we
            // have to shift x to look at its upper 4 bits; also note that
            // we have to mask out the upper bit because the shuffle
            // instruction will zero the corresponding byte if this bit is set
            auto x_low = _mm256_and_si256(x_col, low_4bits_mask);
            auto x_high = _mm256_srli_epi16(x_col, 4);
            x_high = _mm256_and_si256(x_high, low_4bits_mask);

            auto dists_low = _mm256_shuffle_epi8(lut_low, x_low);
            auto dists_high = _mm256_shuffle_epi8(lut_high, x_high);

            // TODO uncomment after debug
            totals = _mm256_adds_epu8(totals, dists_low);
            totals = _mm256_adds_epu8(totals, dists_high);

            codes += 32;
        }
        // _mm256_store_si256((__m256i*)dists_out, totals);
        _mm256_stream_si256((__m256i*)dists_out, totals);
        // luts -= 4 * 32;
        luts -= NBytes / 2 * 32;
        dists_out += 32;
    }
}

// unpacks 16B LUTs into 32B to cut out half the loads, and computes dists
// in blocks; note that this requires reordering the codes into striped blocks
template<int B, int NBytes>
inline void lut_dists_block32_4b_vertical(const uint8_t* codes,
    const uint8_t* luts, uint8_t* dists_out, int64_t N)
{
    static constexpr int packet_width = 32; // how many vecs we operate on at once
    static constexpr int nstripes = B / packet_width; // # of rows of 32B per block
    static_assert(B % packet_width == 0, "B must be a multiple of packet_width");
    static_assert(B > 0, "B must be > 0");
    static_assert(NBytes > 0, "Code length <= 0 is not valid");

    // static_assert(nstripes == 1, "TODO rm after debug"); // TODO rm

    static const __m256i low_4bits_mask = _mm256_set1_epi8(0x0F);
    const int64_t nblocks = N / B;
    assert(N % B == 0);

    // load up luts into SIMD registers
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

    __m256i accumulators[nstripes];

    for (int64_t b = 0; b < nblocks; b++) { // for each block
        for (int i = 0; i < nstripes; i++) {
            accumulators[i] = _mm256_setzero_si256(); // zero dists
        }
        for (int j = 0; j < NBytes; j++) { // for each col (1 byte, 2 codes)

            // TODO also try loading from lut* (ie, not having luts_ar)
            auto lut_low = luts_ar[2 * j];
            auto lut_high = luts_ar[2 * j + 1];

            for (int i = 0; i < nstripes; i++) { // for each stripe
                auto x_col = stream_load_si256i(codes);
                codes += 32;

                auto x_low = _mm256_and_si256(x_col, low_4bits_mask);
                auto x_shft = _mm256_srli_epi16(x_col, 4);
                auto x_high = _mm256_and_si256(x_shft, low_4bits_mask);

                auto dists_low = _mm256_shuffle_epi8(lut_low, x_low);
                auto dists_high = _mm256_shuffle_epi8(lut_high, x_high);

                accumulators[i] = _mm256_adds_epu8(accumulators[i], dists_low);
                accumulators[i] = _mm256_adds_epu8(accumulators[i], dists_high);
            }
        }
        for (uint8_t i = 0; i < nstripes; i++) { // for each stripe
            _mm256_store_si256((__m256i*)dists_out, accumulators[i]);
            dists_out += packet_width;
        }
    } // for each block
}

// version that unpacks 16B LUTs into 32B to cut out half the loads
template<int NBytes>
inline void lut_dists_block32_4b_unpack(const uint8_t* codes,
    const uint8_t* luts, uint8_t* dists_out, int64_t nblocks)
{
    static_assert(NBytes > 0, "Code length <= 0 is not valid");
    static const __m256i low_4bits_mask = _mm256_set1_epi8(0x0F);

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
            // auto x_col = load_si256i(codes);
            auto x_col = stream_load_si256i(codes);
            codes += 32;

            auto lut_low = luts_ar[2 * j];
            auto lut_high = luts_ar[2 * j + 1];
            // unpack lower and upper 16B into two 32B luts
            // NOTE: cast + broadcast seems no faster, and more complicated
            // auto both_luts = load_si256i(luts);
            // // auto lower_128 = _mm256_castsi256_si128(both_luts);
            // // auto lut_low = _mm256_broadcastsi128_si256(lower_128);
            // auto lut_low = _mm256_permute2x128_si256(both_luts, both_luts, 0 + (0 << 4));
            // auto lut_high = _mm256_permute2x128_si256(both_luts, both_luts, 1 + (1 << 4));

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

            totals = _mm256_adds_epu8(totals, dists_low);
            totals = _mm256_adds_epu8(totals, dists_high);
        }
        // _mm256_store_si256((__m256i*)dists_out, totals);
        _mm256_stream_si256((__m256i*)dists_out, totals); // "non-temporal memory hint"
        dists_out += 32;
        // luts -= 8 * 64;
        // luts -= NBytes * 32;
    }
}

template<int NBytes>
inline void lut_dists_block32_4b(const uint8_t* codes, const uint8_t* luts,
    uint8_t* dists_out, int64_t nblocks)
{
    static_assert(NBytes > 0, "Code length <= 0 is not valid");
    static const __m256i low_4bits_mask = _mm256_set1_epi8(0x0F);

    for (int64_t i = 0; i < nblocks; i++) {
        auto totals = _mm256_setzero_si256();
        for (uint8_t j = 0; j < NBytes; j++) {
            // auto x_col = load_si256i(codes);
            auto x_col = stream_load_si256i(codes);
            auto lut_low = load_si256i(luts);
            auto lut_high = load_si256i(luts + 32);
            // auto both_luts = load_si256i(luts);
            // auto lut_low = _mm256_permute2x128_si256(both_luts, both_luts, 0 + (0 << 4));
            // auto lut_high = _mm256_permute2x128_si256(both_luts, both_luts, 1 + (1 << 4));

            // compute distances via lookups; we have one table for the upper
            // 4 bits of each byte in x, and one for the lower 4 bits; the
            // shuffle instruction always looks at the lower 4 bits, so we
            // have to shift x to look at its upper 4 bits; also note that
            // we have to mask out the upper bit because the shuffle
            // instruction will zero the corresponding byte if this bit is set
            auto x_low = _mm256_and_si256(x_col, low_4bits_mask);
            auto x_high = _mm256_srli_epi16(x_col, 4);
            x_high = _mm256_and_si256(x_high, low_4bits_mask);

            auto dists_low = _mm256_shuffle_epi8(lut_low, x_low);
            auto dists_high = _mm256_shuffle_epi8(lut_high, x_high);

            totals = _mm256_adds_epu8(totals, dists_low);
            totals = _mm256_adds_epu8(totals, dists_high);

            codes += 32;
            luts += 64;
        }
        // _mm256_store_si256((__m256i*)dists_out, totals);
        _mm256_stream_si256((__m256i*)dists_out, totals); // "non-temporal memory hint"
        luts -= NBytes * 64;
        // luts -= 8 * 32;
        dists_out += 32;
    }
}

// for debugging; should have same behavior as above (vectorized and
// non-unpacking) func
template<int NBytes>
inline void debug_lut_dists_block32_4b(const uint8_t* codes, const uint8_t* luts,
    uint8_t* dists_out, int64_t nblocks)
{
    static_assert(NBytes > 0, "Code length <= 0 is not valid");

    for (int64_t b = 0; b < nblocks; b++) {
        for (uint8_t i = 0; i < 32; i++) {
            dists_out[i] = 0;
        }

        for (uint8_t j = 0; j < NBytes; j++) {
            for (uint8_t i = 0; i < 32; i++) {
                auto code = codes[i];
                auto low_bits = code & mask_low4b;
                auto high_bits = code >> 4;

                auto offset = 16 * (i >= 16); // look in 2nd 16B of lut
                auto lut_low = luts + offset;
                auto lut_high = luts + 32 + offset;
                dists_out[i] += lut_low[low_bits];
                dists_out[i] += lut_high[high_bits];
            }
            codes += 32;
            luts += 64;
        }
        luts -= NBytes * 64;
        dists_out += 32;
    }
}

// luts must be of size [16][16]
template<int NBytes, class dist_t>
inline void lut_dists_4b(const uint8_t* codes, const dist_t* luts,
    dist_t* dists_out, int64_t N)
{
    static_assert(NBytes > 0, "Code length <= 0 is not valid");

    for (int64_t i = 0; i < N; i++) {
        dists_out[i] = 0;
        auto lut_ptr = luts;
        for (uint8_t j = 0; j < NBytes; j++) {
            uint8_t code_high = static_cast<uint8_t>(codes[j] >> 4);
            uint8_t code_low = static_cast<uint8_t>(codes[j] & mask_low4b);

            // // TODO rm after debug
            // dists_out[i] += popcount(code_left ^ j);
            // dists_out[i] += popcount(code_right ^ j);

            auto lut_low = lut_ptr;
            auto lut_high = lut_low + 16;
            dists_out[i] += lut_low[code_low];
            dists_out[i] += lut_high[code_high];
            // dists_out[i] += 42;
            // auto dist = lut_low[code_low] + lut_high[code_high];
            // std::cout << "---- " << (int)j << "\n";
            // std::cout << "code: " << (int)codes[j] << "\n";
            // std::cout << "code_low: " << (int)code_low << "\n";
            // std::cout << "code_high: " << (int)code_high << "\n";
            // std::cout << "dist: " << (int)dist << "\n";
            // std::cout <i< "dist: " << (int)dist << "\n";
            lut_ptr += 32;
        }
        codes += NBytes;
    }
}

// non-vectorized NBytes-byte codes, where each byte is one codeword
template<int NBytes, class dist_t>
inline void lut_dists_8b(const uint8_t* codes, const dist_t* luts,
    dist_t* dists_out, int64_t N)
{
    for (int64_t i = 0; i < N; i++) {
        dists_out[i] = 0;
        for (int j = 0; j < NBytes; j++) {
            auto lut_ptr = luts + 256 * j;
            dists_out[i] += lut_ptr[codes[j]];
        }
        codes += NBytes;
    }
}


template<int B, int NBytes, class dist_t> // block size, number of dimensions of each vector
inline void lut_dists_8b_vertical(const uint8_t* codes, const dist_t* luts,
   dist_t* dists_out, int64_t N)
{
    static constexpr int lut_sz = 256;
    static constexpr int nstripes = B; // # of rows of per block
    static_assert(B > 0, "Block size B must be > 0");
    const int64_t nblocks = N / B;
    assert(N % B == 0);

    dist_t accumulators[nstripes];

    for (int64_t b = 0; b < nblocks; b++) { // for each block
        for (int i = 0; i < nstripes; i++) {
            accumulators[i] = 0; // zero dists
        }
        for (int j = 0; j < NBytes; j++) { // for each pair of dimensions
            auto lut_ptr = luts + lut_sz * j;
            for (int i = 0; i < nstripes; i++) { // for each stripe
                auto idx = *codes;
                codes++;
                accumulators[i] += lut_ptr[idx];
            }
        }
        for (int i = 0; i < nstripes; i++) { // for each stripe
            dists_out[i] = accumulators[i];
        }
        dists_out += nstripes;
    }
}

template<int NBytes, class dist_t>
inline void lut_dists_12b(const uint8_t* codes, const dist_t* luts,
    dist_t* dists_out, int64_t N)
{
    static constexpr int ncodebook_bits = 12;
    static constexpr int total_bits = 8 * NBytes;
    static constexpr int ncodewords = total_bits / ncodebook_bits;
    static constexpr int codebook_sz = 1 << ncodebook_bits;
    static constexpr bool nbytes_multiple_of_8 = NBytes % 8 == 0;
    static constexpr uint16_t idx_mask = (1 << ncodebook_bits) - 1;
    static constexpr uint8_t low_4bits_mask = 0x0F;
    static constexpr uint8_t high_4bits_mask = 0xF0;
    static_assert(NBytes >= 2, "NBytes < 2 is not meaningful");

    for (int64_t i = 0; i < N; i++) {
        dists_out[i] = 0;
        for (int j = 0; j < ncodewords; j += 2) {
            auto lut_ptr0 = luts + codebook_sz * j;
            auto lut_ptr1 = lut_ptr0 + codebook_sz;

            // packed indices format:
            //
            // codes[j]             codes[j+1]           codes[j+2]
            // -------------------- ----------|---------- --------------------
            //       idx0[:8]       idx0[8:12] idx1[8:12]       idx1[:8]
            uint16_t low_bits = codes[3*j+1] & low_4bits_mask;
            uint16_t high_bits = codes[3*j+1] & high_4bits_mask;
            uint16_t idx0 = codes[3*j] + (low_bits << 8);
            uint16_t idx1 = codes[3*j+2] + (high_bits << 4);

            dists_out[i] += lut_ptr0[idx0];
            dists_out[i] += lut_ptr1[idx1];

            // codes += 3;
        }
        if (ncodewords % 2 != 0) {
            auto lut_ptr = luts + codebook_sz * (ncodewords - 1);
            uint16_t low_bits = codes[NBytes-1] & low_4bits_mask; // use last byte
            auto idx = codes[NBytes-2] + (low_bits << 8); // use 2nd to last byte
            dists_out[i] += lut_ptr[idx];
        }
        codes += NBytes;
        // codes += (NBytes % 3);
    }
}
template<int B, int NBytes, class dist_t>
inline void lut_dists_12b_vertical(const uint8_t* codes, const dist_t* luts,
    dist_t* dists_out, int64_t N)
{
    static constexpr int ncodebook_bits = 12;
    static constexpr int nstripes = B; // # of rows of per block
    static constexpr int total_bits = 8 * NBytes;
    static constexpr int ncodewords = total_bits / ncodebook_bits;
    static constexpr int codebook_sz = 1 << ncodebook_bits;
    static constexpr bool nbytes_multiple_of_8 = NBytes % 8 == 0;
    static constexpr uint16_t idx_mask = (1 << ncodebook_bits) - 1;
    static constexpr uint8_t low_4bits_mask = 0x0F;
    static constexpr uint8_t high_4bits_mask = 0xF0;
    static_assert(NBytes >= 2, "NBytes < 2 is not meaningful");
    // static_assert(NBytes % 2 == 0, "Odd NBytes not implemented!");
    // static_assert(NBytes % 3 == 0)
    static_assert(B > 0, "Block size B must be > 0");
    const int64_t nblocks = N / B;
    assert(N % B == 0);

    dist_t accumulators[nstripes];

    for (int64_t b = 0; b < nblocks; b++) { // for each block
        for (int i = 0; i < nstripes; i++) {
            accumulators[i] = 0; // zero dists
        }
        for (int j = 0; j < ncodewords; j += 2) { // for each pair of bytes
            auto lut_ptr0 = luts + codebook_sz * j;
            auto lut_ptr1 = lut_ptr0 + codebook_sz;

            for (int i = 0; i < nstripes; i++) { // for each stripe

                // packed indices format:
                //
                // codes[j]             codes[j+1]           codes[j+2]
                // -------------------- ----------|---------- --------------------
                //       idx0[:8]       idx0[8:12] idx1[8:12]       idx1[:8]
                uint16_t low_bits = codes[3*j+1] & low_4bits_mask;
                uint16_t high_bits = codes[3*j+1] & high_4bits_mask;
                uint16_t idx0 = codes[3*j] + (low_bits << 8);
                uint16_t idx1 = codes[3*j+2] + (high_bits << 4);

                accumulators[i] += lut_ptr0[idx0];
                accumulators[i] += lut_ptr1[idx1];

                // codes += 3;
            }
        }
        if (NBytes % 3 == 2) {  // one more codeword in the final 2 bytes
            auto lut_ptr = luts + codebook_sz * (ncodewords - 1);
            for (int i = 0; i < nstripes; i++) {
                uint16_t low_bits = codes[NBytes-1] & low_4bits_mask; // use last byte
                auto idx = codes[NBytes-2] + (low_bits << 8); // use 2nd to last byte
                accumulators[i] += lut_ptr[idx];
            }
        }
        // codes += (NBytes % 3);
        codes += NBytes;

        for (int i = 0; i < nstripes; i++) { // for each stripe
            dists_out[i] = accumulators[i];
        }
        dists_out += nstripes;
    }
}

// non-vectorized NBytes-byte codes, where each byte is one codeword
template<int NBytes, class dist_t>
inline void lut_dists_16b(const uint16_t* codes, const dist_t* luts,
    dist_t* dists_out, int64_t N)
{
    // std::cout << "lut dists 16b: using N = " << N << "\n";
    static_assert(NBytes % 2 == 0, "Odd NBytes not implemented!");
    static constexpr int lut_sz = (1 << 16);
    static constexpr int ncodewords = NBytes / 2;

    for (int64_t i = 0; i < N; i++) {
        dists_out[i] = 0;
        for (int j = 0; j < ncodewords; j++) {
            auto lut_ptr = luts + lut_sz * j;
            dists_out[i] += lut_ptr[codes[j]];
        }
        codes += ncodewords;
    }
}
template<int B, int NBytes, class dist_t> // block size, number of dimensions of each vector
inline void lut_dists_16b_vertical(const uint16_t* codes, const dist_t* luts,
   dist_t* dists_out, int64_t N)
{
    static constexpr int lut_sz = 1 << 16;
    static constexpr int nstripes = B; // # of rows of per block
    static constexpr int ncodewords = NBytes / 2;
    static_assert(NBytes % 2 == 0, "Odd NBytes not implemented!");
    static_assert(B > 0, "Block size B must be > 0");
    const int64_t nblocks = N / B;
    assert(N % B == 0);

    dist_t accumulators[nstripes];

    for (int64_t b = 0; b < nblocks; b++) { // for each block
        for (int i = 0; i < nstripes; i++) {
            accumulators[i] = 0; // zero dists
        }
        for (int j = 0; j < ncodewords; j++) { // for each pair of bytes
            auto lut_ptr = luts + lut_sz * j;
            for (int i = 0; i < nstripes; i++) { // for each stripe
                auto idx = *codes;
                codes++;
                accumulators[i] += lut_ptr[idx];
            }
        }
        for (int i = 0; i < nstripes; i++) { // for each stripe
            dists_out[i] = accumulators[i];
        }
        dists_out += nstripes;
    }
}


// // non-vectorized NBytes-byte codes, where each NCodeBookBits is one codeword
// EDIT: actually, I have no idea how to get this to work (remotely cleanly)
// when NBytes isn't a multiple of 8, because 10+ NCodeBookBits can span up
// to 3 bytes, and these can cross arbitrary uint64 (or even page) boundaries,
// so we basically have to do a bunch of math to figure out what bytes each
// code starts in, spans, and ends in
// template<int NBytes, int NCodeBookBits, class dist_t>
// inline void lut_dists_generic(const uint8_t* codes, const dist_t* luts,
//     dist_t* dists_out, int64_t N)
// {
//     static_assert(NCodeBookBits <= 16, "NCodeBookBits > 16 not implemented");
//     static_assert(NBytes <= 32, "NBytes > 32 not implemented");
//     static constexpr int total_bits = 8 * NBytes;
//     static constexpr int ncodewords = total_bits / NCodeBookBits;
//     static constexpr int codebook_sz = 1 << NCodeBookBits;
//     static constexpr bool nbytes_multiple_of_8 = NBytes % 8 == 0;
//     static constexpr uint16_t idx_mask = (1 << NCodeBookBits) - 1;
//     static constexpr uint16_t align_mask = ~static_cast<uint64_t>(0x07);

//     for (int64_t i = 0; i < N; i++) {
//         dists_out[i] = 0;
//         for (int j = 0; j < ncodewords; j++) {
//             auto lut_ptr = luts + codebook_sz * j;
//             auto idx_bits_start = j * NCodeBookBits;
//             auto idx_bits_end = idx_bits_start + NCodeBookBits;
//             uint16_t idx; // uint16 because we assume NCodeBookBits <= 16

//             if (NBytes == 8 || (nbytes_multiple_of_8 && idx_bits_end <= 64)) {
//                 // (first half of) code can be cast to 8B uint
//                 uint64_t code_uint = *(uint64_t*)(codes);
//                 idx = (code_uint >> idx_bits_start) & idx_mask;

//             } else if (nbytes_multiple_of_8 && idx_bits_start >= 64 && idx_bits_end <= 128) {
//                 // second 8B of code can be cast to 8B uint
//                 uint64_t code_uint = *(uint64_t*)(codes + 8);
//                 auto shift_amount = idx_bits_start - 64;
//                 idx = (code_uint >> shift_amount) & idx_mask;

//             } else if (nbytes_multiple_of_8 && idx_bits_start >= 128 && idx_bits_end <= 192) {
//                 // third 8B of code can be cast to 8B uint
//                 uint64_t code_uint = *(uint64_t*)(codes + 16);
//                 auto shift_amount = idx_bits_start - 128;
//                 idx = (code_uint >> shift_amount) & idx_mask;

//             } else if (nbytes_multiple_of_8 && idx_bits_start >= 192 && idx_bits_end <= 256) {
//                 // fourth 8B of code can be cast to 8B uint
//                 uint64_t code_uint = *(uint64_t*)(codes + 24);
//                 auto shift_amount = idx_bits_start - 192;
//                 idx = (code_uint >> shift_amount) & idx_mask;

//             } else {
//                 // // do evil to figure out the uint64 containing the index, and
//                 // // then slice out appropriate bits from there
//                 // uint8_t* uint64_start = codes & align_mask;
//                 // uint64_t code_uint = *(uint64_t*)uint64_start;
//                 // auto shift_amount = idx_bits_start - 192;

//             }
//             dists_out[i] += lut_ptr[idx];
//         }
//         codes += NBytes;
//     }
// }

// apparently I'm not actually calling this func in the tests? and seems to
// be redundant with lut_dists_4b above
template<int NBytes, class dist_t>
inline void lut_dists_8b_stride4b(const uint8_t* codes,
    const dist_t* luts, dist_t* dists_out, int64_t N)
{
    static_assert(NBytes > 0, "Code length <= 0 is not valid");

    // sum LUT distances both along byte boundaries and shifted by 4 bits
    // note that the shifted lookups assume that the corresponding LUT entries
    // are after the 8 entries for the non-shifted lookups
    // static const int NBytes = 8;
    for (int64_t i = 0; i < N; i++) {
        dists_out[i] = 0;
        for (int j = 0; j < NBytes; j++) {
            dists_out[i] += luts[j][codes[j]];
        }
        // TODO possibly shift 4 of these at once using SIMD instrs
        auto codes_as_uint = reinterpret_cast<const uint64_t*>(codes);
        auto shifted = (*codes_as_uint) >> 4;
        auto shifted_codes = reinterpret_cast<const uint8_t*>(shifted);
        for (int j = 0; j < NBytes - 1; j++) {
            dists_out[i] += luts[j + NBytes][shifted_codes[j]];
        }
        codes += NBytes;
    }
}

inline void popcount_8B(const uint8_t* codes, const uint64_t q,
    uint8_t* dists_out, int64_t N)
{
    for (int64_t i = 0; i < N; i++) {
        auto row_ptr = reinterpret_cast<const uint64_t*>(codes + (8 * i));
        dists_out[i] = __builtin_popcountll((*row_ptr) ^ q);
    }
}

template<int NBytes>
inline void popcount_generic(const uint8_t* codes, const uint8_t* q,
    uint8_t* dists_out, int64_t N)
{
    static_assert(NBytes % 8 == 0, "Popcount requires multiples of 8B");
    static_assert(NBytes > 0, "Code length <= 0 is not valid");
    static constexpr int row_width8 = NBytes / 8;

    const uint64_t* codes64 = reinterpret_cast<const uint64_t*>(codes);
    const uint64_t* q64 = reinterpret_cast<const uint64_t*>(q);

    for (int64_t i = 0; i < N; i++) {
        // auto row_ptr = reinterpret_cast<const uint64_t*>(codes + (NBytes * i));
        auto row_ptr = codes64 + row_width8 * i;
        dists_out[i] = 0;
        for (int b = 0; b < row_width8; b++) {
            const uint64_t* q_subvect = q64 + b;
            const uint64_t* subrow_ptr = row_ptr + b;
            dists_out[i] += __builtin_popcountll((*subrow_ptr) ^ (*q_subvect));
        }
    }
}

// returns a * b + c, elementwise; see eigen/src/Core/arch/AVX/PacketMath.h
inline __m256 fma(__m256 a, __m256 b, __m256 c) {
    __m256 res = c;
    __asm__("vfmadd231ps %[a], %[b], %[c]" : [c] "+x" (res) : [a] "x" (a), [b] "x" (b));
    return res;
}

template<int B, int D> // block size, number of dimensions of each vector
inline void float_dists_vertical(const float* X, const float* q,
    float* dists_out, int64_t N)
{
    static constexpr int packet_width = 8; // how many vecs we operate on at once
    static constexpr int nstripes = B / packet_width; // # of rows of 32B per block
    static_assert(B % packet_width == 0, "B must be a multiple of packet_width");
    static_assert(B > 0, "B must be > 0");
    const int64_t nblocks = N / B;
    assert(N % B == 0);

    __m256 accumulators[nstripes];

    for (int64_t b = 0; b < nblocks; b++) { // for each block
        for (int i = 0; i < nstripes; i++) {
            accumulators[i] = _mm256_setzero_ps();
        }

        for (int j = 0; j < D; j++) { // for each dimension
            auto q_broadcast = _mm256_set1_ps(q[j]);
            for (int i = 0; i < nstripes; i++) { // for each stripe
                auto x_col = _mm256_load_ps(X);
                X += packet_width;

                auto diff = _mm256_sub_ps(q_broadcast, x_col);
                accumulators[i] = fma(diff, diff, accumulators[i]);
                // auto prods = fma(diff, diff, accumulators[i]);
                // accumulators[i] = _mm256_add_ps(accumulators[i], prods);
            }
        }
        for (uint8_t i = 0; i < nstripes; i++) { // for each stripe
            _mm256_store_ps(dists_out, accumulators[i]);
            dists_out += packet_width;
        }
    }
}

template<int B, int D> // block size, number of dimensions of each vector
inline void byte_dists_vertical(const uint8_t* X, const uint8_t* q,
    uint16_t* dists_out, int64_t N)
{
    static constexpr int packet_width = 32; // how many vecs we operate on at once
    static constexpr int nstripes = B / packet_width; // # of rows of 32B per block
    static_assert(B % packet_width == 0, "B must be a multiple of packet_width");
    static_assert(B > 0, "B must be > 0");
    const int64_t nblocks = N / B * 2;
    assert(N % B == 0);

    // we assume that pairs of bytes are from the same vector for our
    // maddubs; so pretending q points to int16s makes broadcasting pairs work
    const uint16_t* q16 = reinterpret_cast<const uint16_t*>(q);

    __m256i accumulators[nstripes];

    for (int64_t b = 0; b < nblocks; b++) { // for each block
        for (int i = 0; i < nstripes; i++) {
            accumulators[i] = _mm256_setzero_si256(); // zero dists
        }
        for (int j = 0; j < D / 2; j++) { // for each pair of dimensions
            auto q_broadcast = _mm256_set1_epi16(q16[j]);
            for (int i = 0; i < nstripes; i++) { // for each stripe
                // auto x_col = _mm256_load_si256((__m256i*)X);
                auto x_col = stream_load_si256i((__m256i*)X);
                X += packet_width;

                auto diffs = _mm256_sub_epi8(q_broadcast, x_col);
                auto abs_diffs = _mm256_abs_epi8(diffs);
                auto prods = _mm256_maddubs_epi16(abs_diffs, abs_diffs);
                accumulators[i] = _mm256_adds_epi16(accumulators[i], prods);
            }
        }
        for (uint8_t i = 0; i < nstripes; i++) { // for each stripe
            _mm256_store_si256((__m256i*)dists_out, accumulators[i]);
            // we compute 16 dists because we assume that adjacent pairs of
            // bytes belong to the same vector; this is necessary for the
            // maddubs to be meaningful
            dists_out += packet_width / 2;
        }
    }
}

// just sums up the inputs and stores results; upper bound on how fast
// any of these functions could be
template<int B, int D> // block size, number of dimensions of each vector
inline void sum_inputs(const uint8_t* X, const uint8_t* q_unused,
   uint8_t* dists_out, int64_t N)
{
    static constexpr int packet_width = 32; // how many vecs we operate on at once
    static constexpr int nstripes = B / packet_width; // # of rows of 32B per block
    static_assert(B % packet_width == 0, "B must be a multiple of packet_width");
    static_assert(B > 0, "B must be > 0");
    const int64_t nblocks = N / B;
    assert(N % B == 0);

    __m256i accumulators[nstripes];

    for (int64_t b = 0; b < nblocks; b++) { // for each block
        for (int i = 0; i < nstripes; i++) {
            accumulators[i] = _mm256_setzero_si256(); // zero dists
        }
        for (int j = 0; j < D; j++) { // for each pair of dimensions
            for (int i = 0; i < nstripes; i++) { // for each stripe
                // auto x_col = _mm256_load_si256((__m256i*)X);
                auto x_col = stream_load_si256i((__m256i*)X);
                X += packet_width;
                accumulators[i] = _mm256_adds_epi16(accumulators[i], x_col);
            }
        }
        for (uint8_t i = 0; i < nstripes; i++) { // for each stripe
            _mm256_store_si256((__m256i*)dists_out, accumulators[i]);
            dists_out += packet_width;
        }
    }
}

} // namespace dist
#endif // __MULTI_CODEBOOK_HPP

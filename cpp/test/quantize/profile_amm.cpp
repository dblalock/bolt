//
//  profile_amm.cpp
//  Bolt
//
//  Created by DB on 10/7/19.
//  Copyright © 2019 D Blalock. All rights reserved.
//

#include <stdio.h>
#include <string>
#include <vector>

#ifdef BLAZE
    #include "test/external/catch.hpp"
    #include "src/quantize/bolt.hpp"
    #include "src/quantize/mithral.hpp"
    #include "src/quantize/multisplit.hpp"
    #include "src/utils/debug_utils.hpp"
    #include "src/utils/eigen_utils.hpp"
    #include "src/utils/timing_utils.hpp"
    #include "src/utils/memory.hpp"
    #include "test/testing_utils/testing_utils.hpp"
#else
    #include "catch.hpp"
    #include "bolt.hpp"
    #include "mithral.hpp"
    #include "multisplit.hpp"
    #include "debug_utils.hpp"
    #include "eigen_utils.hpp"
    #include "timing_utils.hpp"
    #include "testing_utils.hpp"
    #include "memory.hpp"
#endif

static constexpr int kNreps = 3;
// static constexpr int kNreps = 1;
static constexpr int kNtrials = 5;

TEST_CASE("amm profile smoketest", "[amm][profile]") {
    static constexpr int64_t nrows_enc = 128*100;   // number of rows to encode
    static constexpr int ncols = 64;               // length of vectors
    static constexpr int bits_per_codebook = 4;
    static constexpr int ncentroids = (1 << bits_per_codebook);
    static constexpr int nbytes = 8;

    static constexpr int nrows = nrows_enc;

    ColMatrix<float> centroids(ncentroids, ncols);
    centroids.setRandom();
    RowMatrix<float> X(nrows, ncols);
    X.setRandom();
    RowMatrix<uint8_t> encoding_out(nrows, nbytes);

    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt encode", kNtrials,
    encoding_out.data(), encoding_out.size(),
        bolt_encode<nbytes>(X.data(), nrows, ncols, centroids.data(),
                            encoding_out.data()));
}

TEST_CASE("amm profile split encode", "[amm][encode][split][profile]") {
    static const int N = 128 * 1000;
    // static const int N = 32;
    static const uint32_t D = 64;
    static const int ncodebooks = 4;
    static const int nsplits_per_codebook = 4;
    static const int total_nsplits = ncodebooks * nsplits_per_codebook;

    ColMatrix<float> X(N, D);
    X.setRandom();
    RowVector<uint32_t> splitdims_(total_nsplits);
    splitdims_.setRandom();
    RowVector<uint32_t> splitdims = splitdims_.unaryExpr(
        [](const uint32_t x) { return x % D; });
    RowVector<int8_t> splitvals(total_nsplits);
    splitvals.setRandom();
    RowVector<float> scales(total_nsplits);
    scales.setRandom();
    RowVector<float> offsets(total_nsplits);
    offsets.setRandom();
    ColMatrix<uint8_t> out(N, ncodebooks);

    // split_encode_8b_colmajor(
    //     X.data(), N, D, splitdims.data(), splitvals.data(), scales.data(),
    //     offsets.data(), ncodebooks, nsplits_per_codebook, out.data());
    // printf("sum of out: %d\n", out.sum());

    // split_encode_4b_colmajor(
    //         X.data(), N, D, splitdims.data(), splitvals.data(), scales.data(),
    //         offsets.data(), ncodebooks, out.data());

    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "split 8b encode     ", kNtrials,
        out.data(), out.size(),
        split_encode_8b_colmajor(
            X.data(), N, D, splitdims.data(), splitvals.data(), scales.data(),
            offsets.data(), ncodebooks, nsplits_per_codebook, out.data()));

    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "split 4b encode     ", kNtrials,
        out.data(), out.size(),
        split_encode_4b_colmajor(
            X.data(), N, D, splitdims.data(), splitvals.data(), scales.data(),
            offsets.data(), ncodebooks, out.data()));

    RowVector<float> splitvals_f32(total_nsplits);
    splitvals_f32.setRandom();
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "split 4b encode alt ", kNtrials,
        out.data(), out.size(),
        split_encode_4b_colmajor_alt(
            X.data(), N, D, splitdims.data(), splitvals_f32.data(), ncodebooks,
            out.data()));
}

TEST_CASE("amm profile multisplit encode", "[amm][encode][multisplit][profile]") {
    static const int N = 1024 * 1000;
    // static const int N = 128;
    static const uint32_t D = 64;
    // static const uint32_t D = 24;
    // static const uint32_t D = 4;
    // static const int ncodebooks = 64;
    // static const int ncodebooks = 32;
    // static const int ncodebooks = 16;
    // static const int ncodebooks = 8;
    static const int ncodebooks = 4;
    // static const int ncodebooks = 1;
    static const int nsplits_per_codebook = 4;
    static const int total_nsplits = ncodebooks * nsplits_per_codebook;
    static const int group_id_nbits = 4;
    static const int max_ngroups = 1 << group_id_nbits;

    ColMatrix<float> X(N, D);
    X.setRandom();
    RowVector<uint32_t> splitdims_(total_nsplits);
    splitdims_.setRandom();
    RowVector<uint32_t> splitdims = splitdims_.unaryExpr(
        [](const int x) { return x % D; });
    ColMatrix<int8_t> all_splitvals(max_ngroups, total_nsplits);
    all_splitvals.setRandom();
    // RowVector<float> scales(total_nsplits);
    RowVector<float> scales(MAX(D, total_nsplits)); // v2 needs D of these
    scales.setRandom();
    // RowVector<float> offsets(total_nsplits);
    RowVector<float> offsets(MAX(D, total_nsplits)); // v2 needs D of these
    offsets.setRandom();
    ColMatrix<uint8_t> out(N, ncodebooks);

    ColMatrix<int8_t> X_i8(N, D);
    X_i8.setRandom();

    // multisplit_encode_4b_colmajor_v2(
    //         X.data(), N, D, splitdims.data(), all_splitvals.data(),
    //         scales.data(), offsets.data(), ncodebooks, out.data(), X_i8.data());


    // multisplit_encode_8b_colmajor(
    //     X.data(), N, D, splitdims.data(), all_splitvals.data(), scales.data(),
    //     offsets.data(), ncodebooks, nsplits_per_codebook, out.data());
    // printf("sum of out: %d\n", out.sum());

    // printf("out.size(): %lu\n", out.size());
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "multisplit encode 8b", kNtrials,
        out.data(), out.size(),
        multisplit_encode_8b_colmajor(
            X.data(), N, D, splitdims.data(), all_splitvals.data(),
            scales.data(), offsets.data(), ncodebooks, nsplits_per_codebook,
            out.data()));

    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "multisplit encode 4b", kNtrials,
        out.data(), out.size(),
        multisplit_encode_4b_colmajor(
            X.data(), N, D, splitdims.data(), all_splitvals.data(),
            scales.data(), offsets.data(), ncodebooks, out.data()));

    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "multisplit deferp 4b", kNtrials,
    //     out.data(), out.size(),
    //     multisplit_encode_4b_colmajor<true>(
    //         X.data(), N, D, splitdims.data(), all_splitvals.data(),
    //         scales.data(), offsets.data(), ncodebooks, out.data()));

    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "multisplit enc i8 4b", kNtrials,
        out.data(), out.size(),
        multisplit_encode_4b_colmajor(
            X_i8.data(), N, D, splitdims.data(), all_splitvals.data(),
            ncodebooks, out.data()));

    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "multisplit enc f  v2", kNtrials,
        out.data(), out.size(),
        multisplit_encode_4b_colmajor_v2(
            X.data(), N, D, splitdims.data(), all_splitvals.data(),
            scales.data(), offsets.data(), ncodebooks, out.data(), X_i8.data()));

    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "multisplit deferp v2", kNtrials,
    //     out.data(), out.size(),
    //     multisplit_encode_4b_colmajor_v2<true>(
    //         X.data(), N, D, splitdims.data(), all_splitvals.data(),
    //         scales.data(), offsets.data(), ncodebooks, out.data(), X_i8.data()));
}

TEST_CASE("bolt + mithral scan speeds", "[amm][bolt][scan][profile]") {
    static constexpr int nblocks = 64 * 1000;
    static constexpr int nrows = nblocks * 32;
    // static constexpr int ncodebooks = 16;
    // static constexpr int ncodebooks = 64;
    static constexpr int ncodebooks = 4;
    static constexpr int ncentroids = 16;
    static constexpr int M = ncodebooks / 2;

    // create random codes from in [0, 15]
    ColMatrix<uint8_t> codes(nrows, ncodebooks);
    codes.setRandom();
    codes = codes.unaryExpr(
        [=](const uint8_t x) { return (uint8_t)(x % ncentroids); });

    // create random luts
    ColMatrix<uint8_t> luts(ncentroids, ncodebooks);
    luts.setRandom();
    luts = luts.array() / ncodebooks; // make max lut value small

    ColMatrix<int8_t> luts_signed(ncentroids, ncodebooks);
    luts_signed.setRandom();
    luts_signed = luts_signed.array() / ncodebooks; // make max lut value small

    // ColMatrix<int8_t> luts_i8(ncentroids, ncodebooks);
    // luts_i8.setRandom();
    // luts_i8 = luts_i8.array() / ncodebooks;

    // do the scan to compute the distances
    RowVector<uint8_t> dists_u8(nrows);
    RowVector<uint16_t> dists_u16(nrows);
    RowVector<uint16_t> dists_u16_safe(nrows);
    // RowVector<int16_t> dists_u16_colmajor(nrows);
    // RowVector<int16_t> dists_u16_colmajor_tile4(nrows);
    RowVector<int16_t> dists_u16_colmajor_mithral(nrows);

    // static constexpr int signed_luts = true;
    static constexpr int signed_luts = false;

    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan uint8                  ", kNtrials,
        dists_u8.data(), nrows,
        (bolt_scan<M, true, signed_luts>(
            codes.data(), luts.data(), dists_u8.data(), nblocks)));
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan uint16                 ", kNtrials,
        dists_u16.data(), nrows,
        (bolt_scan<M, false, signed_luts>(
            codes.data(), luts.data(), dists_u16.data(), nblocks)));
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan uint16 safe            ", kNtrials,
        dists_u16_safe.data(), nrows,
        (bolt_scan<M, true, signed_luts>(
            codes.data(), luts.data(), dists_u16_safe.data(), nblocks)));
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan colmajor", kNtrials,
    //     dists_u16_colmajor.data(), nrows,
    //     mithral_scan_unpacked_colmajor(codes.data(), nblocks, ncodebooks,
    //         luts.data(), dists_u16_colmajor.data()));
    // // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan colmajor tile4 upcast1", kNtrials,
    // //     dists_u16_colmajor_tile4.data(), nrows,
    // //     bolt_scan_colmajor_tile4<1>(codes.data(), nblocks, ncodebooks,
    // //         luts.data(), dists_u16_colmajor_tile4.data()));
    // // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan colmajor tile4 upcast2", kNtrials,
    // //     dists_u16_colmajor_tile4.data(), nrows,
    // //     bolt_scan_colmajor_tile4<2>(codes.data(), nblocks, ncodebooks,
    // //         luts.data(), dists_u16_colmajor_tile4.data()));
    // // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan colmajor tile4 upcast4", kNtrials,
    // //     dists_u16_colmajor_tile4.data(), nrows,
    // //     bolt_scan_colmajor_tile4<4>(codes.data(), nblocks, ncodebooks,
    // //         luts.data(), dists_u16_colmajor_tile4.data()));
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan colmajor tile4 upcast1 packed", kNtrials,
    //     dists_u16_colmajor_tile4.data(), nrows,
    //     bolt_scan_colmajor_tile4_packed<1>(codes.data(), nblocks, ncodebooks,
    //         luts.data(), dists_u16_colmajor_tile4.data()));
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan colmajor tile4 upcast2 packed", kNtrials,
    //     dists_u16_colmajor_tile4.data(), nrows,
    //     bolt_scan_colmajor_tile4_packed<2>(codes.data(), nblocks, ncodebooks,
    //         luts.data(), dists_u16_colmajor_tile4.data()));
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan colmajor tile4 upcast4 packed", kNtrials,
    //     dists_u16_colmajor_tile4.data(), nrows,
    //     mithral_scan_tile4<4>(codes.data(), nblocks, ncodebooks,
    //         luts.data(), dists_u16_colmajor_tile4.data()));

    static constexpr int noutputs = 1;
    static constexpr int noutputs_per_block = 1;

    // // create random codes from in [0, 15]
    // ColMatrix<uint8_t> codes(nrows, ncodebooks);
    // codes.setRandom();
    // codes = codes.array() / ncentroids;

    ColMatrix<int8_t> all_luts(noutputs * ncentroids, ncodebooks);
    luts_signed.setRandom();
    luts_signed = luts_signed.array() / ncodebooks; // make max lut value small

    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "mithral scan UpcastEvery=16      ", kNtrials,
        dists_u16_colmajor_mithral.data(), nrows * noutputs,
        mithral_scan<16>(codes.data(), nrows, ncodebooks,
            noutputs, luts_signed.data(), dists_u16_colmajor_mithral.data()));
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "mithral scan UpcastEvery=8       ", kNtrials,
        dists_u16_colmajor_mithral.data(), nrows * noutputs,
        mithral_scan<8>(codes.data(), nrows, ncodebooks,
            noutputs, luts_signed.data(), dists_u16_colmajor_mithral.data()));
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "mithral scan UpcastEvery=4       ", kNtrials,
        dists_u16_colmajor_mithral.data(), nrows * noutputs,
        mithral_scan<4>(codes.data(), nrows, ncodebooks,
            noutputs, luts_signed.data(), dists_u16_colmajor_mithral.data()));
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "mithral scan UpcastEvery=2       ", kNtrials,
        dists_u16_colmajor_mithral.data(), nrows * noutputs,
        mithral_scan<2>(codes.data(), nrows, ncodebooks,
            noutputs, luts_signed.data(), dists_u16_colmajor_mithral.data()));
}

template<int M, bool Safe=false, class dist_t=void>
void _bolt_query(const uint8_t* codes, int nblocks,
    const float* q, int ncols,
    const float* centroids,
    uint8_t* lut_out, dist_t* dists_out)
{
    bolt_lut<M>(q, ncols, centroids, lut_out);
    bolt_scan<M, Safe>(codes, lut_out, dists_out, nblocks);
}

template<int ncodebooks>
void _amm_bolt(const float* Q, int nrows, int ncols, const float* centroids,
               uint8_t* lut_out, uint16_t* dists_out,
               const uint8_t* codes, int nblocks)
{
    // in contrast to multisplit, this precomputes encodings and computes
    // new LUTs when a query comes in, instead of the reverse
    static constexpr int M = ncodebooks / 2;
    auto q_ptr = Q;
    auto dists_ptr = dists_out;
    for (int i = 0; i < nrows; i++) {  // rows in query matrix, not codes
        _bolt_query<M, true>(
            codes, nblocks, q_ptr, ncols, centroids, lut_out, dists_ptr);
        q_ptr += ncols;
        dists_ptr += nblocks * 32;
    }
}

template<int ncodebooks>
void _template_profile_bolt_amm(uint32_t N, uint32_t D, uint32_t M) {
    static const int ncentroids = 16;
    auto orig_M = M;
    auto orig_D = D;

    auto nblocks = (M + 31) / 32;
    M = 32 * nblocks;
    if (D % ncodebooks) {  // ensure that ncodebooks evenly divides D
        D += (ncodebooks - (D % ncodebooks));
    }

    // stuff for LUT creation
    ColMatrix<float> centroids(ncentroids, D);
    centroids.setRandom();
    RowMatrix<float> Q(N, D);
    Q.setRandom();
    ColMatrix<uint8_t> lut_out(ncentroids, ncodebooks);
    RowVector<float> offsets(D);
    offsets.setRandom();
    float scaleby = 3; // arbitrary number

    // additional stuff for distance computation

    ColMatrix<uint8_t> codes(M, ncodebooks / 2);
    codes.setRandom();
    ColMatrix<uint16_t> dists_u16(N, M);

    std::string msg = string_with_format(
        "amm bolt N, D, M, ncodebooks: %6d, %3d, %3d, %2d \t",
        N, orig_D, orig_M, ncodebooks);
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        dists_u16.data(), dists_u16.size(),
        _amm_bolt<ncodebooks>(Q.data(), N, D, centroids.data(), lut_out.data(),
                  dists_u16.data(), codes.data(), nblocks));
}

void _profile_bolt_amm(uint32_t N, uint32_t D, uint32_t M, int ncodebooks) {
    if (ncodebooks > D) { return; }
    switch(ncodebooks) {
        case 2: _template_profile_bolt_amm<2>(N, D, M); break;
        case 4: _template_profile_bolt_amm<4>(N, D, M); break;
        case 8: _template_profile_bolt_amm<8>(N, D, M); break;
        case 16: _template_profile_bolt_amm<16>(N, D, M); break;
        case 32: _template_profile_bolt_amm<32>(N, D, M); break;
        case 64: _template_profile_bolt_amm<64>(N, D, M); break;
        default: break;
    }
}
// template<int ncodebooks>
// void _amm_bolt(const float* X, int64_t nrows, int ncols,
//                const float* centroids, uint8_t* out_enc, const uint8_t* luts,
//                uint16_t* out_mat, int out_ncols)
// {
//     static constexpr int M = ncodebooks / 2;
//     bolt_encode<M>(X, nrows, ncols, centroids, out_enc);
//     auto nblocks = nrows / 32;
//     auto out_ptr = out_mat;
//     auto lut_ptr = luts;
//     for (int i = 0; i < out_ncols; i++) {
//         bolt_scan_colmajor_tile4_packed<4>(out_enc, nblocks, ncodebooks,
//             luts, out_ptr);
//         out_ptr += nrows;
//         lut_ptr += 16 * ncodebooks;
//     }
// }

TEST_CASE("amm lut+scan bolt", "[amm][matmul][bolt][profile]") {
    std::vector<int> ncodebooks {4, 8, 16, 32, 64};
    for (auto c  : ncodebooks) {
        printf("ncodebooks = %d\n", c);
        _profile_bolt_amm(10000, 512, 10, c);     // cifar10

        // TODO uncomment below

        _profile_bolt_amm(10000, 512, 100, c);    // cifar100
        _profile_bolt_amm(57593, 24, 3, c);       // ecg
        // _profile_bolt_amm(115193, 24, 3, c);      // ecg
        // _profile_bolt_amm(230393, 24, 3, c);      // ecg
        _profile_bolt_amm(49284, 27, 2, c);       // caltech
    }
}

void _amm_mithral_unpacked(const float* X, int64_t nrows, int ncols,
    const uint32_t* splitdims, const int8_t* all_splitvals,
    const float* scales, const float* offsets,
    int ncodebooks, uint8_t* out_enc, int8_t* luts,
    int16_t* out_mat, int out_ncols)
{
    // multisplit_encode_8b_colmajor(
    //     X, nrows, ncols, splitdims, all_splitvals, scales,
    //     offsets, ncodebooks, nsplits_per_codebook, out_enc);
    multisplit_encode_4b_colmajor(
        X, nrows, ncols, splitdims, all_splitvals, scales,
        offsets, ncodebooks, out_enc);
    auto nblocks = nrows / 32;
    auto out_ptr = out_mat;
    auto lut_ptr = luts;
    for (int i = 0; i < out_ncols; i++) {
        _mithral_scan_tile4<4, false>(out_enc, nblocks, ncodebooks,
            luts, out_ptr);
        out_ptr += nrows;
        lut_ptr += 16 * ncodebooks;
    }
}

void _amm_mithral_packed(const float* X, int64_t nrows, int ncols,
    const uint32_t* splitdims, const int8_t* all_splitvals,
    const float* scales, const float* offsets,
    int ncodebooks, uint8_t* out_enc, uint8_t* out_enc_packed, int8_t* luts,
    int16_t* out_mat, int out_ncols)
{
    // multisplit_encode_8b_colmajor(
    //     X, nrows, ncols, splitdims, all_splitvals, scales,
    //     offsets, ncodebooks, nsplits_per_codebook, out_enc);
    multisplit_encode_4b_colmajor(
        X, nrows, ncols, splitdims, all_splitvals, scales,
        offsets, ncodebooks, out_enc);
    zip2_4b_colmajor(out_enc, nrows, ncodebooks, out_enc_packed);
    auto nblocks = nrows / 32;
    auto out_ptr = out_mat;
    auto lut_ptr = luts;
    for (int i = 0; i < out_ncols; i++) {
        _mithral_scan_tile4<4, true>(out_enc, nblocks, ncodebooks,
            luts, out_ptr);
        out_ptr += nrows;
        lut_ptr += 16 * ncodebooks;
    }
}

// template<int UpcastEvery=4>
void _amm_mithral(const float* X, int64_t nrows, int ncols,
    const uint32_t* splitdims, const int8_t* all_splitvals,
    const float* scales, const float* offsets,
    int ncodebooks, uint8_t* out_enc, uint8_t* out_enc_packed, int8_t* luts,
    int16_t* out_mat, int out_ncols)
{
    multisplit_encode_4b_colmajor(
        X, nrows, ncols, splitdims, all_splitvals, scales,
        offsets, ncodebooks, out_enc);
    zip4_4b_colmajor(out_enc, nrows, ncodebooks, out_enc_packed);
    // mithral_scan<2>(out_enc_packed, nrows, ncodebooks, out_ncols, luts, out_mat);
    // mithral_scan<UpcastEvery>(out_enc_packed, (int)nrows, ncodebooks, out_ncols, luts, out_mat);
    mithral_scan<4>(out_enc_packed, (int)nrows, ncodebooks, out_ncols, luts, out_mat);
    // mithral_scan<8>(out_enc_packed, nrows, ncodebooks, out_ncols, luts, out_mat);
    // auto out_ptr = out_mat;
    // auto lut_ptr = luts;
    // for (int i = 0; i < out_ncols; i++) {
    //     mithral_scan<4>(out_enc_packed, nrows, ncodebooks, 1, luts, out_ptr);
    //     out_ptr += nrows;
    //     lut_ptr += 16 * ncodebooks;
    // }
}

void _amm_mithral_just_enc(const float* X, int64_t nrows, int ncols,
    const uint32_t* splitdims, const int8_t* all_splitvals,
    const float* scales, const float* offsets,
    int ncodebooks, uint8_t* out_enc, uint8_t* out_enc_packed)
{
    multisplit_encode_4b_colmajor(
        X, nrows, ncols, splitdims, all_splitvals, scales,
        offsets, ncodebooks, out_enc);
    // multisplit_encode_8b_colmajor(
    //     X, nrows, ncols, splitdims, all_splitvals, scales,
    //     offsets, ncodebooks, 4, out_enc);
    zip4_4b_colmajor(out_enc, nrows, ncodebooks, out_enc_packed);
    // memset(out_mat, 42, nrows * out_ncols * sizeof(out_mat[0]));
}

void _amm_mithral_just_zip(const float* X, int64_t nrows, int ncols,
    const uint32_t* splitdims, const int8_t* all_splitvals,
    const float* scales, const float* offsets,
    int ncodebooks, uint8_t* out_enc, uint8_t* out_enc_packed)
{
    zip4_4b_colmajor(out_enc, nrows, ncodebooks, out_enc_packed);
}


void _amm_mithral_just_zip2(const float* X, int64_t nrows, int ncols,
    const uint32_t* splitdims, const int8_t* all_splitvals,
    const float* scales, const float* offsets,
    int ncodebooks, uint8_t* out_enc, uint8_t* out_enc_packed)
{
    zip2_4b_colmajor(out_enc, nrows, ncodebooks, out_enc_packed);
}

void _amm_mithral_just_zip_bolt(const float* X, int64_t nrows, int ncols,
    const uint32_t* splitdims, const int8_t* all_splitvals,
    const float* scales, const float* offsets,
    int ncodebooks, uint8_t* out_enc, uint8_t* out_enc_packed)
{
    zip_bolt_colmajor(out_enc, nrows, ncodebooks, out_enc_packed);
}

// void _amm_mithral_enc(const float* X, int64_t nrows, int ncols,
//     const uint32_t* splitdims, const int8_t* all_splitvals,
//     const float* scales, const float* offsets,
//     int ncodebooks, uint8_t* out_enc, uint8_t* out_enc_packed, int8_t* luts,
//     int16_t* out_mat, int out_ncols)
// {
//     multisplit_encode_4b_colmajor(
//         X, nrows, ncols, splitdims, all_splitvals, scales,
//         offsets, ncodebooks, out_enc);
//     // multisplit_encode_8b_colmajor(
//     //     X, nrows, ncols, splitdims, all_splitvals, scales,
//     //     offsets, ncodebooks, 4, out_enc);
//     zip4_4b_colmajor(out_enc, nrows, ncodebooks, out_enc_packed);
//     // mithral_scan<2>(out_enc_packed, nrows, ncodebooks, out_ncols, luts, out_mat);
//     mithral_scan<4>(out_enc_packed, 64, ncodebooks, out_ncols, luts, out_mat);
//     // mithral_scan<8>(out_enc_packed, nrows, ncodebooks, out_ncols, luts, out_mat);
//     // auto out_ptr = out_mat;
//     // auto lut_ptr = luts;
//     // for (int i = 0; i < out_ncols; i++) {
//     //     mithral_scan<4>(out_enc_packed, nrows, ncodebooks, 1, luts, out_ptr);
//     //     out_ptr += nrows;
//     //     lut_ptr += 16 * ncodebooks;
//     // }
// }

void _profile_multisplit(uint32_t N, uint32_t D, uint32_t M, int ncodebooks) {
    // static const int N = 128 * 1000;
    // static const uint32_t D = 64;
    int out_ncols = M;
    int ncentroids = 16;
    // static const int ncodebooks = 4;
    int nsplits_per_codebook = 4;
    int total_nsplits = ncodebooks * nsplits_per_codebook;
    int group_id_nbits = 4;
    int max_ngroups = 1 << group_id_nbits;

    auto orig_N = N;

    if (N % 32 > 0) {
        // TODO better way of dealing with block size needs
        N += (32 - (N % 32));
    }
    assert(N % 32 == 0);
    // if (D % ncodebooks > 0) { // TODO rm need for even multiple?
    //     D += ncodebooks - (D % ncodebooks);
    // }

    // printf("N, D, M, ncodebooks: %6d, %3d, %3d, %2d, \t", N, D, M, ncodebooks);
    // printf("total_nsplits: %d\n", total_nsplits);

    // create data + info needed for encoding
    ColMatrix<float> X(N, D);
    X.setRandom();
    RowVector<uint32_t> splitdims_(total_nsplits);
    splitdims_.setRandom();
    RowVector<uint32_t> splitdims = splitdims_.unaryExpr(
        [=](const int x) { return x % D; });
    ColMatrix<int8_t> all_splitvals(max_ngroups, total_nsplits);
    all_splitvals.setRandom();
    RowVector<float> scales(total_nsplits);
    scales.setRandom();
    RowVector<float> offsets(total_nsplits);
    offsets.setRandom();
    ColMatrix<uint8_t> codes(N, ncodebooks);
    codes.setRandom();
    codes = codes.array() / 16;
    ColMatrix<uint8_t> codes_packed(N, ncodebooks / 2);
    codes_packed.setRandom();

    // create random luts
    ColMatrix<int8_t> luts(ncentroids, ncodebooks * out_ncols);
    luts.setRandom();
    luts = luts.array() / ncodebooks; // make max lut value small

    // storage for overall distances
    ColMatrix<int16_t> out_mat(N, out_ncols);

//    std::string msg = string_with_format(
//        "amm multisplit   N, D, M, ncodebooks: %6d, %3d, %3d, %2d \t",
//        orig_N, D, M, ncodebooks);
//    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
//        out_mat.data(), out_mat.size(),
//        _amm_multisplit(
//            X.data(), N, D, splitdims.data(), all_splitvals.data(),
//            scales.data(), offsets.data(), ncodebooks, nsplits_per_codebook,
//            codes.data(), luts.data(), out_mat.data(), out_ncols));

    std::string msg;
    printf("----\n");

    msg = string_with_format(
        "amm mithral      N, D, M, ncodebooks: %6d, %3d, %3d, %2d \t",
        orig_N, D, M, ncodebooks);
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        out_mat.data(), out_mat.size(),
        _amm_mithral(
            X.data(), N, D, splitdims.data(), all_splitvals.data(),
            scales.data(), offsets.data(), ncodebooks,
            codes.data(), codes_packed.data(), luts.data(), out_mat.data(), out_ncols));

    msg = string_with_format(
        "amm mithral unpa N, D, M, ncodebooks: %6d, %3d, %3d, %2d \t",
        orig_N, D, M, ncodebooks);
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        out_mat.data(), out_mat.size(),
        _amm_mithral_unpacked(
            X.data(), N, D, splitdims.data(), all_splitvals.data(),
            scales.data(), offsets.data(), ncodebooks,
            codes.data(), luts.data(), out_mat.data(), out_ncols));

    msg = string_with_format(
        "amm mithral pack N, D, M, ncodebooks: %6d, %3d, %3d, %2d \t",
        orig_N, D, M, ncodebooks);
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        out_mat.data(), out_mat.size(),
        _amm_mithral_packed(
            X.data(), N, D, splitdims.data(), all_splitvals.data(),
            scales.data(), offsets.data(), ncodebooks,
            codes.data(), codes_packed.data(), luts.data(), out_mat.data(), out_ncols));

    // X.setRandom();
    msg = string_with_format(
        "amm mithral enc  N, D, M, ncodebooks: %6d, %3d, %3d, %2d \t",
        orig_N, D, M, ncodebooks);
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        codes_packed.data(), codes_packed.size(),
        _amm_mithral_just_enc(
            X.data(), N, D, splitdims.data(), all_splitvals.data(),
            scales.data(), offsets.data(), ncodebooks,
            codes.data(), codes_packed.data()));

    msg = string_with_format(
        "amm mithral zip4 N, D, M, ncodebooks: %6d, %3d, %3d, %2d \t",
        orig_N, D, M, ncodebooks);
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        codes_packed.data(), codes_packed.size(),
        _amm_mithral_just_zip(
            X.data(), N, D, splitdims.data(), all_splitvals.data(),
            scales.data(), offsets.data(), ncodebooks,
            codes.data(), codes_packed.data()));
    // msg = string_with_format(
    //     "amm mithral zip2 N, D, M, ncodebooks: %6d, %3d, %3d, %2d \t",
    //     orig_N, D, M, ncodebooks);
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
    //     codes_packed.data(), codes_packed.size(),
    //     _amm_mithral_just_zip2(
    //         X.data(), N, D, splitdims.data(), all_splitvals.data(),
    //         scales.data(), offsets.data(), ncodebooks,
    //         codes.data(), codes_packed.data()));
    msg = string_with_format(
        "amm mithral zipb N, D, M, ncodebooks: %6d, %3d, %3d, %2d \t",
        orig_N, D, M, ncodebooks);
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        codes_packed.data(), codes_packed.size(),
        _amm_mithral_just_zip_bolt(
            X.data(), N, D, splitdims.data(), all_splitvals.data(),
            scales.data(), offsets.data(), ncodebooks,
            codes.data(), codes_packed.data()));
}

TEST_CASE("amm enc+scan multisplit", "[amm][multisplit][mithral][matmul][profile]") {
    // _profile_multisplit(128 * 1000, 64, 32, 4);
    std::vector<int> ncodebooks {4, 8, 16, 32, 64};
    // std::vector<int> ncodebooks {4};
    for (auto c  : ncodebooks) {
        printf("ncodebooks = %d\n", c);
        // _profile_multisplit(128 * 1000, 64, 32, c);
        _profile_multisplit(10000, 512, 10, c);     // cifar10

        // TODO uncomment below

        _profile_multisplit(10000, 512, 100, c);    // cifar100
        _profile_multisplit(57593, 24, 3, c);       // ecg
        // _profile_multisplit(115193, 24, 3, c);      // ecg
        // _profile_multisplit(230393, 24, 3, c);      // ecg
        _profile_multisplit(49284, 27, 2, c);       // caltech
    }
}

template<class MatrixT1, class MatrixT2, class MatrixT3>
void _run_matmul(const MatrixT1& X, const MatrixT2& Q, MatrixT3& out) {
   out.noalias() = X * Q;
}

template<class MatrixT1, class MatrixT2, class MatrixT3>
void _run_our_matmul(const MatrixT1& X, const MatrixT2& Q, MatrixT3& out) {
    // not actually faster than the eigen one
    sgemm_colmajor(
        X.data(), Q.data(), (int)X.rows(), (int)X.cols(), (int)Q.cols(), out.data());
}

void _profile_matmul(uint32_t N, uint32_t D, uint32_t M) {
    // using MatrixT = ColMatrix<float>;
    using MatrixT = ColMatrix<float>; // faster for small batches, else slower

    // N = 1024; // TODO rm
    // N = 2048; // TODO rm
    // N = 4096; // TODO rm

    auto orig_N = N;
    auto orig_D = D;
    auto orig_M = M;
    // if (N % 8 > 0) {  // match padding that other algos get
    //     N += (8 - (N % 8));
    // }
    // if ((D % 4 > 0) && (D > 16)) {
    //     D += (4 - (D % 4));
    // }
    // auto target_factor = 2;
    // if ((M % target_factor > 0) && (M > target_factor)) {
    //     M += (target_factor - (M % target_factor));
    // }

    // create random data
    MatrixT X(N, D);
    X.setRandom();
    MatrixT W(D, M);
    W.setRandom();

    // create output matrix to avoid malloc
    MatrixT out(N, M);
    out.setRandom();

    // time it
    {
        std::string msg = string_with_format("blas matmul N, D, M: %6d, %3d, %3d \t",
            orig_N, orig_D, orig_M);
        REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
            out.data(), out.size(),
            _run_matmul(X, W, out));
    }
    {
        std::string msg = string_with_format("our  matmul N, D, M: %6d, %3d, %3d \t",
            orig_N, orig_D, orig_M);
        REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
            out.data(), out.size(),
            _run_our_matmul(X, W, out));
    }
}

TEST_CASE("amm exact matmul", "[amm][matmul][exact][profile]") {
    int N, M;
    // std::vector<int> dvals {2, 4, 6, 8, 12, 16, 24, 32, 48, 64};
    std::vector<int> dvals {2, 4, 8, 16, 32, 64}; // TODO uncomment above

    N = 10000; M = 10;          // cifar10
    for (auto d : dvals) {
        _profile_matmul(N, d, M);
    }
    _profile_matmul(N, 512, M);


    // TODO uncomment below


    N = 10000; M = 100;         // cifar100
    for (auto d : dvals) {
        _profile_matmul(N, d, M);
    }
    _profile_matmul(N, 512, M);

    M = 3;                      // ecg
    int D = 24;
    // std::vector<int> ecg_nvals {57593, 115193, 230393};
    std::vector<int> ecg_nvals {57593};
    for (auto n : ecg_nvals) {
        for (auto d : dvals) {
            if (d > D) { continue; }
            _profile_matmul(n, d, M);
        }
        _profile_matmul(n, D, M);
    }

    N = 49284; M = 2; D = 27;          // caltech
    for (auto d : dvals) {
        if (d > D) { continue; }
        _profile_matmul(N, d, M);
    }
    _profile_matmul(N, D, M);
}

// TEST_CASE("amm profile bolt scan colmajor tile4", "[amm][bolt][profile]") {
//     static constexpr int nblocks = nblocks_scan;
//     static constexpr int nrows = nblocks_scan * 32;
//     // create random codes from in [0, 15]
//     ColMatrix<uint8_t> codes(nrows, ncodebooks);
//     codes.setRandom();
//     codes = codes.array() / 16;

//     // create random luts
//     ColMatrix<uint8_t> luts(ncentroids, ncodebooks);
//     luts.setRandom();
//     luts = luts.array() / (2 * M); // make max lut value small

//     RowVector<uint16_t> dists_u16(nrows);

// }

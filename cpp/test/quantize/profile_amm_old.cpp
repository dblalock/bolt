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
    #include "src/sketch.hpp"
    #include "src/quantize/bolt.hpp"
    #include "src/quantize/mithral_v1.hpp"
    #include "src/quantize/multisplit.hpp"
    #include "src/utils/debug_utils.hpp"
    #include "src/utils/eigen_utils.hpp"
    #include "src/utils/timing_utils.hpp"
    #include "src/utils/memory.hpp"
    // #include "src/utils/arr/array_utils.hpp"  // just for rand_idxs
    #include "test/testing_utils/testing_utils.hpp"
#else
    #include "catch.hpp"
    #include "bolt.hpp"
    #include "mithral_v1.hpp"
    #include "multisplit.hpp"
    #include "debug_utils.hpp"
    #include "eigen_utils.hpp"
    #include "sketch.hpp"
    #include "timing_utils.hpp"
    #include "testing_utils.hpp"
    // #include "array_utils.hpp"  // just for rand_idxs
    #include "memory.hpp"
#endif

static constexpr int kNreps = 3;
// static constexpr int kNreps = 1;
static constexpr int kNtrials = 20;
// static constexpr int kNreps = 1;
// static constexpr int kNtrials = 1;

// static constexpr int CALTECH_N = 49284;
// static constexpr int CALTECH_D = 27;
// static constexpr int CALTECH_M = 2;
// static constexpr int CIFAR10_N = 10000;
// static constexpr int CIFAR10_D = 512;
// static constexpr int CIFAR10_M = 10;
// static constexpr int CIFAR100_N = 10000;
// static constexpr int CIFAR100_D = 512;
// static constexpr int CIFAR100_M = 100;
// static constexpr int UCR_N = 1896;
// static constexpr int UCR_D = 320;
// static constexpr int UCR_M = 128;

struct MatmulTaskShape { int N, D, M; const char* name; };
static constexpr MatmulTaskShape kCaltechTaskShape {49284, 27, 2, "Caltech"};
static constexpr MatmulTaskShape kCifar10TaskShape {10000, 512, 10, "Cifar10"};
static constexpr MatmulTaskShape kCifar100TaskShape {
    10000, 512, 100, "Cifar100"};
static constexpr MatmulTaskShape kUcrTaskShape {1896, 320, 128, "UCR"};


TEST_CASE("amm profile smoketest old", "[amm][profile][old]") {
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

TEST_CASE("amm profile split encode old",
          "[amm][encode][split][profile][old]")
{
    static const int N = 1024 * 1000;
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

TEST_CASE("amm profile multisplit encode old",
          "[amm][encode][multisplit][profile][old]")
{
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
    ColMatrix<int16_t> X_i16(N, D);
    X_i16.setRandom();

    RowVector<int16_t> offsets_i16(total_nsplits);
    offsets_i16.setRandom();
    RowVector<uint8_t> shifts(total_nsplits);
    shifts.setRandom();

    // multisplit_encode_4b_colmajor_v2(
    //         X.data(), N, D, splitdims.data(), all_splitvals.data(),
    //         scales.data(), offsets.data(), ncodebooks, out.data(), X_i8.data());


    // multisplit_encode_8b_colmajor(
    //     X.data(), N, D, splitdims.data(), all_splitvals.data(), scales.data(),
    //     offsets.data(), ncodebooks, nsplits_per_codebook, out.data());
    // printf("sum of out: %d\n", out.sum());

    // printf("out.size(): %lu\n", out.size());
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "multisplit encode 8b         ", kNtrials,
        out.data(), out.size(),
        multisplit_encode_8b_colmajor(
            X.data(), N, D, splitdims.data(), all_splitvals.data(),
            scales.data(), offsets.data(), ncodebooks, nsplits_per_codebook,
            out.data()));

    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "multisplit encode 4b         ", kNtrials,
        out.data(), out.size(),
        multisplit_encode_4b_colmajor(
            X.data(), N, D, splitdims.data(), all_splitvals.data(),
            scales.data(), offsets.data(), ncodebooks, out.data()));

    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "multisplit deferp 4b ", kNtrials,
    //     out.data(), out.size(),
    //     multisplit_encode_4b_colmajor<true>(
    //         X.data(), N, D, splitdims.data(), all_splitvals.data(),
    //         scales.data(), offsets.data(), ncodebooks, out.data()));

    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "multisplit enc i8 4b         ", kNtrials,
        out.data(), out.size(),
        multisplit_encode_4b_colmajor(
            X_i8.data(), N, D, splitdims.data(), all_splitvals.data(),
            ncodebooks, out.data()));

    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "multisplit enc i8 bolt 4b    ", kNtrials,
    //     out.data(), out.size(),
    //     multisplit_encode_4b_colmajor<Layouts::BoltNoPack>(
    //         X_i8.data(), N, D, splitdims.data(), all_splitvals.data(),
    //         ncodebooks, out.data()));

    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "multisplit enc i16 4b        ", kNtrials,
        out.data(), out.size(),
        multisplit_encode_4b_colmajor(
            X_i16.data(), N, D, splitdims.data(), all_splitvals.data(),
            shifts.data(), offsets_i16.data(), ncodebooks, out.data()));

    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "multisplit enc i16 bolt 4b   ", kNtrials,
    //     out.data(), out.size(),
    //     multisplit_encode_4b_colmajor<Layouts::BoltNoPack>(
    //         X_i16.data(), N, D, splitdims.data(), all_splitvals.data(),
    //         shifts.data(), offsets_i16.data(), ncodebooks, out.data()));

    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "multisplit enc f v2          ", kNtrials,
        out.data(), out.size(),
        multisplit_encode_4b_colmajor_v2(
            X.data(), N, D, splitdims.data(), all_splitvals.data(),
            scales.data(), offsets.data(), ncodebooks, out.data(), X_i8.data()));

    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "multisplit deferp v2 ", kNtrials,
    //     out.data(), out.size(),
    //     multisplit_encode_4b_colmajor_v2<true>(
    //         X.data(), N, D, splitdims.data(), all_splitvals.data(),
    //         scales.data(), offsets.data(), ncodebooks, out.data(), X_i8.data()));
}

TEST_CASE("bolt + mithral scan speeds",
          "[amm][bolt][scan][profile][old]")
{
    static constexpr int nblocks = 64 * 1000;
    // static constexpr int nblocks = 2;
    // static constexpr int nblocks = 256;
    static constexpr int nrows = nblocks * 32;
    // static constexpr int ncodebooks = 64;
    // static constexpr int ncodebooks = 32;
    // static constexpr int ncodebooks = 24;
    static constexpr int ncodebooks = 16;
    // static constexpr int ncodebooks = 8;
    // static constexpr int ncodebooks = 4;
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


    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan avg upcast=4           ", kNtrials,
    //     dists_u8_x2.data(), nrows,
    //     (mithral_scan(
    //         codes.data(), luts.data(), dists_u8_x2.data(), nblocks, ncodebooks)));

    RowVector<uint8_t> dists_u8_x2(nrows * 2); // in case decides to upcast

    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan avg upcast=2           ", kNtrials,
        dists_u8_x2.data(), nrows,
        (mithral_scan<M, 2>(
            codes.data(), nblocks, luts.data(), dists_u8_x2.data())));
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan avg upcast=4           ", kNtrials,
        dists_u8_x2.data(), nrows,
        (mithral_scan<M, 4>(
            codes.data(), nblocks, luts.data(), dists_u8_x2.data())));
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan avg upcast=8           ", kNtrials,
        dists_u8_x2.data(), nrows,
        (mithral_scan<M, 8>(
            codes.data(), nblocks, luts.data(), dists_u8_x2.data())));
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan avg upcast=16          ", kNtrials,
        dists_u8_x2.data(), nrows,
        (mithral_scan<M, 16>(
            codes.data(), nblocks, luts.data(), dists_u8_x2.data())));

    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan uint16                 ", kNtrials,
        dists_u16.data(), nrows,
        (bolt_scan<M, false, signed_luts>(
            codes.data(), luts.data(), dists_u16.data(), nblocks)));
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan uint16 safe            ", kNtrials,
        dists_u16_safe.data(), nrows,
        (bolt_scan<M, true, signed_luts>(
            codes.data(), luts.data(), dists_u16_safe.data(), nblocks)));
    // // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan colmajor", kNtrials,
    // //     dists_u16_colmajor.data(), nrows,
    // //     mithral_scan_unpacked_colmajor(codes.data(), nblocks, ncodebooks,
    // //         luts.data(), dists_u16_colmajor.data()));
    // // // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan colmajor tile4 upcast1", kNtrials,
    // // //     dists_u16_colmajor_tile4.data(), nrows,
    // // //     bolt_scan_colmajor_tile4<1>(codes.data(), nblocks, ncodebooks,
    // // //         luts.data(), dists_u16_colmajor_tile4.data()));
    // // // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan colmajor tile4 upcast2", kNtrials,
    // // //     dists_u16_colmajor_tile4.data(), nrows,
    // // //     bolt_scan_colmajor_tile4<2>(codes.data(), nblocks, ncodebooks,
    // // //         luts.data(), dists_u16_colmajor_tile4.data()));
    // // // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan colmajor tile4 upcast4", kNtrials,
    // // //     dists_u16_colmajor_tile4.data(), nrows,
    // // //     bolt_scan_colmajor_tile4<4>(codes.data(), nblocks, ncodebooks,
    // // //         luts.data(), dists_u16_colmajor_tile4.data()));
    // // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan colmajor tile4 upcast1 packed", kNtrials,
    // //     dists_u16_colmajor_tile4.data(), nrows,
    // //     bolt_scan_colmajor_tile4_packed<1>(codes.data(), nblocks, ncodebooks,
    // //         luts.data(), dists_u16_colmajor_tile4.data()));
    // // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan colmajor tile4 upcast2 packed", kNtrials,
    // //     dists_u16_colmajor_tile4.data(), nrows,
    // //     bolt_scan_colmajor_tile4_packed<2>(codes.data(), nblocks, ncodebooks,
    // //         luts.data(), dists_u16_colmajor_tile4.data()));
    // // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan colmajor tile4 upcast4 packed", kNtrials,
    // //     dists_u16_colmajor_tile4.data(), nrows,
    // //     mithral_scan_tile4<4>(codes.data(), nblocks, ncodebooks,
    // //         luts.data(), dists_u16_colmajor_tile4.data()));

    static constexpr int noutputs = 1;
    static constexpr int noutputs_per_block = 1;

    ColMatrix<int8_t> all_luts(noutputs * ncentroids, ncodebooks);
    luts_signed.setRandom();
    luts_signed = luts_signed.array() / ncodebooks; // make max lut value small

    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "mithral scan UpcastEvery=16      ", kNtrials,
    //     dists_u16_colmajor_mithral.data(), nrows * noutputs,
    //     mithral_scan<16>(codes.data(), nrows, ncodebooks,
    //         noutputs, luts_signed.data(), dists_u16_colmajor_mithral.data()));
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "mithral scan tiled UpcastEvery=8     ", kNtrials,
        dists_u16_colmajor_mithral.data(), nrows * noutputs,
        mithral_scan_tiled<8>(codes.data(), nrows, ncodebooks,
            noutputs, luts_signed.data(), dists_u16_colmajor_mithral.data()));
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "mithral scan tiled UpcastEvery=4     ", kNtrials,
        dists_u16_colmajor_mithral.data(), nrows * noutputs,
        mithral_scan_tiled<4>(codes.data(), nrows, ncodebooks,
            noutputs, luts_signed.data(), dists_u16_colmajor_mithral.data()));
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "mithral scan tiled UpcastEvery=2     ", kNtrials,
        dists_u16_colmajor_mithral.data(), nrows * noutputs,
        mithral_scan_tiled<2>(codes.data(), nrows, ncodebooks,
            noutputs, luts_signed.data(), dists_u16_colmajor_mithral.data()));
}

template<int M, bool Safe=false, class dist_t=void>
void _bolt_query(const uint8_t* codes, int nblocks,
    const float* q, int ncols,
    const float* centroids,
    uint8_t* lut_out, dist_t* dists_out)
{
    // TODO use version of lut that requires offsets and scales
    bolt_lut<M, Reductions::DotProd>(q, ncols, centroids, lut_out);
    bolt_scan<M, Safe>(codes, lut_out, dists_out, nblocks);
}

// template<int ncodebooks, bool encode=false>
template<int ncodebooks, bool encode=true>
void _amm_bolt(const float* X, int nrowsX, const float* Q, int nrows, int ncols,
                     const float* centroids,
                     uint8_t* lut_out, uint16_t* dists_out,
                     uint8_t* codes, int nblocks)
{
    static constexpr int nbytes = ncodebooks / 2;
    // in contrast to multisplit, this precomputes encodings and computes
    // new LUTs when a query comes in, instead of the reverse

    if (encode) {
        bolt_encode<nbytes>(X, nrowsX, ncols, centroids, codes);
    }

    auto q_ptr = Q;
    auto dists_ptr = dists_out;
    for (int i = 0; i < nrows; i++) {  // rows in query matrix, not codes
        _bolt_query<nbytes, true>(
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

    // stuff just for encoding; for bolt, we encode the smaller matrix since
    // that's slower
    RowMatrix<float> X(M, D);
    X.setRandom();

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
        _amm_bolt<ncodebooks>(X.data(), M, Q.data(), N, D, centroids.data(),
            lut_out.data(), dists_u16.data(), codes.data(), nblocks));
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

TEST_CASE("amm lut+scan bolt", "[amm][matmul][bolt][profile][old]") {
    std::vector<int> ncodebooks {4, 8, 16, 32, 64};
    for (auto c  : ncodebooks) {
        printf("ncodebooks = %d\n", c);
        _profile_bolt_amm(10000, 512, 10, c);     // cifar10

        // TODO uncomment below

        _profile_bolt_amm(10000, 512, 100, c);    // cifar100
        _profile_bolt_amm(223590, 96, 12, c);       // ecg
//        _profile_bolt_amm(57593, 24, 3, c);       // ecg
        // _profile_bolt_amm(115193, 24, 3, c);      // ecg
        // _profile_bolt_amm(230393, 24, 3, c);      // ecg
        _profile_bolt_amm(49284, 27, 2, c);       // caltech
     }
}


template<class InputT> struct input_type_traits {};
template<> struct input_type_traits<float> {
    using scales_type = float;
    using offsets_type = float;
    const char* name = "f32";
    // using output_type = float;
};
template<> struct input_type_traits<int16_t> {
    using scales_type = uint8_t;
    using offsets_type = int16_t;
    const char* name = "i16";
    // using output_type = int16_t;
};
template<> struct input_type_traits<int8_t> {
    using scales_type = uint8_t;    // doesn't matter; unused
    using offsets_type = uint8_t;  // doesn't matter; unused
    const char* name = "i8";
    // using output_type = int8_t;
};

template<class InputT, class ScaleT, class OffsetT>
void _amm_mithral_unpacked(const InputT* X, int64_t nrows, int ncols,
    const uint32_t* splitdims, const int8_t* all_splitvals,
    const ScaleT* scales, const OffsetT* offsets,
    int ncodebooks, uint8_t* out_enc, int8_t* luts,
    int16_t* out_mat, int out_ncols)
{
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

template<class InputT, class ScaleT, class OffsetT>
void _amm_mithral_packed(const InputT* X, int64_t nrows, int ncols,
    const uint32_t* splitdims, const int8_t* all_splitvals,
    const ScaleT* scales, const OffsetT* offsets,
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

template<class InputT, class ScaleT, class OffsetT>
void _amm_mithral_nolut(const InputT* X, int64_t nrows, int ncols,
    const uint32_t* splitdims, const int8_t* all_splitvals,
    const ScaleT* scales, const OffsetT* offsets,
    int ncodebooks, uint8_t* out_enc, uint8_t* out_enc_packed, int8_t* luts,
    int16_t* out_mat, int out_ncols)
{
    // multisplit_encode_8b_colmajor(
    //     X, nrows, ncols, splitdims, all_splitvals, scales,
    //     offsets, ncodebooks, nsplits_per_codebook, out_enc);
    multisplit_encode_4b_colmajor(
        X, nrows, ncols, splitdims, all_splitvals, scales,
        offsets, ncodebooks, out_enc);
    zip_bolt_colmajor(out_enc, nrows, ncodebooks, out_enc_packed);
    auto nblocks = nrows / 32;
    auto out_ptr = (uint8_t*)out_mat;
    auto lut_ptr = luts;
    static constexpr int UpcastEvery = 8;
    auto out_col_stride = UpcastEvery >= ncodebooks ? nrows : 2 * nrows;
    for (int i = 0; i < out_ncols; i++) {
        mithral_scan<UpcastEvery>(out_enc, nblocks, ncodebooks, (uint8_t*)luts, out_ptr);
        out_ptr += out_col_stride;
        lut_ptr += 16 * ncodebooks;
    }
}

// template<typename T> struct mithral_input_type_traits<> {};
// template<> struct mithral_input_type_traits<float> {
//     using data_t = float;
//     using encoding_scale_t = float;
//     using encoding_offset_t = float;
// };
// template<> struct mithral_input_type_traits<int16_t> {
//     using data_t = int16_t;
//     using encoding_scale_t = uint8_t;  // shifts
//     using encoding_offset_t = int16_t;
// };
// template<> struct mithral_input_type_traits<int8_t> {
//     using data_t = int8_t;
//     using encoding_scale_t = uint8_t;  // unused
//     using encoding_offset_t = uint8_t; // unused
// };

template<class InputT> struct mithral_input_type_traits {};
template<> struct mithral_input_type_traits<float> {
    using encoding_scales_type = float;
    using encoding_offsets_type = float;
    using output_type = int16_t;
};
template<> struct mithral_input_type_traits<int16_t> {
    using encoding_scales_type = uint8_t;
    using encoding_offsets_type = int16_t;
    using output_type = int16_t;
};
template<> struct mithral_input_type_traits<int8_t> {
    using encoding_scales_type = uint8_t;    // doesn't matter; unused
    using encoding_offsets_type = uint8_t;  // doesn't matter; unused
    using output_type = int16_t;
};

// these 3 structs are unused, but are a useful reference
template<class InputT>
struct mithral_encode_params {
    using traits = mithral_input_type_traits<InputT>;
    using scale_t = typename traits::encoding_scales_type;
    using offset_t = typename traits::encoding_offsets_type;

    const InputT* X;
    int nrows;
    int ncols;
    int ncodebooks;
    const uint32_t* splitdims;
    const int8_t* all_splitvals;
    const scale_t* scales;
    const offset_t* offsets;
};
struct mithral_lut_params {
    const float* Q;
    int nrows; /// nrows in Q matrix
    int ncols; /// ncols in Q matrix
    int ncodebooks;
    const float* centroids;
    const int* idxs;
    int nnz_per_centroid;
    float* tmp_lut_f32;
    float out_offset_sum;
    float out_scale;
    uint8_t* out;
};
struct mithral_scan_params {
    const uint8_t* codes;
    int nblocks;
    int ncodebooks;
    const uint8_t* luts;
    uint8_t* dists_out;
};


template<class InputT>
struct mithral_amm {
    using traits = mithral_input_type_traits<InputT>;
    using scale_t = typename traits::encoding_scales_type;
    using offset_t = typename traits::encoding_offsets_type;
    using output_t = typename traits::output_type;
    static constexpr int scan_block_nrows = 32;
    static constexpr int lut_sz = 16;

    // NxD matrix @ DxM matrix
    mithral_amm(int N, int D, int M, int ncodebooks, const float* centroids,
                // for encoding
                const uint32_t* splitdims, const int8_t* splitvals,
                const scale_t* encode_scales, const offset_t* encode_offsets,
                // for lut creation
                const int* idxs, int nnz_per_centroid):
        N(N), D(D), M(M), ncodebooks(ncodebooks), centroids(centroids),
        splitdims(splitdims), splitvals(splitvals),
        encode_scales(encode_scales), encode_offsets(encode_offsets),
        idxs(idxs), nnz_per_centroid(nnz_per_centroid),
        tmp_codes(N, ncodebooks), codes(N, ncodebooks),
        tmp_luts_f32(N, ncodebooks * lut_sz), luts(N, ncodebooks * lut_sz),
        out_mat(N, M)
    {
        luts.setRandom();  // so profiling without LUT creation isn't undefined
    }

    void encode(const InputT* X) {
        // TODO add strides to these funcs so that we can pad number
        // of rows, so scan can rely on nrows being a multiple of 32
        multisplit_encode_4b_colmajor(
            X, N, D, splitdims, splitvals, encode_scales,
            encode_offsets, ncodebooks, tmp_codes.data());
        zip_bolt_colmajor(tmp_codes.data(), N, ncodebooks, codes.data());
    }

    void lut(const float* Q) {
        // printf("nnz_per_centroid=%d ", nnz_per_centroid);
        if (nnz_per_centroid > 0) {
            mithral_lut_sparse(Q, M, D, ncodebooks, centroids,
                idxs, nnz_per_centroid, out_offset_sum, out_scale,
                tmp_luts_f32.data(), luts.data());
        } else {
            // printf("dense lut! ");
            mithral_lut_dense(Q, M, D, ncodebooks, centroids,
                out_offset_sum, out_scale, tmp_luts_f32.data(), luts.data());
        }
    }

    void scan() {
        auto nblocks = N / scan_block_nrows;
        mithral_scan(codes.data(), nblocks, ncodebooks, M,
                     luts.data(), (uint8_t*)out_mat.data());
    }

    // ctor params
    int N;
    int D;
    int M;
    int ncodebooks;
    const float* centroids;
    const uint32_t* splitdims;
    const int8_t* splitvals;
    const scale_t* encode_scales;
    const offset_t* encode_offsets;
    const int* idxs;
    int nnz_per_centroid;

    // storage for intermediate values
    ColMatrix<uint8_t> tmp_codes;
    ColMatrix<uint8_t> codes;
    RowMatrix<float> tmp_luts_f32;
    RowMatrix<uint8_t> luts;

    // outputs
    float out_offset_sum;
    float out_scale;
    ColMatrix<output_t> out_mat;
};

template<class InputT>
struct mithral_amm_task {
    using traits = mithral_input_type_traits<InputT>;
    using scale_t = typename traits::encoding_scales_type;
    using offset_t = typename traits::encoding_offsets_type;
    using output_t = typename traits::output_type;
    static constexpr int scan_block_nrows = 32;
    static constexpr int ncentroids = 16;
    static constexpr int nsplits_per_codebook = 4;
    static constexpr int max_splitvals = 1 << 4;

    mithral_amm_task(int N, int D, int M, int ncodebooks,
                     float lut_work_const):
        N_padded(N % scan_block_nrows == 0 ? N :
            N + (scan_block_nrows - (N % scan_block_nrows))),
        centroids(ncentroids * ncodebooks, D),
        nsplits(ncodebooks * nsplits_per_codebook),
        splitdims(nsplits),
        splitvals(max_splitvals, nsplits),
        encode_scales(nsplits),
        encode_offsets(nsplits),
        nnz_per_centroid(lut_work_const * D / ncodebooks),
        idxs(ncodebooks, nnz_per_centroid),
        amm(N_padded, D, M, ncodebooks, centroids.data(),
            splitdims.data(), splitvals.data(),
            encode_scales.data(), encode_offsets.data(),
            idxs.data(), nnz_per_centroid),
        X(N_padded, D),
        Q(D, M)
    {
        centroids.setRandom();
        splitdims.setRandom();
        for (int i = 0; i < splitdims.size(); i++) {
            splitdims(i) = splitdims(i) % D;
        }
        splitvals.setRandom();
        encode_scales.setRandom();
        encode_offsets.setRandom();

        // randomly initialize idxs, ensuring all are unique and < D
        idxs.setRandom();
        int all_idxs[D];
        for (int i = 0; i < D; i++) {
            all_idxs[i] = i;
        }
        std::random_device rd;
        std::mt19937 g(rd());  // why can't shuffle just create its own...
        for (int c = 0; c < ncodebooks; c++) {  // random sequential idxs
            std::shuffle(all_idxs, all_idxs + D, g);
            std::sort(all_idxs, all_idxs + nnz_per_centroid);
            for (int j = 0; j < nnz_per_centroid; j++) {
                idxs(c, j) = all_idxs[j];
            }
        }

        X.setRandom();
        Q.setRandom();
    }

    void encode() { amm.encode(X.data()); }
    void lut() { amm.lut(Q.data()); }
    void scan() { amm.scan(); }

    void run_matmul(bool create_lut=true) {
        encode();
        if (create_lut) {
            lut();
        }
        scan();
    }

    const ColMatrix<output_t>& output() const { return amm.out_mat; }

    // stuff we pass into the amm object (would be learned during training)
    int N_padded;
    ColMatrix<float> centroids;
    int nsplits;
    RowVector<uint32_t> splitdims;
    ColMatrix<int8_t> splitvals;
    RowVector<scale_t> encode_scales;
    RowVector<offset_t> encode_offsets;
    int nnz_per_centroid;
    RowMatrix<int> idxs;

    // amm object
    mithral_amm<InputT> amm;

    // random data
    ColMatrix<InputT> X;
    ColMatrix<float> Q;
};


// TODO create a struct or something for params because this has gotten
// completely unmanageable
template<class InputT, class ScaleT, class OffsetT>
void _amm_mithral(const InputT* X, const float* W, int64_t nrows, int D, int M,
    const uint32_t* splitdims, const int8_t* all_splitvals,
    const ScaleT* scales, const OffsetT* offsets, const float* centroids,
    int ncodebooks,
    // const int* idxs, const int nnz_per_centroid,
    uint8_t* out_enc, uint8_t* out_enc_packed,
    // float*__restrict__ out_offsets, float& out_offset_sum, float& out_scale,
    // float* out_luts_f32,
    int8_t* out_luts, int16_t* out_mat)
{
    // encode input
    multisplit_encode_4b_colmajor(
        X, nrows, D, splitdims, all_splitvals, scales,
        offsets, ncodebooks, out_enc);
    zip_bolt_colmajor(out_enc, nrows, ncodebooks, out_enc_packed);
    // create luts
    uint8_t* lut_out_ptr = (uint8_t*)out_luts;
    // mithral_lut_sparse(
    //     W, D, ncodebooks, centroids, idxs, nnz_per_centroid,
    //     offsets
    //     luts_f32);


    for (int i = 0; i < M; i++) {
        mithral_lut_v1(W, D, ncodebooks, centroids, lut_out_ptr);
        lut_out_ptr += 16 * ncodebooks;
    }

    // do the amm
    auto nblocks = nrows / 32;
    auto out_ptr = (uint8_t*)out_mat;
    uint8_t* lut_ptr = (uint8_t*)out_luts;
    static constexpr int UpcastEvery = 8;
    auto out_col_stride = UpcastEvery >= ncodebooks ? nrows : 2 * nrows;
    for (int i = 0; i < M; i++) {
        mithral_scan<UpcastEvery>(out_enc, nblocks, ncodebooks, lut_ptr, out_ptr);
        out_ptr += out_col_stride;
        lut_ptr += 16 * ncodebooks;
    }
}


template<class InputT, class ScaleT, class OffsetT>
void _amm_mithral_just_lut(const InputT* X, const float* W, int64_t nrows, int D, int M,
    const uint32_t* splitdims, const int8_t* all_splitvals,
    const ScaleT* scales, const OffsetT* offsets, const float* centroids,
    int ncodebooks, uint8_t* out_enc, uint8_t* out_enc_packed, int8_t* luts,
    int16_t* out_mat)
{
    uint8_t* lut_out_ptr = (uint8_t*)luts;
    for (int i = 0; i < M; i++) {
        mithral_lut_v1(W, D, ncodebooks, centroids, lut_out_ptr);
        lut_out_ptr += 16 * ncodebooks;
    }
}


// template<int Reduction=Reductions::DotProd>
// void bolt_lut(const float* Q, int nrows, int ncols, const float* centroids,
//                   int ncodebooks, uint8_t* out)
// {
//     auto in_ptr = Q;
//     uint8_t* lut_out_ptr = (uint8_t*)out;
//     for (int i = 0; i < nrows; i++) {
//         bolt_lut(in_ptr, nrows, ncols, centroids, ncodebooks, lut_out_ptr);
//         in_ptr += ncols;
//         lut_out_ptr += 16 * ncodebooks;
//     }
// }

template<int ncodebooks>
void _dummy_lut(const float* q, int len, const float* centroids, uint8_t* out)
{
    for (int c = 0; c < ncodebooks; c++) {
        out[c] = 0;
        for (int j = 0; j < len; j++) {
            out[c] += (j % 4) + (j & 0xabcd);
        }
    }
}


TEST_CASE("amm lut old", "[amm][lut][profile][old]") {
    // static constexpr int nrows = 1024*1000;
    // static constexpr int nrows = 128*1000;
    // static constexpr int nrows = 4096;
    // static constexpr int nrows = 24 * 1000;
    // static constexpr int nrows = 24 * 500;
    static constexpr int nrows = 24 * 100;
    // static constexpr int nrows = 24 * 10;
    // static constexpr int nrows = 24;
    // static constexpr int nrows = 6;
    // static constexpr int nrows = 128;
    // static constexpr int64_t nrows = 1;
    // static constexpr int ncols = 24 * 16;               // length of vectors
    // static constexpr int ncols = 12 * 16;               // length of vectors
    // static constexpr int ncols = 1024;               // length of vectors
    static constexpr int ncols = 128;               // length of vectors
    // static constexpr int ncols = 127;               // length of vectors
    // static constexpr int ncols = 32;               // length of vectors
    // static constexpr int ncols = 16;               // length of vectors
    // static constexpr int ncols = 8;               // length of vectors
    // static constexpr int ncols = 1024 * 1024;               // length of vectors
    static constexpr int bits_per_codebook = 4;
    // static constexpr int ncodebooks = 32;
    static constexpr int ncodebooks = 16;
    // static constexpr int ncodebooks = 12;
    // static constexpr int ncodebooks = 8;
    // static constexpr int ncodebooks = 4;
    // static constexpr int ncodebooks = 2;
    static constexpr int ncentroids = (1 << bits_per_codebook);
    // static constexpr int nbytes = ncodebooks / 2;
    // static constexpr int nnz = ncols;  // like 10-15% slower than dense; or
    static constexpr int nnz = ncols * (2.f / ncodebooks);
    printf("nnz: %d\n", nnz);

    ColMatrix<float> centroids(ncodebooks * ncentroids, ncols);
    centroids.setRandom();
    RowMatrix<float> X(nrows, ncols);
    X.setRandom();
    RowMatrix<uint8_t> lut_out(nrows, ncodebooks * ncentroids);
    lut_out.setRandom();

    RowMatrix<float> lut_f32_out(nrows, ncodebooks * ncentroids);
    lut_f32_out.setRandom();

    RowVector<float> offsets(ncodebooks);
    offsets.setRandom();

    RowMatrix<int> idxs(ncodebooks, nnz);
    // idxs.setRandom();
    int all_idxs[ncols];
    for (int i = 0; i < ncols; i++) {
        all_idxs[i] = i;
    }
    std::random_device rd;
    std::mt19937 g(rd());  // why can't shuffle just create its own...
    for (int c = 0; c < ncodebooks; c++) {  // random sequential idxs
        std::shuffle(all_idxs, all_idxs + ncols, g);
        std::sort(all_idxs, all_idxs + nnz);
        for (int j = 0; j < nnz; j++) {
            idxs(c, j) = all_idxs[j];
        }
    }

    // printf("lut_out size: %d\n", (int)lut_out.size());

    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "dummy lut ", kNtrials,
    //     lut_out.data(), lut_out.size(),
    //     _dummy_lut<ncodebooks>(X.data(), ncols,
    //         centroids.data(), lut_out.data()));

    float offset = 0.;
    float scale = 1.;

    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt lut cheating ", kNtrials,
        lut_out.data(), lut_out.size(),
        (bolt_lut<Reductions::DotProd>(X.data(), nrows, ncols,
                centroids.data(), ncodebooks, lut_out.data()) ) );
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt lut          ", kNtrials,
        lut_out.data(), lut_out.size(),
        (bolt_lut<Reductions::DotProd>(X.data(), nrows, ncols, centroids.data(),
            ncodebooks, offsets.data(), scale, lut_out.data()) ) );

    // // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "mithral lut    ", kNtrials,
    // //     lut_out.data(), lut_out.size(),
    // //     (mithral_lut_v1(X.data(), nrows, ncols, ncodebooks,
    // //         centroids.data(), lut_out.data())) );

    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "mithral quant lut 1", kNtrials,
    //     lut_out.data(), lut_out.size(),
    //     (mithral_quantize_luts(lut_f32_out.data(), nrows, ncodebooks,
    //         offset, scale, lut_out.data())));
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "mithral quant lut 2", kNtrials,
    //     lut_out.data(), lut_out.size(),
    //     (mithral_quantize_luts<2>(lut_f32_out.data(), nrows, ncodebooks,
    //         offset, scale, lut_out.data())));
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "mithral quant lut 4", kNtrials,
    //     lut_out.data(), lut_out.size(),
    //     (mithral_quantize_luts<4>(lut_f32_out.data(), nrows, ncodebooks,
    //         offset, scale, lut_out.data())));
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "mithral quant lut 8", kNtrials,
    //     lut_out.data(), lut_out.size(),
    //     (mithral_quantize_luts<8>(lut_f32_out.data(), nrows, ncodebooks,
    //         offset, scale, lut_out.data())));

    //     // lut_out.data(), lut_out.size(),
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "lut dense       2,2", kNtrials,
    //     lut_f32_out.data(), lut_f32_out.size(),
    //     (mithral_lut_dense(X.data(), nrows, ncols, ncodebooks,
    //         centroids.data(), offsets.data(), offset, scale,
    //         lut_f32_out.data(), lut_out.data())) );
    //     // lut_out.data(), lut_out.size(),
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "mithral lut dense ", kNtrials,
        lut_out.data(), lut_out.size(),
        (mithral_lut_dense(X.data(), nrows, ncols, ncodebooks,
            centroids.data(), offset, scale,
            lut_f32_out.data(), lut_out.data())) );
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "mithral lut sparse", kNtrials,
        lut_out.data(), lut_out.size(),
        (mithral_lut_sparse(X.data(), nrows, ncols, ncodebooks,
            centroids.data(), idxs.data(), nnz, offset, scale,
            lut_f32_out.data(), lut_out.data())) );

    // SELF: pick up by putting wrapper funcs in a cpp file; what happens right
    // now is that if we uncomment these calls to the fused func below, the
    // performance gets cut in half (which makes no sense at all); put wrappers
    // in cpp file so they'll just get compiled once and this sort of craziness
    // won't happen



    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "fused   lut f32 2,2", kNtrials,
    //     lut_f32_out.data(), lut_f32_out.size(),
    //     (dense_lut_f32_fused<2,2>(X.data(), nrows, ncols, ncodebooks,
    //         centroids.data(), offsets.data(), offset, scale,
    //         lut_f32_out.data())) );
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "fused   lut f32 2,3", kNtrials,
    //     lut_f32_out.data(), lut_f32_out.size(),
    //     (dense_lut_f32_fused<2,3>(X.data(), nrows, ncols, ncodebooks,
    //         centroids.data(), offsets.data(), offset, scale,
    //         lut_f32_out.data())) );

    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "just quantize lut 1", kNtrials,
        lut_out.data(), lut_out.size(),
        (quantize_luts<1>(lut_f32_out.data(), nrows, ncodebooks,
            offsets.data(), scale, lut_out.data())));
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "just quantize lut 2", kNtrials,
    //     lut_out.data(), lut_out.size(),
    //     (quantize_luts<2>(lut_f32_out.data(), nrows, ncodebooks,
    //         offsets.data(), scale, lut_out.data())));
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "just quantize lut 4", kNtrials,
    //     lut_out.data(), lut_out.size(),
    //     (quantize_luts<4>(lut_f32_out.data(), nrows, ncodebooks,
    //         offsets.data(), scale, lut_out.data())));

    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "sparse lut    1,1", kNtrials,
        lut_f32_out.data(), lut_f32_out.size(),
        (sparse_lut_f32<1,1>(X.data(), nrows, ncols, ncodebooks,
            centroids.data(), idxs.data(), nnz, lut_f32_out.data())) );
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "sparse lut    1,2", kNtrials,
        lut_f32_out.data(), lut_f32_out.size(),
        (sparse_lut_f32<1,2>(X.data(), nrows, ncols, ncodebooks,
            centroids.data(), idxs.data(), nnz, lut_f32_out.data())) );
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "sparse lut    1,3", kNtrials,
        lut_f32_out.data(), lut_f32_out.size(),
        (sparse_lut_f32<1,3>(X.data(), nrows, ncols, ncodebooks,
            centroids.data(), idxs.data(), nnz, lut_f32_out.data())) );
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "sparse lut    2,1", kNtrials,
        lut_f32_out.data(), lut_f32_out.size(),
        (sparse_lut_f32<2,1>(X.data(), nrows, ncols, ncodebooks,
            centroids.data(), idxs.data(), nnz, lut_f32_out.data())) );
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "sparse lut    2,2", kNtrials,
        lut_f32_out.data(), lut_f32_out.size(),
        (sparse_lut_f32<2,2>(X.data(), nrows, ncols, ncodebooks,
            centroids.data(), idxs.data(), nnz, lut_f32_out.data())) );
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "sparse lut    2,3", kNtrials,
        lut_f32_out.data(), lut_f32_out.size(),
        (sparse_lut_f32<2,3>(X.data(), nrows, ncols, ncodebooks,
            centroids.data(), idxs.data(), nnz, lut_f32_out.data())) );


    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "dense lut f32 1,1", kNtrials,
        lut_f32_out.data(), lut_f32_out.size(),
        (dense_lut_f32<1,1>(X.data(), nrows, ncols, ncodebooks,
            centroids.data(), lut_f32_out.data())) );
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "dense lut f32 1,2", kNtrials,
        lut_f32_out.data(), lut_f32_out.size(),
        (dense_lut_f32<1,2>(X.data(), nrows, ncols, ncodebooks,
            centroids.data(), lut_f32_out.data())) );
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "dense lut f32 1,3", kNtrials,
        lut_f32_out.data(), lut_f32_out.size(),
        (dense_lut_f32<1,3>(X.data(), nrows, ncols, ncodebooks,
            centroids.data(), lut_f32_out.data())) );
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "dense lut f32 2,1", kNtrials,
        lut_f32_out.data(), lut_f32_out.size(),
        (dense_lut_f32<2,1>(X.data(), nrows, ncols, ncodebooks,
            centroids.data(), lut_f32_out.data())) );
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "dense lut f32 2,2", kNtrials,
        lut_f32_out.data(), lut_f32_out.size(),
        (dense_lut_f32<2,2>(X.data(), nrows, ncols, ncodebooks,
            centroids.data(), lut_f32_out.data())) );
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "dense lut f32 2,3", kNtrials,
        lut_f32_out.data(), lut_f32_out.size(),
        (dense_lut_f32<2,3>(X.data(), nrows, ncols, ncodebooks,
            centroids.data(), lut_f32_out.data())) );
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "dense lut f32 2,4", kNtrials,
        lut_f32_out.data(), lut_f32_out.size(),
        (dense_lut_f32<2,4>(X.data(), nrows, ncols, ncodebooks,
            centroids.data(), lut_f32_out.data())) );
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "dense lut f32    ", kNtrials,
        lut_f32_out.data(), lut_f32_out.size(),
        (dense_lut_f32(X.data(), nrows, ncols, ncodebooks,
            centroids.data(), lut_f32_out.data())) );
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "dense lut f32 3,1", kNtrials,
    // lut_out.data(), lut_out.size(),
    //     (dense_lut_f32<3,1>(X.data(), nrows, ncols, ncodebooks,
    //         centroids.data(), lut_f32_out.data())) );
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "dense lut f32 3,2", kNtrials,
    // lut_out.data(), lut_out.size(),
    //     (dense_lut_f32<3,2>(X.data(), nrows, ncols, ncodebooks,
    //         centroids.data(), lut_f32_out.data())) );
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "dense lut f32 3,3", kNtrials,
    // lut_out.data(), lut_out.size(),
    //     (dense_lut_f32<3,3>(X.data(), nrows, ncols, ncodebooks,
    //         centroids.data(), lut_f32_out.data())) );
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "dense lut f32 3,4", kNtrials,
    // lut_out.data(), lut_out.size(),
    //     (dense_lut_f32<3,4>(X.data(), nrows, ncols, ncodebooks,
    //         centroids.data(), lut_f32_out.data())) );
}

// template<int UpcastEvery=4>
template<class InputT, class ScaleT, class OffsetT>
void _amm_mithral_tile(const InputT* X, int64_t nrows, int ncols,
    const uint32_t* splitdims, const int8_t* all_splitvals,
    const ScaleT* scales, const OffsetT* offsets,
    int ncodebooks, uint8_t* out_enc, uint8_t* out_enc_packed, int8_t* luts,
    int16_t* out_mat, int out_ncols)
{
    multisplit_encode_4b_colmajor(
        X, nrows, ncols, splitdims, all_splitvals, scales,
        offsets, ncodebooks, out_enc);
    zip4_4b_colmajor(out_enc, nrows, ncodebooks, out_enc_packed);
    // mithral_scan<2>(out_enc_packed, nrows, ncodebooks, out_ncols, luts, out_mat);
    // mithral_scan<UpcastEvery>(out_enc_packed, (int)nrows, ncodebooks, out_ncols, luts, out_mat);
    mithral_scan_tiled<4>(out_enc_packed, (int)nrows, ncodebooks, out_ncols, luts, out_mat);
    // mithral_scan<8>(out_enc_packed, nrows, ncodebooks, out_ncols, luts, out_mat);
    // auto out_ptr = out_mat;
    // auto lut_ptr = luts;
    // for (int i = 0; i < out_ncols; i++) {
    //     mithral_scan<4>(out_enc_packed, nrows, ncodebooks, 1, luts, out_ptr);
    //     out_ptr += nrows;
    //     lut_ptr += 16 * ncodebooks;
    // }
}



template<class InputT, class ScaleT, class OffsetT>
void _amm_mithral_just_enc(const InputT* X, int64_t nrows, int ncols,
    const uint32_t* splitdims, const int8_t* all_splitvals,
    const ScaleT* scales, const OffsetT* offsets,
    int ncodebooks, uint8_t* out_enc, uint8_t* out_enc_packed)
{
    multisplit_encode_4b_colmajor(
        X, nrows, ncols, splitdims, all_splitvals, scales,
        offsets, ncodebooks, out_enc);
    // multisplit_encode_8b_colmajor(
    //     X, nrows, ncols, splitdims, all_splitvals, scales,
    //     offsets, ncodebooks, 4, out_enc);
    zip_bolt_colmajor(out_enc, nrows, ncodebooks, out_enc_packed);
    // zip4_4b_colmajor(out_enc, nrows, ncodebooks, out_enc_packed);
    // memset(out_mat, 42, nrows * out_ncols * sizeof(out_mat[0]));
}

template<class InputT, class ScaleT, class OffsetT>
void _amm_mithral_just_zip4(const InputT* X, int64_t nrows, int ncols,
    const uint32_t* splitdims, const int8_t* all_splitvals,
    const ScaleT* scales, const OffsetT* offsets,
    int ncodebooks, uint8_t* out_enc, uint8_t* out_enc_packed)
{
    zip4_4b_colmajor(out_enc, nrows, ncodebooks, out_enc_packed);
}

template<class InputT, class ScaleT, class OffsetT>
void _amm_mithral_just_zip2(const InputT* X, int64_t nrows, int ncols,
    const uint32_t* splitdims, const int8_t* all_splitvals,
    const ScaleT* scales, const OffsetT* offsets,
    int ncodebooks, uint8_t* out_enc, uint8_t* out_enc_packed)
{
    zip2_4b_colmajor(out_enc, nrows, ncodebooks, out_enc_packed);
}

template<class InputT, class ScaleT, class OffsetT>
void _amm_mithral_just_zip_bolt(const InputT* X, int64_t nrows, int ncols,
    const uint32_t* splitdims, const int8_t* all_splitvals,
    const ScaleT* scales, const OffsetT* offsets,
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

template<class InputT=float>
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

    using scales_t = typename input_type_traits<InputT>::scales_type;
    using offsets_t = typename input_type_traits<InputT>::offsets_type;


    // printf("N, D, M, ncodebooks: %6d, %3d, %3d, %2d, \t", N, D, M, ncodebooks);
    // printf("total_nsplits: %d\n", total_nsplits);

    // create data + info needed for encoding
    ColMatrix<InputT> X(N, D);
    X.setRandom();
    ColMatrix<float> W(D, out_ncols);
    W.setRandom();
    RowVector<uint32_t> splitdims_(total_nsplits);
    splitdims_.setRandom();
    RowVector<uint32_t> splitdims = splitdims_.unaryExpr(
        [=](const int x) { return x % D; });
    ColMatrix<int8_t> all_splitvals(max_ngroups, total_nsplits);
    all_splitvals.setRandom();
    RowVector<scales_t> scales(total_nsplits);
    scales.setRandom();
    RowVector<offsets_t> offsets(total_nsplits);
    offsets.setRandom();
    ColMatrix<uint8_t> codes(N, ncodebooks);
    codes.setRandom();
    codes = codes.array() / 16;
    ColMatrix<uint8_t> codes_packed(N, ncodebooks / 2);
    codes_packed.setRandom();
    ColMatrix<float> centroids(16 * ncodebooks, D);
    centroids.setRandom();

    // create random luts
    ColMatrix<int8_t> luts(ncentroids, ncodebooks * out_ncols);
    luts.setRandom();
    luts = luts.array() / ncodebooks; // make max lut value small

    // storage for overall distances
    ColMatrix<int16_t> out_mat(N, out_ncols);

    // ColMatrix<InputT> X_f(N, D);
    // X_f.setRandom();

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
    std::string dtype_str(input_type_traits<InputT>{}.name);
    // switch(InputT)
    // std::string dtype_str("f32");
    // if (sizeof(InputT) == 1) {
    //     dtype_str = "i8";
    // } else if (sizeof(InputT) == 2) {
    //     dtype_str = "i16";
    // }

    // msg = string_with_format(
    //     "%s amm mithral tile N, D, M, ncodebooks: %6d, %3d, %3d, %2d \t",
    //     dtype_str.c_str(), orig_N, D, M, ncodebooks);
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
    //     out_mat.data(), out_mat.size(),
    //     _amm_mithral_tile(
    //         X.data(), N, D, splitdims.data(), all_splitvals.data(),
    //         scales.data(), offsets.data(), ncodebooks,
    //         codes.data(), codes_packed.data(), luts.data(), out_mat.data(), out_ncols));

    // msg = string_with_format(
    //     "%s amm mithral unpa N, D, M, ncodebooks: %6d, %3d, %3d, %2d \t",
    //     dtype_str.c_str(), orig_N, D, M, ncodebooks);
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
    //     out_mat.data(), out_mat.size(),
    //     _amm_mithral_unpacked(
    //         X.data(), N, D, splitdims.data(), all_splitvals.data(),
    //         scales.data(), offsets.data(), ncodebooks,
    //         codes.data(), luts.data(), out_mat.data(), out_ncols));

    // msg = string_with_format(
    //     "%s amm mithral pack N, D, M, ncodebooks: %6d, %3d, %3d, %2d \t",
    //     dtype_str.c_str(), orig_N, D, M, ncodebooks);
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
    //     out_mat.data(), out_mat.size(),
    //     _amm_mithral_packed(
    //         X.data(), N, D, splitdims.data(), all_splitvals.data(),
    //         scales.data(), offsets.data(), ncodebooks,
    //         codes.data(), codes_packed.data(), luts.data(), out_mat.data(), out_ncols));

    msg = string_with_format(
        "%3s amm mithral      N, D, M, ncodebooks: %6d, %3d, %3d, %2d \t",
        dtype_str.c_str(), orig_N, D, M, ncodebooks);
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        out_mat.data(), out_mat.size(),
        _amm_mithral(
            X.data(), W.data(), N, D, out_ncols,
            splitdims.data(), all_splitvals.data(),
            scales.data(), offsets.data(),
            centroids.data(), ncodebooks,
            codes.data(), codes_packed.data(), luts.data(), out_mat.data()));

    msg = string_with_format(
        "%3s amm mithral lut  N, D, M, ncodebooks: %6d, %3d, %3d, %2d \t",
        dtype_str.c_str(), orig_N, D, M, ncodebooks);
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        out_mat.data(), out_mat.size(),
        _amm_mithral_just_lut(
            X.data(), W.data(), N, D, out_ncols,
            splitdims.data(), all_splitvals.data(),
            scales.data(), offsets.data(),
            centroids.data(), ncodebooks,
            codes.data(), codes_packed.data(), luts.data(), out_mat.data()));

    // X.setRandom();
    msg = string_with_format(
        "%3s amm mithral enc  N, D, M, ncodebooks: %6d, %3d, %3d, %2d \t",
        dtype_str.c_str(), orig_N, D, M, ncodebooks);
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        codes_packed.data(), codes_packed.size(),
        _amm_mithral_just_enc(
            X.data(), N, D, splitdims.data(), all_splitvals.data(),
            scales.data(), offsets.data(), ncodebooks,
            codes.data(), codes_packed.data()));

    // msg = string_with_format(
    //     "%s amm mithral zip4 N, D, M, ncodebooks: %6d, %3d, %3d, %2d \t",
    //     dtype_str.c_str(), orig_N, D, M, ncodebooks);
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
    //     codes_packed.data(), codes_packed.size(),
    //     _amm_mithral_just_zip4(
    //         X.data(), N, D, splitdims.data(), all_splitvals.data(),
    //         scales.data(), offsets.data(), ncodebooks,
    //         codes.data(), codes_packed.data()));
    // msg = string_with_format(
    //     "%s amm mithral zip2 N, D, M, ncodebooks: %6d, %3d, %3d, %2d \t",
    //     dtype_str.c_str(), orig_N, D, M, ncodebooks);
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
    //     codes_packed.data(), codes_packed.size(),
    //     _amm_mithral_just_zip2(
    //         X.data(), N, D, splitdims.data(), all_splitvals.data(),
    //         scales.data(), offsets.data(), ncodebooks,
    //         codes.data(), codes_packed.data()));
    msg = string_with_format(
        "%s amm mithral zipb N, D, M, ncodebooks: %6d, %3d, %3d, %2d \t",
        dtype_str.c_str(), orig_N, D, M, ncodebooks);
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        codes_packed.data(), codes_packed.size(),
        _amm_mithral_just_zip_bolt(
            X.data(), N, D, splitdims.data(), all_splitvals.data(),
            scales.data(), offsets.data(), ncodebooks,
            codes.data(), codes_packed.data()));
}

template<class InputT=float>
void _profile_mithral(const char* dset_name, uint32_t N, uint32_t D, uint32_t M,
                      int ncodebooks, float lut_work_const=2) {
    mithral_amm_task<InputT> task(N, D, M, ncodebooks, lut_work_const);
    // mithral_amm_task<InputT> task_dense(N, D, M, ncodebooks, -1);

    printf("---- ncodebooks=%d\n", ncodebooks);

    std::string msg;
    auto dtype_str = input_type_traits<InputT>{}.name;

    // auto fmt = "%7s, %3s, %22s, N, D, M, C, lut_work_coef:\t"
    //         "%6d, %3d, %3d, %2d, %.1f\t";
    auto fmt_as_cppstring = string_with_format(
        "%s, %-3s, %%-22s, N D M C lut_work_coef:,"
        "%6d, %3d, %3d, %2d, %%4.1f,\t", dset_name, dtype_str,
        N, D, M, ncodebooks);
    auto fmt = fmt_as_cppstring.c_str();
    // printf("fmt string: %s\n", fmt.c_str());
    // fmt = string_with_format()

    // overall AMM time
    msg = string_with_format(fmt, "amm mithral nolut", -1.f);
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        task.output().data(), task.output().size(),
        task.run_matmul(false));

    msg = string_with_format(fmt, "amm mithral sparselut", lut_work_const);
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        task.output().data(), task.output().size(),
        task.run_matmul(true));
    // msg = string_with_format( // time if lut already created
    //     "%3s amm mithral nolut      N, D, M, C, lut_work_coef:\t"
    //         "%6d, %3d, %3d, %2d, %.1f\t",
    //     dtype_str, N, D, M, ncodebooks, -1.f);
        // "%3s amm mithral nolut      N, D, M, C:\t\t\t\t\t"
        //     "%6d, %3d, %3d, %2d\t\t",
        // dtype_str, N, D, M, ncodebooks);
    // msg = string_with_format(fmt, dset_name, "amm mithral nolut",
    //     dtype_str, N, D, M, ncodebooks, -1.f);

    // using dense centroids, which slows down LUT creation
    auto orig_nnz_per_centroid = task.amm.nnz_per_centroid;
    task.amm.nnz_per_centroid = -1;
    // msg = string_with_format(  // still overall AMM time
    //     "%3s amm mithral denselut   N, D, M, C, lut_work_coef:\t"
    //         "%6d, %3d, %3d, %2d, %.1f\t",
    //     dtype_str, N, D, M, ncodebooks, -1.f);
    // msg = string_with_format(fmt, dset_name, "amm mithral denselut",
    //     dtype_str, N, D, M, ncodebooks, -1.f);
    msg = string_with_format(fmt, "amm mithral denselut", -1.f);
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        task.output().data(), task.output().size(),
        task.run_matmul(true));
    // msg = string_with_format(  // just time to create lut with dense centroids
    //     "%3s amm mithral lut dense  N, D, M, C, lut_work_coef:\t"
    //         "%6d, %3d, %3d, %2d, %.1f\t",
    //     dtype_str, N, D, M, ncodebooks, -1.f);
    // msg = string_with_format(fmt, dset_name, "amm mithral lut dense",
    //     dtype_str, N, D, M, ncodebooks, -1.f);
    msg = string_with_format(fmt, "amm mithral lut dense", -1.f);
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        task.output().data(), task.output().size(),
        task.lut());
    task.amm.nnz_per_centroid = orig_nnz_per_centroid;

    // back to sparse centroids
    // msg = string_with_format(
    //     "%3s amm mithral lut sparse N, D, M, C, lut_work_coef:\t"
    //         "%6d, %3d, %3d, %2d, %.1f\t",
    //     dtype_str, N, D, M, ncodebooks, lut_work_const);
    // msg = string_with_format(fmt, dset_name, "amm mithral lut sparse",
    //     dtype_str, N, D, M, ncodebooks, lut_work_const);
    msg = string_with_format(fmt, "amm mithral lut sparse", lut_work_const);
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        task.output().data(), task.output().size(),
        task.lut());
    // msg = string_with_format(
    //     "%3s amm mithral enc        N, D, M, C, lut_work_coef:\t"
    //         "%6d, %3d, %3d, %2d, %.1f\t",
    //     dtype_str, N, D, M, ncodebooks, lut_work_const);
    // msg = string_with_format(fmt, dset_name, "amm mithral enc",
    //     dtype_str, N, D, M, ncodebooks, lut_work_const);
    msg = string_with_format(fmt, "amm mithral enc", -1.f);
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        task.output().data(), task.output().size(),
        task.encode());
    // msg = string_with_format(
    //     "%3s amm mithral scan       N, D, M, C, lut_work_coef:\t"
    //         "%6d, %3d, %3d, %2d, %.1f\t",
    //     dtype_str, N, D, M, ncodebooks, lut_work_const);
    // msg = string_with_format(fmt, dset_name, "amm mithral scan",
    //     dtype_str, N, D, M, ncodebooks, lut_work_const);
    msg = string_with_format(fmt, "amm mithral scan", -1.f);
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        task.output().data(), task.output().size(),
        task.scan());
}

template<class InputT=float>
void _profile_mithral(const MatmulTaskShape& shape, std::vector<int> ncodebooks,
                      float lut_work_const=2)
{
    auto dtype_name = input_type_traits<InputT>{}.name;
    printf("------------------------ %s %s\n", shape.name, dtype_name);
    for (auto c : ncodebooks) {
        _profile_mithral<InputT>(
            shape.name, shape.N, shape.D, shape.M, c, lut_work_const);
    }
}

// TEST_CASE("amm actual matmul multisplit", "[amm][multisplit][mithral][matmul][profile]") {
//     // _profile_multisplit(128 * 1000, 64, 32, 4);
//      std::vector<int> ncodebooks {4, 8, 16, 32, 64};
//     // std::vector<int> ncodebooks {4, 64};
// //    std::vector<int> ncodebooks {4, 16, 64};
//     // std::vector<int> ncodebooks {64};
//     // std::vector<int> ncodebooks {4};
//     for (auto c  : ncodebooks) {
//         printf("ncodebooks = %d\n", c);
//         _profile_multisplit<float>(10000, 512, 10, c);  // cifar10
//         _profile_multisplit<float>(10000, 512, 100, c); // cifar100
// //        223590
//         _profile_multisplit<float>(223590, 96, 12, c);  // ecg
//         _profile_multisplit<int16_t>(223590, 96, 12, c);  // ecg
// //        _profile_multisplit<float>(57593, 24, 3, c);  // ecg
// //        _profile_multisplit<int16_t>(57593, 24, 3, c);  // ecg
//         // _profile_multisplit(115193, 24, 3, c);       // ecg
//         // _profile_multisplit(230393, 24, 3, c);       // ecg
//         _profile_multisplit<float>(49284, 27, 2, c);   // caltech
//         _profile_multisplit<int8_t>(49284, 27, 2, c);   // caltech
//     }
// }

TEST_CASE("amm mithral old", "[amm][multisplit][mithral][matmul][profile][old]") {
     std::vector<int> ncodebooks {4, 8, 16, 32, 64};

     float lut_work_const = 2;
     _profile_mithral<int8_t>(kCaltechTaskShape, ncodebooks, lut_work_const);
     _profile_mithral(kCaltechTaskShape, ncodebooks, lut_work_const);
     _profile_mithral(kCifar10TaskShape, ncodebooks, lut_work_const);
     _profile_mithral(kCifar100TaskShape, ncodebooks, lut_work_const);
     _profile_mithral(kUcrTaskShape, ncodebooks, lut_work_const);

     // std::vector<int> ncodebooks {4};
    // for (auto c  : ncodebooks) {
    //     printf("ncodebooks = %d\n", c);
    //     // printf("----\n");
    //     _profile_mithral<float>(10000, 512, 10, c);  // cifar10
    //     // printf("----\n");
    //     _profile_mithral<float>(10000, 512, 100, c); // cifar100
    //     // printf("----\n");
    //     _profile_mithral<float>(223590, 96, 12, c);  // ecg
    //     // printf("----\n");
    //     _profile_mithral<int16_t>(223590, 96, 12, c);  // ecg
    //     // printf("----\n");
    //     _profile_mithral<float>(49284, 27, 2, c);   // caltech
    //     // printf("----\n");
    //     _profile_mithral<int8_t>(49284, 27, 2, c);   // caltech
    // }
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

void _profile_matmul(const char* dset_name, uint32_t N, uint32_t D, uint32_t M)
{
    using MatrixT = ColMatrix<float>;

    // create random data
    MatrixT X(N, D);
    X.setRandom();
    MatrixT W(D, M);
    W.setRandom();

    // create output matrix to avoid malloc
    MatrixT out(N, M);
    out.setRandom();

    std::string msg;
    auto fmt_as_cppstring = string_with_format(
        "%s, %%-25s, N D M:,   %6d, %3d, %3d,\t\t\t", dset_name, N, D, M);
    auto fmt = fmt_as_cppstring.c_str();

    // time it
    {
        // std::string msg = string_with_format(
        //     "blas matmul               N, D, M:    %6d, %3d, %3d \t\t\t",
        //     orig_N, orig_D, orig_M);
        msg = string_with_format(fmt, "blas matmul");
        REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
            out.data(), out.size(),
            _run_matmul(X, W, out));
    }
    {
        // std::string msg = string_with_format(
        //     "our  matmul               N, D, M:    %6d, %3d, %3d \t\t\t",
        //     orig_N, orig_D, orig_M);
        msg = string_with_format(fmt, "our matmul");
        REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
            out.data(), out.size(),
            _run_our_matmul(X, W, out));
    }
}

template<class MatrixT>
void _run_matmul_fixedW(const MatrixT& X,
                        const MatrixT& W0, MatrixT& sketch_out,
                        const MatrixT& W1, MatrixT& out)
{
   sketch_out.noalias() = X * W0;
   out.noalias() = sketch_out * W1;
}

template<class MatrixT>
void _run_our_matmul_fixedW(const MatrixT& X,
                            const MatrixT& W0, MatrixT& sketch_out,
                            const MatrixT& W1, MatrixT& out)
{
    auto N = (int)X.rows();
    auto D = (int)X.cols();
    auto M = (int)out.cols();
    auto d = (int)W0.cols();
    sgemm_colmajor(X.data(), W0.data(), N, D, d, sketch_out.data());
    sgemm_colmajor(sketch_out.data(), W1.data(), N, d, M, out.data());
}

void _profile_sketch_matmul_fixedW(const char* dset_name, uint32_t N,
    uint32_t D, uint32_t M, uint32_t d)
{
    using MatrixT = ColMatrix<float>;

    // create random matrices of the appropriate sizes
    MatrixT X(N, D); X.setRandom();
    MatrixT W0(D, d); W0.setRandom();
    MatrixT W1(d, M); W1.setRandom();

    // create output matrices to avoid malloc
    MatrixT sketch_out(N, d);
    sketch_out.setRandom();
    MatrixT out(N, M);
    out.setRandom();

    std::string msg;
    auto fmt_as_cppstring = string_with_format(
        "%s, %%-25s, N D M d:, %6d, %3d, %3d, %3d,\t", dset_name, N, D, M, d);
    auto fmt = fmt_as_cppstring.c_str();

    // time it
    // msg = string_with_format("blas sketch fixedW matmul N, D, M, d: %6d, %3d, %3d, %3d \t",
    //     N, D, M, d);
    msg = string_with_format(fmt, "blas sketch fixedW matmul");
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        out.data(), out.size(),
        _run_matmul_fixedW(X, W0, sketch_out, W1, out));
    // msg = string_with_format("our  sketch fixedW matmul N, D, M, d: %6d, %3d, %3d, %3d \t",
    //     N, D, M, d);
    msg = string_with_format(fmt, "our sketch fixedW matmul");
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        out.data(), out.size(),
        _run_our_matmul_fixedW(X, W0, sketch_out, W1, out));
}

template<class MatrixT>
void _run_sketch_matmul(const MatrixT& X, const MatrixT& W, const MatrixT& S,
                        MatrixT& X_sketched, MatrixT& W_sketched, MatrixT& out)
{
   X_sketched.noalias() = X * S;
   W_sketched.noalias() = S.transpose() * W;
   out.noalias() = X_sketched * W_sketched;
}

template<class MatrixT>
void _run_our_sketch_matmul(
    const MatrixT& X, const MatrixT& W, const MatrixT& S, const MatrixT& St,
    MatrixT& X_sketched, MatrixT& W_sketched, MatrixT& out)
{
    auto N = (int)X.rows();
    auto D = (int)X.cols();
    auto M = (int)W.cols();
    auto d = (int)S.cols();
    sgemm_colmajor(X.data(), S.data(), N, D, d, X_sketched.data());
    sgemm_colmajor(St.data(), W.data(), d, D, M, W_sketched.data());
    sgemm_colmajor(X_sketched.data(), W_sketched.data(), N, d, M, out.data());
}

void _profile_sketch_matmul(const char* dset_name, uint32_t N, uint32_t D,
    uint32_t M, uint32_t d)
{
    using MatrixT = ColMatrix<float>;

    // create random matrices of the appropriate sizes
    MatrixT X(N, D); X.setRandom();
    MatrixT W(D, M); W.setRandom();
    MatrixT S(D, d); S.setRandom();
    MatrixT St(S.transpose());

    // create output matrices to avoid malloc
    MatrixT sketch_X(N, d);
    sketch_X.setRandom();
    MatrixT sketch_W(d, M);
    sketch_W.setRandom();
    MatrixT out(N, M);
    out.setRandom();

    std::string msg;
    auto fmt_as_cppstring = string_with_format(
        "%s, %%-25s, N D M d:, %6d, %3d, %3d, %3d,\t", dset_name, N, D, M, d);
    auto fmt = fmt_as_cppstring.c_str();

    // time it
    // msg = string_with_format("blas sketch matmul        N, D, M, d: %6d, %3d, %3d, %3d \t",
    //     N, D, M, d);
    msg = string_with_format(fmt, "blas sketch matmul");
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        out.data(), out.size(),
        _run_sketch_matmul(X, W, S, sketch_X, sketch_W, out));
    // msg = string_with_format("our  sketch matmul        N, D, M, d: %6d, %3d, %3d, %3d \t",
    //     N, D, M, d);
    msg = string_with_format(fmt, "our sketch matmul");
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        out.data(), out.size(),
        _run_our_sketch_matmul(X, W, S, St, sketch_X, sketch_W, out));
}

// template<bool UseOurGemm, bool SketchW, class SketchT, class ColMatrixT>
template<bool SketchW, class SketchT, class ColMatrixT>
void _run_fancy_sketch_matmul(
    const SketchT& sketch, const ColMatrixT& X, const ColMatrixT& Wt,
    ColMatrixT& X_sketched, ColMatrixT& Wt_sketched, ColMatrixT& out)
{
    // printf("\nsketching X\n");
    sketch(X, X_sketched);
    if (SketchW) {
        // printf("sketching W\n");
        // sketch(W, W_sketched, true /*transpose*/);
        sketch(Wt, Wt_sketched);
    }
    // no option to use our gemm here since it would require transposing W
    out.noalias() = X_sketched * Wt_sketched.transpose();
    // if (UseOurGemm) {
    //     auto N = (int)X_sketched.rows();
    //     auto d = (int)X_sketched.cols();
    //     auto M = (int)W_sketched.cols();
    //     sgemm_colmajor(X_sketched.data(), W_sketched.data(),
    //                    N, d, M, out.data());
    // } else {
    //     out.noalias() = X_sketched * W_sketched;
    // }
}

void _profile_osnap(const char* dset_name, uint32_t N, uint32_t D,
                    uint32_t M, uint32_t d, int nsketches)
{
    using MatrixT = ColMatrix<float>;
    MatrixT X(N, D); X.setRandom();
    MatrixT Wt(M, D); Wt.setRandom();

    // create output matrices to avoid malloc
    MatrixT sketch_X(N, d);
    sketch_X.setRandom();
    // MatrixT sketch_Wt(d, M);
    MatrixT sketch_Wt(M, d);
    sketch_Wt.setRandom();
    MatrixT out(N, M);
    out.setRandom();

    std::string msg;
    auto fmt_as_cppstring = string_with_format(
        "%s, %%-25s, N D M d, s:, %6d, %3d, %3d, %3d, %2d\t",
        dset_name, N, D, M, d, nsketches);
    auto fmt = fmt_as_cppstring.c_str();

    auto sketch = OsnapSketch(D, d, nsketches);

    //
    // sketching W takes almost no time, even for cifar100, so just
    // report fixedW resuls to err on side of optimism and halve the
    // execution time
    //
    // msg = string_with_format(fmt, "osnap");
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
    //     out.data(), out.size(),
    //     (_run_fancy_sketch_matmul<true>(
    //         sketch, X, Wt, sketch_X, sketch_Wt, out)));

    msg = string_with_format(fmt, "osnap fixedW");
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        out.data(), out.size(),
        (_run_fancy_sketch_matmul<false>(
            sketch, X, Wt, sketch_X, sketch_Wt, out)));
}

void _profile_osnap(std::vector<int> dvals, std::vector<int> nsketches,
                    MatmulTaskShape shape)
{
    // assert(false); // are we in release mode?
    auto N = shape.N;
    auto D = shape.D;
    auto M = shape.M;
    printf("------------------------ %s\n", shape.name);
    for (auto d : dvals) {
        for (auto s : nsketches) {
            if (s > d) { continue; }
            _profile_osnap(shape.name, N, D, M, d, s);
        }
    }
}

// void _profile_matmul_methods(std::vector<int> dvals, int N, int D, int M) {
void _profile_matmul_methods(std::vector<int> dvals, MatmulTaskShape shape) {
    auto N = shape.N;
    auto D = shape.D;
    auto M = shape.M;
    printf("------------------------ %s\n", shape.name);
    for (auto d : dvals) {
        _profile_sketch_matmul(shape.name, N, D, M, d);
        _profile_sketch_matmul_fixedW(shape.name, N, D, M, d);
    }
    _profile_matmul(shape.name, N, D, M);
}

TEST_CASE("amm linear approx matmul old", "[amm][matmul][linear][profile][old]") {
    int N, D, M;
    // std::vector<int> dvals {2, 4, 6, 8, 12, 16, 24, 32, 48, 64};
    std::vector<int> dvals {2, 4, 8, 16, 32, 64, 128}; // TODO uncomment above

    _profile_matmul_methods(dvals, kCaltechTaskShape);
    _profile_matmul_methods(dvals, kCifar10TaskShape);
    _profile_matmul_methods(dvals, kCifar100TaskShape);
    _profile_matmul_methods(dvals, kUcrTaskShape);
}

TEST_CASE("amm osnap old", "[amm][matmul][osnap][linear][profile][old]") {
    int N, D, M;
    // std::vector<int> dvals {2, 4, 6, 8, 12, 16, 24, 32, 48, 64};
    std::vector<int> dvals {2, 4, 8, 16, 32, 64, 128}; // TODO uncomment above
    // std::vector<int> dvals {2}; // TODO uncomment above
    std::vector<int> nsketches {1, 2, 4};
    // std::vector<int> nsketches {1};

    _profile_osnap(dvals, nsketches, kCaltechTaskShape);
    _profile_osnap(dvals, nsketches, kCifar10TaskShape);
    _profile_osnap(dvals, nsketches, kCifar100TaskShape);
    _profile_osnap(dvals, nsketches, kUcrTaskShape);
}

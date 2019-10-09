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
    #include "src/quantize/multisplit.hpp"
    #include "src/utils/debug_utils.hpp"
    #include "src/utils/eigen_utils.hpp"
    #include "src/utils/timing_utils.hpp"
    #include "src/utils/memory.hpp"
    #include "test/testing_utils/testing_utils.hpp"
#else
    #include "catch.hpp"
    #include "bolt.hpp"
    #include "multisplit.hpp"
    #include "debug_utils.hpp"
    #include "eigen_utils.hpp"
    #include "timing_utils.hpp"
    #include "testing_utils.hpp"
    #include "memory.hpp"
#endif

static constexpr int kNreps = 5;
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

TEST_CASE("amm profile split encode", "[amm][split][profile]") {
    static const int N = 128 * 1000;
    static const uint32_t D = 64;
    static const int ncodebooks = 4;
    static const int nsplits_per_codebook = 4;
    static const int total_nsplits = ncodebooks * nsplits_per_codebook;

    ColMatrix<float> X(N, D);
    X.setRandom();
    RowVector<uint32_t> splitdims_(total_nsplits);
    splitdims_.setRandom();
    RowVector<uint32_t> splitdims = splitdims_.unaryExpr(
        [](const int x) { return x % D; });
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

    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "split encode", kNtrials,
        out.data(), out.size(),
        split_encode_8b_colmajor(
            X.data(), N, D, splitdims.data(), splitvals.data(), scales.data(),
            offsets.data(), ncodebooks, nsplits_per_codebook, out.data()));
}

TEST_CASE("amm profile multisplit encode", "[amm][multisplit][profile]") {
    static const int N = 128 * 1000;
    static const uint32_t D = 64;
    static const int ncodebooks = 4;
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
    RowVector<float> scales(total_nsplits);
    scales.setRandom();
    RowVector<float> offsets(total_nsplits);
    offsets.setRandom();
    ColMatrix<uint8_t> out(N, ncodebooks);

    // multisplit_encode_8b_colmajor(
    //     X.data(), N, D, splitdims.data(), all_splitvals.data(), scales.data(),
    //     offsets.data(), ncodebooks, nsplits_per_codebook, out.data());
    // printf("sum of out: %d\n", out.sum());

    // printf("out.size(): %lu\n", out.size());
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "multisplit encode", kNtrials,
        out.data(), out.size(),
        multisplit_encode_8b_colmajor(
            X.data(), N, D, splitdims.data(), all_splitvals.data(),
            scales.data(), offsets.data(), ncodebooks, nsplits_per_codebook,
            out.data()));
    }

TEST_CASE("bolt scan speed with colmajor", "[amm][bolt][mcq][profile]") {
    static constexpr int nblocks = 16 * 1000;
    static constexpr int nrows = nblocks * 32;
    static constexpr int ncodebooks = 16;
    static constexpr int ncentroids = 16;
    static constexpr int M = ncodebooks / 2;

    // create random codes from in [0, 15]
    ColMatrix<uint8_t> codes(nrows, ncodebooks);
    codes.setRandom();
    codes = codes.array() / ncentroids;

    // create random luts
    ColMatrix<uint8_t> luts(ncentroids, ncodebooks);
    luts.setRandom();
    luts = luts.array() / ncodebooks; // make max lut value small
    // ColMatrix<int8_t> luts_i8(ncentroids, ncodebooks);
    // luts_i8.setRandom();
    // luts_i8 = luts_i8.array() / ncodebooks;

    // do the scan to compute the distances
    RowVector<uint8_t> dists_u8(nrows);
    RowVector<uint16_t> dists_u16(nrows);
    RowVector<uint16_t> dists_u16_safe(nrows);
    RowVector<int16_t> dists_u16_colmajor(nrows);
    RowVector<int16_t> dists_u16_colmajor_tile4(nrows);

    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan uint8", kNtrials,
        dists_u8.data(), nrows,
        bolt_scan<M>(codes.data(), luts.data(), dists_u8.data(), nblocks));
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan uint16", kNtrials,
        dists_u16.data(), nrows,
        bolt_scan<M>(codes.data(), luts.data(), dists_u16.data(), nblocks));
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan uint16 safe", kNtrials,
        dists_u16_safe.data(), nrows,
        bolt_scan<M>(codes.data(), luts.data(), dists_u16_safe.data(), nblocks));
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan colmajor", kNtrials,
        dists_u16_colmajor.data(), nrows,
        bolt_scan_unpacked_colmajor(codes.data(), nblocks, ncodebooks,
            luts.data(), dists_u16_colmajor.data()));
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan colmajor tile4 upcast1", kNtrials,
    //     dists_u16_colmajor_tile4.data(), nrows,
    //     bolt_scan_colmajor_tile4<1>(codes.data(), nblocks, ncodebooks,
    //         luts.data(), dists_u16_colmajor_tile4.data()));
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan colmajor tile4 upcast2", kNtrials,
    //     dists_u16_colmajor_tile4.data(), nrows,
    //     bolt_scan_colmajor_tile4<2>(codes.data(), nblocks, ncodebooks,
    //         luts.data(), dists_u16_colmajor_tile4.data()));
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan colmajor tile4 upcast4", kNtrials,
    //     dists_u16_colmajor_tile4.data(), nrows,
    //     bolt_scan_colmajor_tile4<4>(codes.data(), nblocks, ncodebooks,
    //         luts.data(), dists_u16_colmajor_tile4.data()));
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan colmajor tile4 upcast1 packed", kNtrials,
        dists_u16_colmajor_tile4.data(), nrows,
        _bolt_scan_colmajor_tile4_packed<1>(codes.data(), nblocks, ncodebooks,
            luts.data(), dists_u16_colmajor_tile4.data()));
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan colmajor tile4 upcast2 packed", kNtrials,
        dists_u16_colmajor_tile4.data(), nrows,
        _bolt_scan_colmajor_tile4_packed<2>(codes.data(), nblocks, ncodebooks,
            luts.data(), dists_u16_colmajor_tile4.data()));
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan colmajor tile4 upcast4 packed", kNtrials,
        dists_u16_colmajor_tile4.data(), nrows,
        _bolt_scan_colmajor_tile4_packed<4>(codes.data(), nblocks, ncodebooks,
            luts.data(), dists_u16_colmajor_tile4.data()));
}

void _amm_multisplit(const float* X, int64_t nrows, int ncols,
    const uint32_t* splitdims, const int8_t* all_splitvals,
    const float* scales, const float* offsets,
    int ncodebooks, int nsplits_per_codebook,
    uint8_t* out_enc, int8_t* luts, int16_t* out_mat, int out_ncols)
{
    multisplit_encode_8b_colmajor(
        X, nrows, ncols, splitdims, all_splitvals, scales,
        offsets, ncodebooks, nsplits_per_codebook, out_enc);

    auto nblocks = nrows / 32;
    auto out_ptr = out_mat;
    auto lut_ptr = luts;
    for (int i = 0; i < out_ncols; i++) {
        _bolt_scan_colmajor_tile4_packed<4>(out_enc, nblocks, ncodebooks,
            luts, out_ptr);
        out_ptr += nrows;
        lut_ptr += 16 * ncodebooks;
    }
}

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

    if (N % 32 > 0) {
        N -= (N % 32);  // TODO better way of dealing with block size needs
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

    // create random luts
    ColMatrix<int8_t> luts(ncentroids, ncodebooks * out_ncols);
    luts.setRandom();
    luts = luts.array() / ncodebooks; // make max lut value small

    // storage for overall distances
    ColMatrix<int16_t> out_mat(N, out_ncols);

    std::string msg = string_with_format(
        "amm multisplit N, D, M, ncodebooks: %6d, %3d, %3d, %2d \t",
        N, D, M, ncodebooks);
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        out_mat.data(), out_mat.size(),
        _amm_multisplit(
            X.data(), N, D, splitdims.data(), all_splitvals.data(),
            scales.data(), offsets.data(), ncodebooks, nsplits_per_codebook,
            codes.data(), luts.data(), out_mat.data(), out_ncols));
}

template<class MatrixT1, class MatrixT2, class MatrixT3>
void _run_matmul(const MatrixT1& X, const MatrixT2& Q, MatrixT3& out) {
   out.noalias() = X * Q;
}

void _profile_matmul(uint32_t N, uint32_t D, uint32_t M) {
    // using MatrixT = ColMatrix<float>;
    using MatrixT = ColMatrix<float>; // faster for small batches, else slower

    // create random data
    MatrixT X(N, D);
    X.setRandom();
    MatrixT W(D, M);
    W.setRandom();

    // create output matrix to avoid malloc
    MatrixT out(N, M);

    // printf("N, D, M: %6d, %3d, %3d, \t", N, D, M);

    // time it
    std::string msg = string_with_format("matmul N, D, M: %6d, %3d, %3d \t",
        N, D, M);
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        out.data(), out.size(),
        _run_matmul(X, W, out));
}

TEST_CASE("amm enc+scan multisplit", "[amm][multisplit][profile]") {
    // _profile_multisplit(128 * 1000, 64, 32, 4);
    std::vector<int> ncodebooks {4, 8, 16, 32, 64};
    // std::vector<int> ncodebooks {4};
    for (auto c  : ncodebooks) {
        printf("ncodebooks = %d\n", c);
        // _profile_multisplit(128 * 1000, 64, 32, c);
        _profile_multisplit(10000, 512, 10, c);     // cifar10
        _profile_multisplit(10000, 512, 100, c);    // cifar100
        _profile_multisplit(57593, 24, 3, c);       // ecg
        _profile_multisplit(115193, 24, 3, c);      // ecg
        _profile_multisplit(230393, 24, 3, c);      // ecg
        _profile_multisplit(49284, 27, 2, c);       // caltech
    }
}

TEST_CASE("amm exact matmul", "[amm][exact][profile]") {
    int N, M;
    std::vector<int> dvals {2, 4, 6, 8, 12, 16, 24, 27, 32, 48, 64};
    
    N = 10000; M = 10;          // cifar10
    for (auto d : dvals) {
        _profile_matmul(N, d, M);
    }
    _profile_matmul(N, 512, M);
    
    N = 10000; M = 100;         // cifar100
    for (auto d : dvals) {
        _profile_matmul(N, d, M);
    }
    _profile_matmul(N, 512, M);
    
    M = 3;                      // ecg
    std::vector<int> ecg_nvals {57593, 115193, 230393};
    for (auto n : ecg_nvals) {
        for (auto d : dvals) {
            _profile_matmul(n, d, M);
        }
    }
    
    N = 49284; M = 2;           // caltech
    for (auto d : dvals) {
        _profile_matmul(N, d, M);
    }
    
//    _profile_matmul(10000, 512, 100);   // cifar100
//    _profile_matmul(57593, 24, 3);      // ecg
//    _profile_matmul(115193, 24, 3);     // ecg
//    _profile_matmul(230393, 24, 3);     // ecg
//    _profile_matmul(49284, 27, 2);      // caltech
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

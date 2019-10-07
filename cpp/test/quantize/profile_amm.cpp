//
//  profile_amm.cpp
//  Bolt
//
//  Created by DB on 10/7/19.
//  Copyright © 2019 D Blalock. All rights reserved.
//

#include <stdio.h>
#include <string>

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

static constexpr int kNreps = 2;
// static constexpr int kNreps = 1;
static constexpr int kNtrials = 2;

// TEST_CASE("amm profile smoketest", "[amm][profile]") {
//     static constexpr int64_t nrows_enc = 10*1000;   // number of rows to encode
//     static constexpr int ncols = 128;               // length of vectors
//     static constexpr int bits_per_codebook = 4;
//     static constexpr int ncentroids = (1 << bits_per_codebook);
//     static constexpr int nbytes = 8;

//     static constexpr int nrows = nrows_enc;

//     ColMatrix<float> centroids(ncentroids, ncols);
//     centroids.setRandom();
//     RowMatrix<float> X(nrows, ncols);
//     X.setRandom();
//     RowMatrix<uint8_t> encoding_out(nrows, nbytes);

//     REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt encode", kNtrials,
//     encoding_out.data(), nrows,
//         bolt_encode<nbytes>(X.data(), nrows, ncols, centroids.data(),
//                             encoding_out.data()));
// }

TEST_CASE("amm profile split encode", "[amm][split][profile]") {
    static const int N = 128;
    static const int D = 8;
    static const int ncodebooks = 4;
    static const int nsplits_per_codebook = 4;
    static const int total_nsplits = ncodebooks * nsplits_per_codebook;

    ColMatrix<float> X(N, D);
    X.setRandom();
    RowVector<int> splitdims_(total_nsplits);
    splitdims_.setRandom();
    RowVector<int> splitdims = splitdims_.unaryExpr(
        [](const int x) { return x % D; });
    RowVector<int8_t> splitvals(total_nsplits);
    splitvals.setRandom();
    RowVector<float> scales(total_nsplits);
    scales.setRandom();
    RowVector<float> offsets(total_nsplits);
    scales.setRandom();
    ColMatrix<uint8_t> out(N, total_nsplits);

    split_encode_8b_colmajor(
        X.data(), N, D, splitdims.data(), splitvals.data(), scales.data(),
        offsets.data(), ncodebooks, nsplits_per_codebook, out.data());
    printf("sum of out: %d\n", out.sum());
}

TEST_CASE("amm profile multisplit encode", "[amm][split][profile]") {
    static const int N = 128;
    static const int D = 8;
    static const int ncodebooks = 4;
    static const int nsplits_per_codebook = 4;
    static const int total_nsplits = ncodebooks * nsplits_per_codebook;
    static const int group_id_nbits = 4;
    static const int max_ngroups = 1 << group_id_nbits;

    ColMatrix<float> X(N, D);
    X.setRandom();
    RowVector<int> splitdims_(total_nsplits);
    splitdims_.setRandom();
    RowVector<int> splitdims = splitdims_.unaryExpr(
        [](const int x) { return x % D; });
    ColMatrix<int8_t> all_splitvals(max_ngroups, total_nsplits);
    all_splitvals.setRandom();
    RowVector<float> scales(total_nsplits);
    scales.setRandom();
    RowVector<float> offsets(total_nsplits);
    scales.setRandom();
    ColMatrix<uint8_t> out(N, total_nsplits);

    multisplit_encode_8b_colmajor(
        X.data(), N, D, splitdims.data(), all_splitvals.data(), scales.data(),
        offsets.data(), ncodebooks, nsplits_per_codebook, out.data());
    printf("sum of out: %d\n", out.sum());
}

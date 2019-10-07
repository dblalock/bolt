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

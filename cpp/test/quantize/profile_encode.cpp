//
//  profile_vq_encoding.cpp
//  Bolt
//
//  Created by DB on 12/10/19.
//  Copyright © 2019 D Blalock. All rights reserved.
//

#ifdef BLAZE
    #include "src/quantize/multisplit.hpp"
    #include "test/quantize/amm_common.hpp"
#else
    #include "multisplit.hpp"
    #include "amm_common.hpp"
#endif

TEST_CASE("vq encode timing", "[amm][encode][profile]") {
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

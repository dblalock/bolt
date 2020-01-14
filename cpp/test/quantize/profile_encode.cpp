//
//  profile_vq_encoding.cpp
//  Bolt
//
//  Created by DB on 12/10/19.
//  Copyright © 2019 D Blalock. All rights reserved.
//

#ifdef BLAZE
    #include "src/quantize/bolt.hpp"
    #include "src/quantize/multisplit.hpp"
    #include "src/quantize/product_quantize.hpp"
    #include "test/quantize/amm_common.hpp"
#else
    #include "bolt.hpp"
    #include "multisplit.hpp"
    #include "product_quantize.hpp"
    #include "amm_common.hpp"
#endif

// void _profile_encode(int N, int D, int ncodebooks) {
void _profile_encode(int N, int D, int nbytes) {
    static constexpr int nsplits_per_codebook = 4;
    static constexpr int group_id_nbits = 4;
    static constexpr int max_ngroups = 1 << group_id_nbits;
    int ncodebooks = 2 * nbytes;
    int total_nsplits = ncodebooks * nsplits_per_codebook;
    int ncodebooks_pq = nbytes;

    // shared
    ColMatrix<float> X(N, D); X.setRandom();
    ColMatrix<uint8_t> out(N, ncodebooks); out.setRandom();

    // mithral-specific (it's a lot of stuff to handle different dtypes)
    RowVector<uint32_t> splitdims_(total_nsplits);
    splitdims_.setRandom();
    RowVector<uint32_t> splitdims = splitdims_.unaryExpr(
        [=](const uint32_t x) { return x % D; });
    // RowVector<uint32_t> splitdims(total_nsplits); splitdims.setZero(); // TODO rm
    ColMatrix<int8_t> all_splitvals(max_ngroups, total_nsplits);
    all_splitvals.setRandom();
    RowVector<float> scales(MAX(D, total_nsplits)); // v2 needs D of these
    scales.setRandom();
    RowVector<float> offsets(MAX(D, total_nsplits)); // v2 needs D of these
    offsets.setRandom();
    ColMatrix<int8_t> X_i8(N, D); X_i8.setRandom();
    ColMatrix<int16_t> X_i16(N, D); X_i16.setRandom();
    RowVector<int16_t> offsets_i16(total_nsplits); offsets_i16.setRandom();
    RowVector<uint8_t> shifts(total_nsplits); shifts.setRandom();

    std::string msg;
    auto fmt_as_cppstring = string_with_format(
        "%%-22s, N D C B:, %7d, %3d, %%3d, %2d,\t", N, D, nbytes);
    auto fmt = fmt_as_cppstring.c_str();

    // ------------------------ mithral
    msg = string_with_format(fmt, "mithral encode f32", ncodebooks);
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        out.data(), out.size(),
        mithral_encode(
            X.data(), N, D, splitdims.data(), all_splitvals.data(),
            scales.data(), offsets.data(), ncodebooks, out.data()));

    msg = string_with_format(fmt, "mithral encode i8", ncodebooks);
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        out.data(), out.size(),
        mithral_encode(
            X_i8.data(), N, D, splitdims.data(), all_splitvals.data(),
            ncodebooks, out.data()));

    msg = string_with_format(fmt, "mithral encode i16", ncodebooks);
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        out.data(), out.size(),
        mithral_encode(
            X_i16.data(), N, D, splitdims.data(), all_splitvals.data(),
            shifts.data(), offsets_i16.data(), ncodebooks, out.data()));


    if (D < ncodebooks) { return; } // subsequent methods can't handle this

    // ------------------------ bolt
    ColMatrix<float> centroids(16, D); centroids.setRandom();

    msg = string_with_format(fmt, "bolt encode", ncodebooks);
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        out.data(), out.size(),
        bolt_encode(X.data(), N, D, ncodebooks, centroids.data(), out.data()));

    // ------------------------ pq
    ColMatrix<float> centroids256(256, D); centroids256.setRandom();

    msg = string_with_format(fmt, "pq encode", ncodebooks_pq);
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        out.data(), out.size(),
        pq_encode_8b(X.data(), N, D, ncodebooks,
                     centroids256.data(), out.data()) );

    // ------------------------ opq
    ColMatrix<float> R(D, D); R.setRandom();
    RowMatrix<float> X_tmp(N, D);
    msg = string_with_format(fmt, "opq encode", ncodebooks_pq);
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        out.data(), out.size(),
        opq_encode_8b(X, ncodebooks, centroids256.data(),
                      R, X_tmp, out.data()) );
}

TEST_CASE("vq encode timing", "[amm][encode][profile]") {
    static constexpr int nrows = 1 << 14;

    // NOTE: python code needs header to not be included in the csv file
    printf("algo, _0, N, D, C, B, "
           "_1, latency0, _2, latency1, _3, latency2, _4, "
           "latency3, _5, latency4, _6\n");

    std::vector<int> all_ncols = {32, 64, 128, 256, 512, 1024};
    std::vector<int> all_nbytes = {8, 16, 32};
    for (auto ncols : all_ncols) {
        for (auto b : all_nbytes) {
            _profile_encode(nrows, ncols, b);
        }
    }
}

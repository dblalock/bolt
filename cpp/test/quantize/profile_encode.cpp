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

void _profile_encode(int N, int D, int ncodebooks) {
    static constexpr int nsplits_per_codebook = 4;
    static constexpr int group_id_nbits = 4;
    static constexpr int max_ngroups = 1 << group_id_nbits;
    int total_nsplits = ncodebooks * nsplits_per_codebook;

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
        "%%-22s, N D C:, %7d, %3d, %2d,\t", N, D, ncodebooks);
    auto fmt = fmt_as_cppstring.c_str();

    // ------------------------ mithral
    msg = string_with_format(fmt, "mithral encode 4b f32");
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        out.data(), out.size(),
        mithral_encode(
            X.data(), N, D, splitdims.data(), all_splitvals.data(),
            scales.data(), offsets.data(), ncodebooks, out.data()));

    msg = string_with_format(fmt, "mithral encode 4b i8");
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        out.data(), out.size(),
        mithral_encode(
            X_i8.data(), N, D, splitdims.data(), all_splitvals.data(),
            ncodebooks, out.data()));

    msg = string_with_format(fmt, "mithral encode i16");
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        out.data(), out.size(),
        mithral_encode(
            X_i16.data(), N, D, splitdims.data(), all_splitvals.data(),
            shifts.data(), offsets_i16.data(), ncodebooks, out.data()));


    if (D < ncodebooks) { return; } // subsequent methods can't handle this

    // ------------------------ bolt
    ColMatrix<float> centroids(16, D); centroids.setRandom();

    msg = string_with_format(fmt, "bolt encode");
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        out.data(), out.size(),
        bolt_encode(X.data(), N, D, ncodebooks, centroids.data(), out.data()));

    // ------------------------ pq
    ColMatrix<float> centroids256(256, D); centroids256.setRandom();
    // ColMatrix<float> out_f32(N, ncodebooks); out_f32.setRandom();

    msg = string_with_format(fmt, "pq encode");
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        out.data(), out.size(),
        pq_encode_8b(X.data(), N, D, ncodebooks,
                     centroids256.data(), out.data()) );

    // ------------------------ opq
    ColMatrix<float> R(D, D); R.setRandom();
    RowMatrix<float> X_tmp(N, D);
    msg = string_with_format(fmt, "opq encode");
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        out.data(), out.size(),
        opq_encode_8b(X, ncodebooks, centroids256.data(),
                      R, X_tmp, out.data()) );
}

TEST_CASE("vq encode timing", "[amm][encode][profile]") {
    static constexpr int nrows = 1 << 14;

    printf("algo, _0, N, D, C, "
           "_1, latency0, _2, latency1, _3, latency2, _4\n");
    
    std::vector<int> all_ncols = {32, 64, 128, 256, 512, 1024};
//    std::vector<int> all_ncols = {512, 1024};
    std::vector<int> all_ncodebooks = {16, 32, 64};
    for (auto ncols : all_ncols) {
        for (auto c : all_ncodebooks) {
            // fails at OPQ 102400, 512, 16; pq works...
            _profile_encode(nrows, ncols, c);
        }
    }
}

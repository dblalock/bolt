//
//  profile_scan.cpp
//  Bolt
//
//  Created by DB on 12/10/19.
//  Copyright © 2019 D Blalock. All rights reserved.
//

#ifdef BLAZE
    #include "src/quantize/product_quantize.hpp"
    #include "test/quantize/amm_common.hpp"
#else
    #include "amm_common.hpp"
    #include "product_quantize.hpp"
#endif

void _profile_scan(int nrows, int ncodebooks) {
    int nblocks = nrows / 32;

    // for mithral / bolt
    ColMatrix<uint8_t> codes16(nrows, ncodebooks); codes16.setRandom();
    codes16 = codes16.unaryExpr(
        [=](const uint8_t x) { return (uint8_t)(x % 16); });
    ColMatrix<uint8_t> luts16(16, ncodebooks); luts16.setRandom();
    luts16 = luts16.array() / ncodebooks; // make max lut value small
    RowVector<uint8_t> dists_u8(nrows);
    RowVector<uint16_t> dists_u16(nrows);

    // for pq / opq
    ColMatrix<uint8_t> codes256(nrows, ncodebooks);
    codes256.setRandom();
    codes256 = codes256.unaryExpr(
        [=](const uint8_t x) { return (uint8_t)(x % 256); });
    ColMatrix<float> luts_f32(256, ncodebooks); luts_f32.setRandom();
    RowVector<float> dists_f32(nrows); dists_f32.setRandom();

    std::string msg;
    auto fmt_as_cppstring = string_with_format(
        "%%-22s, N C:, %7d, %2d,\t", nrows, ncodebooks);
    auto fmt = fmt_as_cppstring.c_str();

    // ------------------------ mithral
    RowVector<uint8_t> dists_u8_x2(nrows * 2); // to handle upcast
    msg = string_with_format(fmt, "mithral scan upcast4");
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        dists_u8_x2.data(), nrows,
        (mithral_scan<4>(codes16.data(), nblocks, ncodebooks,
                      luts16.data(), dists_u8_x2.data())));
    msg = string_with_format(fmt, "mithral scan upcast8");
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        dists_u8_x2.data(), nrows,
        (mithral_scan(codes16.data(), nblocks, ncodebooks,
                      luts16.data(), dists_u8_x2.data())));
    msg = string_with_format(fmt, "mithral scan upcast16");
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        dists_u8_x2.data(), nrows,
        (mithral_scan<16>(codes16.data(), nblocks, ncodebooks,
                      luts16.data(), dists_u8_x2.data())));

    // ------------------------ bolt
    static constexpr bool signed_luts = true; // bolt uses signed for dotprod
    msg = string_with_format(fmt, "bolt scan uint8");
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        dists_u8.data(), nrows,
        (bolt_scan<false, signed_luts>(codes16.data(), nblocks, ncodebooks,
                                       luts16.data(), dists_u8.data())));
    msg = string_with_format(fmt, "bolt scan uint16");
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        dists_u16.data(), nrows,
        (bolt_scan<false, signed_luts>(codes16.data(), nblocks, ncodebooks,
                                       luts16.data(), dists_u16.data())));
    msg = string_with_format(fmt, "bolt scan safe uint16");
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        dists_u16.data(), nrows,
        (bolt_scan<true, signed_luts>(codes16.data(), nblocks, ncodebooks,
                                      luts16.data(), dists_u16.data())));

    // ------------------------ pq (same as opq)
    msg = string_with_format(fmt, "pq scan");
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        dists_f32.data(), nrows,
        pq_scan_8b(codes256.data(), nrows, ncodebooks, luts_f32.data(),
                   dists_f32.data()));

}


TEST_CASE("vq scan timing", "[amm][scan][profile]") {
    // static constexpr int nrows = 100 * 1024;
    static constexpr int nrows = 1000 * 1000;

    printf("algo, _0, N, C, _1, latency0, _2, latency1, _3, latency2, _4\n");

    std::vector<int> all_ncodebooks {4, 8, 16, 32, 64};
    for (auto c : all_ncodebooks) {
        _profile_scan(nrows, c);
    }
}

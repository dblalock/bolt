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

void _profile_scan(int nrows, int ncols, int ncodebooks) {
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
        "%%-22s, N D C:, %7d, %3d, %2d\t",
        nrows, ncols, ncodebooks);
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
    static constexpr int nrows = 100 * 1024;

    printf("algo, _0, N, C, _1, latency0, _2, latency1, _3, latency2, _4\n");

    // std::vector<int> all_ncols {32, 64, 128, 256, 512, 1024};
    // std::vector<int> all_ncodebooks {16, 32, 64};
    std::vector<int> all_ncols {128};
    std::vector<int> all_ncodebooks {16};

    for (auto ncols : all_ncols) {
        for (auto c : all_ncodebooks) {
            _profile_scan(nrows, ncols, c);
        }
    }

    // static constexpr int nblocks = 64 * 1000;
    // // static constexpr int nblocks = 2;
    // // static constexpr int nblocks = 256;
    // static constexpr int nrows = nblocks * 32;
    // // static constexpr int ncodebooks = 64;
    // // static constexpr int ncodebooks = 32;
    // // static constexpr int ncodebooks = 24;
    // static constexpr int ncodebooks = 16;
    // // static constexpr int ncodebooks = 8;
    // // static constexpr int ncodebooks = 4;
    // static constexpr int ncentroids = 16;
    // static constexpr int M = ncodebooks / 2;

    // // create random codes from in [0, 15]
    // ColMatrix<uint8_t> codes(nrows, ncodebooks);
    // codes.setRandom();
    // codes = codes.unaryExpr(
    //     [=](const uint8_t x) { return (uint8_t)(x % ncentroids); });

    // // create random luts
    // ColMatrix<uint8_t> luts(ncentroids, ncodebooks);
    // luts.setRandom();
    // luts = luts.array() / ncodebooks; // make max lut value small

    // ColMatrix<int8_t> luts_signed(ncentroids, ncodebooks);
    // luts_signed.setRandom();
    // luts_signed = luts_signed.array() / ncodebooks; // make max lut value small

    // // ColMatrix<int8_t> luts_i8(ncentroids, ncodebooks);
    // // luts_i8.setRandom();
    // // luts_i8 = luts_i8.array() / ncodebooks;

    // // do the scan to compute the distances
    // RowVector<uint8_t> dists_u8(nrows);
    // RowVector<uint16_t> dists_u16(nrows);
    // RowVector<uint16_t> dists_u16_safe(nrows);
    // // RowVector<int16_t> dists_u16_colmajor(nrows);
    // // RowVector<int16_t> dists_u16_colmajor_tile4(nrows);
    // RowVector<int16_t> dists_u16_colmajor_mithral(nrows);

    // // static constexpr int signed_luts = true;
    // static constexpr int signed_luts = false;

    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan uint8                  ", kNtrials,
    //     dists_u8.data(), nrows,
    //     (bolt_scan<M, true, signed_luts>(
    //         codes.data(), luts.data(), dists_u8.data(), nblocks)));

    // // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan avg upcast=4           ", kNtrials,
    // //     dists_u8_x2.data(), nrows,
    // //     (mithral_scan(
    // //         codes.data(), luts.data(), dists_u8_x2.data(), nblocks, ncodebooks)));

    // RowVector<uint8_t> dists_u8_x2(nrows * 2); // in case decides to upcast

    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan avg upcast=2           ", kNtrials,
    //     dists_u8_x2.data(), nrows,
    //     (mithral_scan<M, 2>(
    //         codes.data(), nblocks, luts.data(), dists_u8_x2.data())));
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan avg upcast=4           ", kNtrials,
    //     dists_u8_x2.data(), nrows,
    //     (mithral_scan<M, 4>(
    //         codes.data(), nblocks, luts.data(), dists_u8_x2.data())));
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan avg upcast=8           ", kNtrials,
    //     dists_u8_x2.data(), nrows,
    //     (mithral_scan<M, 8>(
    //         codes.data(), nblocks, luts.data(), dists_u8_x2.data())));
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan avg upcast=16          ", kNtrials,
    //     dists_u8_x2.data(), nrows,
    //     (mithral_scan<M, 16>(
    //         codes.data(), nblocks, luts.data(), dists_u8_x2.data())));

    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan uint16                 ", kNtrials,
    //     dists_u16.data(), nrows,
    //     (bolt_scan<M, false, signed_luts>(
    //         codes.data(), luts.data(), dists_u16.data(), nblocks)));
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan uint16 safe            ", kNtrials,
    //     dists_u16_safe.data(), nrows,
    //     (bolt_scan<M, true, signed_luts>(
    //         codes.data(), luts.data(), dists_u16_safe.data(), nblocks)));
}
